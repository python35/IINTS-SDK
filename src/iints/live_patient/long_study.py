from __future__ import annotations

import csv
import importlib
import json
import math
import platform
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

import iints
from iints.analysis.study_analysis import analyze_study_directory
from iints.api.base_algorithm import InsulinAlgorithm
from iints.live_patient.runtime import (
    DailyEventTemplate,
    RuntimeScenarioProfile,
    _load_algorithm_instance_silent,
    get_runtime_scenario_profile,
)
from iints.utils.run_io import write_json
from iints.validation import compute_run_metrics

WEEKDAY_ORDER: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_BUILTIN_ALGORITHM_REGISTRY: dict[str, tuple[str, str]] = {
    "Clinical Baseline": ("iints.core.algorithms.clinical_baseline", "ClinicalBaselineAlgorithm"),
    "PID Controller": ("iints.core.algorithms.pid_controller", "PIDController"),
    "Standard Pump": ("iints.core.algorithms.standard_pump_algo", "StandardPumpAlgorithm"),
    "Correction Bolus": ("iints.core.algorithms.correction_bolus", "CorrectionBolus"),
}

_COMPLETE_STATUSES = {"completed", "skipped_existing"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    cleaned: list[str] = []
    previous_dash = False
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-") or "item"


@dataclass(frozen=True)
class EdgeLongStudyConfig:
    duration_days: int
    algorithms: tuple[str, ...]
    week_schedule: dict[str, str]
    seeds: tuple[int, ...]
    output_dir: str = "/media/pi/usb_ssd/results/long_study"
    scratch_dir: str = "/tmp/iints_edge_long_study"
    snapshot_interval_hours: int = 24
    patient_config: str = "default_patient"
    patient_model_type: str = "auto"
    time_step_minutes: int = 5
    duration_minutes_per_day: int = 1440
    start_weekday: str = "monday"

    def weekday_for_day(self, day_index: int) -> str:
        start_index = WEEKDAY_ORDER.index(self.start_weekday)
        return WEEKDAY_ORDER[(start_index + day_index) % len(WEEKDAY_ORDER)]

    def profile_for_day(self, day_index: int) -> str:
        return self.week_schedule[self.weekday_for_day(day_index)]

    def snapshot_every_days(self) -> int:
        return max(1, int(math.ceil(self.snapshot_interval_hours / 24.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_days": self.duration_days,
            "algorithms": list(self.algorithms),
            "week_schedule": dict(self.week_schedule),
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
            "scratch_dir": self.scratch_dir,
            "snapshot_interval_hours": self.snapshot_interval_hours,
            "patient_config": self.patient_config,
            "patient_model_type": self.patient_model_type,
            "time_step_minutes": self.time_step_minutes,
            "duration_minutes_per_day": self.duration_minutes_per_day,
            "start_weekday": self.start_weekday,
        }


@dataclass(frozen=True)
class LongStudyAlgorithmSpec:
    raw_value: str
    slug: str
    display_name: str
    source_type: str
    source_ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "raw_value": self.raw_value,
            "slug": self.slug,
            "display_name": self.display_name,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class LongStudySnapshotResult:
    archive: str
    input_dir: str
    output_dir: str
    generated_at_utc: str

    def to_dict(self) -> dict[str, str]:
        return {
            "archive": self.archive,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "generated_at_utc": self.generated_at_utc,
        }


class LongStudyConfigError(ValueError):
    pass


class EdgeLongStudyExecutionError(RuntimeError):
    pass


def render_edge_long_study_config_template(
    *,
    output_dir: str = "/media/pi/usb_ssd/results/long_study",
    scratch_dir: str = "/tmp/iints_edge_long_study",
    algorithms: list[str] | None = None,
) -> str:
    payload = {
        "duration_days": 14,
        "algorithms": algorithms
        or [
            "algorithms/example_algorithm.py",
            "PID Controller",
            "Clinical Baseline",
        ],
        "week_schedule": {
            "monday": "school_day",
            "tuesday": "sport_day",
            "wednesday": "school_day",
            "thursday": "bad_carb_count",
            "friday": "school_day",
            "saturday": "sport_day",
            "sunday": "relaxed_day",
        },
        "seeds": [1, 2, 3, 4, 5],
        "snapshot_interval_hours": 24,
        "output_dir": output_dir,
        "scratch_dir": scratch_dir,
        "patient_config": "default_patient",
        "patient_model_type": "auto",
        "time_step_minutes": 5,
        "duration_minutes_per_day": 1440,
        "start_weekday": "monday",
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _normalize_week_schedule(raw_value: Any) -> dict[str, str]:
    if not isinstance(raw_value, dict):
        raise LongStudyConfigError("`week_schedule` must be a mapping from weekday to scenario profile.")
    schedule: dict[str, str] = {}
    for weekday in WEEKDAY_ORDER:
        value = raw_value.get(weekday)
        if not isinstance(value, str) or not value.strip():
            raise LongStudyConfigError(f"`week_schedule.{weekday}` must be a non-empty scenario profile name.")
        schedule[weekday] = value.strip()
    return schedule


def load_edge_long_study_config(config_path: Path) -> EdgeLongStudyConfig:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise LongStudyConfigError(f"Could not read long-study config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise LongStudyConfigError("Long-study config must be a top-level mapping.")

    duration_days = int(payload.get("duration_days", 0) or 0)
    if duration_days <= 0:
        raise LongStudyConfigError("`duration_days` must be greater than zero.")

    raw_algorithms = payload.get("algorithms")
    if not isinstance(raw_algorithms, list) or not raw_algorithms:
        raise LongStudyConfigError("`algorithms` must be a non-empty list of algorithm paths or built-in labels.")
    algorithms = tuple(str(item).strip() for item in raw_algorithms if str(item).strip())
    if not algorithms:
        raise LongStudyConfigError("`algorithms` must contain at least one non-empty entry.")

    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise LongStudyConfigError("`seeds` must be a non-empty list of integers.")
    try:
        seeds = tuple(int(item) for item in raw_seeds)
    except (TypeError, ValueError) as exc:
        raise LongStudyConfigError("`seeds` must contain only integers.") from exc

    snapshot_interval_hours = int(payload.get("snapshot_interval_hours", 24) or 24)
    if snapshot_interval_hours <= 0:
        raise LongStudyConfigError("`snapshot_interval_hours` must be greater than zero.")

    time_step_minutes = int(payload.get("time_step_minutes", 5) or 5)
    if time_step_minutes <= 0:
        raise LongStudyConfigError("`time_step_minutes` must be greater than zero.")

    duration_minutes_per_day = int(payload.get("duration_minutes_per_day", 1440) or 1440)
    if duration_minutes_per_day <= 0:
        raise LongStudyConfigError("`duration_minutes_per_day` must be greater than zero.")

    start_weekday = str(payload.get("start_weekday", "monday") or "monday").strip().lower()
    if start_weekday not in WEEKDAY_ORDER:
        raise LongStudyConfigError(f"`start_weekday` must be one of: {', '.join(WEEKDAY_ORDER)}.")

    schedule = _normalize_week_schedule(payload.get("week_schedule", {}))
    for profile_name in schedule.values():
        get_runtime_scenario_profile(profile_name)

    output_dir = str(payload.get("output_dir", "/media/pi/usb_ssd/results/long_study") or "/media/pi/usb_ssd/results/long_study").strip()
    if not output_dir:
        raise LongStudyConfigError("`output_dir` must be a non-empty path.")

    scratch_dir = str(payload.get("scratch_dir", "/tmp/iints_edge_long_study") or "/tmp/iints_edge_long_study").strip()
    if not scratch_dir:
        raise LongStudyConfigError("`scratch_dir` must be a non-empty path.")

    patient_config = str(payload.get("patient_config", "default_patient") or "default_patient").strip()
    patient_model_type = str(payload.get("patient_model_type", "auto") or "auto").strip()

    return EdgeLongStudyConfig(
        duration_days=duration_days,
        algorithms=algorithms,
        week_schedule=schedule,
        seeds=seeds,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        snapshot_interval_hours=snapshot_interval_hours,
        patient_config=patient_config,
        patient_model_type=patient_model_type,
        time_step_minutes=time_step_minutes,
        duration_minutes_per_day=duration_minutes_per_day,
        start_weekday=start_weekday,
    )


def _resolve_config_path(config_path: str | Path, project_dir: str | Path | None = None) -> Path:
    candidate = Path(config_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if project_dir is not None:
        project_root = Path(project_dir).expanduser().resolve()
        project_candidate = project_root / candidate
        if project_candidate.exists():
            return project_candidate.resolve()
    return candidate.resolve()


def _resolve_output_dir(path_value: str, project_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()


def _resolve_scratch_dir(path_value: str, project_dir: Path, output_root: Path) -> Path:
    path = Path(path_value).expanduser()
    base = path.resolve() if path.is_absolute() else (project_dir / path).resolve()
    return base / output_root.name


def _load_builtin_algorithm(display_name: str) -> InsulinAlgorithm:
    if display_name not in _BUILTIN_ALGORITHM_REGISTRY:
        raise LongStudyConfigError(
            f"Unknown built-in algorithm '{display_name}'. Supported built-ins: {', '.join(sorted(_BUILTIN_ALGORITHM_REGISTRY))}."
        )
    module_name, class_name = _BUILTIN_ALGORITHM_REGISTRY[display_name]
    module = importlib.import_module(module_name)
    algorithm_class = getattr(module, class_name)
    instance = algorithm_class()
    if not isinstance(instance, InsulinAlgorithm):
        raise LongStudyConfigError(f"Built-in algorithm '{display_name}' is not a valid InsulinAlgorithm.")
    return instance


def _resolve_algorithm_specs(algorithms: tuple[str, ...], project_dir: Path) -> list[LongStudyAlgorithmSpec]:
    specs: list[LongStudyAlgorithmSpec] = []
    seen_slugs: set[str] = set()
    for raw_value in algorithms:
        expanded = Path(raw_value).expanduser()
        candidate = expanded if expanded.is_absolute() else (project_dir / expanded)
        if candidate.is_file():
            instance = _load_algorithm_instance_silent(candidate)
            display_name = instance.get_algorithm_metadata().name
            slug = _slugify(display_name or candidate.stem)
            source_type = "path"
            source_ref = str(candidate.resolve())
        else:
            instance = _load_builtin_algorithm(raw_value)
            display_name = instance.get_algorithm_metadata().name
            slug = _slugify(display_name)
            source_type = "builtin"
            source_ref = raw_value
        unique_slug = slug
        suffix = 2
        while unique_slug in seen_slugs:
            unique_slug = f"{slug}-{suffix}"
            suffix += 1
        seen_slugs.add(unique_slug)
        specs.append(
            LongStudyAlgorithmSpec(
                raw_value=raw_value,
                slug=unique_slug,
                display_name=display_name,
                source_type=source_type,
                source_ref=source_ref,
            )
        )
    return specs


def _instantiate_algorithm(spec: LongStudyAlgorithmSpec) -> InsulinAlgorithm:
    if spec.source_type == "path":
        return _load_algorithm_instance_silent(Path(spec.source_ref))
    return _load_builtin_algorithm(spec.source_ref)


def _template_to_event_payload(template: DailyEventTemplate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "start_time": int(template.minute_of_day),
        "event_type": template.event_type,
        "value": float(template.value),
        "duration": int(template.duration),
        "absorption_delay_minutes": int(template.absorption_delay_minutes),
    }
    if template.reported_value is not None:
        payload["reported_value"] = float(template.reported_value)
    if template.label:
        payload["label"] = template.label
    if template.isf is not None:
        payload["isf"] = float(template.isf)
    if template.icr is not None:
        payload["icr"] = float(template.icr)
    if template.basal_rate is not None:
        payload["basal_rate"] = float(template.basal_rate)
    if template.dia_minutes is not None:
        payload["dia_minutes"] = float(template.dia_minutes)
    return payload


def build_long_study_day_scenario(
    *,
    profile: RuntimeScenarioProfile,
    day_number: int,
    weekday: str,
    sequence_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scenario_name": f"Long Study Day {day_number:02d} - {profile.name}",
        "scenario_version": "1.0",
        "description": (
            f"Continuous edge long-study day {day_number:02d} scheduled as {weekday}. "
            f"Profile: {profile.description}"
        ),
        "stress_events": [_template_to_event_payload(template) for template in profile.templates],
        "long_study_metadata": {
            "profile": profile.name,
            "weekday": weekday,
            "day_number": day_number,
            "sequence_seed": sequence_seed,
            "warm_start_minutes": profile.warm_start_minutes,
        },
    }


def _append_index_row(index_path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "day_number",
        "weekday",
        "scenario_profile",
        "algorithm_slug",
        "algorithm_name",
        "algorithm_source_type",
        "seed",
        "run_dir",
        "results_csv",
        "run_id",
        "tir_70_180",
        "tir_below_70",
        "tir_above_180",
        "mean_glucose",
        "supervisor_interventions",
        "terminated_early",
        "status",
    ]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()
    with index_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _iter_export_files(root: Path, *, exclude_paths: set[Path] | None = None) -> list[Path]:
    excludes = {path.resolve() for path in (exclude_paths or set())}
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        resolved = path.resolve()
        if any(resolved == excluded or excluded in resolved.parents for excluded in excludes):
            continue
        if path.is_file():
            files.append(path)
    return files


def _sync_directory_tree(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        target = destination_root / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _load_index_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        return []
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _resume_start_day(index_path: Path, *, expected_algorithm_slugs: set[str], expected_seeds: set[int]) -> int:
    rows = _load_index_rows(index_path)
    if not rows:
        return 1

    expected_pairs = {(algorithm_slug, seed) for algorithm_slug in expected_algorithm_slugs for seed in expected_seeds}
    day_pairs: dict[int, set[tuple[str, int]]] = {}
    for row in rows:
        status = str(row.get("status", "")).strip().lower()
        if status not in _COMPLETE_STATUSES:
            continue
        try:
            day_number = int(str(row.get("day_number", "")).strip())
            seed = int(str(row.get("seed", "")).strip())
        except (TypeError, ValueError):
            continue
        algorithm_slug = str(row.get("algorithm_slug", "")).strip()
        if algorithm_slug not in expected_algorithm_slugs:
            continue
        day_pairs.setdefault(day_number, set()).add((algorithm_slug, seed))

    last_contiguous_complete = 0
    for day_number in sorted(day_pairs):
        if day_number != last_contiguous_complete + 1:
            break
        if day_pairs[day_number] >= expected_pairs:
            last_contiguous_complete = day_number
        else:
            break
    return last_contiguous_complete + 1


def create_edge_study_snapshot(
    input_dir: str | Path,
    *,
    output: str | Path,
) -> LongStudySnapshotResult:
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Long-study directory not found: {input_path}")

    output_path = Path(output).expanduser().resolve()
    generated_at = _now_utc()
    if output_path.suffixes[-2:] == [".tar", ".gz"]:
        archive_path = output_path
        archive_dir = archive_path.parent
    elif output_path.suffix == ".tgz":
        archive_path = output_path
        archive_dir = archive_path.parent
    else:
        archive_dir = output_path
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        archive_path = archive_dir / f"{input_path.name}_snapshot_{timestamp}.tar.gz"

    archive_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in _iter_export_files(input_path, exclude_paths={archive_dir}):
            archive.add(path, arcname=f"{input_path.name}/{path.relative_to(input_path).as_posix()}")

    result = LongStudySnapshotResult(
        archive=str(archive_path),
        input_dir=str(input_path),
        output_dir=str(archive_dir),
        generated_at_utc=generated_at,
    )
    if archive_path.suffix == ".gz":
        manifest_path = archive_path.with_name(archive_path.name + ".json")
    else:
        manifest_path = archive_path.with_suffix(archive_path.suffix + ".json")
    write_json(manifest_path, result.to_dict())
    return result


def export_edge_study_archive(
    input_dir: str | Path,
    *,
    output: str | Path,
) -> dict[str, str]:
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Long-study directory not found: {input_path}")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_export_files(input_path, exclude_paths={output_path}):
            archive.write(path, arcname=f"{input_path.name}/{path.relative_to(input_path).as_posix()}")
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    manifest = {
        "archive": str(output_path),
        "input_dir": str(input_path),
        "generated_at_utc": _now_utc(),
    }
    write_json(manifest_path, manifest)
    return {
        "archive": str(output_path),
        "manifest": str(manifest_path),
        "input_dir": str(input_path),
    }


def _load_run_id(run_dir: Path) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return ""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("run_id", ""))


def run_edge_long_study(
    *,
    config_path: str | Path,
    project_dir: str | Path = ".",
    resume: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_dir).expanduser().resolve()
    resolved_config_path = _resolve_config_path(config_path, project_root)
    config = load_edge_long_study_config(resolved_config_path)
    output_root = _resolve_output_dir(config.output_dir, project_root)
    scratch_root = _resolve_scratch_dir(config.scratch_dir, project_root, output_root)
    if scratch_root == output_root:
        raise LongStudyConfigError("`scratch_dir` must resolve to a different path than `output_dir`.")

    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_dir = scratch_root / "protocol"
    days_dir = scratch_root / "days"
    snapshots_dir = output_root / "snapshots"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    days_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    algorithm_specs = _resolve_algorithm_specs(config.algorithms, project_root)
    expected_algorithm_slugs = {spec.slug for spec in algorithm_specs}
    total_runs = config.duration_days * len(config.seeds) * len(algorithm_specs)
    start_day_number = (
        _resume_start_day(
            output_root / "long_study_index.csv",
            expected_algorithm_slugs=expected_algorithm_slugs,
            expected_seeds=set(config.seeds),
        )
        if resume
        else 1
    )

    if start_day_number > config.duration_days:
        summary = analyze_study_directory(output_root)
        summary_path = output_root / "study_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        return {
            "config_path": str(resolved_config_path),
            "project_dir": str(project_root),
            "output_dir": str(output_root),
            "scratch_dir": str(scratch_root),
            "protocol_dir": str(output_root / "protocol"),
            "summary_json": str(summary_path),
            "index_csv": str(output_root / "long_study_index.csv"),
            "progress_json": str(output_root / "study_progress.json"),
            "export_readme": str(output_root / "EXPORT_README.md"),
            "completed_runs": 0,
            "skipped_runs": 0,
            "total_runs": total_runs,
            "elapsed_seconds": 0.0,
            "snapshots": [],
            "resume": resume,
            "resume_start_day": start_day_number,
        }

    write_json(
        protocol_dir / "long_study_manifest.json",
        {
            "kind": "edge_long_study",
            "generated_at_utc": _now_utc(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "project_dir": str(project_root),
            "config_path": str(resolved_config_path),
            "output_dir": str(output_root),
            "scratch_dir": str(scratch_root),
            "algorithm_count": len(algorithm_specs),
            "seed_count": len(config.seeds),
            "duration_days": config.duration_days,
            "snapshot_interval_hours": config.snapshot_interval_hours,
            "resume": resume,
            "resume_start_day": start_day_number,
        },
    )
    (protocol_dir / "edge_long_study.resolved.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    write_json(protocol_dir / "algorithm_registry.json", {"algorithms": [spec.to_dict() for spec in algorithm_specs]})
    _sync_directory_tree(scratch_root, output_root)

    started_at = time.time()
    completed_runs = 0
    skipped_runs = 0
    snapshot_records: list[dict[str, str]] = []
    index_path = scratch_root / "long_study_index.csv"
    final_index_path = output_root / "long_study_index.csv"
    if resume and final_index_path.is_file():
        shutil.copy2(final_index_path, index_path)

    for day_index in range(start_day_number - 1, config.duration_days):
        day_number = day_index + 1
        weekday = config.weekday_for_day(day_index)
        profile_name = config.profile_for_day(day_index)
        profile = get_runtime_scenario_profile(profile_name)
        scenario_payload = build_long_study_day_scenario(
            profile=profile,
            day_number=day_number,
            weekday=weekday,
            sequence_seed=config.seeds[0],
        )
        day_folder = days_dir / f"day_{day_number:02d}_{profile.name}"
        day_folder.mkdir(parents=True, exist_ok=True)
        run_rows: list[dict[str, Any]] = []

        for algorithm_spec in algorithm_specs:
            for seed in config.seeds:
                scratch_run_dir = scratch_root / "algorithms" / algorithm_spec.slug / f"seed_{seed}" / f"day_{day_number:02d}_{profile.name}"
                final_run_dir = output_root / "algorithms" / algorithm_spec.slug / f"seed_{seed}" / f"day_{day_number:02d}_{profile.name}"
                scratch_run_dir.mkdir(parents=True, exist_ok=True)
                final_results_csv = final_run_dir / "results.csv"
                if final_results_csv.is_file():
                    skipped_runs += 1
                    payload = {
                        "day_number": day_number,
                        "weekday": weekday,
                        "scenario_profile": profile.name,
                        "algorithm_slug": algorithm_spec.slug,
                        "algorithm_name": algorithm_spec.display_name,
                        "algorithm_source_type": algorithm_spec.source_type,
                        "seed": seed,
                        "run_dir": str(final_run_dir),
                        "results_csv": str(final_results_csv),
                        "run_id": _load_run_id(final_run_dir),
                        "tir_70_180": "",
                        "tir_below_70": "",
                        "tir_above_180": "",
                        "mean_glucose": "",
                        "supervisor_interventions": "",
                        "terminated_early": "",
                        "status": "skipped_existing",
                    }
                    _append_index_row(index_path, payload)
                    run_rows.append(payload)
                    continue

                if progress_callback is not None:
                    progress_callback(
                        f"Running day {day_number}/{config.duration_days} ({weekday}) for {algorithm_spec.display_name}, seed {seed}"
                    )

                algorithm_instance = _instantiate_algorithm(algorithm_spec)
                try:
                    outputs = iints.run_simulation(
                        algorithm=algorithm_instance,
                        scenario=scenario_payload,
                        patient_config=config.patient_config,
                        patient_model_type=config.patient_model_type,
                        duration_minutes=config.duration_minutes_per_day,
                        time_step=config.time_step_minutes,
                        seed=seed,
                        output_dir=scratch_run_dir,
                        compare_baselines=False,
                        export_audit=True,
                        generate_report=False,
                    )
                except Exception as exc:
                    raise EdgeLongStudyExecutionError(
                        f"Long-study run failed for {algorithm_spec.display_name}, seed {seed}, day {day_number}: {exc}"
                    ) from exc

                metrics = compute_run_metrics(
                    outputs["results"],
                    safety_report=outputs.get("safety_report"),
                    duration_minutes=config.duration_minutes_per_day,
                )
                day_manifest = {
                    "kind": "edge_long_study_run",
                    "generated_at_utc": _now_utc(),
                    "day_number": day_number,
                    "weekday": weekday,
                    "scenario_profile": profile.name,
                    "scenario_description": profile.description,
                    "algorithm": algorithm_spec.to_dict(),
                    "seed": seed,
                    "project_dir": str(project_root),
                    "scratch_run_dir": str(scratch_run_dir),
                    "run_dir": str(final_run_dir),
                    "results_csv": str(final_run_dir / "results.csv"),
                    "metrics": metrics,
                }
                write_json(scratch_run_dir / "study_day_manifest.json", day_manifest)
                _sync_directory_tree(scratch_run_dir, final_run_dir)
                completed_runs += 1

                payload = {
                    "day_number": day_number,
                    "weekday": weekday,
                    "scenario_profile": profile.name,
                    "algorithm_slug": algorithm_spec.slug,
                    "algorithm_name": algorithm_spec.display_name,
                    "algorithm_source_type": algorithm_spec.source_type,
                    "seed": seed,
                    "run_dir": str(final_run_dir),
                    "results_csv": str(final_run_dir / "results.csv"),
                    "run_id": outputs.get("run_id", ""),
                    "tir_70_180": round(float(metrics.get("tir_70_180", 0.0)), 4),
                    "tir_below_70": round(float(metrics.get("tir_below_70", 0.0)), 4),
                    "tir_above_180": round(float(metrics.get("tir_above_180", 0.0)), 4),
                    "mean_glucose": round(float(metrics.get("mean_glucose", 0.0)), 4),
                    "supervisor_interventions": int(metrics.get("supervisor_interventions", 0.0)),
                    "terminated_early": int(metrics.get("terminated_early", 0.0)),
                    "status": "completed",
                }
                _append_index_row(index_path, payload)
                run_rows.append(payload)

        write_json(
            day_folder / "day_summary.json",
            {
                "day_number": day_number,
                "weekday": weekday,
                "scenario_profile": profile.name,
                "generated_at_utc": _now_utc(),
                "runs": run_rows,
            },
        )
        _sync_directory_tree(day_folder, output_root / "days" / day_folder.name)
        shutil.copy2(index_path, final_index_path)

        write_json(
            scratch_root / "study_progress.json",
            {
                "generated_at_utc": _now_utc(),
                "completed_runs": completed_runs,
                "skipped_runs": skipped_runs,
                "total_runs": total_runs,
                "last_completed_day": day_number,
                "output_dir": str(output_root),
                "scratch_dir": str(scratch_root),
                "resume": resume,
                "resume_start_day": start_day_number,
            },
        )
        shutil.copy2(scratch_root / "study_progress.json", output_root / "study_progress.json")

        if day_number % config.snapshot_every_days() == 0 or day_number == config.duration_days:
            snapshot = create_edge_study_snapshot(output_root, output=snapshots_dir)
            snapshot_records.append(snapshot.to_dict())

    summary = analyze_study_directory(output_root)
    summary_path = output_root / "study_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    elapsed_seconds = round(time.time() - started_at, 3)
    export_readme = output_root / "EXPORT_README.md"
    export_readme.write_text(
        "\n".join(
            [
                "# Edge Long Study Export",
                "",
                "This study stays local on the Pi, but it is intentionally stored as normal folders plus CSV/JSON manifests.",
                "",
                "## Copy it to another device",
                "",
                "```bash",
                "iints edge study-export --project-dir . --input-dir " + str(output_root),
                "```",
                "",
                "## Snapshot it before unplugging storage",
                "",
                "```bash",
                "iints edge study-snapshot --project-dir . --input-dir " + str(output_root),
                "```",
                "",
                "## Analyze it on a laptop",
                "",
                "```bash",
                "iints analyze " + str(output_root),
                "```",
                "",
                "## Storage safety",
                "",
                "- daily runs are written to scratch first: `" + str(scratch_root) + "`",
                "- the study is synced to the final output after each day",
                "- for the safest Pi setup, point `output_dir` at a USB SSD instead of the SD card",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "config_path": str(resolved_config_path),
        "project_dir": str(project_root),
        "output_dir": str(output_root),
        "scratch_dir": str(scratch_root),
        "protocol_dir": str(output_root / "protocol"),
        "summary_json": str(summary_path),
        "index_csv": str(final_index_path),
        "progress_json": str(output_root / "study_progress.json"),
        "export_readme": str(export_readme),
        "completed_runs": completed_runs,
        "skipped_runs": skipped_runs,
        "total_runs": total_runs,
        "elapsed_seconds": elapsed_seconds,
        "snapshots": snapshot_records,
        "resume": resume,
        "resume_start_day": start_day_number,
    }
