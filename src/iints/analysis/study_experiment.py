from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StudyExperimentConfig:
    source_path: Path
    name: str
    preset: str
    title: str
    profile_set: str
    scenarios: list[str]
    seeds: list[int]
    duration_minutes: int | None
    time_step: int
    include_default_baselines: bool
    extra_algorithms: list[str]
    prepare_ai: bool
    gate_profile: str | None
    gate_profiles_path: Path | None
    fail_on_gate: bool
    external_reference_label: str
    candidate_algorithm: Path
    output_dir: Path | None
    carelink_metrics: Path | None
    reference_csv: Path | None


def _resolve_optional_path(base_dir: Path, raw_value: Any) -> Path | None:
    if raw_value in (None, "", "null"):
        return None
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require_mapping(payload: Any, label: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping in the experiment file.")
    return dict(payload)


def _normalize_str_list(raw_value: Any, label: str) -> list[str]:
    if raw_value in (None, "", []):
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
        return values
    raise ValueError(f"{label} must be a list or comma-separated string.")


def _normalize_seed_list(raw_value: Any) -> list[int]:
    if raw_value in (None, "", []):
        return [1, 2, 3, 4, 5]
    if isinstance(raw_value, str):
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [int(item) for item in raw_value]
    raise ValueError("experiment.seeds must be a list or comma-separated string.")


def load_study_experiment_config(path: str | Path) -> StudyExperimentConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Study experiment config not found: {source_path}")

    raw_payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Study experiment config must be a YAML mapping.")

    experiment = _require_mapping(raw_payload.get("experiment"), "experiment")
    algorithm = _require_mapping(raw_payload.get("algorithm"), "algorithm")
    paths = _require_mapping(raw_payload.get("paths"), "paths")
    study = _require_mapping(raw_payload.get("study"), "study")
    base_dir = source_path.parent

    candidate_raw = algorithm.get("candidate") or paths.get("candidate_algorithm")
    if not candidate_raw:
        raise ValueError("Study experiment config must define algorithm.candidate or paths.candidate_algorithm.")

    candidate_algorithm = _resolve_optional_path(base_dir, candidate_raw)
    if candidate_algorithm is None:
        raise ValueError("Study experiment config candidate path could not be resolved.")

    return StudyExperimentConfig(
        source_path=source_path,
        name=str(experiment.get("name") or source_path.stem),
        preset=str(experiment.get("preset") or "default"),
        title=str(experiment.get("title") or "IINTS Scientific Validation Protocol"),
        profile_set=str(experiment.get("profile_set") or "clinic_safe_core"),
        scenarios=_normalize_str_list(study.get("scenarios", experiment.get("scenarios")), "study.scenarios"),
        seeds=_normalize_seed_list(experiment.get("seeds")),
        duration_minutes=int(experiment["duration_minutes"]) if experiment.get("duration_minutes") is not None else None,
        time_step=int(experiment.get("time_step", 5)),
        include_default_baselines=bool(experiment.get("include_default_baselines", True)),
        extra_algorithms=_normalize_str_list(algorithm.get("extra_algorithms", experiment.get("extra_algorithms")), "algorithm.extra_algorithms"),
        prepare_ai=bool(experiment.get("prepare_ai", True)),
        gate_profile=str(experiment["gate_profile"]).strip() if experiment.get("gate_profile") not in (None, "") else None,
        gate_profiles_path=_resolve_optional_path(base_dir, paths.get("gate_profiles_path") or experiment.get("gate_profiles_path")),
        fail_on_gate=bool(experiment.get("fail_on_gate", False)),
        external_reference_label=str(
            experiment.get("external_reference_label") or "CareLink personal workbench metrics"
        ),
        candidate_algorithm=candidate_algorithm,
        output_dir=_resolve_optional_path(base_dir, paths.get("output_dir")),
        carelink_metrics=_resolve_optional_path(base_dir, paths.get("carelink_metrics")),
        reference_csv=_resolve_optional_path(base_dir, paths.get("reference_csv")),
    )


def build_study_experiment_template(
    *,
    preset: str,
    title: str,
    profile_set: str,
    seeds: list[int],
    candidate_algorithm: str,
    scenarios: list[str],
    include_default_baselines: bool,
    extra_algorithms: list[str] | None = None,
    external_reference_label: str = "CareLink personal workbench metrics",
    default_output_dir: str | None = None,
) -> dict[str, Any]:
    if default_output_dir is None:
        default_output_dir = "results/eucys_study" if preset == "eucys" else "results/study_bundle"
    return {
        "experiment": {
            "name": "iints_scientific_validation",
            "preset": preset,
            "title": title,
            "profile_set": profile_set,
            "seeds": list(seeds),
            "duration_minutes": None,
            "time_step": 5,
            "include_default_baselines": include_default_baselines,
            "prepare_ai": True,
            "gate_profile": None,
            "fail_on_gate": False,
            "external_reference_label": external_reference_label,
        },
        "study": {
            "scenarios": list(scenarios),
        },
        "algorithm": {
            "candidate": candidate_algorithm,
            "extra_algorithms": list(extra_algorithms or []),
        },
        "paths": {
            "output_dir": default_output_dir,
            "carelink_metrics": None,
            "reference_csv": None,
            "gate_profiles_path": None,
        },
    }


def render_study_experiment_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False)
