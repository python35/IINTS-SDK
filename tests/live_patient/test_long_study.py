from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from iints.live_patient.long_study import (
    export_edge_study_archive,
    load_edge_long_study_config,
    render_edge_long_study_config_template,
    run_edge_long_study,
)


class _FakeSummary:
    def __init__(self, run_count: int) -> None:
        self.run_count = run_count

    def to_dict(self) -> dict[str, object]:
        return {
            "run_count": self.run_count,
            "aggregate": {"mean_tir_70_180": 85.0},
            "aggregate_stats": {},
            "certification_comparison": {},
            "failure_analysis": {},
        }


def test_load_edge_long_study_config_accepts_week_schedule(tmp_path: Path) -> None:
    config_path = tmp_path / "edge_long_study.yaml"
    config_path.write_text(
        render_edge_long_study_config_template(output_dir="results/demo_long_study"),
        encoding="utf-8",
    )

    config = load_edge_long_study_config(config_path)

    assert config.duration_days == 14
    assert config.week_schedule["monday"] == "school_day"
    assert config.week_schedule["sunday"] == "relaxed_day"
    assert config.snapshot_every_days() == 1
    assert config.scratch_dir == "/tmp/iints_edge_long_study"


def test_run_edge_long_study_writes_nested_outputs(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "pi_demo"
    project_dir.mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "edge_long_study.yaml"
    config_path.write_text(
        "\n".join(
            [
                "duration_days: 2",
                "algorithms:",
                "  - Clinical Baseline",
                "  - PID Controller",
                "  - Correction Bolus",
                "week_schedule:",
                "  monday: school_day",
                "  tuesday: sport_day",
                "  wednesday: school_day",
                "  thursday: bad_carb_count",
                "  friday: school_day",
                "  saturday: sport_day",
                "  sunday: relaxed_day",
                "seeds: [1, 2]",
                "snapshot_interval_hours: 24",
                "output_dir: results/long_study",
                "scratch_dir: scratch_stage",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    seen_run_dirs: list[Path] = []

    def _fake_run_simulation(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        seen_run_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "time_minutes": [0.0, 5.0, 10.0],
                "glucose_actual_mgdl": [110.0, 125.0, 118.0],
                "safety_triggered": [False, False, True],
            }
        )
        df.to_csv(output_dir / "results.csv", index=False)
        (output_dir / "audit").mkdir(exist_ok=True)
        (output_dir / "audit" / "audit_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "run_metadata.json").write_text(json.dumps({"run_id": f"seed-{kwargs['seed']}"}), encoding="utf-8")
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        return {
            "results": df,
            "safety_report": {"bolus_interventions_count": 1, "terminated_early": False},
            "run_id": f"seed-{kwargs['seed']}",
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("iints.live_patient.long_study.iints.run_simulation", _fake_run_simulation)
    monkeypatch.setattr(
        "iints.live_patient.long_study.analyze_study_directory",
        lambda root: _FakeSummary(run_count=12),
    )

    outputs = run_edge_long_study(config_path=config_path, project_dir=project_dir)

    output_root = Path(outputs["output_dir"])
    assert output_root.is_dir()
    assert outputs["completed_runs"] == 12
    assert outputs["scratch_dir"].endswith("scratch_stage/long_study")
    assert seen_run_dirs
    assert all("scratch_stage" in str(path) for path in seen_run_dirs)
    assert (output_root / "protocol" / "edge_long_study.resolved.yaml").is_file()
    assert (output_root / "protocol" / "algorithm_registry.json").is_file()
    assert (output_root / "days" / "day_01_school_day" / "day_summary.json").is_file()
    assert (output_root / "algorithms" / "clinical-baseline" / "seed_1" / "day_01_school_day" / "study_day_manifest.json").is_file()
    assert (output_root / "study_summary.json").is_file()
    assert (output_root / "long_study_index.csv").is_file()
    assert (output_root / "EXPORT_README.md").is_file()
    snapshots = sorted((output_root / "snapshots").glob("*.tar.gz"))
    assert snapshots


def test_run_edge_long_study_resume_skips_completed_days(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "pi_demo"
    output_root = project_dir / "results" / "long_study"
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = project_dir / "edge_long_study.yaml"
    config_path.write_text(
        "\n".join(
            [
                "duration_days: 3",
                "algorithms:",
                "  - Clinical Baseline",
                "week_schedule:",
                "  monday: school_day",
                "  tuesday: sport_day",
                "  wednesday: school_day",
                "  thursday: bad_carb_count",
                "  friday: school_day",
                "  saturday: sport_day",
                "  sunday: relaxed_day",
                "seeds: [1]",
                "snapshot_interval_hours: 24",
                "output_dir: results/long_study",
                "scratch_dir: scratch_stage",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "algorithms" / "clinical-baseline" / "seed_1" / "day_01_school_day").mkdir(parents=True, exist_ok=True)
    (output_root / "algorithms" / "clinical-baseline" / "seed_1" / "day_01_school_day" / "results.csv").write_text(
        "time_minutes,glucose_actual_mgdl\n0,110\n",
        encoding="utf-8",
    )
    (output_root / "algorithms" / "clinical-baseline" / "seed_1" / "day_01_school_day" / "run_metadata.json").write_text(
        json.dumps({"run_id": "existing-day-1"}),
        encoding="utf-8",
    )
    (output_root / "long_study_index.csv").write_text(
        "\n".join(
            [
                "day_number,weekday,scenario_profile,algorithm_slug,algorithm_name,algorithm_source_type,seed,run_dir,results_csv,run_id,tir_70_180,tir_below_70,tir_above_180,mean_glucose,supervisor_interventions,terminated_early,status",
                "1,monday,school_day,clinical-baseline,Clinical Baseline,builtin,1,/tmp/day1,/tmp/day1/results.csv,existing-day-1,90,1,5,120,1,0,completed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    seen_run_dirs: list[Path] = []

    def _fake_run_simulation(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        seen_run_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "time_minutes": [0.0, 5.0, 10.0],
                "glucose_actual_mgdl": [115.0, 122.0, 118.0],
            }
        )
        df.to_csv(output_dir / "results.csv", index=False)
        (output_dir / "audit").mkdir(exist_ok=True)
        (output_dir / "audit" / "audit_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        (output_dir / "run_metadata.json").write_text(json.dumps({"run_id": f"resume-{kwargs['seed']}"}), encoding="utf-8")
        (output_dir / "config.json").write_text("{}", encoding="utf-8")
        return {
            "results": df,
            "safety_report": {"bolus_interventions_count": 0, "terminated_early": False},
            "run_id": f"resume-{kwargs['seed']}",
            "output_dir": str(output_dir),
        }

    monkeypatch.setattr("iints.live_patient.long_study.iints.run_simulation", _fake_run_simulation)
    monkeypatch.setattr(
        "iints.live_patient.long_study.analyze_study_directory",
        lambda root: _FakeSummary(run_count=3),
    )

    outputs = run_edge_long_study(config_path=config_path, project_dir=project_dir, resume=True)

    assert outputs["resume"] is True
    assert outputs["resume_start_day"] == 2
    assert seen_run_dirs
    assert all("day_01_school_day" not in str(path) for path in seen_run_dirs)
    assert any("day_02_sport_day" in str(path) for path in seen_run_dirs)
    assert any("day_03_school_day" in str(path) for path in seen_run_dirs)


def test_export_edge_study_archive_writes_zip(tmp_path: Path) -> None:
    study_root = tmp_path / "long_study"
    run_dir = study_root / "algorithms" / "clinical-baseline" / "seed_1" / "day_01_school_day"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.csv").write_text("time_minutes,glucose_actual_mgdl\n0,110\n", encoding="utf-8")

    exported = export_edge_study_archive(study_root, output=tmp_path / "long_study_export.zip")

    archive_path = Path(exported["archive"])
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "long_study/algorithms/clinical-baseline/seed_1/day_01_school_day/results.csv" in names

    snapshot_dir = study_root / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_archive = snapshot_dir / "manual_snapshot.tar.gz"
    with tarfile.open(snapshot_archive, "w:gz") as archive:
        archive.add(run_dir / "results.csv", arcname="results.csv")
    assert snapshot_archive.is_file()
