import json
import zipfile
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.jetson.endurance import (
    EnduranceConfig,
    build_endurance_service_file,
    export_endurance_archive,
    load_endurance_status,
    parse_duration_to_minutes,
    run_endurance_study,
)


runner = CliRunner()


def _run_short_endurance(output_dir: Path):
    config = EnduranceConfig(
        algo_path="builtin:clinical_baseline",
        predictor_path=None,
        duration="1h",
        duration_minutes=parse_duration_to_minutes("1h"),
        time_step_minutes=5,
        output_dir=str(output_dir),
        profile="normal",
        seed=42,
        patient_model="custom",
        sensor_profile="clinical_cgm",
        checkpoint_interval_minutes=30,
        hardware_sample_interval_minutes=30,
        status_interval_steps=3,
    )
    return run_endurance_study(algorithm=ClinicalBaselineAlgorithm(), predictor=None, config=config)


def test_parse_duration_to_minutes() -> None:
    assert parse_duration_to_minutes("1h") == 60
    assert parse_duration_to_minutes("7d") == 10080
    assert parse_duration_to_minutes("2w") == 20160


def test_run_endurance_writes_expected_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "jetson_1h"

    result = _run_short_endurance(output_dir)

    assert result["status"]["status"] == "completed"
    assert result["status"]["completed_steps"] == 12
    assert (output_dir / "protocol" / "test_config.yaml").is_file()
    assert (output_dir / "protocol" / "hardware_info.json").is_file()
    assert (output_dir / "raw" / "steps.csv").is_file()
    assert (output_dir / "raw" / "interventions.csv").is_file()
    assert (output_dir / "raw" / "critical_events.csv").is_file()
    assert (output_dir / "raw" / "hardware_metrics.csv").is_file()
    assert (output_dir / "daily" / "day_01_summary.json").is_file()
    assert (output_dir / "final" / "test_summary.json").is_file()
    assert (output_dir / "final" / "tir_timeseries.csv").is_file()
    assert (output_dir / "final" / "supervisor_analysis.json").is_file()
    assert (output_dir / "final" / "worst_case_events.json").is_file()
    assert (output_dir / "final" / "ENDURANCE_REPORT.md").is_file()
    assert (output_dir / "final" / "main_figure.png").is_file()
    assert (output_dir / "research" / "predictor_training.csv").is_file()
    assert (output_dir / "research" / "controller_teacher_dataset.csv").is_file()
    assert (output_dir / "research" / "training_manifest.json").is_file()
    assert (output_dir / "research" / "README.md").is_file()

    steps = pd.read_csv(output_dir / "raw" / "steps.csv")
    summary = json.loads((output_dir / "final" / "test_summary.json").read_text())
    assert len(steps) == 12
    assert summary["expected_steps"] == 12
    assert summary["checkpoint_interval_minutes"] == 30
    assert 0.0 <= summary["total_tir_70_180_pct"] <= 100.0
    assert summary["execution_mode"] == "accelerated"
    assert (output_dir / "snapshots" / "snapshot_000030m.json").is_file()
    assert (output_dir / "snapshots" / "snapshot_000060m.json").is_file()

    training_manifest = json.loads((output_dir / "research" / "training_manifest.json").read_text())
    assert training_manifest["row_count"] == 12
    assert training_manifest["ministral_training_supported"] is False
    assert "controller_teacher_dataset.csv" in training_manifest["controller_dataset_path"]


def test_status_and_export_helpers(tmp_path: Path) -> None:
    output_dir = tmp_path / "jetson_1h"
    archive_path = tmp_path / "jetson_1h.zip"
    _run_short_endurance(output_dir)

    status = load_endurance_status(output_dir)
    archive = export_endurance_archive(output_dir, archive_path)

    assert status["status"] == "completed"
    assert archive == archive_path
    with zipfile.ZipFile(archive_path) as archive_file:
        names = set(archive_file.namelist())
    assert "status.json" in names
    assert "raw/steps.csv" in names
    assert "final/ENDURANCE_REPORT.md" in names


def test_resume_continues_from_latest_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "jetson_resume"
    first_config = EnduranceConfig(
        algo_path="builtin:clinical_baseline",
        predictor_path=None,
        duration="1h",
        duration_minutes=parse_duration_to_minutes("1h"),
        time_step_minutes=5,
        output_dir=str(output_dir),
        profile="normal",
        seed=42,
        patient_model="custom",
        sensor_profile="clinical_cgm",
        checkpoint_interval_minutes=15,
        hardware_sample_interval_minutes=15,
        status_interval_steps=1,
    )

    def request_stop(status: dict) -> None:
        if status["completed_steps"] == 3:
            (output_dir / "STOP_REQUESTED").write_text("stop", encoding="utf-8")

    first = run_endurance_study(
        algorithm=ClinicalBaselineAlgorithm(),
        predictor=None,
        config=first_config,
        progress_callback=request_stop,
    )
    assert first["status"]["status"] == "stopped"
    assert (output_dir / "snapshots" / "snapshot_000015m.json").is_file()

    (output_dir / "STOP_REQUESTED").unlink()
    resumed = run_endurance_study(
        algorithm=ClinicalBaselineAlgorithm(),
        predictor=None,
        config=EnduranceConfig(**{**first_config.__dict__, "resume": True}),
    )

    assert resumed["status"]["status"] == "completed"
    assert resumed["status"]["completed_steps"] == 12
    assert resumed["status"]["resume_count"] == 1


def test_wall_clock_mode_paces_to_requested_duration(tmp_path: Path) -> None:
    output_dir = tmp_path / "jetson_wall_clock"
    config = EnduranceConfig(
        algo_path="builtin:clinical_baseline",
        predictor_path=None,
        duration="30m",
        duration_minutes=parse_duration_to_minutes("30m"),
        time_step_minutes=5,
        output_dir=str(output_dir),
        profile="normal",
        seed=42,
        patient_model="custom",
        sensor_profile="clinical_cgm",
        checkpoint_interval_minutes=30,
        hardware_sample_interval_minutes=30,
        status_interval_steps=1,
        execution_mode="wall_clock",
    )

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

    fake_clock = FakeClock()
    result = run_endurance_study(
        algorithm=ClinicalBaselineAlgorithm(),
        predictor=None,
        config=config,
        monotonic_fn=fake_clock.monotonic,
        sleep_fn=fake_clock.sleep,
    )

    assert result["status"]["execution_mode"] == "wall_clock"
    assert result["status"]["wall_elapsed_seconds"] == 1800.0
    assert result["status"]["wall_clock_target_seconds"] == 1800
    assert sum(fake_clock.sleeps) == 1800.0


def test_jetson_endurance_cli_status_and_export(tmp_path: Path) -> None:
    output_dir = tmp_path / "jetson_1h"
    archive_path = tmp_path / "export.zip"
    _run_short_endurance(output_dir)

    status_result = runner.invoke(app, ["jetson", "endurance", "status", "--output-dir", str(output_dir)])
    export_result = runner.invoke(
        app,
        ["jetson", "endurance", "export", "--output-dir", str(output_dir), "--output", str(archive_path)],
    )

    assert status_result.exit_code == 0
    assert "IINTS Jetson Endurance Status" in status_result.stdout
    assert export_result.exit_code == 0
    assert archive_path.is_file()


def test_jetson_endurance_cli_start_runs_short_study(tmp_path: Path) -> None:
    algo_path = tmp_path / "demo_algo.py"
    output_dir = tmp_path / "jetson_start"
    algo_path.write_text(
        "from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm\n\n"
        "class DemoAlgo(ClinicalBaselineAlgorithm):\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "jetson",
            "endurance",
            "start",
            "--algo",
            str(algo_path),
            "--duration",
            "30m",
            "--output-dir",
            str(output_dir),
            "--profile",
            "normal",
            "--patient-model",
            "custom",
            "--sensor-profile",
            "clinical_cgm",
        ],
    )

    assert result.exit_code == 0
    assert "Endurance artifacts written to" in result.stdout
    assert (output_dir / "status.json").is_file()
    assert (output_dir / "final" / "test_summary.json").is_file()


def test_service_file_contains_resume_command() -> None:
    unit = build_endurance_service_file(
        algo="algorithms/example_algorithm.py",
        predictor="models/lstm_predictor.pt",
        duration="7d",
        output_dir="results/jetson_7day",
        profile="mixed_adversarial",
        seed=42,
        working_directory="/opt/iints",
    )

    assert "WorkingDirectory=/opt/iints" in unit
    assert "iints jetson endurance start" in unit
    assert "--resume" in unit
    assert "--predictor models/lstm_predictor.pt" in unit


def test_service_file_can_request_wall_clock_mode() -> None:
    unit = build_endurance_service_file(
        algo="algorithms/example_algorithm.py",
        predictor=None,
        duration="24h",
        output_dir="results/jetson_research_day",
        wall_clock=True,
    )

    assert "--wall-clock" in unit
