from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def _write_algorithm(path: Path) -> None:
    path.write_text(
        """
from iints import AlgorithmInput, InsulinAlgorithm


class DemoAlgorithm(InsulinAlgorithm):
    def predict_insulin(self, data: AlgorithmInput):
        return {"total_insulin_delivered": 0.0}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_doctor_blocks_aggressive_glucose_decay(tmp_path) -> None:
    algo = tmp_path / "algorithm.py"
    patient = tmp_path / "patient.yaml"
    scenario = tmp_path / "scenario.json"
    report_json = tmp_path / "doctor.json"
    _write_algorithm(algo)
    patient.write_text(
        "\n".join(
            [
                "basal_insulin_rate: 0.2",
                "insulin_sensitivity: 40.0",
                "carb_factor: 15.0",
                "initial_glucose: 140.0",
                "glucose_decay_rate: 0.05",
            ]
        ),
        encoding="utf-8",
    )
    scenario.write_text(
        json.dumps(
            {
                "scenario_name": "late meal",
                "scenario_version": "1.0",
                "stress_events": [{"start_time": 60, "event_type": "meal", "value": 45}],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "run-doctor",
            "--algo",
            str(algo),
            "--patient-config-path",
            str(patient),
            "--scenario-path",
            str(scenario),
            "--duration",
            "120",
            "--time-step",
            "5",
            "--output-json",
            str(report_json),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert any(check["name"] == "glucose_decay" and check["status"] == "fail" for check in payload["checks"])


def test_evidence_build_creates_public_bundle(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "time_minutes": [0, 5, 10, 15],
            "glucose_actual_mgdl": [110.0, 120.0, 135.0, 128.0],
            "delivered_insulin_units": [0.0, 0.0, 0.2, 0.0],
        }
    ).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run_manifest.json").write_text('{"seed": 42}\n', encoding="utf-8")
    (run_dir / "safety_report.json").write_text('{"critical_events": 0}\n', encoding="utf-8")

    output_dir = tmp_path / "evidence"
    result = runner.invoke(
        app,
        [
            "evidence",
            "build",
            "--run",
            f"normal={run_dir}",
            "--output-dir",
            str(output_dir),
            "--title",
            "Demo Evidence",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "MODEL_CARD.md").is_file()
    assert (output_dir / "evidence_summary.json").is_file()
    assert (output_dir / "run_index.csv").is_file()


def test_top_level_pump_compile_and_bench_test(tmp_path) -> None:
    algo = tmp_path / "algorithm.py"
    bundle = tmp_path / "bundle"
    report_json = tmp_path / "bench.json"
    _write_algorithm(algo)

    compile_result = runner.invoke(
        app,
        [
            "pump",
            "compile",
            "--algorithm",
            str(algo),
            "--output-dir",
            str(bundle),
        ],
    )
    assert compile_result.exit_code == 0

    bench_result = runner.invoke(
        app,
        [
            "pump",
            "bench-test",
            "--bundle-dir",
            str(bundle),
            "--output-json",
            str(report_json),
        ],
    )
    assert bench_result.exit_code == 0
    assert json.loads(report_json.read_text(encoding="utf-8"))["passed"] is True


def test_research_train_local_ai_alias_is_available() -> None:
    result = runner.invoke(app, ["research", "train-local-ai", "--help"])

    assert result.exit_code == 0
    assert "--run" in result.stdout
