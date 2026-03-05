from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.validation.replay import ReplayCheckResult, ReplayRunDigest
from iints.validation.golden import GoldenBenchmarkPack, GoldenScenarioSpec


runner = CliRunner()


def test_validation_profiles_command_lists_profiles() -> None:
    result = runner.invoke(app, ["validation-profiles"])
    assert result.exit_code == 0
    assert "research_default" in result.stdout
    assert "strict_safety" in result.stdout


def test_validate_run_command_passes_for_stable_trace(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "time_minutes": [idx * 5 for idx in range(25)],  # 120 minutes
            "glucose_actual_mgdl": [120.0] * 25,
            "safety_triggered": [False] * 25,
        }
    )
    results_csv = tmp_path / "results.csv"
    report_json = tmp_path / "validation_report.json"
    df.to_csv(results_csv, index=False)

    result = runner.invoke(
        app,
        [
            "validate-run",
            "--results-csv",
            str(results_csv),
            "--profile",
            "screening",
            "--output-json",
            str(report_json),
        ],
    )

    assert result.exit_code == 0
    assert report_json.is_file()
    payload = json.loads(report_json.read_text())
    assert payload["passed"] is True
    assert payload["profile"]["id"] == "screening"


def test_run_full_rejects_missing_predictor_file() -> None:
    result = runner.invoke(
        app,
        [
            "run-full",
            "--algo",
            "examples/mytest_algorithm.py",
            "--predictor",
            "models/does_not_exist.pt",
        ],
    )
    assert result.exit_code != 0
    assert "Predictor checkpoint not found" in result.stdout


def test_contract_verify_command_passes_default_grid(tmp_path) -> None:
    out_json = tmp_path / "contract_report.json"
    result = runner.invoke(
        app,
        [
            "contract-verify",
            "--glucose-min",
            "70",
            "--glucose-max",
            "95",
            "--glucose-step",
            "5",
            "--trend-min",
            "-2",
            "--trend-max",
            "1",
            "--trend-step",
            "1",
            "--proposed-doses",
            "0.0,0.5,1.0",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()


def test_replay_check_command_writes_report(tmp_path, monkeypatch) -> None:
    out_json = tmp_path / "replay.json"

    def _fake_replay_check(**kwargs):
        return ReplayCheckResult(
            passed=True,
            seed=42,
            digests=[
                ReplayRunDigest(results_hash="a" * 64, safety_hash="b" * 64, rows=120),
                ReplayRunDigest(results_hash="a" * 64, safety_hash="b" * 64, rows=120),
            ],
        )

    monkeypatch.setattr("iints.cli.cli.run_deterministic_replay_check", _fake_replay_check)

    result = runner.invoke(
        app,
        [
            "replay-check",
            "--algo",
            "examples/mytest_algorithm.py",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out_json.read_text())
    assert payload["passed"] is True
    assert len(payload["digests"]) == 2


def test_golden_benchmark_command_passes_with_mocked_run(tmp_path, monkeypatch) -> None:
    pack = GoldenBenchmarkPack(
        name="test-pack",
        version=1,
        scenarios=[
            GoldenScenarioSpec(
                preset="baseline_t1d",
                description="demo",
                expected={"tir_70_180": {"min": 50.0}},
            )
        ],
    )

    def _fake_get_preset(name: str):
        return {
            "name": name,
            "patient_config": "default_patient",
            "duration_minutes": 30,
            "time_step_minutes": 5,
            "scenario": {},
        }

    def _fake_run_simulation(**kwargs):
        return {
            "results": pd.DataFrame(
                {
                    "time_minutes": [0, 5, 10, 15, 20, 25],
                    "glucose_actual_mgdl": [110, 112, 109, 111, 110, 108],
                    "safety_triggered": [False] * 6,
                }
            ),
            "safety_report": {"terminated_early": False},
        }

    def _fake_compute_metrics(results_df, safety_report=None, duration_minutes=None):
        return {"tir_70_180": 95.0, "terminated_early": 0.0}

    monkeypatch.setattr("iints.cli.cli.load_golden_benchmark_pack", lambda path=None: pack)
    monkeypatch.setattr("iints.cli.cli._get_preset", _fake_get_preset)
    monkeypatch.setattr("iints.cli.cli.iints.run_simulation", _fake_run_simulation)
    monkeypatch.setattr("iints.cli.cli.compute_run_metrics", _fake_compute_metrics)

    out_json = tmp_path / "golden.json"
    result = runner.invoke(
        app,
        [
            "golden-benchmark",
            "--algo",
            "examples/mytest_algorithm.py",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out_json.read_text())
    assert payload["passed"] is True
    assert payload["pack"]["name"] == "test-pack"


def test_research_evaluate_forecast_with_gate(tmp_path) -> None:
    input_csv = tmp_path / "forecast.csv"
    out_json = tmp_path / "forecast_metrics.json"
    df = pd.DataFrame(
        {
            "glucose_actual_mgdl": [100.0, 120.0, 140.0, 160.0, 110.0],
            "predicted_glucose_ai_30min": [101.0, 121.0, 139.0, 159.0, 110.0],
            "predictor_uncertainty_std_mgdl": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    df.to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "research",
            "evaluate-forecast",
            "--input-csv",
            str(input_csv),
            "--gate-profile",
            "research_default",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out_json.read_text())
    assert payload["calibration_gate"]["profile"] == "research_default"
    assert payload["calibration_gate"]["passed"] is True


def test_research_registry_list_and_promote(tmp_path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "run_id": "run-1",
                    "stage": "candidate",
                    "test_rmse": 12.3,
                    "test_mae": 8.1,
                    "timestamp_utc": "2026-03-05T10:00:00Z",
                    "model_path": "models/run-1/predictor.pt",
                }
            ]
        )
    )

    list_result = runner.invoke(
        app,
        [
            "research",
            "registry-list",
            "--registry",
            str(registry),
        ],
    )
    assert list_result.exit_code == 0
    assert "run-1" in list_result.stdout

    promote_result = runner.invoke(
        app,
        [
            "research",
            "registry-promote",
            "--registry",
            str(registry),
            "--run-id",
            "run-1",
            "--stage",
            "validated",
        ],
    )
    assert promote_result.exit_code == 0
    rows = json.loads(registry.read_text())
    assert rows[0]["stage"] == "validated"


def test_sources_command_lists_evidence_manifest() -> None:
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    assert "IINTS-AF Evidence Sources" in result.stdout
    assert "ada_2026" in result.stdout


def test_sources_command_writes_json_manifest(tmp_path) -> None:
    out_json = tmp_path / "sources.json"
    result = runner.invoke(
        app,
        [
            "sources",
            "--category",
            "trial",
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text())
    assert payload["category"] == "trial"
    assert payload["count"] >= 1
