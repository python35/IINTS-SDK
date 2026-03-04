from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app


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
