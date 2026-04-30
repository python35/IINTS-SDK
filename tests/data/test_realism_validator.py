from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.importer import import_cgm_dataframe, load_demo_dataframe
from iints.data.realism_validator import validate_realism_dataset


runner = CliRunner()


def _flat_meal_trace() -> pd.DataFrame:
    rows = []
    meals = {420: 45.0, 720: 65.0, 1080: 80.0}
    for timestamp in range(0, 1440, 5):
        rows.append(
            {
                "timestamp": timestamp,
                "glucose": 112.0 + (timestamp / 1440.0) * 9.0,
                "carbs": meals.get(timestamp, 0.0),
                "insulin": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_realism_validator_accepts_bundled_demo_trace() -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    report = validate_realism_dataset(standard_df)

    assert report.verdict == "likely_realistic"
    assert report.metrics["meal_count"] == 4
    assert report.metrics["insulin_event_count"] == 4
    assert any(check.code == "meal_response" and check.status == "passed" for check in report.checks)


def test_realism_validator_flags_flat_too_neat_trace() -> None:
    report = validate_realism_dataset(_flat_meal_trace())

    assert report.verdict == "likely_unrealistic"
    statuses = {check.code: check.status for check in report.checks}
    assert statuses["glucose_variability"] == "failed"
    assert statuses["event_balance"] == "failed"


def test_data_realism_check_cli_writes_json_and_gates_verdict(tmp_path) -> None:
    input_csv = tmp_path / "flat.csv"
    output_json = tmp_path / "realism.json"
    _flat_meal_trace().to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "realism-check",
            str(input_csv),
            "--output-json",
            str(output_json),
            "--min-realism-verdict",
            "likely_realistic",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output_json.read_text())
    assert payload["verdict"] == "likely_unrealistic"
    assert "checks" in payload
