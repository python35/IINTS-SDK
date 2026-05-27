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


def _late_insulin_trace() -> pd.DataFrame:
    rows = []
    meals = {420: 45.0, 720: 65.0, 1080: 80.0}
    insulin = {540: 3.5, 840: 5.5, 1200: 6.5}
    for timestamp in range(0, 1440, 5):
        glucose = 109.0
        for meal_time, carbs in meals.items():
            dt = timestamp - meal_time
            if dt > 0:
                glucose += min(70.0, carbs * 0.9 * (1.0 - pow(2.718281828, -dt / 55.0))) * pow(2.718281828, -max(dt - 80.0, 0.0) / 180.0)
        glucose += 4.0 * ((timestamp % 1440) / 1440.0)
        rows.append(
            {
                "timestamp": timestamp,
                "glucose": round(glucose, 1),
                "carbs": meals.get(timestamp, 0.0),
                "insulin": insulin.get(timestamp, 0.0),
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
    assert any(check.code == "causal_alignment" and check.status == "passed" for check in report.checks)


def test_realism_validator_can_compare_against_reference_envelope() -> None:
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")

    report = validate_realism_dataset(standard_df, reference="free_living_t1d")

    assert report.reference_profile is not None
    assert report.reference_profile.id == "free_living_t1d"
    assert any(check.code == "reference_envelope" for check in report.checks)
    assert any(comparison.metric_key == "cv_pct" for comparison in report.reference_comparisons)


def test_realism_validator_flags_flat_too_neat_trace() -> None:
    report = validate_realism_dataset(_flat_meal_trace())

    assert report.verdict == "likely_unrealistic"
    statuses = {check.code: check.status for check in report.checks}
    assert statuses["glucose_variability"] == "failed"
    assert statuses["event_balance"] == "failed"


def test_realism_validator_flags_late_insulin_causal_mismatch() -> None:
    report = validate_realism_dataset(_late_insulin_trace())

    statuses = {check.code: check.status for check in report.checks}
    assert statuses["meal_response"] == "passed"
    assert statuses["event_balance"] == "passed"
    assert statuses["causal_alignment"] == "failed"


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


def test_data_realism_check_cli_writes_html_dashboard(tmp_path) -> None:
    input_csv = tmp_path / "demo.csv"
    output_html = tmp_path / "realism_dashboard.html"
    load_demo_dataframe().to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "realism-check",
            str(input_csv),
            "--reference",
            "free_living_t1d",
            "--output-html",
            str(output_html),
        ],
    )

    assert result.exit_code == 0
    content = output_html.read_text()
    assert "Reference Envelope" in content
    assert "Free-Living T1D Daily Envelope" in content


def test_data_realism_check_cli_can_write_strict_gate_payload(tmp_path) -> None:
    input_csv = tmp_path / "demo.csv"
    output_json = tmp_path / "strict_realism.json"
    load_demo_dataframe().to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "realism-check",
            str(input_csv),
            "--output-json",
            str(output_json),
            "--strict-real-data-gate",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(output_json.read_text())
    assert payload["strict_real_data_gate"]["status"] == "blocked"
    assert any("No empirical reference profile" in item for item in payload["strict_real_data_gate"]["critical_failures"])
