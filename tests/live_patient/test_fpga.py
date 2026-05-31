from __future__ import annotations

import json
from pathlib import Path

from iints.live_patient.fpga import (
    create_fpga_lab,
    evaluate_fpga_safety_reference,
    normalize_fpga_event,
    run_fpga_safety_simulation,
)


def test_fpga_reference_flags_low_glucose_as_critical() -> None:
    result = evaluate_fpga_safety_reference(
        {
            "minute": 5,
            "glucose_mgdl": 64,
            "trend_mgdl_per_min": -1.0,
            "sensor_status": "OK",
        }
    )

    assert result["risk_label"] == "CRITICAL"
    assert result["risk_score"] == 3
    assert result["check_required"] is True


def test_fpga_reference_flags_sensor_error() -> None:
    result = evaluate_fpga_safety_reference(
        {
            "minute": 5,
            "glucose_mgdl": 130,
            "trend_mgdl_per_min": 0.0,
            "sensor_status": "ERROR",
        }
    )

    assert result["risk_label"] == "SENSOR_ERROR"
    assert "sensor status" in result["reasons"][0]


def test_fpga_event_normalization_handles_csv_bool_strings() -> None:
    result = normalize_fpga_event(
        {
            "minute": "10",
            "glucose": "110",
            "trend": "-0.3",
            "sensor_status": "ok",
            "meal_event": "false",
            "insulin_event": "1",
        }
    )

    assert result["meal_event"] is False
    assert result["insulin_event"] is True
    assert result["sensor_status"] == "OK"


def test_create_fpga_lab_writes_workspace(tmp_path: Path) -> None:
    outputs = create_fpga_lab(tmp_path / "fpga_lab")
    root = Path(outputs["output_dir"])

    assert (root / "rtl" / "iints_fpga_safety_core.v").is_file()
    assert (root / "scenarios" / "fpga_demo_events.json").is_file()
    assert (root / "scenarios" / "night_hypo_risk.json").is_file()
    assert (root / "FPGA_STORY.md").is_file()
    contract = json.loads((root / "fpga_safety_contract.json").read_text(encoding="utf-8"))
    assert contract["medical_device"] is False
    assert contract["insulin_delivery_enabled"] is False
    assert "not a medical device" in (root / "README.md").read_text(encoding="utf-8")


def test_run_fpga_mock_simulation_writes_comparison_artifacts(tmp_path: Path) -> None:
    summary = run_fpga_safety_simulation(output_dir=tmp_path / "fpga_run")

    assert summary.mismatch_count == 0
    assert summary.results_csv.is_file()
    assert summary.comparison_json.is_file()
    assert summary.report_md.is_file()
    assert (summary.output_dir / "events.csv").is_file()
    assert (summary.output_dir / "results.json").is_file()
    assert (summary.output_dir / "manifest.json").is_file()
    assert (summary.output_dir / "report.md").is_file()
    comparison = json.loads(summary.comparison_json.read_text(encoding="utf-8"))
    assert comparison["passed"] is True
    assert comparison["event_count"] > 0
    assert "Software Reference Logic" in summary.report_md.read_text(encoding="utf-8")
