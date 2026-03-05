from __future__ import annotations

from pathlib import Path

from iints.research.calibration_gate import (
    evaluate_calibration_gate,
    load_calibration_gate_profiles,
)
from iints.research.model_registry import (
    append_registry_entry,
    list_registry,
    promote_registry_run,
)


def test_load_default_calibration_gate_profiles() -> None:
    profiles = load_calibration_gate_profiles()
    assert "research_default" in profiles
    assert "strict" in profiles


def test_calibration_gate_evaluation_flags_failures() -> None:
    profile = load_calibration_gate_profiles()["strict"]
    report = {
        "within_20_mgdl_pct": 65.0,           # below strict 70
        "false_hypo_alarm_rate_pct": 8.0,     # pass
        "missed_hypo_rate_pct": 12.0,         # fail (>10)
        "interval_95_coverage_pct": 75.0,     # fail (<80)
    }
    checks = evaluate_calibration_gate(report, profile)
    assert checks["within_20_mgdl_pct"]["passed"] is False
    assert checks["false_hypo_alarm_rate_pct"]["passed"] is True
    assert checks["missed_hypo_rate_pct"]["passed"] is False
    assert checks["interval_95_coverage_pct"]["passed"] is False


def test_model_registry_promotion_flow(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    append_registry_entry(
        registry,
        {
            "run_id": "run-001",
            "stage": "candidate",
            "test_rmse": 18.2,
            "timestamp_utc": "2026-03-05T00:00:00Z",
        },
    )
    rows = list_registry(registry)
    assert len(rows) == 1
    assert rows[0]["stage"] == "candidate"

    blocked = promote_registry_run(registry, run_id="run-001", stage="production", force=False)
    assert blocked.updated is False

    validated = promote_registry_run(registry, run_id="run-001", stage="validated")
    assert validated.updated is True
    promoted = promote_registry_run(registry, run_id="run-001", stage="production")
    assert promoted.updated is True

    rows_after = list_registry(registry)
    assert rows_after[0]["stage"] == "production"
