from __future__ import annotations

import pandas as pd
import pytest

from iints.validation.run_validation import compute_run_metrics, evaluate_run, load_validation_profiles


def _make_results(glucose_values: list[float], time_step: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_minutes": [idx * time_step for idx in range(len(glucose_values))],
            "glucose_actual_mgdl": glucose_values,
            "safety_triggered": [False for _ in glucose_values],
            "supervisor_latency_ms": [0.15 for _ in glucose_values],
        }
    )


def test_load_validation_profiles_default_bundle() -> None:
    profiles = load_validation_profiles()
    assert "research_default" in profiles
    assert "strict_safety" in profiles
    assert profiles["research_default"].checks


def test_evaluate_run_passes_research_default_for_stable_trace() -> None:
    profiles = load_validation_profiles()
    df = _make_results([115.0] * 49)  # 240 minutes at 5-minute intervals
    report = evaluate_run(
        df,
        profile=profiles["research_default"],
        safety_report={"bolus_interventions_count": 0, "terminated_early": False},
    )
    assert report.passed is True
    assert report.required_checks_passed == report.required_checks_total


def test_evaluate_run_fails_when_hypoglycemia_burden_is_high() -> None:
    profiles = load_validation_profiles()
    # 240-minute run with frequent severe lows to trigger profile failures.
    glucose = [90.0] * 10 + [52.0] * 20 + [85.0] * 19
    df = _make_results(glucose)
    report = evaluate_run(
        df,
        profile=profiles["research_default"],
        safety_report={"bolus_interventions_count": 0, "terminated_early": False},
    )
    assert report.passed is False
    failed_metrics = {check.rule.metric for check in report.checks if not check.passed}
    assert "tir_below_54" in failed_metrics


def test_compute_run_metrics_requires_glucose_column() -> None:
    with pytest.raises(ValueError, match="glucose_actual_mgdl"):
        compute_run_metrics(pd.DataFrame({"time_minutes": [0, 5, 10]}))

