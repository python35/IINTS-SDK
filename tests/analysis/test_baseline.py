from __future__ import annotations

import pandas as pd

from iints.analysis.baseline import run_baseline_comparison


def test_baseline_comparison_preserves_completion_context() -> None:
    primary_results = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [120.0, 118.0, 116.0],
        }
    )
    primary_safety = {
        "terminated_early": True,
        "termination_reason": {
            "reason": "critical hypoglycemia",
            "current_time_minutes": 15,
        },
    }

    comparison = run_baseline_comparison(
        patient_params={
            "basal_insulin_rate": 0.0,
            "insulin_sensitivity": 40.0,
            "carb_factor": 15.0,
            "glucose_decay_rate": 0.0,
            "initial_glucose": 120.0,
        },
        stress_event_payloads=[],
        duration=60,
        time_step=5,
        primary_label="Primary",
        primary_results=primary_results,
        primary_safety=primary_safety,
        compare_standard_pump=False,
        seed=42,
    )

    row = comparison["rows"][0]
    assert row["requested_duration_minutes"] == 60
    assert row["completed_duration_minutes"] == 15
    assert row["completion_ratio_pct"] == 25.0
    assert row["terminated_early"] is True
    assert row["termination_reason"] == "critical hypoglycemia"


def test_baseline_comparison_filters_disabled_model_specific_defaults() -> None:
    primary_results = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [120.0, 119.0, 118.0],
        }
    )

    comparison = run_baseline_comparison(
        patient_params={
            "basal_insulin_rate": 0.0,
            "insulin_sensitivity": 40.0,
            "carb_factor": 15.0,
            "glucose_decay_rate": 0.0,
            "initial_glucose": 120.0,
            "stem_cell_engraftment_percent": 0.0,
            "stem_cell_subq_fraction": 0.0,
            "immune_rejection_rate": 0.0,
        },
        patient_model_type="custom",
        stress_event_payloads=[],
        duration=10,
        time_step=5,
        primary_label="Primary",
        primary_results=primary_results,
        primary_safety={},
        compare_standard_pump=False,
        seed=42,
    )

    assert [row["algorithm"] for row in comparison["rows"]] == [
        "Primary",
        "Clinical Baseline",
        "Standard PID",
    ]


def test_baseline_comparison_uses_bergman_for_enabled_stem_cell_experiment() -> None:
    primary_results = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [150.0, 149.0, 148.0],
        }
    )

    comparison = run_baseline_comparison(
        patient_params={
            "basal_insulin_rate": 0.0,
            "insulin_sensitivity": 40.0,
            "carb_factor": 15.0,
            "glucose_decay_rate": 0.0,
            "initial_glucose": 150.0,
            "stem_cell_engraftment_percent": 50.0,
            "stem_cell_subq_fraction": 0.25,
            "immune_rejection_rate": 0.0,
        },
        patient_model_type="bergman",
        stress_event_payloads=[],
        duration=10,
        time_step=5,
        primary_label="Primary",
        primary_results=primary_results,
        primary_safety={},
        compare_standard_pump=False,
        seed=42,
    )

    assert len(comparison["rows"]) == 3
