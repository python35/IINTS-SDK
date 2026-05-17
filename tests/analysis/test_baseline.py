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
