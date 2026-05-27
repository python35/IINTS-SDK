from __future__ import annotations

from iints.research.local_ai_gate import (
    review_closed_loop_evaluation,
    review_controller_training_artifacts,
)


def test_training_gate_blocks_tiny_or_unsafe_controller_training_set() -> None:
    result = review_controller_training_artifacts(
        {
            "rows": 4,
            "max_teacher_insulin_units": 6.0,
            "teacher_source_columns": ["delivered_insulin_units"],
        },
        train_metrics={
            "unsafe_hypo_proposal_rows": 2,
            "over_5u_proposal_rows": 1,
            "max_prediction_units": 6.5,
        },
    )

    assert result.status == "blocked"
    assert result.passed is False
    assert len(result.critical_failures) >= 3


def test_closed_loop_gate_blocks_hypo_regression_against_baseline() -> None:
    result = review_closed_loop_evaluation(
        {
            "clinical_baseline": {
                "mean_tir_70_180_pct": 72.0,
                "mean_time_below_70_pct": 2.0,
                "mean_time_below_54_pct": 0.0,
                "mean_supervisor_intervention_rate_pct": 1.0,
            },
            "candidate": {
                "mean_tir_70_180_pct": 75.0,
                "mean_time_below_70_pct": 4.5,
                "mean_time_below_54_pct": 1.0,
                "mean_supervisor_intervention_rate_pct": 5.0,
                "mean_completion_pct": 100.0,
                "terminated_early_runs": 0,
                "delta_vs_clinical_baseline": {
                    "tir_70_180_pct": 3.0,
                    "time_below_70_pct": 2.5,
                    "time_below_54_pct": 1.0,
                    "supervisor_intervention_rate_pct": 4.0,
                },
            },
        }
    )

    assert result.status == "blocked"
    assert any("severe hypo burden" in item for item in result.critical_failures)


def test_closed_loop_gate_passes_safe_candidate() -> None:
    result = review_closed_loop_evaluation(
        {
            "clinical_baseline": {
                "mean_tir_70_180_pct": 72.0,
                "mean_time_below_70_pct": 2.0,
                "mean_time_below_54_pct": 0.0,
                "mean_supervisor_intervention_rate_pct": 1.0,
            },
            "candidate": {
                "mean_tir_70_180_pct": 73.0,
                "mean_time_below_70_pct": 2.1,
                "mean_time_below_54_pct": 0.0,
                "mean_supervisor_intervention_rate_pct": 1.2,
                "mean_completion_pct": 100.0,
                "terminated_early_runs": 0,
                "delta_vs_clinical_baseline": {
                    "tir_70_180_pct": 1.0,
                    "time_below_70_pct": 0.1,
                    "time_below_54_pct": 0.0,
                    "supervisor_intervention_rate_pct": 0.2,
                },
            },
        }
    )

    assert result.status == "passed"
    assert result.passed is True
