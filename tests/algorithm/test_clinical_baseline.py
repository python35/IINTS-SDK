from __future__ import annotations

from iints.api.base_algorithm import AlgorithmInput
from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm


def test_clinical_baseline_delivers_meal_and_correction_insulin() -> None:
    algorithm = ClinicalBaselineAlgorithm()

    result = algorithm.predict_insulin(
        AlgorithmInput(
            current_glucose=195.0,
            predicted_glucose_30min=220.0,
            predicted_glucose_30min_std=8.0,
            insulin_on_board=0.8,
            carb_intake=60.0,
            time_step=5.0,
        )
    )

    assert result["total_insulin_delivered"] > 0.0
    assert result["meal_bolus"] > 0.0
    assert result["correction_bolus"] > 0.0


def test_clinical_baseline_reduces_basal_when_glucose_is_falling() -> None:
    algorithm = ClinicalBaselineAlgorithm()

    result = algorithm.predict_insulin(
        AlgorithmInput(
            current_glucose=82.0,
            glucose_trend_mgdl_min=-2.0,
            insulin_on_board=1.2,
            time_step=5.0,
        )
    )

    assert result["basal_insulin"] < (0.75 / 60.0) * 5.0
    assert result["correction_bolus"] == 0.0
