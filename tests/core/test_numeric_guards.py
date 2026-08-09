import math

import pytest

from iints.api.base_algorithm import AlgorithmInput
from iints.core.algorithms.pid_controller import PIDController
from iints.core.patient.models import CustomPatientModel
from iints.core.simulator import Simulator


@pytest.mark.parametrize(
    ("insulin_action_duration", "insulin_peak_time"),
    [
        (0.0, 75.0),
        (60.0, 0.0),
        (75.0, 75.0),
    ],
)
def test_custom_patient_model_rejects_invalid_insulin_curve_parameters(
    insulin_action_duration: float,
    insulin_peak_time: float,
) -> None:
    with pytest.raises(ValueError):
        CustomPatientModel(
            insulin_action_duration=insulin_action_duration,
            insulin_peak_time=insulin_peak_time,
        )


def test_custom_patient_model_rejects_zero_carb_absorption_duration() -> None:
    with pytest.raises(ValueError):
        CustomPatientModel(carb_absorption_duration_minutes=0.0)


def test_predict_glucose_handles_zero_carb_absorption_minutes() -> None:
    simulator = Simulator(
        patient_model=CustomPatientModel(),
        algorithm=PIDController(),
    )

    predicted = simulator._predict_glucose(
        current_glucose=100.0,
        trend_mgdl_min=0.0,
        iob_units=0.0,
        cob_grams=10.0,
        isf=50.0,
        icr=10.0,
        dia_minutes=240.0,
        horizon_minutes=30,
        carb_absorption_minutes=0.0,
    )

    assert math.isfinite(predicted)
    assert predicted == pytest.approx(150.0)


def test_pid_controller_integral_is_clamped_under_sustained_error() -> None:
    controller = PIDController()
    high_glucose = AlgorithmInput(current_glucose=400.0, time_step=5.0)
    low_glucose = AlgorithmInput(current_glucose=40.0, time_step=5.0)

    for _ in range(200):
        result = controller.predict_insulin(high_glucose)
        assert result["total_insulin_delivered"] <= controller.max_insulin

    assert controller.integral <= controller.integral_limit

    for _ in range(200):
        result = controller.predict_insulin(low_glucose)
        assert result["total_insulin_delivered"] >= controller.min_insulin

    assert controller.integral >= -controller.integral_limit
