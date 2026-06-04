from __future__ import annotations

from iints.core.algorithms.mock_algorithms import ConstantDoseAlgorithm
from iints.core.patient.models import PatientModel
from iints.core.simulator import Simulator


def _simulator() -> Simulator:
    return Simulator(
        patient_model=PatientModel(initial_glucose=120.0),
        algorithm=ConstantDoseAlgorithm(dose=0.0),
        time_step=5,
    )


def test_xai_meal_response_uses_previous_trend_not_previous_glucose() -> None:
    simulator = _simulator()
    simulator.patient_model.carbs_on_board = 30.0
    simulator._xai_previous_glucose_trend = 0.0

    events = simulator._build_explainable_events(
        current_time=610.0,
        actual_glucose_reading=150.0,
        glucose_trend=0.8,
        predicted_glucose_30=175.0,
        safety_triggered=False,
        safety_result={},
    )

    assert events == ["At 10:10 glucose started rising after meal/breakfast."]


def test_xai_event_cooldown_prevents_repeated_meal_spam() -> None:
    simulator = _simulator()
    simulator.patient_model.carbs_on_board = 30.0
    simulator._xai_previous_glucose_trend = 0.0

    first = simulator._build_explainable_events(
        current_time=610.0,
        actual_glucose_reading=150.0,
        glucose_trend=0.8,
        predicted_glucose_30=175.0,
        safety_triggered=False,
        safety_result={},
    )
    second = simulator._build_explainable_events(
        current_time=620.0,
        actual_glucose_reading=156.0,
        glucose_trend=0.9,
        predicted_glucose_30=180.0,
        safety_triggered=False,
        safety_result={},
    )

    assert first
    assert second == []


def test_xai_absorption_anomaly_requires_actual_prediction_miss() -> None:
    simulator = _simulator()
    simulator.patient_model.carbs_on_board = 20.0
    simulator._predictor_history = [
        {"predicted_glucose_heuristic_30min": 130.0}
        for _ in range(simulator._predictor_horizon_steps)
    ]

    events = simulator._build_explainable_events(
        current_time=615.0,
        actual_glucose_reading=150.5,
        glucose_trend=0.6,
        predicted_glucose_30=155.0,
        safety_triggered=False,
        safety_result={},
    )

    assert events == ["At 10:15 the model detected faster-than-expected absorption."]


def test_xai_supervisor_message_matches_reason() -> None:
    simulator = _simulator()

    predicted_low_events = simulator._build_explainable_events(
        current_time=765.0,
        actual_glucose_reading=92.0,
        glucose_trend=-0.4,
        predicted_glucose_30=64.0,
        safety_triggered=True,
        safety_result={"insulin_reduction": 1.0, "safety_reason": "PREDICTED_LOW"},
    )

    assert predicted_low_events == [
        "At 12:45 supervisor intervention prevented predicted glucose below 70 mg/dL."
    ]

    simulator = _simulator()
    generic_events = simulator._build_explainable_events(
        current_time=765.0,
        actual_glucose_reading=180.0,
        glucose_trend=0.1,
        predicted_glucose_30=190.0,
        safety_triggered=True,
        safety_result={"insulin_reduction": 1.0, "safety_reason": "MAX_IOB"},
    )

    assert generic_events == [
        "At 12:45 supervisor intervention reduced insulin after an independent safety check."
    ]
