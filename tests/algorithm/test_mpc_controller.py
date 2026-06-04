from __future__ import annotations

import pytest

from iints.api.base_algorithm import AlgorithmInput
from iints.core.algorithms.mpc_controller import MPCController


def _input(**overrides: object) -> AlgorithmInput:
    payload = {
        "current_glucose": 180.0,
        "time_step": 5.0,
        "insulin_on_board": 0.0,
        "carb_intake": 0.0,
        "patient_state": {},
        "current_time": 0.0,
        "glucose_trend_mgdl_min": 0.0,
        "predicted_glucose_30min": 180.0,
        "isf": 55.0,
    }
    payload.update(overrides)
    return AlgorithmInput(**payload)


def test_mpc_holds_insulin_when_low_or_falling() -> None:
    controller = MPCController()

    result = controller.predict_insulin(
        _input(
            current_glucose=86.0,
            predicted_glucose_30min=82.0,
            glucose_trend_mgdl_min=-1.3,
        )
    )

    assert result["total_insulin_delivered"] == pytest.approx(0.0)
    assert result["research_only"] is True
    assert "falling" in result["safety_reason"].lower()


def test_mpc_output_is_non_negative_and_bounded() -> None:
    controller = MPCController(settings={"max_insulin_u_per_step": 0.8})

    result = controller.predict_insulin(
        _input(current_glucose=240.0, predicted_glucose_30min=250.0)
    )

    assert 0.0 <= result["total_insulin_delivered"] <= 0.8
    assert result["mpc_recommended_units"] == pytest.approx(result["total_insulin_delivered"])
    physiology = result["mpc_physics_state"]
    assert "active_insulin" in physiology
    assert "insulin_effect" in physiology
    assert "delivered_insulin" in physiology


def test_mpc_tapers_to_zero_when_iob_is_high() -> None:
    controller = MPCController(
        settings={
            "iob_taper_start_units": 1.0,
            "high_iob_guard_units": 2.0,
            "max_insulin_u_per_step": 1.0,
        }
    )

    result = controller.predict_insulin(
        _input(current_glucose=210.0, predicted_glucose_30min=215.0, insulin_on_board=2.5)
    )

    assert result["total_insulin_delivered"] == pytest.approx(0.0)
    assert "active insulin" in result["safety_reason"].lower()
