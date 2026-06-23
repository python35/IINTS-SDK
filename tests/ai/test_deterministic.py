from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given, strategies as st
import pytest

from iints.ai.deterministic import calculate_deterministic_dose


def test_deterministic_dose_matches_golden_vectors() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "deterministic_dose_golden.json"
    vectors = json.loads(fixture.read_text(encoding="utf-8"))

    for vector in vectors:
        result = calculate_deterministic_dose(vector["input"])
        for field, expected in vector["expected"].items():
            assert getattr(result, field) == expected, vector["name"]


def test_deterministic_dose_is_identical_for_identical_inputs() -> None:
    payload = {
        "current_glucose": 185.0,
        "predicted_glucose_30min": 170.0,
        "glucose_trend_mgdl_min": 0.4,
        "insulin_on_board": 1.2,
        "mpc_recommended_units": 0.45,
        "deterministic_glucagon_candidate_mg": 0.0,
    }

    first = calculate_deterministic_dose(payload)
    second = calculate_deterministic_dose(dict(reversed(list(payload.items()))))

    assert first == second
    assert first.final_insulin_units == 0.45
    assert first.final_glucagon_mg == 0.0
    assert first.input_fingerprint_sha256 == second.input_fingerprint_sha256


def test_fixed_safety_guards_hold_insulin_without_ai_judgment() -> None:
    result = calculate_deterministic_dose(
        {
            "current_glucose": 105.0,
            "predicted_glucose_30min": 82.0,
            "glucose_trend_mgdl_min": -1.2,
            "insulin_on_board": 4.5,
            "mpc_recommended_units": 0.7,
        }
    )

    assert result.safety_hold is True
    assert result.final_insulin_units == 0.0
    assert "predicted_glucose_at_or_below_fixed_guard" in result.reasons
    assert "falling_trend_at_or_below_fixed_guard" in result.reasons
    assert "iob_at_or_above_fixed_guard" in result.reasons


def test_deterministic_candidates_are_clamped_to_explicit_hard_caps() -> None:
    result = calculate_deterministic_dose(
        {
            "current_glucose": 190.0,
            "mpc_recommended_units": 1.1,
            "hard_max_insulin_units": 0.6,
            "deterministic_glucagon_candidate_mg": 0.25,
            "hard_max_glucagon_mg": 0.1,
        }
    )

    assert result.final_insulin_units == 0.6
    assert result.final_glucagon_mg == 0.1
    assert "insulin_clamped_to_hard_cap" in result.reasons
    assert "glucagon_clamped_to_hard_cap" in result.reasons


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not-a-number"])
def test_deterministic_dose_rejects_invalid_mpc_values(value: object) -> None:
    with pytest.raises(ValueError, match="mpc_recommended_units"):
        calculate_deterministic_dose({"mpc_recommended_units": value})


@given(
    mpc=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    cap=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_deterministic_insulin_never_exceeds_controller_or_cap(mpc: float, cap: float) -> None:
    result = calculate_deterministic_dose(
        {
            "current_glucose": 180.0,
            "mpc_recommended_units": mpc,
            "hard_max_insulin_units": cap,
        }
    )

    assert 0.0 <= result.final_insulin_units <= min(mpc, cap)


@given(
    glucose=st.floats(min_value=20.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    mpc=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_low_glucose_property_always_holds_insulin(glucose: float, mpc: float) -> None:
    result = calculate_deterministic_dose(
        {"current_glucose": glucose, "mpc_recommended_units": mpc}
    )

    assert result.safety_hold is True
    assert result.final_insulin_units == 0.0
