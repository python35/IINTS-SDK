"""Tests for the dawn insulin-resistance component.

The dawn phenomenon was previously modelled only as an additive glucose
inflow. A closed-loop controller can dose an inflow away, so an inflow-only
dawn makes an algorithm look better overnight than it would be in a patient.
These tests pin the resistance component that was added alongside it, and in
particular the property that motivated it: under resistance, extra insulin
buys less glucose lowering.
"""
from __future__ import annotations

import math

import pytest

from iints.core.patient.models import CustomPatientModel
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.patient.bergman_model import BergmanPatientModel
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.patient.physiology import (
    dawn_glucose_rate_mgdl_min,
    dawn_insulin_sensitivity_multiplier,
    dawn_window_fraction,
)

# Every backend that accepts the dawn settings, so the parameter cannot be
# wired into some of them and silently ignored by the rest.
BACKENDS = [
    CustomPatientModel,
    BergmanPatientModel,
    HovorkaPatientModel,
    AdvancedMetabolicModel,
]

DAWN_START = 4.0
DAWN_END = 8.0
DAWN_MIDPOINT_MINUTE = 6.0 * 60.0


# --- The window itself -------------------------------------------------------


def test_window_is_zero_outside_and_one_at_the_midpoint() -> None:
    for minute in (0.0, 3.9 * 60.0, 8.1 * 60.0, 20.0 * 60.0):
        assert dawn_window_fraction(
            minute, start_hour=DAWN_START, end_hour=DAWN_END
        ) == 0.0
    assert dawn_window_fraction(
        DAWN_MIDPOINT_MINUTE, start_hour=DAWN_START, end_hour=DAWN_END
    ) == pytest.approx(1.0)


def test_window_closes_continuously_at_both_edges() -> None:
    # A raised cosine reaches zero at the edges, so no discontinuous jump in
    # either dawn term at the configured start and end.
    for edge in (DAWN_START * 60.0, DAWN_END * 60.0):
        assert dawn_window_fraction(
            edge, start_hour=DAWN_START, end_hour=DAWN_END
        ) == pytest.approx(0.0, abs=1e-12)


def test_both_dawn_terms_share_one_window() -> None:
    # The inflow and the resistance must peak at the same minute; if they can
    # drift apart, a run's two dawn components describe different mornings.
    for minute in range(0, 1_440, 7):
        window = dawn_window_fraction(
            float(minute), start_hour=DAWN_START, end_hour=DAWN_END
        )
        rate = dawn_glucose_rate_mgdl_min(
            float(minute),
            peak_strength_mgdl_per_hour=18.0,
            start_hour=DAWN_START,
            end_hour=DAWN_END,
        )
        multiplier = dawn_insulin_sensitivity_multiplier(
            float(minute),
            peak_resistance_fraction=0.4,
            start_hour=DAWN_START,
            end_hour=DAWN_END,
        )
        assert rate == pytest.approx(18.0 * window / 60.0)
        assert multiplier == pytest.approx(1.0 - 0.4 * window)


def test_window_repeats_daily() -> None:
    assert dawn_window_fraction(
        DAWN_MIDPOINT_MINUTE + 1_440.0, start_hour=DAWN_START, end_hour=DAWN_END
    ) == pytest.approx(1.0)


# --- The multiplier ----------------------------------------------------------


def test_multiplier_is_a_no_op_at_zero_resistance() -> None:
    for minute in (0.0, DAWN_MIDPOINT_MINUTE, 1_000.0):
        assert dawn_insulin_sensitivity_multiplier(
            minute,
            peak_resistance_fraction=0.0,
            start_hour=DAWN_START,
            end_hour=DAWN_END,
        ) == 1.0


def test_multiplier_reaches_the_requested_loss_at_the_peak() -> None:
    assert dawn_insulin_sensitivity_multiplier(
        DAWN_MIDPOINT_MINUTE,
        peak_resistance_fraction=0.35,
        start_hour=DAWN_START,
        end_hour=DAWN_END,
    ) == pytest.approx(0.65)


def test_multiplier_never_abolishes_insulin_action() -> None:
    # Sensitivity may fall a long way but never to zero: a zero multiplier
    # would make glucose independent of every dose the algorithm gives.
    worst = dawn_insulin_sensitivity_multiplier(
        DAWN_MIDPOINT_MINUTE,
        peak_resistance_fraction=0.999,
        start_hour=DAWN_START,
        end_hour=DAWN_END,
    )
    assert 0.0 < worst < 1.0


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5, math.nan, math.inf])
def test_multiplier_rejects_out_of_range_resistance(fraction) -> None:
    with pytest.raises(ValueError):
        dawn_insulin_sensitivity_multiplier(
            DAWN_MIDPOINT_MINUTE,
            peak_resistance_fraction=fraction,
            start_hour=DAWN_START,
            end_hour=DAWN_END,
        )


# --- Backend wiring ----------------------------------------------------------


def _common(**overrides):
    settings = {
        "initial_glucose": 200.0,
        "basal_insulin_rate": 0.0,
        "dawn_phenomenon_strength": 0.0,
        "dawn_start_hour": DAWN_START,
        "dawn_end_hour": DAWN_END,
        "max_glucose_rate_mgdl_per_min": 10.0,
    }
    settings.update(overrides)
    return settings


def _run_dawn(model_type, *, resistance, insulin_per_step, minutes=None):
    """Step a backend across the dawn window at a fixed insulin rate."""

    model = model_type(**_common(dawn_insulin_resistance_fraction=resistance))
    window = minutes if minutes is not None else range(4 * 60, 8 * 60, 5)
    for minute in window:
        model.update(5.0, insulin_per_step, current_time=float(minute))
    return model.get_current_glucose()


@pytest.mark.parametrize("model_type", BACKENDS)
def test_zero_resistance_is_an_exact_no_op(model_type) -> None:
    # Existing runs, presets and calibrated profiles must be untouched by the
    # new parameter, so the default has to reproduce the old trajectory.
    without = model_type(**_common())
    explicit_zero = model_type(**_common(dawn_insulin_resistance_fraction=0.0))
    for minute in range(4 * 60, 8 * 60, 5):
        without.update(5.0, 0.05, current_time=float(minute))
        explicit_zero.update(5.0, 0.05, current_time=float(minute))
    assert explicit_zero.get_current_glucose() == pytest.approx(
        without.get_current_glucose()
    )


@pytest.mark.parametrize("model_type", BACKENDS)
def test_resistance_raises_glucose_during_dawn(model_type) -> None:
    sensitive = _run_dawn(model_type, resistance=0.0, insulin_per_step=0.1)
    resistant = _run_dawn(model_type, resistance=0.6, insulin_per_step=0.1)
    assert resistant > sensitive


@pytest.mark.parametrize("model_type", BACKENDS)
def test_resistance_does_nothing_outside_the_dawn_window(model_type) -> None:
    evening = range(18 * 60, 22 * 60, 5)
    sensitive = _run_dawn(
        model_type, resistance=0.0, insulin_per_step=0.1, minutes=evening
    )
    resistant = _run_dawn(
        model_type, resistance=0.6, insulin_per_step=0.1, minutes=evening
    )
    assert resistant == pytest.approx(sensitive)


@pytest.mark.parametrize("model_type", BACKENDS)
def test_extra_insulin_buys_less_lowering_under_resistance(model_type) -> None:
    """The property that motivated the change.

    An additive glucose inflow can be dosed away: more insulin cancels more of
    it, one for one. A loss of sensitivity cannot, because the extra insulin is
    itself less effective. So the glucose lowering achieved by the *same* dose
    increment must be smaller when resistance is present.
    """

    low, high = 0.05, 0.15

    sensitive_lowering = _run_dawn(
        model_type, resistance=0.0, insulin_per_step=low
    ) - _run_dawn(model_type, resistance=0.0, insulin_per_step=high)
    resistant_lowering = _run_dawn(
        model_type, resistance=0.6, insulin_per_step=low
    ) - _run_dawn(model_type, resistance=0.6, insulin_per_step=high)

    assert sensitive_lowering > 0.0
    assert resistant_lowering > 0.0
    assert resistant_lowering < sensitive_lowering


@pytest.mark.parametrize("model_type", BACKENDS)
def test_backends_reject_out_of_range_resistance(model_type) -> None:
    for fraction in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="dawn_insulin_resistance_fraction"):
            model_type(**_common(dawn_insulin_resistance_fraction=fraction))


def test_profile_forwards_resistance_to_the_patient_config() -> None:
    from iints.core.patient.profile import PatientProfile

    config = PatientProfile(dawn_insulin_resistance_fraction=0.25).to_patient_config()
    assert config["dawn_insulin_resistance_fraction"] == 0.25
