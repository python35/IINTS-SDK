"""Pin the sensor rate-of-change ceilings to the measured step distribution.

The ceilings were 20 mg/dL per 5 min, which sits below the 99th percentile of real
5-minute steps in all three prepared cohorts and so rejected genuine physiology as
sensor error. These tests keep the two constants consistent with each other and above
the highest measured p99.9, so a future edit cannot quietly restore a ceiling that
discards roughly 1 in 100 real steps.

The empirical values are documented in src/iints/core/safety/config.py.
"""

from __future__ import annotations

import pytest

from iints.core.safety.config import (
    SENSOR_FAIL_SOFT_MAX_FOLLOW_PER_5_MIN_MGDL,
    SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL,
    SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL,
    SafetyConfig,
)
from iints.core.safety.input_validator import InputValidator

# Highest p99.9 of absolute 5-minute steps across AZT1D, HUPA-UCM and OhioT1DM.
MEASURED_MAX_P99_9_MGDL_PER_5_MIN = 54.0


def test_five_minute_ceiling_admits_measured_p99_9() -> None:
    assert SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL >= MEASURED_MAX_P99_9_MGDL_PER_5_MIN, (
        "the 5-minute ceiling rejects steps that occur in the real cohorts; "
        "see the measured distribution in core/safety/config.py"
    )


def test_the_two_ceilings_state_the_same_claim() -> None:
    implied_per_minute = SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL / 5.0
    assert implied_per_minute == pytest.approx(SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL, abs=0.5), (
        "the per-5-min and per-minute ceilings disagree, so data quality checks and "
        "input validation would reject different steps"
    )


def test_a_steep_but_real_fall_is_not_rejected() -> None:
    """A 34 mg/dL fall in 5 minutes occurs in all three cohorts and must pass.

    This is the safety-relevant direction: the old ceiling raised on exactly the
    rapid fall that the supervisor exists to respond to.
    """
    validator = InputValidator(safety_config=SafetyConfig())
    validator.validate_glucose(180.0, current_time=0.0)
    assert validator.validate_glucose(146.0, current_time=5.0) == 146.0


def test_fail_soft_damping_stays_tighter_than_the_plausibility_ceiling() -> None:
    """The two limits answer opposite questions and must not be merged again.

    The ceiling decides what counts as a real step, so it has to be permissive.
    The damping decides how fast the reported value may follow a reading that was
    already rejected, so it has to stay tight — that is the path protecting the
    algorithm from injected sensor corruption.
    """
    assert SENSOR_FAIL_SOFT_MAX_FOLLOW_PER_5_MIN_MGDL < SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL
    assert SafetyConfig().fail_soft_max_follow_per_5_min == SENSOR_FAIL_SOFT_MAX_FOLLOW_PER_5_MIN_MGDL


def test_an_artifact_sized_jump_is_still_rejected() -> None:
    validator = InputValidator(safety_config=SafetyConfig())
    validator.validate_glucose(120.0, current_time=0.0)
    with pytest.raises(ValueError, match="RATE_OF_CHANGE_ERROR"):
        validator.validate_glucose(300.0, current_time=5.0)
