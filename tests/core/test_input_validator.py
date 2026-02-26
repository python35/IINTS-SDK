import pytest

from iints.core.safety.input_validator import InputValidator


def test_input_validator_rejects_negative_insulin():
    validator = InputValidator()
    assert validator.validate_insulin(-0.5) == 0.0


def test_input_validator_rejects_unrealistic_glucose_jump():
    validator = InputValidator(max_glucose_delta_per_5_min=20.0)

    validator.validate_glucose(100.0, current_time=0.0)

    with pytest.raises(ValueError, match="RATE_OF_CHANGE_ERROR"):
        validator.validate_glucose(200.0, current_time=5.0)
