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


def test_input_validator_rejects_out_of_sensor_range_values():
    validator = InputValidator()

    with pytest.raises(ValueError, match="BIOLOGICAL_PLAUSIBILITY_ERROR"):
        validator.validate_glucose(39.0, current_time=0.0)

    with pytest.raises(ValueError, match="BIOLOGICAL_PLAUSIBILITY_ERROR"):
        validator.validate_glucose(501.0, current_time=5.0)


@pytest.mark.parametrize("dose", [float("nan"), float("inf"), "invalid"])
def test_input_validator_rejects_malformed_insulin(dose):
    validator = InputValidator()
    with pytest.raises(ValueError, match="insulin dose"):
        validator.validate_insulin(dose)


def test_input_validator_rejects_invalid_configuration_and_snapshot():
    with pytest.raises(ValueError):
        InputValidator(min_glucose=500.0, max_glucose=40.0)

    validator = InputValidator()
    with pytest.raises(ValueError, match="both glucose and time"):
        validator.set_state({"last_valid_glucose": 120.0})
    with pytest.raises(ValueError, match="outside configured bounds"):
        validator.set_state(
            {"last_valid_glucose": 900.0, "last_validation_time": 0.0}
        )
