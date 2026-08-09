import math
from typing import Any, Dict, Optional

from iints.core.safety.config import (
    SENSOR_GLUCOSE_MAX_MGDL,
    SENSOR_GLUCOSE_MIN_MGDL,
    SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL,
    SafetyConfig,
)

class InputValidator:
    """
    A biological validation filter for sensor inputs to ensure they are
    physiologically plausible before being used by an algorithm.
    This component makes the system robust against common sensor errors.
    """
    def __init__(self,
                 min_glucose: float = SENSOR_GLUCOSE_MIN_MGDL,
                 max_glucose: float = SENSOR_GLUCOSE_MAX_MGDL,
                 max_glucose_delta_per_5_min: float = SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL,
                 safety_config: Optional[SafetyConfig] = None):
        """
        Initializes the validator with plausible biological limits.

        Args:
            min_glucose (float): Broad fail-soft lower bound for incoming CGM/sensor values (mg/dL).
            max_glucose (float): Broad fail-soft upper bound for incoming CGM/sensor values (mg/dL).
            max_glucose_delta_per_5_min (float): The maximum plausible change in glucose
                                                 over a 5-minute period (mg/dL).
        """
        if safety_config is not None:
            min_glucose = safety_config.min_glucose
            max_glucose = safety_config.max_glucose
            max_glucose_delta_per_5_min = safety_config.max_glucose_delta_per_5_min

        values = {
            "min_glucose": float(min_glucose),
            "max_glucose": float(max_glucose),
            "max_glucose_delta_per_5_min": float(max_glucose_delta_per_5_min),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("input-validator limits must all be finite")
        if not 0.0 <= values["min_glucose"] < values["max_glucose"]:
            raise ValueError("min_glucose must be non-negative and below max_glucose")
        if values["max_glucose_delta_per_5_min"] <= 0.0:
            raise ValueError("max_glucose_delta_per_5_min must be positive")

        self.min_glucose = values["min_glucose"]
        self.max_glucose = values["max_glucose"]
        self.max_glucose_delta_per_5_min = values[
            "max_glucose_delta_per_5_min"
        ]
        self.last_valid_glucose: Optional[float] = None
        self.last_validation_time: Optional[float] = None

    def reset(self) -> None:
        """Resets the state of the validator for a new simulation."""
        self.last_valid_glucose = None
        self.last_validation_time = None

    def get_state(self) -> Dict[str, Any]:
        return {
            "last_valid_glucose": self.last_valid_glucose,
            "last_validation_time": self.last_validation_time,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        glucose = state.get("last_valid_glucose")
        timestamp = state.get("last_validation_time")
        if (glucose is None) != (timestamp is None):
            raise ValueError(
                "validator snapshot must provide both glucose and time or neither"
            )
        if glucose is None:
            self.reset()
            return
        assert timestamp is not None
        try:
            restored_glucose = float(glucose)
            restored_time = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("validator snapshot values must be numeric") from exc
        if not all(math.isfinite(value) for value in (restored_glucose, restored_time)):
            raise ValueError("validator snapshot values must be finite")
        if not self.min_glucose <= restored_glucose <= self.max_glucose:
            raise ValueError("validator snapshot glucose is outside configured bounds")
        self.last_valid_glucose = restored_glucose
        self.last_validation_time = restored_time

    def validate_glucose(self, glucose_value: float, current_time: float) -> float:
        """
        Validates a glucose reading against absolute and rate-of-change limits.

        Args:
            glucose_value (float): The incoming glucose reading from the sensor.
            current_time (float): The current simulation time in minutes.

        Returns:
            float: The validated glucose value.

        Raises:
            ValueError: If the value is outside biological plausibility limits.
        """
        try:
            glucose = float(glucose_value)
            timestamp = float(current_time)
        except (TypeError, ValueError) as exc:
            raise ValueError("glucose and current_time must be numeric") from exc
        if not math.isfinite(glucose) or not math.isfinite(timestamp):
            raise ValueError("glucose and current_time must be finite")

        # 1. Broad CGM/sensor plausibility check
        if not (self.min_glucose <= glucose <= self.max_glucose):
            raise ValueError(
                f"BIOLOGICAL_PLAUSIBILITY_ERROR: Glucose {glucose} mg/dL is outside the "
                f"valid range [{self.min_glucose}, {self.max_glucose}]."
            )

        # 2. Rate-of-change check for unrealistic jumps
        if self.last_valid_glucose is not None and self.last_validation_time is not None:
            time_delta = timestamp - self.last_validation_time
            if time_delta < 0.0:
                raise ValueError("current_time must not move backwards")
            if time_delta > 0:
                # Normalize the max allowed delta to the actual time step
                allowed_delta = self.max_glucose_delta_per_5_min * (time_delta / 5.0)
                glucose_delta = abs(glucose - self.last_valid_glucose)

                if glucose_delta > allowed_delta:
                    raise ValueError(
                        f"RATE_OF_CHANGE_ERROR: Glucose jump of {glucose_delta:.1f} mg/dL over "
                        f"{time_delta:.1f} min is unrealistic (max allowed: {allowed_delta:.1f} mg/dL)."
                    )

        # If all checks pass, update state and return the value
        self.last_valid_glucose = glucose
        self.last_validation_time = timestamp
        return glucose

    def validate_insulin(self, dose: float) -> float:
        """Fail safely for negative requests and reject malformed requests."""
        try:
            numeric = float(dose)
        except (TypeError, ValueError) as exc:
            raise ValueError("insulin dose must be numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError("insulin dose must be finite")
        if numeric < 0.0:
            return 0.0
        return numeric
