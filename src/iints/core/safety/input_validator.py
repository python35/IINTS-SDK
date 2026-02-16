from typing import Optional

from iints.core.safety.config import SafetyConfig

class InputValidator:
    """
    A biological validation filter for sensor inputs to ensure they are
    physiologically plausible before being used by an algorithm.
    This component makes the system robust against common sensor errors.
    """
    def __init__(self,
                 min_glucose: float = 20.0,
                 max_glucose: float = 600.0,
                 max_glucose_delta_per_5_min: float = 35.0,
                 safety_config: Optional[SafetyConfig] = None):
        """
        Initializes the validator with plausible biological limits.

        Args:
            min_glucose (float): The absolute minimum plausible glucose value (mg/dL).
            max_glucose (float): The absolute maximum plausible glucose value (mg/dL).
            max_glucose_delta_per_5_min (float): The maximum plausible change in glucose
                                                 over a 5-minute period (mg/dL).
        """
        if safety_config is not None:
            min_glucose = safety_config.min_glucose
            max_glucose = safety_config.max_glucose
            max_glucose_delta_per_5_min = safety_config.max_glucose_delta_per_5_min

        self.min_glucose = min_glucose
        self.max_glucose = max_glucose
        self.max_glucose_delta_per_5_min = max_glucose_delta_per_5_min
        self.last_valid_glucose: Optional[float] = None
        self.last_validation_time: Optional[float] = None

    def reset(self):
        """Resets the state of the validator for a new simulation."""
        self.last_valid_glucose = None
        self.last_validation_time = None

    def get_state(self) -> dict:
        return {
            "last_valid_glucose": self.last_valid_glucose,
            "last_validation_time": self.last_validation_time,
        }

    def set_state(self, state: dict) -> None:
        self.last_valid_glucose = state.get("last_valid_glucose")
        self.last_validation_time = state.get("last_validation_time")

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
        # 1. Absolute biological plausibility check
        if not (self.min_glucose <= glucose_value <= self.max_glucose):
            raise ValueError(
                f"BIOLOGICAL_PLAUSIBILITY_ERROR: Glucose {glucose_value} mg/dL is outside the "
                f"valid range [{self.min_glucose}, {self.max_glucose}]."
            )

        # 2. Rate-of-change check for unrealistic jumps
        if self.last_valid_glucose is not None and self.last_validation_time is not None:
            time_delta = current_time - self.last_validation_time
            if time_delta > 0:
                # Normalize the max allowed delta to the actual time step
                allowed_delta = self.max_glucose_delta_per_5_min * (time_delta / 5.0)
                glucose_delta = abs(glucose_value - self.last_valid_glucose)

                if glucose_delta > allowed_delta:
                    raise ValueError(
                        f"RATE_OF_CHANGE_ERROR: Glucose jump of {glucose_delta:.1f} mg/dL over "
                        f"{time_delta:.1f} min is unrealistic (max allowed: {allowed_delta:.1f} mg/dL)."
                    )

        # If all checks pass, update state and return the value
        self.last_valid_glucose = glucose_value
        self.last_validation_time = current_time
        return glucose_value

    def validate_insulin(self, dose: float) -> float:
        """Validates that a proposed insulin dose is non-negative."""
        if dose < 0:
            raise ValueError(f"INVALID_DOSE_ERROR: Proposed insulin dose {dose} U cannot be negative.")
        return dose
