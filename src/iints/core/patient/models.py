import copy

import numpy as np
from typing import Dict, Any, Optional

from .physiology import (
    dawn_glucose_rate_mgdl_min,
    dawn_insulin_sensitivity_multiplier,
    validated_activity_events,
    validated_snapshot_bool,
    validated_snapshot_scalar,
)


class PatientModelDomainError(RuntimeError):
    """The model left its declared numerical/physiological validity domain."""

    def __init__(
        self,
        message: str,
        *,
        current_glucose: float,
        proposed_glucose: float,
    ) -> None:
        super().__init__(message)
        self.current_glucose = float(current_glucose)
        self.proposed_glucose = float(proposed_glucose)

# Use custom patient model as default to avoid simglucose dependency issues
# from simglucose.simulation.env import T1DSimEnv
# from simglucose.patient.t1dpatient import T1DPatient
# from simglucose.sensor.cgm import CGMSensor
# from simglucose.actuator.pump import InsulinPump
# from simglucose.controller.base import Action

class CustomPatientModel:
    """
    A simplified patient model for simulating blood glucose dynamics.
    This model is intended for educational and stress-testing purposes, not for clinical accuracy.
    """
    def __init__(self, basal_insulin_rate: float = 0.8, insulin_sensitivity: float = 50.0,
                 carb_factor: float = 10.0, glucose_decay_rate: float = 0.001,
                 initial_glucose: float = 120.0, glucose_absorption_rate: float = 0.03,
                 basal_glucose_target: Optional[float] = None,
                 insulin_action_duration: float = 300.0, # minutes, e.g., 5 hours
                 insulin_peak_time: float = 75.0, # minutes
                 meal_mismatch_epsilon: float = 1.0, # Legacy effective meal-appearance multiplier
                 dawn_phenomenon_strength: float = 0.0, # mg/dL per hour
                 dawn_insulin_resistance_fraction: float = 0.0, # peak fraction of insulin sensitivity lost
                 dawn_start_hour: float = 4.0,
                 dawn_end_hour: float = 8.0,
                 carb_absorption_duration_minutes: float = 240.0,
                 max_glucose_rate_mgdl_per_min: float = 3.0,
                 molecular_affinity_scalar: float = 1.0):
        """
        Initializes the patient model with simplified parameters.

        Args:
            basal_insulin_rate (float): Basal insulin rate in U/hr.
            insulin_sensitivity (float): How much 1 unit of insulin lowers glucose (mg/dL per Unit).
            carb_factor (float): How many carbs (g) are covered by 1 unit of insulin.
            glucose_decay_rate (float): Rate at which glucose naturally decreases (e.g., due to metabolism).
            initial_glucose (float): Starting blood glucose level (mg/dL).
            glucose_absorption_rate (float): Rate at which carbs are absorbed into glucose.
            insulin_action_duration (float): Duration of insulin action (DIA) in minutes.
            insulin_peak_time (float): Time to peak insulin activity in minutes.
            meal_mismatch_epsilon (float): Legacy name for the effective
                carbohydrate-appearance multiplier applied to the physical meal
                input. Announced-versus-actual mismatch is represented by the
                simulator scenario's ``reported_value``. Defaults to 1.0.
            molecular_affinity_scalar (float): Multiplier representing molecular binding affinity (1.0 = normal, 0.2 = 80% loss of function).
        """
        numeric_parameters = {
            "basal_insulin_rate": basal_insulin_rate,
            "insulin_sensitivity": insulin_sensitivity,
            "carb_factor": carb_factor,
            "glucose_decay_rate": glucose_decay_rate,
            "initial_glucose": initial_glucose,
            "glucose_absorption_rate": glucose_absorption_rate,
            "insulin_action_duration": insulin_action_duration,
            "insulin_peak_time": insulin_peak_time,
            "meal_mismatch_epsilon": meal_mismatch_epsilon,
            "dawn_phenomenon_strength": dawn_phenomenon_strength,
            "dawn_insulin_resistance_fraction": dawn_insulin_resistance_fraction,
            "dawn_start_hour": dawn_start_hour,
            "dawn_end_hour": dawn_end_hour,
            "carb_absorption_duration_minutes": carb_absorption_duration_minutes,
            "max_glucose_rate_mgdl_per_min": max_glucose_rate_mgdl_per_min,
            "molecular_affinity_scalar": molecular_affinity_scalar,
        }
        if not all(np.isfinite(float(value)) for value in numeric_parameters.values()):
            raise ValueError("patient parameters must contain only finite values")
        if basal_insulin_rate < 0.0:
            raise ValueError("basal_insulin_rate must be non-negative")
        if insulin_sensitivity <= 0.0 or carb_factor <= 0.0:
            raise ValueError("insulin_sensitivity and carb_factor must be positive")
        if glucose_decay_rate < 0.0 or glucose_absorption_rate <= 0.0:
            raise ValueError("glucose rates must be positive or zero as documented")
        if initial_glucose < 20.0:
            raise ValueError("initial_glucose must be at least 20 mg/dL")
        if insulin_action_duration <= 0.0:
            raise ValueError("insulin_action_duration must be positive")
        if not 0.0 < insulin_peak_time < insulin_action_duration:
            raise ValueError("insulin_peak_time must lie inside insulin_action_duration")
        if meal_mismatch_epsilon < 0.0 or carb_absorption_duration_minutes <= 0.0:
            raise ValueError("meal multiplier must be non-negative and absorption duration positive")
        if max_glucose_rate_mgdl_per_min <= 0.0:
            raise ValueError("max_glucose_rate_mgdl_per_min must be positive")
        if not 0.0 <= molecular_affinity_scalar <= 2.0:
            raise ValueError("molecular_affinity_scalar must be between 0 and 2")
        if dawn_phenomenon_strength < 0.0:
            raise ValueError("dawn_phenomenon_strength must be non-negative")
        if not 0.0 <= dawn_insulin_resistance_fraction < 1.0:
            raise ValueError(
                "dawn_insulin_resistance_fraction must satisfy 0 <= fraction < 1"
            )
        if not 0.0 <= dawn_start_hour < dawn_end_hour <= 24.0:
            raise ValueError("dawn hours must satisfy 0 <= start < end <= 24")
        if basal_glucose_target is not None and (
            not np.isfinite(basal_glucose_target)
            or float(basal_glucose_target) < 20.0
        ):
            raise ValueError(
                "basal_glucose_target must be finite and at least 20 mg/dL"
            )

        self.basal_insulin_rate = float(basal_insulin_rate)
        self.molecular_affinity_scalar = float(molecular_affinity_scalar)
        self.nominal_insulin_sensitivity = float(insulin_sensitivity)
        # Keep the configured ISF separate from the experimental molecular
        # scalar; the scalar is applied exactly once in the action equation.
        self.insulin_sensitivity = self.nominal_insulin_sensitivity
        self.carb_factor = float(carb_factor)
        self.glucose_decay_rate = float(glucose_decay_rate)
        self.glucose_absorption_rate = float(glucose_absorption_rate)
        self.basal_glucose_target = basal_glucose_target if basal_glucose_target is not None else initial_glucose
        self.insulin_action_duration = insulin_action_duration
        self.insulin_peak_time = insulin_peak_time
        self.meal_mismatch_epsilon = meal_mismatch_epsilon
        self.dawn_phenomenon_strength = dawn_phenomenon_strength
        self.dawn_insulin_resistance_fraction = dawn_insulin_resistance_fraction
        self.dawn_start_hour = dawn_start_hour
        self.dawn_end_hour = dawn_end_hour
        self.carb_absorption_duration_minutes = carb_absorption_duration_minutes
        self.max_glucose_rate_mgdl_per_min = max_glucose_rate_mgdl_per_min


        self.initial_glucose = initial_glucose
        self.current_glucose = initial_glucose
        self.insulin_on_board = 0.0 # Units of insulin still active
        self.carbs_on_board = 0.0   # Grams of carbs still being absorbed
        self.meal_effect_delay = 30 # minutes for carb absorption to peak

        # Exercise state
        self.is_exercising = False
        self.exercise_intensity = 0.0  # 0.0 to 1.0
        self.exercise_glucose_consumption_rate = 1.5  # mg/dL per minute at max intensity

        # Stress state
        self.is_stressed = False
        self.stress_intensity = 0.0  # 0.0 to 1.0
        self._last_unsupported_event: Optional[Dict[str, Any]] = None

        self.reset()  # Call reset to ensure initial state consistency

    def _effective_insulin_curve(self) -> tuple[float, float]:
        """Return the validated triangular action duration and peak."""
        return float(self.insulin_action_duration), float(self.insulin_peak_time)

    def _effective_carb_absorption_duration(self) -> float:
        """Keep meal mass until the configured chain is effectively absorbed."""
        return max(
            float(self.carb_absorption_duration_minutes),
            12.0 / float(self.glucose_absorption_rate),
        )

    @staticmethod
    def _triangular_action_cdf(age: float, duration: float, peak: float) -> float:
        """CDF of a unit-area triangular insulin-action profile."""
        t = float(np.clip(age, 0.0, duration))
        if t <= peak:
            return t * t / (duration * peak)
        remaining = duration - t
        return 1.0 - remaining * remaining / (duration * (duration - peak))

    @staticmethod
    def _two_compartment_absorption_cdf(age: float, rate_per_min: float) -> float:
        """CDF for two equal first-order meal compartments."""
        t = max(0.0, float(age))
        rate = float(rate_per_min)
        return float(np.clip(1.0 - np.exp(-rate * t) * (1.0 + rate * t), 0.0, 1.0))

    def _guard_glucose_transition(self, proposed_glucose: float, time_step: float) -> float:
        """Reject non-finite or explicitly out-of-envelope transitions."""
        if not np.isfinite(proposed_glucose):
            raise PatientModelDomainError(
                "patient model produced non-finite glucose",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        if proposed_glucose < 20.0:
            raise PatientModelDomainError(
                "patient model produced glucose below its declared 20 mg/dL domain",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        max_rate = float(self.max_glucose_rate_mgdl_per_min)
        max_delta = max_rate * float(time_step)
        requested_delta = float(proposed_glucose) - float(self.current_glucose)
        if abs(requested_delta) > max_delta + 1e-9:
            raise PatientModelDomainError(
                "patient model exceeded the configured glucose-rate envelope: "
                f"{requested_delta / time_step:.3f} mg/dL/min",
                current_glucose=self.current_glucose,
                proposed_glucose=proposed_glucose,
            )
        return float(proposed_glucose)

    def reset(self):
        """Resets the patient's state to initial conditions."""
        self.current_glucose = self.initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.active_insulin_doses = [] # List of {'amount': float, 'age': float}
        self.active_carb_intakes = [] # (carb_amount, time_since_intake)
        self.is_exercising = False
        self.exercise_intensity = 0.0
        self.is_stressed = False
        self.stress_intensity = 0.0

    def start_exercise(self, intensity: float):
        """Starts an exercise session."""
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Exercise intensity must be between 0.0 and 1.0")
        self.is_exercising = True
        self.exercise_intensity = intensity

    def stop_exercise(self):
        """Stops an exercise session."""
        self.is_exercising = False
        self.exercise_intensity = 0.0

    def start_stress(self, intensity: float):
        """Starts a physiological stress/illness session."""
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Stress intensity must be between 0.0 and 1.0")
        self.is_stressed = True
        self.stress_intensity = intensity

    def stop_stress(self):
        """Stops a physiological stress/illness session."""
        self.is_stressed = False
        self.stress_intensity = 0.0

    def update(self, time_step: float, delivered_insulin: float, carb_intake: float = 0.0, current_time: Optional[float] = None, **kwargs) -> float:
        """
        Updates the patient's glucose level over a given time step.

        Args:
            time_step (float): The duration of the simulation step in minutes.
            delivered_insulin (float): Total insulin delivered in this time step (e.g., bolus + basal).
            carb_intake (float): Carbohydrates consumed in this time step (grams).
            **kwargs: Additional factors like exercise, stress (not yet implemented in detail).

        Returns:
            float: The new current blood glucose level.
        """
        if not np.isfinite(time_step) or float(time_step) <= 0.0:
            raise ValueError("time_step must be finite and positive")
        if not np.isfinite(delivered_insulin) or float(delivered_insulin) < 0.0:
            raise ValueError("delivered_insulin must be finite and non-negative")
        if not np.isfinite(carb_intake) or float(carb_intake) < 0.0:
            raise ValueError("carb_intake must be finite and non-negative")
        if current_time is not None and not np.isfinite(current_time):
            raise ValueError("current_time must be finite when provided")
        delivered_glucagon = float(kwargs.pop("delivered_glucagon_mg", 0.0))
        if not np.isfinite(delivered_glucagon) or delivered_glucagon < 0.0:
            raise ValueError("delivered_glucagon_mg must be finite and non-negative")
        if delivered_glucagon > 0.0:
            raise NotImplementedError(
                "CustomPatientModel has no glucagon PK/PD; use Hovorka or Bergman mode"
            )
        if kwargs:
            raise TypeError(f"unsupported CustomPatientModel update fields: {sorted(kwargs)}")
        previous_state = copy.deepcopy(self.get_state())
        insulin_action_duration, insulin_peak_time = self._effective_insulin_curve()
        carb_absorption_duration = self._effective_carb_absorption_duration()

        # --- Insulin effect ---
        # Add new insulin dose
        if delivered_insulin > 0.001:
            self.active_insulin_doses.append({'amount': delivered_insulin, 'age': 0.0})

        # Integrate the normalized triangular activity profile over this exact
        # interval. Using CDF differences avoids changing total dose effect when
        # the simulator time step changes.
        total_insulin_action = 0.0
        retained_doses = []
        for dose in self.active_insulin_doses:
            old_age = float(dose['age'])
            new_age = old_age + float(time_step)
            old_fraction = self._triangular_action_cdf(old_age, insulin_action_duration, insulin_peak_time)
            new_fraction = self._triangular_action_cdf(new_age, insulin_action_duration, insulin_peak_time)
            total_insulin_action += float(dose['amount']) * (new_fraction - old_fraction)
            dose['age'] = new_age
            if new_age < insulin_action_duration:
                retained_doses.append(dose)
        self.active_insulin_doses = retained_doses
        self.insulin_on_board = sum(
            float(dose['amount'])
            * (1.0 - self._triangular_action_cdf(
                float(dose['age']), insulin_action_duration, insulin_peak_time
            ))
            for dose in self.active_insulin_doses
        )

        # Stress decreases insulin sensitivity up to 70%
        stress_isf_multiplier = 1.0 - 0.7 * self.stress_intensity if self.is_stressed else 1.0
        # Dawn resistance. This backend is stepped without a clock in some
        # callers; with no time of day there is no dawn window to place, so
        # the multiplier stays 1.0 rather than assuming a phase.
        dawn_isf_multiplier = 1.0
        if current_time is not None:
            dawn_isf_multiplier = dawn_insulin_sensitivity_multiplier(
                current_time,
                peak_resistance_fraction=self.dawn_insulin_resistance_fraction,
                start_hour=self.dawn_start_hour,
                end_hour=self.dawn_end_hour,
            )
        effective_isf = (
            self.insulin_sensitivity
            * self.molecular_affinity_scalar
            * stress_isf_multiplier
            * dawn_isf_multiplier
        )
        insulin_effect = total_insulin_action * effective_isf

        # --- Carb effect ---
        # ``carb_intake`` is the physical meal input. The simulator supplies
        # announced carbohydrates separately to the algorithm. This legacy
        # multiplier is therefore an effective appearance/bioavailability term,
        # not a second carb-counting error.
        true_carbs = carb_intake * self.meal_mismatch_epsilon
        # Add new carbs
        if true_carbs > 0:
            self.active_carb_intakes.append({'amount': true_carbs, 'time_since_intake': 0.0})

        # Integrate a normalized two-compartment appearance profile. The total
        # meal effect is ISF/ICR per gram in this phenomenological model.
        carb_effect = 0.0
        new_active_carb_intakes = []
        for carb_event in self.active_carb_intakes:
            old_age = float(carb_event['time_since_intake'])
            new_age = old_age + float(time_step)
            old_effective_age = min(old_age, carb_absorption_duration)
            new_effective_age = min(new_age, carb_absorption_duration)
            old_fraction = self._two_compartment_absorption_cdf(
                old_effective_age, self.glucose_absorption_rate
            )
            new_fraction = self._two_compartment_absorption_cdf(
                new_effective_age, self.glucose_absorption_rate
            )
            carb_effect += (
                float(carb_event['amount'])
                * (self.nominal_insulin_sensitivity / self.carb_factor)
                * (new_fraction - old_fraction)
            )
            carb_event['time_since_intake'] = new_age
            if new_age < carb_absorption_duration:
                new_active_carb_intakes.append(carb_event)
        self.active_carb_intakes = new_active_carb_intakes
        self.carbs_on_board = sum(
            float(carb_event['amount'])
            * (1.0 - self._two_compartment_absorption_cdf(
                float(carb_event['time_since_intake']), self.glucose_absorption_rate
            ))
            for carb_event in self.active_carb_intakes
        )


        # --- Exercise Effect ---
        exercise_effect = 0.0
        if self.is_exercising:
            exercise_effect = self.exercise_intensity * self.exercise_glucose_consumption_rate * time_step


        # --- Basal metabolic glucose production/consumption (simplified) ---
        # Homeostatic drift toward a basal target (prevents runaway decline)
        # Stress increases endogenous glucose production, shifting the basal target up
        stress_bg_increase = 50.0 * self.stress_intensity if self.is_stressed else 0.0
        effective_basal_target = self.basal_glucose_target + stress_bg_increase
        basal_glucose_change = -self.glucose_decay_rate * (self.current_glucose - effective_basal_target) * time_step

        # --- Dawn phenomenon effect ---
        dawn_effect = 0.0
        if current_time is not None:
            dawn_effect = dawn_glucose_rate_mgdl_min(
                current_time,
                peak_strength_mgdl_per_hour=self.dawn_phenomenon_strength,
                start_hour=self.dawn_start_hour,
                end_hour=self.dawn_end_hour,
            ) * time_step

        # --- Update glucose ---
        delta_glucose = carb_effect - insulin_effect - exercise_effect + basal_glucose_change + dawn_effect
        proposed_glucose = self.current_glucose + delta_glucose
        try:
            self.current_glucose = self._guard_glucose_transition(
                proposed_glucose, time_step
            )
        except Exception:
            self.set_state(previous_state)
            raise

        return self.current_glucose

    def get_current_glucose(self) -> float:
        """Returns the current blood glucose level."""
        return self.current_glucose

    def trigger_event(self, event_type: str, value: Any):
        """
        Triggers a specific event for stress testing (e.g., missed meal, sensor error).

        Args:
            event_type (str): Type of event ('missed_meal', 'sensor_error', 'exercise', etc.).
            value (Any): Value associated with the event (e.g., carb amount for missed meal).
        """
        if event_type == 'missed_meal':
            self._last_unsupported_event = {"event_type": event_type, "value": value, "applied": False}
        elif event_type == 'sensor_error':
            self._last_unsupported_event = {"event_type": event_type, "value": value, "applied": False}
        else:
            self._last_unsupported_event = {"event_type": event_type, "value": value, "applied": False}

    # Helper function for visualization/logging
    def get_patient_state(self) -> Dict[str, float]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
            "max_glucose_rate_mgdl_per_min": self.max_glucose_rate_mgdl_per_min,
        }

    def get_ratio_state(self) -> Dict[str, float]:
        return {
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
        }

    def set_ratio_state(
        self,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        basal_rate: Optional[float] = None,
        dia_minutes: Optional[float] = None,
    ) -> None:
        if isf is not None:
            value = float(isf)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("isf must be finite and positive")
            self.nominal_insulin_sensitivity = value
            self.insulin_sensitivity = value
        if icr is not None:
            value = float(icr)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("icr must be finite and positive")
            self.carb_factor = value
        if basal_rate is not None:
            value = float(basal_rate)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("basal_rate must be finite and non-negative")
            self.basal_insulin_rate = value
        if dia_minutes is not None:
            value = float(dia_minutes)
            if not np.isfinite(value) or value <= self.insulin_peak_time:
                raise ValueError("dia_minutes must be finite and greater than the insulin peak")
            self.insulin_action_duration = value

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "active_insulin_doses": self.active_insulin_doses,
            "active_carb_intakes": self.active_carb_intakes,
            "is_exercising": self.is_exercising,
            "exercise_intensity": self.exercise_intensity,
            "is_stressed": self.is_stressed,
            "stress_intensity": self.stress_intensity,
            "last_unsupported_event": getattr(self, "_last_unsupported_event", None),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        restored_glucose = validated_snapshot_scalar(
            state.get("current_glucose", self.current_glucose),
            name="current_glucose",
            minimum=20.0,
        )
        self.current_glucose = restored_glucose
        self.insulin_on_board = validated_snapshot_scalar(
            state.get("insulin_on_board", self.insulin_on_board),
            name="insulin_on_board",
            minimum=0.0,
        )
        self.carbs_on_board = validated_snapshot_scalar(
            state.get("carbs_on_board", self.carbs_on_board),
            name="carbs_on_board",
            minimum=0.0,
        )
        self.active_insulin_doses = validated_activity_events(
            state.get("active_insulin_doses", []),
            name="active_insulin_doses",
            age_key="age",
        )
        self.active_carb_intakes = validated_activity_events(
            state.get("active_carb_intakes", []),
            name="active_carb_intakes",
            age_key="time_since_intake",
        )
        self.is_exercising = validated_snapshot_bool(
            state.get("is_exercising", False), name="is_exercising"
        )
        self.exercise_intensity = validated_snapshot_scalar(
            state.get("exercise_intensity", 0.0),
            name="exercise_intensity",
            minimum=0.0,
            maximum=1.0,
        )
        self.is_stressed = validated_snapshot_bool(
            state.get("is_stressed", False), name="is_stressed"
        )
        self.stress_intensity = validated_snapshot_scalar(
            state.get("stress_intensity", 0.0),
            name="stress_intensity",
            minimum=0.0,
            maximum=1.0,
        )
        self._last_unsupported_event = state.get("last_unsupported_event")

# Alias for easy import
PatientModel = CustomPatientModel

# SimglucosePatientModel commented out due to dependency issues
# Uncomment and install simglucose for its open-source virtual patients.
