import numpy as np
from typing import Dict, Any

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
                 carb_factor: float = 10.0, glucose_decay_rate: float = 0.05,
                 initial_glucose: float = 120.0, glucose_absorption_rate: float = 0.03,
                 insulin_action_duration: float = 300.0, # minutes, e.g., 5 hours
                 insulin_peak_time: float = 75.0, # minutes
                 meal_mismatch_epsilon: float = 1.0): # Factor for meal mismatch
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
            meal_mismatch_epsilon (float): The multiplier for carb intake to simulate meal size errors. 
                                         `true_carbs = announced_carbs * meal_mismatch_epsilon`. Defaults to 1.0.
        """
        self.basal_insulin_rate = basal_insulin_rate
        self.insulin_sensitivity = insulin_sensitivity
        self.carb_factor = carb_factor
        self.glucose_decay_rate = glucose_decay_rate
        self.glucose_absorption_rate = glucose_absorption_rate
        self.insulin_action_duration = insulin_action_duration
        self.insulin_peak_time = insulin_peak_time
        self.meal_mismatch_epsilon = meal_mismatch_epsilon


        self.initial_glucose = initial_glucose
        self.current_glucose = initial_glucose
        self.insulin_on_board = 0.0 # Units of insulin still active
        self.carbs_on_board = 0.0   # Grams of carbs still being absorbed
        self.meal_effect_delay = 30 # minutes for carb absorption to peak

        # Exercise state
        self.is_exercising = False
        self.exercise_intensity = 0.0 # 0.0 to 1.0
        self.exercise_glucose_consumption_rate = 1.5 # mg/dL per minute at max intensity

        self.reset() # Call reset to ensure initial state consistency

    def reset(self):
        """Resets the patient's state to initial conditions."""
        self.current_glucose = self.initial_glucose
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.active_insulin_doses = [] # List of {'amount': float, 'age': float}
        self.active_carb_intakes = [] # (carb_amount, time_since_intake)
        self.is_exercising = False
        self.exercise_intensity = 0.0

    def start_exercise(self, intensity: float):
        """Starts an exercise session."""
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Exercise intensity must be between 0.0 and 1.0")
        self.is_exercising = True
        self.exercise_intensity = intensity
        print(f"INFO: Patient started exercise with intensity {intensity:.2f}")

    def stop_exercise(self):
        """Stops an exercise session."""
        self.is_exercising = False
        self.exercise_intensity = 0.0
        print("INFO: Patient stopped exercise.")

    def update(self, time_step: float, delivered_insulin: float, carb_intake: float = 0.0, **kwargs) -> float:
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
        # Convert time_step to hours for basal rate
        time_step_hours = time_step / 60.0

        # --- Insulin effect ---
        # Add new insulin dose
        if delivered_insulin > 0.001:
            self.active_insulin_doses.append({'amount': delivered_insulin, 'age': 0.0})

        # Update ages and remove old doses
        for dose in self.active_insulin_doses:
            dose['age'] += time_step
        
        self.active_insulin_doses = [d for d in self.active_insulin_doses if d['age'] <= self.insulin_action_duration]

        # Calculate IOB (Insulin on Board) using a linear decay model for remaining insulin
        iob = 0.0
        for dose in self.active_insulin_doses:
            remaining_fraction = (self.insulin_action_duration - dose['age']) / self.insulin_action_duration
            iob += dose['amount'] * remaining_fraction
        self.insulin_on_board = iob

        # Calculate Insulin Action for this time step using a bilinear activity curve
        total_insulin_action = 0.0
        for dose in self.active_insulin_doses:
            if dose['age'] < self.insulin_peak_time:
                # Action ramps up
                action_factor = dose['age'] / self.insulin_peak_time
            else:
                # Action ramps down
                action_factor = (self.insulin_action_duration - dose['age']) / (self.insulin_action_duration - self.insulin_peak_time)
            
            # Normalize the action so the total effect of 1U of insulin equals 1U * insulin_sensitivity
            # The area under the bilinear activity curve is 0.5 * peak_time + 0.5 * (duration - peak_time) = 0.5 * duration
            # The action for this step is a fraction of the total dose effect.
            dose_action_this_step = dose['amount'] * action_factor * (time_step / (0.5 * self.insulin_action_duration))
            total_insulin_action += dose_action_this_step

        insulin_effect = total_insulin_action * self.insulin_sensitivity

        # --- Carb effect ---
        # The 'carb_intake' parameter represents the "announced carbs" by the AI.
        # The 'true_carbs' are the actual carbs the patient's body processes,
        # simulated using the meal_mismatch_epsilon parameter.
        true_carbs = carb_intake * self.meal_mismatch_epsilon
        # Add new carbs
        if true_carbs > 0:
            self.active_carb_intakes.append({'amount': true_carbs, 'time_since_intake': 0.0})

        # Process active carbs
        carb_effect = 0.0
        new_active_carb_intakes = []
        for carb_event in self.active_carb_intakes:
            carb_event['time_since_intake'] += time_step
            # Simple model: carbs absorb over time, peaking around meal_effect_delay
            # This is a very rough approximation
            absorption_factor = 0.0
            if carb_event['time_since_intake'] <= 240: # Carbs absorb for ~4 hours
                absorption_factor = self.glucose_absorption_rate * (np.exp(-carb_event['time_since_intake'] / self.meal_effect_delay) - np.exp(-carb_event['time_since_intake'] / (self.meal_effect_delay * 0.5)))
                carb_effect += carb_event['amount'] * absorption_factor
                new_active_carb_intakes.append(carb_event)
            # Carbs are "gone" after a while, or their effect is negligible
        self.active_carb_intakes = new_active_carb_intakes


        # --- Exercise Effect ---
        exercise_effect = 0.0
        if self.is_exercising:
            exercise_effect = self.exercise_intensity * self.exercise_glucose_consumption_rate * time_step


        # --- Basal metabolic glucose production/consumption (simplified) ---
        basal_glucose_change = -self.glucose_decay_rate * self.current_glucose * time_step

        # --- Update glucose ---
        delta_glucose = carb_effect - insulin_effect - exercise_effect + basal_glucose_change
        self.current_glucose = max(20, self.current_glucose + delta_glucose) # Prevent glucose from going too low (hypoglycemia)

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
            print(f"STRESS EVENT: Missed meal of {value}g carbs!")
            # This event primarily affects the *simulator's* input to the algorithm,
            # but the patient model needs to know if carbs are actually consumed.
            # For now, it's a print statement. Actual effect handled in simulator.
        elif event_type == 'sensor_error':
            print(f"STRESS EVENT: Sensor error - returning {value} as glucose reading.")
            # This event will be handled by the simulator intercepting glucose readings.
        else:
            print(f"Unknown stress event: {event_type}")

    # Helper function for visualization/logging
    def get_patient_state(self) -> Dict[str, float]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_glucose": self.current_glucose,
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "active_insulin_doses": self.active_insulin_doses,
            "active_carb_intakes": self.active_carb_intakes,
            "is_exercising": self.is_exercising,
            "exercise_intensity": self.exercise_intensity,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.current_glucose = state.get("current_glucose", self.current_glucose)
        self.insulin_on_board = state.get("insulin_on_board", self.insulin_on_board)
        self.carbs_on_board = state.get("carbs_on_board", self.carbs_on_board)
        self.active_insulin_doses = state.get("active_insulin_doses", [])
        self.active_carb_intakes = state.get("active_carb_intakes", [])
        self.is_exercising = state.get("is_exercising", False)
        self.exercise_intensity = state.get("exercise_intensity", 0.0)

# Alias for easy import
PatientModel = CustomPatientModel

# SimglucosePatientModel commented out due to dependency issues
# Uncomment and install simglucose for FDA-approved virtual patients
