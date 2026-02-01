import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from iints.core.patient.models import PatientModel
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput
from iints.core.supervisor import IndependentSupervisor, SafetyLevel
from iints.core.safety import InputValidator
import numpy as np

class StressEvent:
    """Represents a discrete event that can occur during a simulation for stress testing."""
    def __init__(self, start_time: int, event_type: str, value: Any = None, reported_value: Any = None, absorption_delay_minutes: int = 0, duration: int = 0):
        """
        Args:
            start_time (int): The simulation time (in minutes) when the event should occur.
            event_type (str): Type of event (e.g., 'meal', 'missed_meal', 'sensor_error', 'exercise').
            value (Any): Value associated with the event (e.g., carb amount, error value).
            reported_value (Any): Value reported to the algorithm (if different from actual value, e.g., for "prul" scenarios).
            absorption_delay_minutes (int): Delay in minutes before carbohydrates from this event start affecting glucose.
            duration (int): The duration of the event in minutes (e.g., for exercise).
        """
        self.start_time = start_time
        self.event_type = event_type
        self.value = value
        self.reported_value = reported_value # New attribute
        self.absorption_delay_minutes = absorption_delay_minutes # New attribute
        self.duration = duration

    def __str__(self):
        reported_str = f", Reported: {self.reported_value}" if self.reported_value is not None else ""
        delay_str = f", Delay: {self.absorption_delay_minutes}m" if self.absorption_delay_minutes > 0 else ""
        duration_str = f", Duration: {self.duration}m" if self.duration > 0 else ""
        return f"Event(Time: {self.start_time}m, Type: {self.event_type}, Value: {self.value}{reported_str}{delay_str}{duration_str})"

class Simulator:
    """
    Orchestrates the interaction between a patient model and an insulin algorithm
    over a simulated period, including stress-test scenarios.
    """
    def __init__(self, patient_model: PatientModel, algorithm: InsulinAlgorithm, time_step: int = 5, seed: Optional[int] = None, audit_log_path: Optional[str] = None):
        """
        Initializes the simulator.

        Args:
            patient_model (PatientModel): The patient model to simulate.
            algorithm (InsulinAlgorithm): The insulin delivery algorithm to test.
            time_step (int): The duration of each simulation step in minutes.
            seed (Optional[int]): Random seed for reproducible simulations.
            audit_log_path (Optional[str]): If provided, path to write a detailed JSON audit log.
        """
        self.patient_model = patient_model
        self.algorithm = algorithm
        self.time_step = time_step
        self.simulation_data: List[Any] = [] # To store results
        self.stress_events: List[StressEvent] = []
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed) # Set numpy seed for reproducibility
            # Potentially set other seeds here if other random modules are used (e.g., random.seed(self.seed))
        self.supervisor = IndependentSupervisor()
        self.input_validator = InputValidator()
        self.meal_queue: List[Dict[str, Any]] = [] # Initialize meal queue for delayed absorption
        self.audit_log_path = audit_log_path
        if self.audit_log_path:
            # Clear the log file at the beginning of a simulation run
            try:
                with open(self.audit_log_path, 'w') as f:
                    f.write("") # Overwrite the file
            except IOError as e:
                print(f"Warning: Could not clear audit log file at {self.audit_log_path}. Error: {e}")

    def _write_audit_log(self, data: Dict[str, Any]):
        """Writes a single step's audit data to the JSON log file."""
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, 'a') as f:
                    f.write(json.dumps(data, default=str) + '\n')
            except IOError as e:
                print(f"Warning: Could not write to audit log file at {self.audit_log_path}. Error: {e}")

    def add_stress_event(self, event: StressEvent):
        """Adds a stress event to be triggered during the simulation."""
        self.stress_events.append(event)
        self.stress_events.sort(key=lambda e: e.start_time) # Keep events sorted by time

    def run(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Alias for run_batch to ensure backward compatibility."""
        return self.run_batch(duration_minutes)

    def run_batch(self, duration_minutes: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs the entire simulation and returns the results as a single DataFrame.

        Args:
            duration_minutes (int): Total simulation duration in minutes.

        Returns:
            pd.DataFrame: A DataFrame containing the complete simulation results.
            Dict[str, Any]: A dictionary containing the safety report from the supervisor.
        """
        all_records = list(self.run_live(duration_minutes))
        simulation_results_df = pd.DataFrame(all_records)
        safety_report = self.supervisor.get_safety_report()
        return simulation_results_df, safety_report

    def run_live(self, duration_minutes: int):
        """
        Runs the simulation as a generator, yielding the record of each time step.

        Args:
            duration_minutes (int): Total simulation duration in minutes.

        Yields:
            Dict[str, Any]: The data record for each simulation time step.
        """
        self.patient_model.reset()
        self.algorithm.reset()
        self.supervisor.reset()
        self.input_validator.reset() # Reset input validator for new run
        self.simulation_data = []
        self.meal_queue = [] # Reset meal queue for new run
        current_time = 0

        while current_time <= duration_minutes:
            patient_carb_intake_this_step = 0.0
            algo_carb_intake_this_step = 0.0
            actual_glucose_reading = self.patient_model.get_current_glucose()
            # Validate the raw sensor reading
            actual_glucose_reading = self.input_validator.validate_glucose(actual_glucose_reading, float(current_time))
            glucose_to_algorithm = actual_glucose_reading

            # Process newly triggered stress events
            events_to_process_now = [] # Events that affect algorithm input immediately (e.g., reported carbs, sensor error)
            events_to_queue_for_patient = [] # Meal events that affect patient after delay

            for event in self.stress_events:
                if current_time == event.start_time:
                    print(f"[{current_time} min] Triggering stress event: {event}")
                    if event.event_type == 'meal':
                        events_to_queue_for_patient.append(event)
                        # Algorithm gets carb info based on reported_value if available, otherwise actual value
                        algo_carb_intake_this_step += event.reported_value if event.reported_value is not None else event.value
                    elif event.event_type == 'missed_meal':
                        # Patient consumes carbs immediately, algorithm not aware
                        patient_carb_intake_this_step += event.value
                        algo_carb_intake_this_step = 0.0 # Algorithm gets 0 carbs
                    elif event.event_type == 'sensor_error':
                        glucose_to_algorithm = event.value
                    elif event.event_type == 'exercise':
                        self.patient_model.start_exercise(event.value)
                        # Schedule the end of the exercise
                        end_event = StressEvent(start_time=current_time + event.duration, event_type='exercise_end')
                        self.add_stress_event(end_event)
                    elif event.event_type == 'exercise_end':
                        self.patient_model.stop_exercise()

                    events_to_process_now.append(event) # Mark for removal from stress_events list

            # Remove processed stress events that don't need to be queued for patient absorption
            self.stress_events = [e for e in self.stress_events if e not in events_to_process_now]

            # Add newly triggered meal events to the meal queue for delayed absorption
            for event in events_to_queue_for_patient:
                self.meal_queue.append({
                    'event': event,
                    'absorption_start_time': current_time + event.absorption_delay_minutes
                })

            # Process meals from the queue that are now ready for absorption
            meals_absorbed_this_step = []
            for meal_entry in self.meal_queue:
                if current_time >= meal_entry['absorption_start_time']:
                    patient_carb_intake_this_step += meal_entry['event'].value # Actual carbs for the patient model
                    meals_absorbed_this_step.append(meal_entry)
            
            # Remove absorbed meals from the queue
            self.meal_queue = [meal_entry for meal_entry in self.meal_queue if meal_entry not in meals_absorbed_this_step]

            # Validate glucose passed to algorithm (could be modified by stress events)
            # This ensures even sensor error stress events are validated
            glucose_to_algorithm = self.input_validator.validate_glucose(glucose_to_algorithm, float(current_time))

            # --- Algorithm Input ---
            algo_input = AlgorithmInput(
                current_glucose=glucose_to_algorithm,
                time_step=self.time_step,
                insulin_on_board=self.patient_model.insulin_on_board,
                carb_intake=algo_carb_intake_this_step, # Use algo's perspective of carbs
                patient_state=self.patient_model.get_patient_state(),
                current_time=float(current_time) # Pass current_time
            )

            # --- Algorithm Calculation ---
            insulin_output = self.algorithm.predict_insulin(algo_input)
            algo_recommended_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            # Validate the algorithm's output to prevent negative insulin requests
            algo_recommended_insulin = self.input_validator.validate_insulin(algo_recommended_insulin)
            
            # Get the why_log from the algorithm
            algorithm_why_log = self.algorithm.get_why_log()

            # --- Safety Supervision ---
            start_perf_time = time.perf_counter()
            safety_result = self.supervisor.evaluate_safety(
                current_glucose=glucose_to_algorithm,
                proposed_insulin=algo_recommended_insulin,
                current_time=float(current_time),
                current_iob=self.patient_model.insulin_on_board
            )
            supervisor_latency_ms = (time.perf_counter() - start_perf_time) * 1000

            delivered_insulin = safety_result["approved_insulin"]
            overridden = safety_result["insulin_reduction"] > 0
            safety_level = safety_result["safety_level"]
            safety_actions = "; ".join(safety_result["actions_taken"])

            # --- Audit Logging ---
            self._write_audit_log({
                "timestamp": current_time,
                "cgm": actual_glucose_reading,
                "ai_suggestion": algo_recommended_insulin,
                "supervisor_override": overridden,
                "final_dose": delivered_insulin,
                "hidden_state_summary": self.algorithm.get_state()
            })

            # --- Patient Model Update ---
            self.patient_model.update(
                time_step=self.time_step,
                delivered_insulin=delivered_insulin,
                carb_intake=patient_carb_intake_this_step # Use actual carbs for patient
            )

            # --- Record Data ---
            record = {
                "time_minutes": current_time,
                "glucose_actual_mgdl": actual_glucose_reading,
                "glucose_to_algo_mgdl": glucose_to_algorithm,
                "delivered_insulin_units": delivered_insulin,
                "algo_recommended_insulin_units": algo_recommended_insulin,
                "basal_insulin_units": insulin_output.get("basal_insulin", 0.0),
                "bolus_insulin_units": insulin_output.get("bolus_insulin", 0.0) + insulin_output.get("meal_bolus", 0.0), # Combine for simplicity
                "correction_bolus_units": insulin_output.get("correction_bolus", 0.0),
                "carb_intake_grams": patient_carb_intake_this_step,
                "patient_iob_units": self.patient_model.insulin_on_board,
                "patient_cob_grams": self.patient_model.carbs_on_board,
                "uncertainty": insulin_output.get("uncertainty", 0.0),
                "fallback_triggered": insulin_output.get("fallback_triggered", False),
                "safety_level": safety_level.value,
                "safety_actions": safety_actions,
                "supervisor_latency_ms": supervisor_latency_ms,
                "algorithm_why_log": [entry.to_dict() for entry in algorithm_why_log], # Convert WhyLogEntry to dict for serialization
                **{f"algo_state_{k}": v for k, v in self.algorithm.get_state().items()} # Include algorithm internal state
            }
            
            yield record

            current_time += self.time_step