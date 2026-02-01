import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.patient.models import PatientModel
from src.algorithm.correction_bolus import CorrectionBolus
from src.algorithm.lstm_algorithm import LSTMInsulinAlgorithm
from src.algorithm.hybrid_algorithm import HybridInsulinAlgorithm
from src.simulation.simulator import Simulator, StressEvent
from src.analysis.diabetes_metrics import DiabetesMetrics
from src.analysis.sensor_filtering import SensorNoiseModel, KalmanFilter
from src.safety.supervisor import IndependentSupervisor
from src.analysis.validator import ReverseEngineeringValidator
from src.learning.autonomous_optimizer import AutonomousLearningSystem
from src.patient.patient_factory import PatientFactory

# --- Configuration ---
OUTPUT_DIR = 'results/plots/final_analysis_cli'
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# --- Helper Classes & Functions ---

class SafetyAwareSimulator(Simulator):
    """Simulator with integrated safety supervisor."""
    def __init__(self, patient_model, algorithm, time_step=5, enable_safety=True, audit_log_path: Optional[str] = None):
        super().__init__(patient_model, algorithm, time_step, audit_log_path=audit_log_path)
        self.safety_supervisor = IndependentSupervisor() if enable_safety else None
        
    def run(self, duration_minutes: int) -> pd.DataFrame:
        self.patient_model.reset()
        self.algorithm.reset()
        if self.safety_supervisor:
            self.safety_supervisor.reset()
        
        self.simulation_data = []
        current_time = 0
        
        while current_time <= duration_minutes:
            actual_glucose_reading = self.patient_model.get_current_glucose()
            
            carb_intake_this_step = 0.0
            patient_carb_intake = 0.0
            for event in self.stress_events:
                if current_time == event.start_time:
                    if event.event_type == 'meal':
                        carb_intake_this_step = event.value
                        patient_carb_intake = event.value
                    elif event.event_type == 'missed_meal':
                        carb_intake_this_step = 0
                        patient_carb_intake = event.value

            insulin_output = self.algorithm.calculate_insulin(
                current_glucose=actual_glucose_reading, 
                time_step=self.time_step, 
                carb_intake=carb_intake_this_step
            )
            proposed_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            
            # Safety supervisor evaluation
            if self.safety_supervisor:
                safety_result = self.safety_supervisor.evaluate_safety(
                    current_glucose=actual_glucose_reading,
                    proposed_insulin=proposed_insulin,
                    current_time=current_time,
                    current_iob=self.patient_model.insulin_on_board
                )
                delivered_insulin = safety_result["approved_insulin"]
                safety_level = safety_result["safety_level"].value
                actions_taken = safety_result["actions_taken"]
            else:
                delivered_insulin = proposed_insulin
                safety_level = "safe"
                actions_taken = []
            
            self.patient_model.update(self.time_step, delivered_insulin, patient_carb_intake)

            record = {
                "time_minutes": current_time, 
                "glucose_actual_mgdl": actual_glucose_reading,
                "proposed_insulin_units": proposed_insulin,
                "delivered_insulin_units": delivered_insulin,
                "safety_level": safety_level,
                "safety_actions": "; ".join(actions_taken) if actions_taken else ""
            }
            self.simulation_data.append(record)
            current_time += self.time_step
            
        return pd.DataFrame(self.simulation_data)

class NoisySimulator(Simulator):
    """A simulator that adds random noise to glucose readings."""
    def __init__(self, patient_model, algorithm, time_step=5, noise_level=15, audit_log_path: Optional[str] = None):
        super().__init__(patient_model, algorithm, time_step, audit_log_path=audit_log_path)
        self.noise_level = noise_level
        
    def run(self, duration_minutes: int) -> pd.DataFrame:
        self.patient_model.reset()
        self.algorithm.reset()
        self.simulation_data = []
        current_time = 0
        while current_time <= duration_minutes:
            actual_glucose_reading = self.patient_model.get_current_glucose()
            noise = np.random.normal(0, self.noise_level)
            glucose_to_algorithm = actual_glucose_reading + noise
            
            carb_intake_this_step = 0.0
            patient_carb_intake = 0.0
            for event in self.stress_events:
                if current_time == event.start_time:
                    if event.event_type == 'meal':
                        carb_intake_this_step = event.value
                        patient_carb_intake = event.value
                    elif event.event_type == 'missed_meal':
                        carb_intake_this_step = 0 # Algo doesn't know
                        patient_carb_intake = event.value # Patient still gets carbs

            insulin_output = self.algorithm.calculate_insulin(
                current_glucose=glucose_to_algorithm, 
                time_step=self.time_step, 
                carb_intake=carb_intake_this_step
            )
            delivered_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            self.patient_model.update(self.time_step, delivered_insulin, patient_carb_intake)

            record = { "time_minutes": current_time, "glucose_actual_mgdl": actual_glucose_reading,
                       "glucose_to_algo_mgdl": glucose_to_algorithm, "delivered_insulin_units": delivered_insulin }
            self.simulation_data.append(record)
            current_time += self.time_step
        return pd.DataFrame(self.simulation_data)

def calculate_metrics(df, baseline=120):
    """Calculates comprehensive diabetes metrics."""
    basic_metrics = DiabetesMetrics.calculate_all_metrics(df, baseline)
    
    # Legacy metrics for compatibility
    peak_glucose = df['glucose_actual_mgdl'].max()
    time_to_peak = df['glucose_actual_mgdl'].idxmax() * 5
    overshoot = peak_glucose - baseline
    
    basic_metrics.update({
        "time_to_peak_min": time_to_peak,
        "max_overshoot_mgdl": overshoot
    })
    
    return basic_metrics

def run_and_analyze(scenario_name, algorithm_name, save_plot=True, audit_log_path: Optional[str] = None):
    """Runs a single specified scenario and algorithm, prints metrics, and saves a plot."""
    print(f"--- Running Analysis ---")
    print(f"Scenario: {scenario_name}")
    print(f"Algorithm: {algorithm_name}")

    # Define scenarios
    scenarios = {
        "Standard_Meal": {'events': [StressEvent(start_time=60, event_type='meal', value=60)], 'initial_glucose': 120, 'is_noisy': False},
        "Unannounced_Meal": {'events': [StressEvent(start_time=60, event_type='missed_meal', value=60)], 'initial_glucose': 120, 'is_noisy': False},
        "Hyperglycemia_Correction": {'events': [], 'initial_glucose': 250, 'is_noisy': False},
        "Noisy_Standard_Meal": {'events': [StressEvent(start_time=60, event_type='meal', value=60)], 'initial_glucose': 120, 'is_noisy': True, 'noise_level': 20}
    }
    params = scenarios[scenario_name]
    
    patient = PatientModel(initial_glucose=params['initial_glucose'])
    
    if algorithm_name == "rule_based":
        algo = CorrectionBolus()
    elif algorithm_name == "lstm":
        algo = LSTMInsulinAlgorithm()
    elif algorithm_name == "hybrid":
        algo = HybridInsulinAlgorithm()
    else:
        raise ValueError("Unknown algorithm specified.")

    if params['is_noisy']:
        sim = NoisySimulator(patient, algo, noise_level=params.get('noise_level', 20), audit_log_path=audit_log_path)
    else:
        sim = Simulator(patient, algo, audit_log_path=audit_log_path)

    for event in params['events']:
        sim.add_stress_event(event)
        
    df = sim.run(duration_minutes=480)
    
    # Validation & Data Integrity Check
    validator = ReverseEngineeringValidator()
    
    # Run baseline for drift detection
    baseline_algo = CorrectionBolus()
    baseline_sim = Simulator(PatientModel(initial_glucose=params['initial_glucose']), baseline_algo)
    for event in params['events']:
        baseline_sim.add_stress_event(event)
    baseline_df = baseline_sim.run(duration_minutes=480)
    baseline_insulin = baseline_df['delivered_insulin_units'].tolist()
    
    # Validate results
    validation_results = validator.validate_simulation_results(
        df, 
        baseline_results=baseline_insulin
    )
    reliability_report = validator.generate_reliability_report(validation_results)
    
    metrics = calculate_metrics(df, params['initial_glucose'])

    # --- Print Metrics ---
    print("\n--- Quantitative Metrics ---")
    print(f"Peak Glucose: {metrics['peak_glucose_mgdl']:.1f} mg/dL")
    print(f"TIR (70-180): {metrics['tir_percentage']:.1f}%")
    print(f"CV: {metrics['cv_percentage']:.1f}%")
    print(f"HBGI: {metrics['hbgi']:.2f}")
    print(f"LBGI: {metrics['lbgi']:.2f}")
    
    # Hybrid stats if applicable
    if hasattr(algo, 'get_stats'):
        stats = algo.get_stats()
        print(f"\n--- Hybrid Algorithm Stats ---")
        print(f"LSTM Usage: {stats['lstm_usage']:.1%}")
        print(f"Rule Usage: {stats['rule_usage']:.1%}")
    
    # Data Reliability Report
    print(f"\n--- Data Reliability Report ---")
    print(f"Overall Reliability: {reliability_report['overall_reliability_score']:.1f}% ({reliability_report['overall_level'].upper()})")
    print(f"Recommendation: {reliability_report['recommendation']}")
    if reliability_report['total_issues'] > 0:
        print(f"Issues Found: {reliability_report['total_issues']}")
        for issue in reliability_report['issues'][:3]:  # Show first 3 issues
            print(f"  - {issue}")
    if reliability_report['total_warnings'] > 0:
        print(f"Warnings: {reliability_report['total_warnings']}")
    
    # Autonomous Learning (if reliability is low)
    if reliability_report['overall_reliability_score'] < 80.0:
        print(f"\n--- Autonomous Learning Triggered ---")
        print(f"Low reliability detected. Initiating clinical learning cycle...")
        
        model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithm', 'trained_lstm_model.pth')
        learning_system = AutonomousLearningSystem(model_path)
        
        # Extract validation errors for learning
        validation_errors = []
        for issue in reliability_report['issues']:
            validation_errors.append({'message': issue, 'glucose_value': 120, 'ai_insulin': 1.0, 'baseline_insulin': 0.5})
        
        improved = learning_system.continuous_learning_cycle(validation_errors)
        
        if improved:
            print(" Model successfully improved using clinical protocols")
            learning_report = learning_system.get_learning_report()
            print(f"New Safety Score: {learning_report['latest_safety_score']:.1f}%")
        else:
            print("  Model improvement did not meet safety threshold")
    
    # --- Save Plot ---
    if save_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'Analysis for {algorithm_name} in {scenario_name}', fontsize=16)
        
        # Glucose Plot
        ax1.plot(df['time_minutes'], df['glucose_actual_mgdl'], label='Actual Glucose', color='blue')
        if 'glucose_to_algo_mgdl' in df.columns:
            ax1.plot(df['time_minutes'], df['glucose_to_algo_mgdl'], label='Noisy Glucose Input', color='lightblue', linestyle=':')
        ax1.set_ylabel("Glucose (mg/dL)")
        ax1.set_title("Glucose Response")
        ax1.legend()
        
        # Insulin Plot
        ax2.plot(df['time_minutes'], df['delivered_insulin_units'].cumsum(), label='Cumulative Insulin', color='green')
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Total Insulin (Units)")
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f'{scenario_name.lower()}_{algorithm_name.lower()}.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath)
        print(f"\nPlot saved to {filepath}")
        plt.close()
    
    print("\n--- Analysis Complete ---")


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single simulation for insulin algorithm analysis.")
    parser.add_argument(
        '--scenario', 
        type=str, 
        required=True, 
        choices=['Standard_Meal', 'Unannounced_Meal', 'Hyperglycemia_Correction', 'Noisy_Standard_Meal'],
        help='The simulation scenario to run.'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['rule_based', 'lstm', 'hybrid'],
        help='The algorithm to use for the simulation.'
    )
    parser.add_argument(
        '--no-safety',
        action='store_true',
        help='Disable safety supervisor for testing.'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='If set, the script will not save a plot.'
    )
    parser.add_argument(
        '--audit-log',
        type=str,
        default=None,
        help='Path to a file to write a detailed JSON audit log.'
    )

    args = parser.parse_args()
    
    run_and_analyze(args.scenario, args.algorithm, save_plot=not args.no_plot, audit_log_path=args.audit_log)