import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
import warnings

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import patient models, algorithms, and simulator
from iints.core.patient.models import PatientModel # Using CustomPatientModel aliased as PatientModel
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.lstm_algorithm import LSTMInsulinAlgorithm
from iints.core.simulator import Simulator, StressEvent

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- Simulation and Plotting Functions (from run_analysis.py) ---
def run_and_plot_comparison(scenario_name, events, initial_glucose=120, duration_minutes=360, output_dir='analysis_plots'):
    """Helper function to run two algorithms and save their comparison plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Instantiate patient and algorithms ---
    patient_rb = PatientModel(initial_glucose=initial_glucose)
    patient_ml = PatientModel(initial_glucose=initial_glucose)
    
    algo_rb = CorrectionBolus()
    algo_ml = LSTMInsulinAlgorithm()
    
    # --- Run simulations ---
    sim_rb = Simulator(patient_model=patient_rb, algorithm=algo_rb)
    sim_ml = Simulator(patient_model=patient_ml, algorithm=algo_ml)
    
    for event in events:
        sim_rb.add_stress_event(event)
        sim_ml.add_stress_event(event)
        
    df_rb = sim_rb.run(duration_minutes=duration_minutes)
    df_ml = sim_ml.run(duration_minutes=duration_minutes)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Comparative Analysis: {scenario_name}', fontsize=16)
    
    # Glucose Plot
    ax1.plot(df_rb['time_minutes'], df_rb['glucose_actual_mgdl'], label='Rule-Based Glucose', color='blue')
    ax1.plot(df_ml['time_minutes'], df_ml['glucose_actual_mgdl'], label='Hybrid LSTM-Based Glucose', color='green', linestyle='--')
    
    # Highlight fallback periods
    fallback_times = df_ml[df_ml['fallback_triggered'] == True]
    if not fallback_times.empty:
        ax1.scatter(fallback_times['time_minutes'], fallback_times['glucose_actual_mgdl'], color='red', s=50, zorder=5, label='Fallback to Rule-Based')

    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.legend()
    ax1.set_title('Glucose Response')
    
    # Insulin Plot (Per-Step with Uncertainty)
    ax2.plot(df_rb['time_minutes'], df_rb['delivered_insulin_units'], label='Rule-Based Insulin', color='blue', alpha=0.7)
    ax2.plot(df_ml['time_minutes'], df_ml['delivered_insulin_units'], label='Hybrid LSTM-Based Insulin', color='green', linestyle='--', alpha=0.7)

    # Scale uncertainty (std dev of raw prediction) to match delivered insulin scale
    uncertainty_scaled = df_ml['uncertainty'] * 10
    lower_bound = df_ml['delivered_insulin_units'] - uncertainty_scaled
    upper_bound = df_ml['delivered_insulin_units'] + uncertainty_scaled
    ax2.fill_between(df_ml['time_minutes'], lower_bound.clip(0), upper_bound, color='green', alpha=0.2, label='LSTM Uncertainty')

    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Insulin Delivered (Units per step)')
    ax2.legend()
    ax2.set_title('Insulin Delivery')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'{scenario_name.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(output_dir, filename))
    print(f'Saved plot to {os.path.join(output_dir, filename)}')
    plt.close()

class NoisySimulator(Simulator):
    def __init__(self, patient_model, algorithm, time_step=5, noise_level=15):
        super().__init__(patient_model, algorithm, time_step)
        self.noise_level = noise_level
        print(f"NoisySimulator initialized with noise level: {self.noise_level}")
        
    def run(self, duration_minutes: int) -> pd.DataFrame:
        self.patient_model.reset()
        self.algorithm.reset()
        self.simulation_data = []
        current_time = 0

        while current_time <= duration_minutes:
            actual_glucose_reading = self.patient_model.get_current_glucose()
            # --- Key difference: Add noise to the glucose reading for the algorithm ---
            noise = np.random.normal(0, self.noise_level)
            glucose_to_algorithm = actual_glucose_reading + noise
            
            carb_intake_this_step = 0.0
            for event in self.stress_events:
                if current_time == event.start_time and event.event_type == 'meal':
                    carb_intake_this_step += event.value
            
            insulin_output = self.algorithm.calculate_insulin(
                current_glucose=glucose_to_algorithm, 
                time_step=self.time_step, 
                carb_intake=carb_intake_this_step
            )
            delivered_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            
            self.patient_model.update(self.time_step, delivered_insulin, carb_intake_this_step)

            # --- Comprehensive Record Keeping ---
            record = {
                "time_minutes": current_time,
                "glucose_actual_mgdl": actual_glucose_reading,
                "glucose_to_algo_mgdl": glucose_to_algorithm,
                "delivered_insulin_units": delivered_insulin,
                "uncertainty": insulin_output.get("uncertainty", 0.0),
                "fallback_triggered": insulin_output.get("fallback_triggered", False),
                **{f"algo_state_{k}": v for k, v in self.algorithm.get_state().items()}
            }
            self.simulation_data.append(record)
            current_time += self.time_step

        return pd.DataFrame(self.simulation_data)

def run_and_plot_noisy_comparison(scenario_name, events, initial_glucose=120, duration_minutes=360, noise_level=15, output_dir='analysis_plots'):
    os.makedirs(output_dir, exist_ok=True)
    patient_rb = PatientModel(initial_glucose=initial_glucose)
    patient_ml = PatientModel(initial_glucose=initial_glucose)
    
    algo_rb = CorrectionBolus()
    algo_ml = LSTMInsulinAlgorithm()
    
    sim_rb = NoisySimulator(patient_model=patient_rb, algorithm=algo_rb)
    sim_ml = NoisySimulator(patient_model=patient_ml, algorithm=algo_ml)
    
    for event in events:
        sim_rb.add_stress_event(event)
        sim_ml.add_stress_event(event)
        
    df_rb = sim_rb.run(duration_minutes=duration_minutes)
    df_ml = sim_ml.run(duration_minutes=duration_minutes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Stress Test: {scenario_name}', fontsize=16)
    
    ax1.plot(df_rb['time_minutes'], df_rb['glucose_actual_mgdl'], label='Rule-Based Actual Glucose', color='blue', alpha=0.6)
    ax1.plot(df_rb['time_minutes'], df_rb['glucose_to_algo_mgdl'], label='Rule-Based Noisy Input', color='lightblue', linestyle=':')
    ax1.plot(df_ml['time_minutes'], df_ml['glucose_actual_mgdl'], label='Hybrid LSTM Actual Glucose', color='green', linestyle='--', alpha=0.6)
    ax1.plot(df_ml['time_minutes'], df_ml['glucose_to_algo_mgdl'], label='Hybrid LSTM Noisy Input', color='lightgreen', linestyle=':')
    
    # Highlight fallback periods
    fallback_times = df_ml[df_ml['fallback_triggered'] == True]
    if not fallback_times.empty:
        ax1.scatter(fallback_times['time_minutes'], fallback_times['glucose_actual_mgdl'], color='red', s=50, zorder=5, label='Fallback to Rule-Based')

    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.legend()
    ax1.set_title('Glucose Response with Sensor Noise')

    ax2.plot(df_rb['time_minutes'], df_rb['delivered_insulin_units'], label='Rule-Based Insulin Delivery', color='blue', alpha=0.7)
    ax2.plot(df_ml['time_minutes'], df_ml['delivered_insulin_units'], label='Hybrid LSTM Insulin Delivery', color='green', linestyle='--', alpha=0.7)

    # Scale uncertainty (std dev of raw prediction) to match delivered insulin scale
    uncertainty_scaled = df_ml['uncertainty'] * 10
    lower_bound = df_ml['delivered_insulin_units'] - uncertainty_scaled
    upper_bound = df_ml['delivered_insulin_units'] + uncertainty_scaled
    ax2.fill_between(df_ml['time_minutes'], lower_bound.clip(0), upper_bound, color='green', alpha=0.2, label='LSTM Uncertainty')

    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Insulin Delivered (Units per step)')
    ax2.legend()
    ax2.set_title('Insulin Delivery Stability')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = f'stress_test_{scenario_name.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(output_dir, filename))
    print(f'Saved plot to {os.path.join(output_dir, filename)}')
    plt.close()

def run_all_simulations():
    print("--- Running Comparative Analysis ---")
    events_a = [StressEvent(start_time=60, event_type='meal', value=60)]
    run_and_plot_comparison('Standard Meal', events_a)
    
    events_b = [StressEvent(start_time=60, event_type='missed_meal', value=60)]
    run_and_plot_comparison('Unannounced Meal', events_b)
    
    events_c = []
    run_and_plot_comparison('Hyperglycemia Correction', events_c, initial_glucose=250)
    
    print("\n--- Running Stress Test Analysis ---")
    meal_event = [StressEvent(start_time=60, event_type='meal', value=60)]
    run_and_plot_noisy_comparison('Standard Meal with Sensor Noise', meal_event, noise_level=20)

    print("\n--- Running Crashtest Scenario (Extreme Hyperglycemia) ---")
    events_d = [StressEvent(start_time=30, event_type='meal', value=150), # Large, early meal
                StressEvent(start_time=90, event_type='sensor_error', value=350)] # Erratic sensor reading
    run_and_plot_comparison('Extreme Hyperglycemia Crashtest', events_d, initial_glucose=400, duration_minutes=180)

    print("\n--- Running Dummytest Scenario (Stable Basal) ---")
    events_e = [] # No events
    run_and_plot_comparison('Stable Basal Dummytest', events_e, initial_glucose=100, duration_minutes=180)
    
    print("\n--- All Simulations Complete ---")

# --- SHAP Analysis Function (from run_shap_analysis.py) ---
def run_shap_analysis_function(output_dir='analysis_plots'):
    """
    Runs SHAP analysis on the trained LSTM model to explain its predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Running SHAP Explainability Analysis ---")

    # 1. Load the pre-trained algorithm
    # Suppress the FutureWarning from torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        algo_ml = LSTMInsulinAlgorithm()
    model = algo_ml.model
    model.eval()

    # 2. Define the prediction wrapper function for SHAP
    def predict_fn(x):
        # The model expects a 3D tensor: (batch, sequence, features)
        input_tensor = torch.tensor(x, dtype=torch.float32).reshape(x.shape[0], 1, algo_ml.settings['input_features'])
        with torch.no_grad():
            predictions = model(input_tensor).numpy().flatten()
        return predictions

    # 3. Create a background dataset for the explainer
    # Features: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    background_data = np.array([
        [3, 120, 72, 29, 32, 0.47, 33], # Normal glucose
        [3, 180, 72, 29, 32, 0.47, 33], # High glucose
        [3, 80, 72, 29, 32, 0.47, 33],  # Low glucose
        [5, 140, 80, 35, 35, 0.5, 45],  # Another patient profile
        [1, 100, 60, 20, 28, 0.3, 25],  # Younger patient profile
    ])
    
    background_summary = shap.kmeans(background_data, 5)
    print(f"\nUsing a background summary of {len(background_summary.data)} samples for SHAP.")

    # 4. Create the SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, background_summary)

    # 5. Generate SHAP values for a few example instances to explain
    data_to_explain = np.array([
        [4, 190, 80, 34, 33, 0.6, 50]
    ])
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'PedigreeFunc', 'Age']

    print(f"\nCalculating SHAP values for sample instance: {data_to_explain}")
    # Suppress the UserWarning from shap.KernelExplainer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shap_values = explainer.shap_values(data_to_explain)

    # 6. Generate and save SHAP plots
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features=data_to_explain, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plot_path_bar = os.path.join(output_dir, 'shap_feature_importance.png')
    plt.savefig(plot_path_bar)
    print(f"Saved SHAP feature importance plot to {plot_path_bar}")
    plt.close()

    # Force plot for the single prediction
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], data_to_explain[0,:], feature_names=feature_names, show=False)
    plot_path_force = os.path.join(output_dir, 'shap_force_plot.html')
    # shap.save_html(plot_path_force, force_plot) # This needs web browser support, just printing for now.
    print(f"SHAP force plot HTML generated at {plot_path_force} (Requires opening in browser to view)")

    print("\n--- SHAP Analysis Complete ---")

# --- Main execution block ---
if __name__ == '__main__':
    run_all_simulations()
    run_shap_analysis_function()
    
    print("\n--- All Comprehensive Analysis Complete ---")
