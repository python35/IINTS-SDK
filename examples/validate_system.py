import sys
import os
import torch
import warnings

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iints.core.patient.models import PatientModel # Using CustomPatientModel aliased as PatientModel
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.lstm_algorithm import LSTMInsulinAlgorithm
from iints.core.simulator import Simulator, StressEvent

def check_python_version():
    """Check if Python version is 3.8 or newer (relaxed requirement)."""
    print("Checking Python version...")
    if sys.version_info >= (3, 8):
        if sys.version_info >= (3, 9):
            print(f" Python version is {sys.version.split(' ')[0]} (>= 3.9). Full compatibility.")
        else:
            print(f"  Python version is {sys.version.split(' ')[0]} (3.8). Basic functionality available.")
            print("Note: Some advanced features may require Python 3.9+")
        return True
    else:
        print(f" Python version is {sys.version.split(' ')[0]} (< 3.8). Upgrade required.")
        return False

def check_trained_model():
    """Check if the trained LSTM model file exists."""
    print("Checking for trained LSTM model...")
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'iints', 'core', 'algorithms', 'trained_lstm_model.pth')
    if os.path.exists(model_path):
        print(f" Trained LSTM model found at: {model_path}")
        return True
    else:
        print(f" Trained LSTM model NOT found at: {model_path}")
        print("Please ensure 'trained_lstm_model.pth' is present in 'src/algorithm/'.")
        return False

def check_analysis_plots_directory():
    """Check if the analysis_plots directory exists and is writable."""
    print("Checking 'analysis_plots' directory...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots', 'analysis_plots')
    if not os.path.exists(output_dir):
        print(f"'analysis_plots' directory not found. Attempting to create: {output_dir}")
        try:
            os.makedirs(output_dir)
            print(f" 'analysis_plots' directory created successfully.")
            return True
        except OSError as e:
            print(f" Failed to create 'analysis_plots' directory: {e}")
            return False
    else:
        if os.access(output_dir, os.W_OK):
            print(f" 'analysis_plots' directory exists and is writable: {output_dir}")
            return True
        else:
            print(f" 'analysis_plots' directory exists but is NOT writable: {output_dir}")
            return False

def run_basic_simulation_sanity_check():
    """Run a very basic, minimal simulation step to check algorithm loading."""
    print("Running basic simulation sanity check...")
    try:
        # Suppress the FutureWarning from torch.load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
        patient = PatientModel(initial_glucose=120)
        algo_rb = CorrectionBolus()
        algo_ml = LSTMInsulinAlgorithm()
        
        sim_rb = Simulator(patient_model=patient, algorithm=algo_rb, time_step=5)
        sim_ml = Simulator(patient_model=patient, algorithm=algo_ml, time_step=5)

        # Run one step
        sim_rb.run(duration_minutes=5)
        sim_ml.run(duration_minutes=5)
        print(" Basic simulation sanity check passed. Algorithms loaded and ran one step.")
        return True
    except Exception as e:
        print(f" Basic simulation sanity check FAILED: {e}")
        print("Please check your algorithm implementations and dependencies.")
        return False

if __name__ == "__main__":
    print("--- System Validation ---")
    all_checks_passed = True

    all_checks_passed &= check_python_version()
    all_checks_passed &= check_trained_model()
    all_checks_passed &= check_analysis_plots_directory()
    all_checks_passed &= run_basic_simulation_sanity_check()

    if all_checks_passed:
        print("\n All system validation checks passed! The framework is ready for analysis.")
        sys.exit(0)
    else:
        print("\n  One or more system validation checks failed. Please address the issues before proceeding.")
        sys.exit(1)
