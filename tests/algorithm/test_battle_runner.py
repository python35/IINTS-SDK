import pytest
import pandas as pd
import numpy as np
from iints.core.algorithms.battle_runner import BattleRunner
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.simulator import Simulator # Import Simulator to ensure its seed is used

# Mock Algorithm to test determinism, as PIDController and CorrectionBolus might not use np.random
class DeterministicAlgorithm(PIDController):
    """A simple algorithm for testing determinism."""
    def __init__(self):
        super().__init__()
        self.internal_random_state = None # To store if it uses random internally

    def predict_insulin(self, algo_input) -> dict:
        # Simulate some internal "randomness" that should be affected by seed
        # For a truly deterministic algorithm like PID, this won't change unless its parameters are randomized
        # We need something that actually uses np.random
        if self.internal_random_state is None:
            # Generate a random offset based on the seed
            self.internal_random_state = np.random.rand() * 5 # A random value between 0 and 5
            
        base_insulin = super().predict_insulin(algo_input).get("total_insulin_delivered", 0.0)
        # Add the "random" internal state to make it dependent on seed
        return {"total_insulin_delivered": base_insulin + self.internal_random_state}


def create_sample_data(seed=None):
    """Helper function to create reproducible sample data."""
    if seed is not None:
        np.random.seed(seed)
    
    n_points = 288  # 24 hours at 5-min intervals
    time = np.arange(n_points) * 5
    
    # Simple glucose curve with some noise
    glucose = 120 + 30 * np.sin(time / (24 * 12 / (2 * np.pi))) + np.random.normal(0, 5, n_points)
    glucose = np.clip(glucose, 40, 400)
    
    # Introduce some carb intake
    carbs = np.zeros(n_points)
    carbs[int(60/5)] = 50 # Meal at 60 min
    carbs[int(300/5)] = 70 # Meal at 300 min
    
    data = pd.DataFrame({
        'time': time,
        'glucose': glucose,
        'carbs': carbs,
        'insulin': 0.0 # Placeholder, not used for input data
    })
    return data

def test_battle_runner_produces_report_and_details():
    """
    Test that BattleRunner produces a report and per-algorithm simulation data.
    """
    algorithms = {
        "Deterministic Algo": DeterministicAlgorithm(),
        "Correction Bolus": CorrectionBolus()
    }

    sample_data = create_sample_data(seed=123)
    runner = BattleRunner(algorithms=algorithms, patient_data=sample_data, scenario_name="deterministic_test")

    report, detailed_data = runner.run_battle()

    assert report["winner"] in algorithms
    assert len(report["rankings"]) == len(algorithms)
    assert set(detailed_data.keys()) == set(algorithms.keys())
    assert all(not df.empty for df in detailed_data.values())
