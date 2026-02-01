import pytest
import pandas as pd
import numpy as np
from src.algorithm.battle_runner import BattleRunner
from src.algorithm.pid_controller import PIDController
from src.algorithm.correction_bolus import CorrectionBolus
from src.simulation.simulator import Simulator # Import Simulator to ensure its seed is used

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
        'timestamp': time,
        'glucose': glucose,
        'carbs': carbs,
        'insulin': 0.0 # Placeholder, not used for input data
    })
    return data

def test_battle_runner_deterministic_behavior():
    """
    Test that BattleRunner produces deterministic results when a seed is provided.
    """
    
    test_seed = 123
    
    # Run 1
    runner1 = BattleRunner()
    runner1.register_algorithm("Deterministic Algo", DeterministicAlgorithm())
    runner1.register_algorithm("Correction Bolus", CorrectionBolus())
    
    sample_data1 = create_sample_data(seed=test_seed)
    report1 = runner1.run_battle(sample_data1, scenario="deterministic_test", seed=test_seed, reset_patient=True)
    
    # Run 2
    runner2 = BattleRunner()
    runner2.register_algorithm("Deterministic Algo", DeterministicAlgorithm())
    runner2.register_algorithm("Correction Bolus", CorrectionBolus())
    
    sample_data2 = create_sample_data(seed=test_seed)
    report2 = runner2.run_battle(sample_data2, scenario="deterministic_test", seed=test_seed, reset_patient=True)
    
    # Compare results
    assert report1.winner == report2.winner, "Winners should be identical"
    assert len(report1.rankings) == len(report2.rankings), "Number of rankings should be identical"
    
    for i in range(len(report1.rankings)):
        rank1 = report1.rankings[i]
        rank2 = report2.rankings[i]
        
        assert rank1['participant'] == rank2['participant'], f"Participant mismatch at rank {i}"
        assert np.isclose(rank1['overall_score'], rank2['overall_score']), f"Overall score mismatch at rank {i}"
        assert np.isclose(rank1['tir'], rank2['tir']), f"TIR mismatch at rank {i}"
        # Add more assertions for other key metrics if desired
        
    # Check simulation data itself for a participant (e.g., Deterministic Algo)
    algo_data1 = next(p.results.simulation_data for p in report1.participants if p.name == "Deterministic Algo")
    algo_data2 = next(p.results.simulation_data for p in report2.participants if p.name == "Deterministic Algo")
    
    pd.testing.assert_frame_equal(algo_data1, algo_data2, check_exact=False, rtol=1e-5), \
        "Simulation data frames for Deterministic Algo should be identical"

    # Check that different seeds produce different results (to confirm randomness is truly seeded)
    seed_diff = 456
    runner_diff = BattleRunner()
    runner_diff.register_algorithm("Deterministic Algo", DeterministicAlgorithm())
    runner_diff.register_algorithm("Correction Bolus", CorrectionBolus())
    
    sample_data_diff = create_sample_data(seed=seed_diff)
    report_diff = runner_diff.run_battle(sample_data_diff, scenario="deterministic_test_diff", seed=seed_diff, reset_patient=True)

    algo_data_diff = next(p.results.simulation_data for p in report_diff.participants if p.name == "Deterministic Algo")
    
    # It's highly improbable that results from different seeds will be identical for non-trivial simulations
    # So, we assert that at least one key metric is different for a participant
    # Compare a critical column, like glucose_actual_mgdl or delivered_insulin_units
    assert not algo_data1['glucose_actual_mgdl'].equals(algo_data_diff['glucose_actual_mgdl']) or \
           not algo_data1['delivered_insulin_units'].equals(algo_data_diff['delivered_insulin_units']), \
        "Simulation data for Deterministic Algo should be different for different seeds."
