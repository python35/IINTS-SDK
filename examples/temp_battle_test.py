import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iints.core.algorithms.battle_runner import BattleRunner
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.hybrid_algorithm import HybridInsulinAlgorithm
from iints.emulation.medtronic_780g import Medtronic780GEmulator
from iints.emulation.tandem_controliq import TandemControlIQEmulator
from iints.emulation.omnipod_5 import Omnipod5Emulator

def run_test():
    """Demonstrate battle runner functionality"""
    print("=" * 70)
    print("BATTLE RUNNER DEMONSTRATION")
    print("=" * 70)
    
    runner = BattleRunner()
    
    # Register algorithms
    print("\n  Registering algorithms...")
    runner.register_algorithm("PID Controller", PIDController(), "#2196F3")
    runner.register_algorithm("Correction Bolus", CorrectionBolus(), "#4CAF50")
    runner.register_algorithm("Hybrid", HybridInsulinAlgorithm(), "#FF9800")
    
    # Generate sample data
    print("\n Generating sample data...")
    np.random.seed(42)
    n_points = 288  # 24 hours at 5-min intervals
    
    time = np.arange(n_points) * 5
    glucose = 120 + 30 * np.sin(time / (24 * 12 / (2 * np.pi))) + np.random.normal(0, 15, n_points)
    glucose = np.clip(glucose, 40, 400)
    
    data = pd.DataFrame({
        'timestamp': time,
        'glucose': glucose,
        'carbs': np.random.choice([0, 30, 60], n_points, p=[0.8, 0.15, 0.05]),
        'insulin': 0.0
    })
    
    # Run battle
    print("\n Starting battle...")
    report = runner.run_battle(data, scenario="standard_meal")
    
    # Print summary
    print("\n" + "=" * 70)
    print(report.get_summary())
    
    # Comparison table
    print("\n Comparison Table")
    print("-" * 70)
    table = runner.get_comparison_table(report)
    print(table.to_string(index=False))
    
    # Save report
    runner.save_report(report, "results/reports/battle_report_test.json")
    
    print("\n" + "=" * 70)
    print("BATTLE RUNNER DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
