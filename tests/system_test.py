import sys
import os
import pandas as pd
from pathlib import Path
from rich.console import Console
import numpy as np
from typing import Optional
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()
test_results = []

def run_test(test_name, func, *args, **kwargs):
    console.print(f"[bold blue]Running Test: {test_name}...[/bold blue]")
    try:
        func(*args, **kwargs)
        console.print(f"[bold green]PASS: {test_name}[/bold green]")
        test_results.append((test_name, "PASS"))
    except Exception as e:
        console.print(f"[bold red]FAIL: {test_name} - {e}[/bold red]")
        test_results.append((test_name, "FAIL"))

# --- Test Cases ---

def test_data_parsing():
    from iints.data.universal_parser import UniversalParser
    
    # Create a dummy CSV file for testing
    dummy_csv_content = """timestamp,glucose_mg_dl,carbs,insulin
0,120,0,0
5,125,10,0.5
10,130,0,0
15,135,20,1.0
20,140,0,0
25,145,0,0
"""
    dummy_csv_path = Path("/tmp/dummy_patient_data.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content)

    parser = UniversalParser()
    result = parser.parse(str(dummy_csv_path))
    
    if not result.success:
        raise Exception(f"Failed to parse dummy CSV: {result.errors}")
    if result.data_pack.data.empty:
        raise Exception("Parsed data is empty.")
    if 'glucose' not in result.data_pack.data.columns:
        raise Exception("Glucose column not found in parsed data.")
    if result.data_pack.confidence_score < 0.8:
        raise Exception(f"Low confidence score for parsed data: {result.data_pack.confidence_score}")
    
    os.remove(dummy_csv_path) # Clean up

def test_algorithm_simulation():
    from iints.core.simulator import Simulator
    from iints.core.patient.models import PatientModel
    from iints.core.algorithms.pid_controller import PIDController # Use a simple algorithm

    patient = PatientModel(initial_glucose=120)
    algorithm = PIDController()
    simulator = Simulator(patient_model=patient, algorithm=algorithm, time_step=5)

    df_results, _ = simulator.run(duration_minutes=60)

    if df_results.empty:
        raise Exception("Simulation returned empty results.")
    if 'glucose_actual_mgdl' not in df_results.columns:
        raise Exception("Actual glucose column not found in simulation results.")
    if df_results['glucose_actual_mgdl'].isnull().any():
        raise Exception("Simulation results contain null glucose values.")

def test_battle_runner_execution():
    from iints.core.algorithms.battle_runner import BattleRunner
    from iints.core.algorithms.pid_controller import PIDController
    from iints.core.algorithms.correction_bolus import CorrectionBolus
    from iints.core.patient.models import PatientModel # For data generation consistency

    # Create dummy patient data
    n_points = 12 * 4 # 4 hours of 5-min intervals
    time = np.arange(n_points) * 5
    glucose_values = 120 + 10 * np.sin(np.linspace(0, 2*np.pi, n_points))
    carbs_values = np.zeros(n_points)
    carbs_values[10] = 30 # Meal
    
    patient_data_df = pd.DataFrame({
        'time': time,
        'glucose': glucose_values,
        'carbs': carbs_values,
        'insulin': np.zeros(n_points)
    })

    algorithms = {
        "PID": PIDController(),
        "CorrectionBolus": CorrectionBolus()
    }

    runner = BattleRunner(algorithms, patient_data_df)
    report, detailed_data = runner.run_battle()

    if not report:
        raise Exception("BattleRunner returned empty report.")
    if not detailed_data:
        raise Exception("BattleRunner returned empty detailed data.")
    if "winner" not in report:
        raise Exception("Battle report missing 'winner' key.")
    if not report["rankings"]:
        raise Exception("Battle report missing rankings.")

def test_report_generation():
    from examples.generate_clinical_report import ClinicalReportGenerator
    from iints.data.universal_parser import UniversalParser
    from iints.analysis.clinical_metrics import ClinicalMetricsResult
    
    # Mock data for report generation
    # Create a dummy CSV file for testing
    dummy_csv_content = """timestamp,glucose_mg_dl,carbs,insulin
0,120,0,0
5,125,10,0.5
10,130,0,0
15,135,20,1.0
20,140,0,0
25,145,0,0
"""
    dummy_csv_path = Path("/tmp/dummy_report_data.csv")
    with open(dummy_csv_path, "w") as f:
        f.write(dummy_csv_content)

    parser = UniversalParser()
    parse_result = parser.parse(str(dummy_csv_path))
    patient_data = parse_result.data_pack.data
    os.remove(dummy_csv_path) # Clean up

    mock_results = {
        'patient_id': 'system_test_patient',
        'original_performance': ClinicalMetricsResult(
            tir_70_180=60.0, tir_70_140=40.0, tir_70_110=20.0,
            tir_below_70=5.0, tir_below_54=1.0, tir_above_180=35.0,
            tir_above_250=10.0, cv=35.0, sd=50.0, gmi=7.0,
            mean_glucose=150.0, median_glucose=140.0, hi=0.5,
            lbgi=2.0, hbgi=8.0, readings_per_day=288.0, data_coverage=100.0
        ).to_dict(),
        'algorithm_results': {
            'pid': ClinicalMetricsResult(
                tir_70_180=75.0, tir_70_140=55.0, tir_70_110=30.0,
                tir_below_70=3.0, tir_below_54=0.5, tir_above_180=22.0,
                tir_above_250=5.0, cv=28.0, sd=40.0, gmi=6.5,
                mean_glucose=130.0, median_glucose=125.0, hi=0.3,
                lbgi=1.0, hbgi=5.0, readings_per_day=288.0, data_coverage=100.0
            ).to_dict()
        }
    }
    
    # Create a dummy directory for report output
    report_output_dir = Path("/tmp/system_test_reports")
    report_output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ClinicalReportGenerator()
    report_path = generator.generate_report(
        patient_id=mock_results['patient_id'],
        results=mock_results, # Pass mock_results as 'results'
        patient_data=patient_data
    )

    if not Path(report_path).exists():
        raise Exception(f"Report file not created: {report_path}")
    if not str(report_path).endswith(".pdf"):
        raise Exception(f"Report file is not a PDF: {report_path}")
    
    # Clean up
    os.remove(report_path)
    report_output_dir.rmdir()


def test_visualization_generation():
    from iints.visualization.cockpit import ClinicalCockpit
    
    # Create dummy simulation data
    n_points = 60 # 5 hours of 5-min intervals
    timestamps = np.arange(0, n_points * 5, 5)
    glucose = 120 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points)) + np.random.normal(0, 5, n_points)
    simulation_data = pd.DataFrame({
        'time_minutes': timestamps,
        'glucose_actual_mgdl': glucose,
        'delivered_insulin_units': np.random.uniform(0, 1, n_points),
        'patient_iob_units': np.cumsum(np.random.uniform(0, 0.1, n_points)),
        'uncertainty': np.random.uniform(0.1, 0.4, n_points)
    })

    cockpit = ClinicalCockpit()
    
    viz_output_path = Path("/tmp/system_test_viz.png")
    fig = cockpit.visualize_results(
        simulation_data=simulation_data,
        save_path=str(viz_output_path)
    )

    if not viz_output_path.exists():
        raise Exception(f"Visualization file not created: {viz_output_path}")
    
    os.remove(viz_output_path) # Clean up

def test_terminal_app_initialization():
    terminal_app_path = Path(__file__).parent.parent / "bin" / "main.py"
    if not terminal_app_path.exists():
        pytest.skip("Terminal app entrypoint not present in this package layout.")
    from bin.main import IINTSTerminalApp
    try:
        app = IINTSTerminalApp()
        # Just check if it initializes without error, no need to run interactively
    except Exception as e:
        raise Exception(f"Failed to initialize IINTSTerminalApp: {e}")

def main():
    console.print("\n[bold magenta]=== IINTS-AF System Health Check ===[/bold magenta]\n")

    run_test("Data Parsing Functionality", test_data_parsing)
    run_test("Algorithm Simulation Execution", test_algorithm_simulation)
    run_test("Battle Runner Execution", test_battle_runner_execution)
    run_test("Clinical Report Generation", test_report_generation)
    run_test("Visualization Generation", test_visualization_generation)
    run_test("Terminal App Initialization", test_terminal_app_initialization)

    console.print("\n[bold magenta]=== Test Summary ===[/bold magenta]")
    passes = [res for res in test_results if res[1] == "PASS"]
    fails = [res for res in test_results if res[1] == "FAIL"]

    console.print(f"[bold green]Total Passed: {len(passes)}[/bold green]")
    console.print(f"[bold red]Total Failed: {len(fails)}[/bold red]")

    if fails:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
