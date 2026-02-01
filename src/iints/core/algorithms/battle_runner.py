import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from iints.core.simulator import Simulator, StressEvent
from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.core.patient.models import PatientModel

class BattleRunner:
    """Runs a battle between different insulin algorithms."""

    def __init__(self, algorithms: Dict[str, Any], patient_data: pd.DataFrame, stress_events: Optional[List[StressEvent]] = None, scenario_name: str = "standard"):
        """
        Initializes the BattleRunner.

        Args:
            algorithms: A dictionary of algorithm names to their instances.
            patient_data: A DataFrame with 'time', 'glucose', and 'carbs' columns.
            stress_events: An optional list of StressEvent objects to apply during simulation.
            scenario_name: The name of the scenario being run.
        """
        self.algorithms = algorithms
        self.patient_data = patient_data
        self.metrics_calculator = ClinicalMetricsCalculator()
        self.stress_events = stress_events if stress_events is not None else []
        self.scenario_name = scenario_name

    def run_battle(self, 
                   isf_override: Optional[float] = None, 
                   icr_override: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]: # Modify return type
        """
        Runs the simulation for each algorithm and returns a battle report and detailed simulation data.

        Args:
            isf_override (Optional[float]): Override value for Insulin Sensitivity Factor.
            icr_override (Optional[float]): Override value for Insulin-to-Carb Ratio.

        Returns:
            Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]: A tuple containing:
                - A dictionary containing the battle report.
                - A dictionary mapping algorithm names to their full simulation results DataFrame.
        """
        battle_results = {}
        detailed_simulation_data = {} # New dictionary to store full dfs
        
        for algo_name, algo_instance in self.algorithms.items():
            print(f"Running simulation for {algo_name}...")

            # Apply overrides if provided
            if isf_override is not None and hasattr(algo_instance, 'set_isf'):
                algo_instance.set_isf(isf_override)
            if icr_override is not None and hasattr(algo_instance, 'set_icr'):
                algo_instance.set_icr(icr_override)

            initial_glucose = self.patient_data['glucose'].iloc[0]
            patient_model = PatientModel(initial_glucose=initial_glucose)
            
            simulator = Simulator(patient_model=patient_model, algorithm=algo_instance, time_step=5)
            
            # Add predefined stress events to the simulator
            for event in self.stress_events:
                simulator.add_stress_event(event)

            duration = self.patient_data['time'].max()
            
            # Existing carb events from patient_data are also added as stress events
            # This needs careful consideration if scenario also adds carb events
            for index, row in self.patient_data.iterrows():
                if row['carbs'] > 0:
                    simulator.add_stress_event(StressEvent(start_time=int(row['time']), event_type='meal', value=row['carbs']))
            
            # Unpack the tuple returned by simulator.run()
            simulation_results_df, algorithm_safety_report = simulator.run(duration_minutes=duration)
            
            detailed_simulation_data[algo_name] = simulation_results_df # Store the full df
            
            glucose_series = simulation_results_df['glucose_actual_mgdl']
            metrics = self.metrics_calculator.calculate(glucose=glucose_series, duration_hours=duration/60)
            
            # Calculate uncertainty score
            uncertainty_score = simulation_results_df['uncertainty'].mean() if 'uncertainty' in simulation_results_df.columns else 0.0
            
            metrics_dict = metrics.to_dict()
            metrics_dict['uncertainty_score'] = uncertainty_score
            
            # Add safety report metrics
            bolus_interventions_count = algorithm_safety_report.get('bolus_interventions_count', 0)
            metrics_dict['bolus_interventions_count'] = bolus_interventions_count
            
            # The existing 'safety_events_count' will now be based on the bolus_interventions_count
            metrics_dict['safety_events_count'] = bolus_interventions_count 

            battle_results[algo_name] = metrics_dict
            
        winner = max(battle_results, key=lambda algo: battle_results[algo]['tir_70_180'])
        
        report = {
            "battle_name": "Algorithm Battle",
            "winner": winner,
            "scenario_name": self.scenario_name, # Add scenario name
            "rankings": sorted(
                [
                    {
                        "participant": algo_name,
                        "overall_score": result['tir_70_180'],
                        "tir": result['tir_70_180'],
                        "tir_tight": result.get('tir_70_140', 0),
                        "cv": result['cv'],
                        "time_below_70": result['tir_below_70'],
                        "time_above_180": result['tir_above_180'],
                        "gmi": result['gmi'],
                        "lbgi": result['lbgi'],
                        "uncertainty_score": result['uncertainty_score'],
                        "safety_events_count": result['safety_events_count'],
                        "bolus_interventions_count": result['bolus_interventions_count'],
                    }
                    for algo_name, result in battle_results.items()
                ],
                key=lambda x: x["overall_score"],
                reverse=True,
            ),
            "detailed_simulation_data": {algo_name: df.to_json() for algo_name, df in detailed_simulation_data.items()} # Convert DataFrames to JSON strings
        }
        
        return report, detailed_simulation_data

    def print_battle_report(self, battle_report: Dict[str, Any]):
        """
        Prints the battle report to the console.

        Args:
            battle_report: The battle report dictionary.
        """
        print("\nBATTLE REPORT")
        print("="*70)
        print(f"Winner: {battle_report['winner']}")
        print("\nRankings:")
        for rank in battle_report['rankings']:
            print(f"- {rank['participant']}: TIR = {rank['tir']:.1f}%, CV = {rank['cv']:.1f}%, Safety Events = {rank['safety_events_count']}")