#!/usr/bin/env python3
"""
IINTS-AF Population Study Runner
High-level research tool for multi-patient algorithm comparison
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iints.data.adapter import DataAdapter
from iints.core.patient.patient_factory import PatientFactory
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.lstm_algorithm import LSTMInsulinAlgorithm
from iints.core.algorithms.hybrid_algorithm import HybridInsulinAlgorithm
from iints.core.simulator import Simulator, StressEvent

class PopulationRunner:
    """Professional population study runner"""
    
    def __init__(self, output_dir="population_studies"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_adapter = DataAdapter()
        
    def run_population_study(self, config: Dict) -> str:
        """Run comprehensive population study"""
        study_id = f"POP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        study_dir = self.output_dir / study_id
        study_dir.mkdir()
        
        print(f"Population Study: {study_id}")
        print(f"Configuration: {config['name']}")
        print(f"Patients: {config['patient_count']}")
        print(f"Algorithms: {len(config['algorithms'])}")
        print(f"Scenarios: {len(config['scenarios'])}")
        
        # Generate patient population
        patients = self._generate_patient_population(config['patient_count'])
        
        # Run experiments
        results = []
        total_experiments = len(patients) * len(config['algorithms']) * len(config['scenarios'])
        current_exp = 0
        
        for patient_idx, patient in enumerate(patients):
            for algo_name in config['algorithms']:
                for scenario_name in config['scenarios']:
                    current_exp += 1
                    progress = current_exp / total_experiments * 100
                    print(f"Progress: {progress:.1f}% - Patient {patient_idx+1}, {algo_name}, {scenario_name}")
                    
                    try:
                        result = self._run_single_experiment(
                            patient, algo_name, scenario_name, config
                        )
                        result.update({
                            'patient_id': patient_idx,
                            'algorithm': algo_name,
                            'scenario': scenario_name,
                            'study_id': study_id
                        })
                        results.append(result)
                        
                    except Exception as e:
                        print(f"[FAIL] Experiment failed: {e}")
                        continue
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(study_dir / "population_results.csv", index=False)
        
        # Generate analysis
        analysis = self._analyze_population_results(results_df)
        
        with open(study_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save configuration
        with open(study_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Population study complete: {study_id}")
        print(f"Results saved to: {study_dir}")
        
        return study_id
    
    def _generate_patient_population(self, count: int) -> List:
        """Generate diverse patient population"""
        patients = []
        
        # Use patient factory for diversity
        diversity_set = PatientFactory.get_patient_diversity_set()
        
        # Extend if needed
        while len(patients) < count:
            patients.extend(diversity_set)
        
        return patients[:count]
    
    def _run_single_experiment(self, patient, algo_name: str, scenario_name: str, config: Dict) -> Dict:
        """Run single experiment and collect metrics"""
        
        # Initialize algorithm
        algorithms = {
            'rule_based': CorrectionBolus(),
            'lstm': LSTMInsulinAlgorithm(),
            'hybrid': HybridInsulinAlgorithm()
        }
        
        algorithm = algorithms[algo_name]
        simulator = Simulator(patient, algorithm)
        
        # Add scenario events
        scenario_configs = {
            "standard_meal": [StressEvent(60, 'meal', 60)],
            "unannounced_meal": [StressEvent(60, 'missed_meal', 60)],
            "hyperglycemia": [],
            "dawn_phenomenon": [StressEvent(360, 'dawn', 30)]
        }
        
        for event in scenario_configs.get(scenario_name, []):
            simulator.add_stress_event(event)
        
        # Run simulation
        df = simulator.run(duration_minutes=config.get('duration', 480))
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(df, algorithm)
        
        # Reset states
        algorithm.reset()
        patient.reset()
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, df: pd.DataFrame, algorithm) -> Dict:
        """Calculate comprehensive research metrics"""
        
        # Basic glucose metrics
        glucose_values = df['glucose_actual_mgdl']
        
        metrics = {
            # Glucose control
            'mean_glucose': float(glucose_values.mean()),
            'glucose_std': float(glucose_values.std()),
            'cv_percentage': float(glucose_values.std() / glucose_values.mean() * 100),
            'peak_glucose': float(glucose_values.max()),
            'min_glucose': float(glucose_values.min()),
            
            # Time in ranges
            'tir_70_180': float(((glucose_values >= 70) & (glucose_values <= 180)).mean() * 100),
            'tir_70_140': float(((glucose_values >= 70) & (glucose_values <= 140)).mean() * 100),
            'time_above_180': float((glucose_values > 180).mean() * 100),
            'time_below_70': float((glucose_values < 70).mean() * 100),
            
            # Insulin metrics
            'total_insulin': float(df['delivered_insulin_units'].sum()),
            'mean_insulin': float(df['delivered_insulin_units'].mean()),
            'insulin_variability': float(df['delivered_insulin_units'].std()),
            
            # Algorithm-specific metrics
            'decision_count': len(df),
            'non_zero_decisions': int((df['delivered_insulin_units'] > 0).sum())
        }
        
        # Algorithm-specific stats
        if hasattr(algorithm, 'get_stats'):
            algo_stats = algorithm.get_stats()
            metrics.update({
                f'algo_{key}': value for key, value in algo_stats.items()
            })
        
        return metrics
    
    def _analyze_population_results(self, df: pd.DataFrame) -> Dict:
        """Analyze population study results"""
        
        analysis = {
            'study_summary': {
                'total_experiments': len(df),
                'algorithms_tested': df['algorithm'].unique().tolist(),
                'scenarios_tested': df['scenario'].unique().tolist(),
                'patients_tested': df['patient_id'].nunique()
            },
            'algorithm_comparison': {},
            'scenario_analysis': {},
            'statistical_tests': {}
        }
        
        # Algorithm comparison
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            
            analysis['algorithm_comparison'][algo] = {
                'mean_tir': float(algo_data['tir_70_180'].mean()),
                'std_tir': float(algo_data['tir_70_180'].std()),
                'mean_cv': float(algo_data['cv_percentage'].mean()),
                'std_cv': float(algo_data['cv_percentage'].std()),
                'mean_insulin': float(algo_data['total_insulin'].mean()),
                'worst_case_tir': float(algo_data['tir_70_180'].min()),
                'best_case_tir': float(algo_data['tir_70_180'].max())
            }
        
        # Scenario analysis
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            
            analysis['scenario_analysis'][scenario] = {
                'mean_peak_glucose': float(scenario_data['peak_glucose'].mean()),
                'glucose_variability': float(scenario_data['glucose_std'].mean()),
                'insulin_usage': float(scenario_data['total_insulin'].mean())
            }
        
        return analysis

def main():
    """Run population study with example configuration"""
    
    # Example configuration
    config = {
        "name": "Algorithm Comparison Study",
        "description": "Compare rule-based, LSTM, and hybrid algorithms across diverse patient population",
        "patient_count": 10,
        "algorithms": ["rule_based", "lstm", "hybrid"],
        "scenarios": ["standard_meal", "unannounced_meal"],
        "duration": 480,  # 8 hours
        "hypothesis": "Hybrid algorithm shows more consistent performance across patient diversity"
    }
    
    runner = PopulationRunner()
    study_id = runner.run_population_study(config)
    
    print(f"\nPopulation study completed: {study_id}")
    print("Check population_studies/ directory for detailed results")

if __name__ == "__main__":
    main()
