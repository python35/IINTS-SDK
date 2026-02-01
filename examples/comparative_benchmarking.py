#!/usr/bin/env python3
"""
Comparative Algorithm Benchmarking - IINTS-AF
Head-to-head performance comparison with statistical significance testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.algorithm.lstm_algorithm import LSTMInsulinAlgorithm
from src.algorithm.hybrid_algorithm import HybridInsulinAlgorithm
from src.algorithm.fixed_basal_bolus import FixedBasalBolus
from src.algorithm.pid_controller import PIDController
from src.analysis.clinical_tir_analyzer import ClinicalTIRAnalyzer

class AlgorithmBenchmark:
    """Comparative benchmarking of diabetes management algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'LSTM_AI': LSTMInsulinAlgorithm(),
            'Hybrid_Control': HybridInsulinAlgorithm(), 
            'Fixed_Dose': FixedBasalBolus(),
            'Industry_PID': PIDController()
        }
        
        self.tir_analyzer = ClinicalTIRAnalyzer()
        self.results = {}
    
    def run_comparative_study(self, patients=['559', '563', '570', '575'], days=7):
        """Run head-to-head comparison across multiple patients"""
        
        print("COMPARATIVE ALGORITHM BENCHMARKING")
        print("=" * 50)
        print(f"Patients: {len(patients)}")
        print(f"Algorithms: {len(self.algorithms)}")
        print(f"Simulation Days: {days}")
        print()
        
        all_results = []
        
        for patient_id in patients:
            print(f"Testing Patient {patient_id}...")
            
            # Generate synthetic patient data for demo
            glucose_data = self._generate_patient_data(patient_id, days)
            
            for algo_name, algorithm in self.algorithms.items():
                print(f"  Running {algo_name}...")
                
                # Simulate algorithm performance
                sim_results = self._simulate_algorithm(algorithm, glucose_data)
                
                # Analyze TIR performance
                glucose_values = sim_results['glucose_history']
                tir_analysis = self.tir_analyzer.analyze_glucose_zones(glucose_values)
                
                # Calculate additional metrics
                metrics = self._calculate_performance_metrics(sim_results, tir_analysis)
                
                # Store results
                result = {
                    'patient_id': patient_id,
                    'algorithm': algo_name,
                    'tir_percentage': tir_analysis['target_range']['percentage'],
                    'hypoglycemia_percentage': tir_analysis['low']['percentage'] + tir_analysis['very_low']['percentage'],
                    'hyperglycemia_percentage': tir_analysis['high']['percentage'] + tir_analysis['very_high']['percentage'],
                    'mean_glucose': metrics['mean_glucose'],
                    'glucose_variability': metrics['glucose_cv'],
                    'safety_activations': metrics['safety_activations'],
                    'total_insulin': metrics['total_insulin'],
                    'clinical_risk_score': metrics['clinical_risk_score']
                }
                
                all_results.append(result)
        
        # Convert to DataFrame for analysis
        self.results_df = pd.DataFrame(all_results)
        
        # Perform statistical analysis
        self.statistical_analysis = self._perform_statistical_tests()
        
        # Generate comparison report
        self.comparison_report = self._generate_comparison_report()
        
        print("\n[OK] Comparative benchmarking complete!")
        return self.results_df
    
    def _generate_patient_data(self, patient_id, days):
        """Generate synthetic patient glucose data for testing"""
        
        # Different patient profiles
        patient_profiles = {
            '559': {'baseline': 140, 'variability': 30, 'trend_bias': 0.1},
            '563': {'baseline': 160, 'variability': 45, 'trend_bias': -0.05},
            '570': {'baseline': 120, 'variability': 25, 'trend_bias': 0.15},
            '575': {'baseline': 180, 'variability': 50, 'trend_bias': -0.1}
        }
        
        profile = patient_profiles.get(patient_id, patient_profiles['559'])
        
        # Generate time series (5-minute intervals)
        timesteps = days * 24 * 12  # 5-min intervals
        
        # Base glucose with circadian rhythm
        time_hours = np.linspace(0, days * 24, timesteps)
        circadian = 20 * np.sin(2 * np.pi * time_hours / 24)  # Daily rhythm
        
        # Add meal spikes (3 meals per day)
        meal_spikes = np.zeros(timesteps)
        for day in range(days):
            for meal_time in [8, 13, 19]:  # Breakfast, lunch, dinner
                meal_idx = int((day * 24 + meal_time) * 12 / 24)
                if meal_idx < timesteps:
                    # Meal spike lasting ~2 hours
                    spike_duration = 24  # 2 hours in 5-min intervals
                    for i in range(spike_duration):
                        if meal_idx + i < timesteps:
                            meal_spikes[meal_idx + i] = 60 * np.exp(-i/8)  # Exponential decay
        
        # Combine components
        glucose_base = (profile['baseline'] + 
                       circadian + 
                       meal_spikes + 
                       np.random.normal(0, profile['variability'], timesteps))
        
        # Add trend
        trend = np.linspace(0, profile['trend_bias'] * timesteps, timesteps)
        glucose_data = glucose_base + trend
        
        # Ensure physiological bounds
        glucose_data = np.clip(glucose_data, 40, 400)
        
        return {
            'glucose_history': glucose_data,
            'timestamps': time_hours,
            'patient_id': patient_id
        }
    
    def _simulate_algorithm(self, algorithm, glucose_data):
        """Simulate algorithm performance on glucose data"""
        
        glucose_history = glucose_data['glucose_history']
        insulin_history = []
        
        # Reset algorithm state if it has reset method
        if hasattr(algorithm, 'reset_controller'):
            algorithm.reset_controller()
        
        # Simulate algorithm decisions
        for i, glucose in enumerate(glucose_history):
            # Calculate trend (last 3 readings)
            if i >= 2:
                trend = [glucose_history[i-2], glucose_history[i-1], glucose]
            else:
                trend = [glucose] * 3
            
            # Get insulin recommendation
            try:
                if hasattr(algorithm, 'calculate_insulin'):
                    result = algorithm.calculate_insulin(glucose, i * 5/60)  # time in hours
                    if isinstance(result, dict):
                        insulin_dose = result.get('total_insulin_delivered', 0)
                    else:
                        insulin_dose = result
                elif hasattr(algorithm, 'predict_insulin'):
                    insulin_dose = algorithm.predict_insulin(glucose, trend)
                else:
                    insulin_dose = 0
            except Exception as e:
                print(f"Algorithm error: {e}")
                insulin_dose = 0
            
            insulin_history.append(insulin_dose)
        
        return {
            'glucose_history': glucose_history,
            'insulin_history': insulin_history,
            'timestamps': glucose_data['timestamps']
        }
    
    def _calculate_performance_metrics(self, sim_results, tir_analysis):
        """Calculate comprehensive performance metrics"""
        
        glucose_values = sim_results['glucose_history']
        insulin_values = sim_results['insulin_history']
        
        # Basic statistics
        mean_glucose = np.mean(glucose_values)
        glucose_cv = (np.std(glucose_values) / mean_glucose) * 100
        
        # Safety metrics
        safety_activations = sum(1 for g in glucose_values if g < 54 or g > 300)
        
        # Insulin efficiency
        total_insulin = sum(insulin_values)
        
        # Clinical risk score (weighted combination)
        hypo_risk = tir_analysis['low']['percentage'] + (tir_analysis['very_low']['percentage'] * 2)
        hyper_risk = tir_analysis['high']['percentage'] + (tir_analysis['very_high']['percentage'] * 1.5)
        variability_risk = min(glucose_cv / 10, 10)  # Cap at 10
        
        clinical_risk_score = hypo_risk + hyper_risk + variability_risk
        
        return {
            'mean_glucose': round(mean_glucose, 1),
            'glucose_cv': round(glucose_cv, 2),
            'safety_activations': safety_activations,
            'total_insulin': round(total_insulin, 2),
            'clinical_risk_score': round(clinical_risk_score, 2)
        }
    
    def _perform_statistical_tests(self):
        """Perform statistical significance testing between algorithms"""
        
        print("\nSTATISTICAL ANALYSIS")
        print("-" * 30)
        
        # Group by algorithm
        algo_groups = self.results_df.groupby('algorithm')
        
        # Key metrics for comparison
        metrics = ['tir_percentage', 'hypoglycemia_percentage', 'clinical_risk_score']
        
        statistical_results = {}
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            
            # Get data for each algorithm
            lstm_data = self.results_df[self.results_df['algorithm'] == 'LSTM_AI'][metric]
            pid_data = self.results_df[self.results_df['algorithm'] == 'Industry_PID'][metric]
            hybrid_data = self.results_df[self.results_df['algorithm'] == 'Hybrid_Control'][metric]
            fixed_data = self.results_df[self.results_df['algorithm'] == 'Fixed_Dose'][metric]
            
            # Paired t-tests (LSTM vs others)
            comparisons = {
                'LSTM_vs_PID': stats.ttest_rel(lstm_data, pid_data),
                'LSTM_vs_Hybrid': stats.ttest_rel(lstm_data, hybrid_data),
                'LSTM_vs_Fixed': stats.ttest_rel(lstm_data, fixed_data)
            }
            
            statistical_results[metric] = {}
            
            for comparison, (t_stat, p_value) in comparisons.items():
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"  {comparison}: p={p_value:.4f} {significance}")
                
                statistical_results[metric][comparison] = {
                    't_statistic': round(t_stat, 3),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05,
                    'significance_level': significance
                }
        
        return statistical_results
    
    def _generate_comparison_report(self):
        """Generate executive summary of algorithm comparison"""
        
        # Calculate mean performance by algorithm
        summary_stats = self.results_df.groupby('algorithm').agg({
            'tir_percentage': ['mean', 'std'],
            'hypoglycemia_percentage': ['mean', 'std'],
            'clinical_risk_score': ['mean', 'std'],
            'mean_glucose': ['mean', 'std']
        }).round(2)
        
        # Find best performing algorithm for each metric
        best_tir = self.results_df.groupby('algorithm')['tir_percentage'].mean().idxmax()
        lowest_hypo = self.results_df.groupby('algorithm')['hypoglycemia_percentage'].mean().idxmin()
        lowest_risk = self.results_df.groupby('algorithm')['clinical_risk_score'].mean().idxmin()
        
        report = f"""
COMPARATIVE ALGORITHM BENCHMARKING REPORT
=========================================

EXECUTIVE SUMMARY:
- Best TIR Performance: {best_tir}
- Lowest Hypoglycemia Risk: {lowest_hypo}  
- Lowest Clinical Risk Score: {lowest_risk}

PERFORMANCE SUMMARY:
{summary_stats.to_string()}

STATISTICAL SIGNIFICANCE:
"""
        
        for metric, comparisons in self.statistical_analysis.items():
            report += f"\n{metric.upper()}:\n"
            for comparison, stats in comparisons.items():
                if stats['significant']:
                    report += f"  {comparison}: p={stats['p_value']} {stats['significance_level']} (SIGNIFICANT)\n"
                else:
                    report += f"  {comparison}: p={stats['p_value']} (not significant)\n"
        
        return report
    
    def create_comparison_visualizations(self):
        """Create publication-ready comparison visualizations"""
        
        results_dir = Path("results/algorithm_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. TIR Comparison Box Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        # TIR Performance
        sns.boxplot(data=self.results_df, x='algorithm', y='tir_percentage', ax=axes[0,0])
        axes[0,0].set_title('Time in Range (70-180 mg/dL)')
        axes[0,0].set_ylabel('TIR Percentage (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hypoglycemia Risk
        sns.boxplot(data=self.results_df, x='algorithm', y='hypoglycemia_percentage', ax=axes[0,1])
        axes[0,1].set_title('Hypoglycemia Risk (<70 mg/dL)')
        axes[0,1].set_ylabel('Hypoglycemia Percentage (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Clinical Risk Score
        sns.boxplot(data=self.results_df, x='algorithm', y='clinical_risk_score', ax=axes[1,0])
        axes[1,0].set_title('Clinical Risk Score (Lower = Better)')
        axes[1,0].set_ylabel('Risk Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Mean Glucose
        sns.boxplot(data=self.results_df, x='algorithm', y='mean_glucose', ax=axes[1,1])
        axes[1,1].set_title('Mean Glucose Control')
        axes[1,1].set_ylabel('Mean Glucose (mg/dL)')
        axes[1,1].axhline(y=120, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'algorithm_comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create performance matrix
        performance_matrix = self.results_df.groupby('algorithm').agg({
            'tir_percentage': 'mean',
            'hypoglycemia_percentage': 'mean', 
            'clinical_risk_score': 'mean',
            'glucose_variability': 'mean'
        }).round(1)
        
        # Normalize for heatmap (higher is better for TIR, lower is better for others)
        normalized_matrix = performance_matrix.copy()
        normalized_matrix['hypoglycemia_percentage'] = 100 - normalized_matrix['hypoglycemia_percentage']
        normalized_matrix['clinical_risk_score'] = 100 - normalized_matrix['clinical_risk_score']
        normalized_matrix['glucose_variability'] = 100 - normalized_matrix['glucose_variability']
        
        sns.heatmap(normalized_matrix.T, annot=performance_matrix.T, fmt='.1f', 
                   cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Performance Score'})
        ax.set_title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Performance Metric')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {results_dir}")
        
        return results_dir
    
    def export_results(self):
        """Export comprehensive results for further analysis"""
        
        results_dir = Path("results/algorithm_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Export raw data
        self.results_df.to_csv(results_dir / 'comparative_results.csv', index=False)
        
        # Export statistical analysis
        with open(results_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(self.statistical_analysis, f, indent=2)
        
        # Export comparison report
        with open(results_dir / 'comparison_report.txt', 'w') as f:
            f.write(self.comparison_report)
        
        # Export executive summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = results_dir / f'executive_summary_{timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write("ALGORITHM BENCHMARKING EXECUTIVE SUMMARY\n")
            f.write("=\n")