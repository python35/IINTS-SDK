#!/usr/bin/env python3
"""
Data Validation Test - IINTS-AF
Demonstrates data integrity and validation capabilities.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.validator import ReverseEngineeringValidator, ReliabilityLevel

def test_clean_data():
    """Test validation with clean, reliable data."""
    print("=== Testing Clean Data ===")
    
    # Generate clean glucose data
    time_points = list(range(0, 480, 5))  # 8 hours, 5-min intervals
    glucose_values = [120 + 20 * np.sin(t/60) + np.random.normal(0, 2) for t in time_points]
    insulin_values = [max(0, (g - 120) / 50) for g in glucose_values]
    
    df = pd.DataFrame({
        'time_minutes': time_points,
        'glucose_actual_mgdl': glucose_values,
        'delivered_insulin_units': insulin_values
    })
    
    validator = ReverseEngineeringValidator()
    results = validator.validate_simulation_results(df)
    report = validator.generate_reliability_report(results)
    
    print(f"Overall Reliability: {report['overall_reliability_score']:.1f}% ({report['overall_level'].upper()})")
    print(f"Issues: {report['total_issues']}, Warnings: {report['total_warnings']}")
    
    return report

def test_corrupted_data():
    """Test validation with corrupted, unreliable data."""
    print("\\n=== Testing Corrupted Data ===")
    
    # Generate corrupted data
    time_points = list(range(0, 480, 5))
    glucose_values = []
    
    for i, t in enumerate(time_points):
        if i < 20:
            glucose_values.append(120 + np.random.normal(0, 5))
        elif i < 40:
            # Impossible glucose spike
            glucose_values.append(400 + np.random.normal(0, 50))
        elif i < 60:
            # Impossible crash
            glucose_values.append(10 + np.random.normal(0, 5))
        else:
            glucose_values.append(120 + np.random.normal(0, 5))
    
    # Add some negative insulin values (impossible)
    insulin_values = [max(-1, (g - 120) / 50) for g in glucose_values]
    insulin_values[30] = -2.0  # Impossible negative insulin
    insulin_values[50] = 10.0  # Dangerously high insulin
    
    df = pd.DataFrame({
        'time_minutes': time_points,
        'glucose_actual_mgdl': glucose_values,
        'delivered_insulin_units': insulin_values
    })
    
    validator = ReverseEngineeringValidator()
    results = validator.validate_simulation_results(df)
    report = validator.generate_reliability_report(results)
    
    print(f"Overall Reliability: {report['overall_reliability_score']:.1f}% ({report['overall_level'].upper()})")
    print(f"Issues: {report['total_issues']}, Warnings: {report['total_warnings']}")
    
    if report['issues']:
        print("Critical Issues Found:")
        for issue in report['issues'][:5]:  # Show first 5 issues
            print(f"  - {issue}")
    
    return report

def test_algorithmic_drift():
    """Test algorithmic drift detection."""
    print("\\n=== Testing Algorithmic Drift Detection ===")
    
    # AI algorithm that drifts from baseline
    baseline_insulin = [1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0]
    ai_insulin = [1.1, 2.5, 0.7, 3.0, 1.0, 0.1, 2.8, 0.2]  # Significant drift
    
    time_points = list(range(0, 40, 5))
    glucose_values = [150] * len(time_points)
    
    df = pd.DataFrame({
        'time_minutes': time_points,
        'glucose_actual_mgdl': glucose_values,
        'delivered_insulin_units': ai_insulin
    })
    
    validator = ReverseEngineeringValidator()
    results = validator.validate_simulation_results(df, baseline_results=baseline_insulin)
    report = validator.generate_reliability_report(results)
    
    print(f"Overall Reliability: {report['overall_reliability_score']:.1f}% ({report['overall_level'].upper()})")
    print(f"Drift Issues: {len([i for i in report['issues'] if 'drift' in i.lower()])}")
    
    if 'algorithmic_drift' in results:
        drift_result = results['algorithmic_drift']
        print(f"Drift Detection Score: {drift_result.reliability_score:.1f}%")
    
    return report

def test_monte_carlo_reliability():
    """Test Monte Carlo statistical reliability."""
    print("\\n=== Testing Monte Carlo Reliability ===")
    
    # Simulate multiple runs with varying reliability
    print("High Reliability (Low Variance):")
    stable_runs = []
    for _ in range(20):
        run = [1.0 + np.random.normal(0, 0.1) for _ in range(10)]  # Low variance
        stable_runs.append(run)
    
    validator = ReverseEngineeringValidator()
    results = validator.validate_simulation_results(
        pd.DataFrame({'time_minutes': range(10), 'glucose_actual_mgdl': [120]*10, 'delivered_insulin_units': [1.0]*10}),
        monte_carlo_results=stable_runs
    )
    report = validator.generate_reliability_report(results)
    print(f"  Reliability: {report['overall_reliability_score']:.1f}%")
    
    print("\\nLow Reliability (High Variance):")
    unstable_runs = []
    for _ in range(5):  # Too few runs
        run = [1.0 + np.random.normal(0, 2.0) for _ in range(10)]  # High variance
        unstable_runs.append(run)
    
    results = validator.validate_simulation_results(
        pd.DataFrame({'time_minutes': range(10), 'glucose_actual_mgdl': [120]*10, 'delivered_insulin_units': [1.0]*10}),
        monte_carlo_results=unstable_runs
    )
    report = validator.generate_reliability_report(results)
    print(f"  Reliability: {report['overall_reliability_score']:.1f}%")
    
    return report

def main():
    """Run all validation tests."""
    print("=== IINTS-AF Data Validation & Integrity Testing ===")
    print("Demonstrating reverse engineering data reliability checks")
    
    # Run all tests
    clean_report = test_clean_data()
    corrupted_report = test_corrupted_data()
    drift_report = test_algorithmic_drift()
    mc_report = test_monte_carlo_reliability()
    
    print("\\n=== Summary ===")
    print(f"Clean Data Reliability: {clean_report['overall_reliability_score']:.1f}%")
    print(f"Corrupted Data Reliability: {corrupted_report['overall_reliability_score']:.1f}%")
    print(f"Drift Detection Reliability: {drift_report['overall_reliability_score']:.1f}%")
    print(f"Monte Carlo Reliability: {mc_report['overall_reliability_score']:.1f}%")
    
    print("\\n=== Validation Framework Ready ===")
    print("Data integrity validation ensures reliable reverse engineering results")

if __name__ == '__main__':
    main()