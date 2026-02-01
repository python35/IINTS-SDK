#!/usr/bin/env python3
"""
Quick Clinical Report Generator
Generate professional PDF reports for Science-Expo
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Generate clinical report for first available patient"""
    from scripts.generate_clinical_report import ClinicalReportGenerator
    from src.data.adapter import DataAdapter
    import pandas as pd
    
    print("IINTS-AF Clinical Report Generator")
    print("=" * 50)
    
    # Check for Ohio patients
    adapter = DataAdapter()
    patients = adapter.get_available_ohio_patients()
    
    if not patients:
        print(" No Ohio T1DM patients found")
        print("Run: python tools/import_ohio.py /path/to/ohio/dataset")
        return
    
    # Use first available patient
    patient_id = patients[0]
    print(f"Generating report for Ohio T1DM Patient {patient_id}")
    
    # Get benchmark results
    print("Running clinical benchmark analysis...")
    results = adapter.clinical_benchmark_comparison(patient_id, ['rule_based', 'lstm', 'hybrid'])
    
    # Load patient data
    print("Loading patient glucose data...")
    patient_data = pd.read_csv(f"data_packs/public/ohio_t1dm/patient_{patient_id}/timeseries.csv")
    
    # Generate report
    print("Creating professional PDF report...")
    generator = ClinicalReportGenerator()
    report_path = generator.generate_report(patient_id, results, patient_data)
    
    print(f"Clinical report generated: {report_path}")
    print("Professional PDF ready for Science-Expo presentation!")
    
    # Show key metrics
    original_tir = results['original_performance']['tir_70_180']
    best_algo = max(results['algorithm_results'].items(), key=lambda x: x[1]['tir_70_180'])
    ai_tir = best_algo[1]['tir_70_180']
    improvement = ai_tir - original_tir
    
    print(f"\nKey Results:")
    print(f"   Original Patient TIR: {original_tir:.1f}%")
    print(f"   Best AI Algorithm: {best_algo[0].upper()}")
    print(f"   AI Model TIR: {ai_tir:.1f}%")
    print(f"   Improvement: +{improvement:.1f}%")
    
    if improvement > 10:
        print("Excellent clinical improvement!")
    elif improvement > 5:
        print("Good clinical improvement!")
    else:
        print("Modest clinical improvement")

if __name__ == "__main__":
    main()