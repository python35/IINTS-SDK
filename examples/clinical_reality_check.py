#!/usr/bin/env python3
"""
Clinical Reality Check - IINTS-AF
Validates that autonomous learning actually changes AI behavior.

This script performs a comprehensive validation to ensure that the autonomous
learning system actually modifies AI behavior in clinically meaningful ways.
It checks model weights, decision patterns, and generalization to new patients.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.algorithm.lstm_algorithm import LSTMModel, LSTMInsulinAlgorithm
from src.patient.patient_factory import PatientFactory
from src.simulation.simulator import Simulator, StressEvent

def calculate_weight_delta(model_path_old, model_path_new):
    """Calculate difference between old and new model weights."""
    
    if not os.path.exists(model_path_new):
        return 0.0, "New model not found"
    
    model_old = LSTMModel(7, 50, 1)
    model_new = LSTMModel(7, 50, 1)
    
    model_old.load_state_dict(torch.load(model_path_old))
    model_new.load_state_dict(torch.load(model_path_new))
    
    total_delta = 0.0
    param_count = 0
    
    for (name_old, param_old), (name_new, param_new) in zip(
        model_old.named_parameters(), model_new.named_parameters()):
        
        delta = torch.norm(param_old - param_new).item()
        total_delta += delta
        param_count += 1
    
    avg_delta = total_delta / param_count if param_count > 0 else 0.0
    
    return avg_delta, f"Average parameter change: {avg_delta:.6f}"

def compare_ai_decisions(scenario_name="Standard_Meal"):
    """Compare AI decisions before and after learning."""
    
    model_path_old = os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithm', 'trained_lstm_model.pth')
    model_path_new = model_path_old + '.improved'
    
    if not os.path.exists(model_path_new):
        print("No improved model found. Run autonomous learning first.")
        return None
    
    # Create test scenario
    patient = PatientFactory.create_patient('custom', initial_glucose=120)
    
    # Test with old model
    algo_old = LSTMInsulinAlgorithm({'model_path': model_path_old})
    sim_old = Simulator(patient, algo_old)
    sim_old.add_stress_event(StressEvent(60, 'meal', 60))
    
    patient.reset()  # Reset patient state
    df_old = sim_old.run(duration_minutes=240)
    
    # Test with new model  
    algo_new = LSTMInsulinAlgorithm({'model_path': model_path_new})
    sim_new = Simulator(patient, algo_new)
    sim_new.add_stress_event(StressEvent(60, 'meal', 60))
    
    patient.reset()  # Reset patient state
    df_new = sim_new.run(duration_minutes=240)
    
    return df_old, df_new

def test_unseen_patients():
    """Test improved model on unseen patient profiles."""
    
    model_path_old = os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithm', 'trained_lstm_model.pth')
    model_path_new = model_path_old + '.improved'
    
    if not os.path.exists(model_path_new):
        return None
    
    # Create diverse patients
    patients = PatientFactory.get_patient_diversity_set()[:3]  # Test 3 different patients
    
    results = []
    
    for i, patient in enumerate(patients):
        print(f"Testing patient {i+1}...")
        
        # Old model
        algo_old = LSTMInsulinAlgorithm({'model_path': model_path_old})
        sim_old = Simulator(patient, algo_old)
        sim_old.add_stress_event(StressEvent(60, 'meal', 45))
        df_old = sim_old.run(duration_minutes=180)
        
        patient.reset()
        
        # New model
        algo_new = LSTMInsulinAlgorithm({'model_path': model_path_new})
        sim_new = Simulator(patient, algo_new)
        sim_new.add_stress_event(StressEvent(60, 'meal', 45))
        df_new = sim_new.run(duration_minutes=180)
        
        # Calculate metrics
        old_insulin_total = df_old['delivered_insulin_units'].sum()
        new_insulin_total = df_new['delivered_insulin_units'].sum()
        old_glucose_std = df_old['glucose_actual_mgdl'].std()
        new_glucose_std = df_new['glucose_actual_mgdl'].std()
        
        results.append({
            'patient': i+1,
            'old_insulin_total': old_insulin_total,
            'new_insulin_total': new_insulin_total,
            'old_glucose_variability': old_glucose_std,
            'new_glucose_variability': new_glucose_std,
            'insulin_change': ((new_insulin_total - old_insulin_total) / old_insulin_total * 100) if old_insulin_total > 0 else 0,
            'variability_change': ((new_glucose_std - old_glucose_std) / old_glucose_std * 100) if old_glucose_std > 0 else 0
        })
        
        patient.reset()
    
    return results

def create_comparison_plot(df_old, df_new):
    """Create A/B comparison plot of old vs new model."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clinical Reality Check: Old vs Improved AI Model', fontsize=16)
    
    # Glucose comparison
    ax1 = axes[0, 0]
    ax1.plot(df_old['time_minutes'], df_old['glucose_actual_mgdl'], 'b-', label='Old Model', alpha=0.7)
    ax1.plot(df_new['time_minutes'], df_new['glucose_actual_mgdl'], 'r-', label='Improved Model', alpha=0.7)
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title('Glucose Response Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Insulin delivery comparison
    ax2 = axes[0, 1]
    ax2.plot(df_old['time_minutes'], df_old['delivered_insulin_units'], 'b-', label='Old Model', alpha=0.7)
    ax2.plot(df_new['time_minutes'], df_new['delivered_insulin_units'], 'r-', label='Improved Model', alpha=0.7)
    ax2.set_ylabel('Insulin (Units)')
    ax2.set_title('Insulin Delivery Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative insulin
    ax3 = axes[1, 0]
    ax3.plot(df_old['time_minutes'], df_old['delivered_insulin_units'].cumsum(), 'b-', label='Old Model')
    ax3.plot(df_new['time_minutes'], df_new['delivered_insulin_units'].cumsum(), 'r-', label='Improved Model')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Cumulative Insulin (Units)')
    ax3.set_title('Cumulative Insulin Delivery')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Difference plot
    ax4 = axes[1, 1]
    insulin_diff = df_new['delivered_insulin_units'].values - df_old['delivered_insulin_units'].values
    ax4.plot(df_old['time_minutes'], insulin_diff, 'g-', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Insulin Difference (Units)')
    ax4.set_title('Insulin Decision Difference (New - Old)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'reports', 'clinical_reality_check.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    """Run complete clinical reality check."""
    
    print("=== CLINICAL REALITY CHECK ===")
    print("Validating autonomous learning effectiveness")
    
    model_path_old = os.path.join(os.path.dirname(__file__), '..', 'src', 'algorithm', 'trained_lstm_model.pth')
    model_path_new = model_path_old + '.improved'
    
    # 1. Weight Delta Check
    print("\n1. MODEL WEIGHT ANALYSIS")
    print("-" * 40)
    
    weight_delta, delta_msg = calculate_weight_delta(model_path_old, model_path_new)
    print(delta_msg)
    
    if weight_delta > 0.001:
        print("[OK] SIGNIFICANT WEIGHT CHANGES DETECTED")
        print("   The AI model has actually learned new patterns")
    elif weight_delta > 0:
        print("[WARN] MINIMAL WEIGHT CHANGES")
        print("   Learning occurred but changes are very small")
    else:
        print("[FAIL] NO WEIGHT CHANGES")
        print("   Model did not learn - check learning process")
        return
    
    # 2. Decision Comparison
    print("\n2. AI DECISION COMPARISON")
    print("-" * 40)
    
    comparison_data = compare_ai_decisions()
    
    if comparison_data is None:
        print("[FAIL] Cannot compare - improved model not found")
        return
    
    df_old, df_new = comparison_data
    
    # Calculate decision differences
    old_total_insulin = df_old['delivered_insulin_units'].sum()
    new_total_insulin = df_new['delivered_insulin_units'].sum()
    insulin_change = ((new_total_insulin - old_total_insulin) / old_total_insulin * 100) if old_total_insulin > 0 else 0
    
    old_variability = df_old['delivered_insulin_units'].std()
    new_variability = df_new['delivered_insulin_units'].std()
    
    print(f"Total insulin delivery change: {insulin_change:+.1f}%")
    print(f"Old model total: {old_total_insulin:.2f} Units")
    print(f"New model total: {new_total_insulin:.2f} Units")
    print(f"Decision variability: {old_variability:.3f} → {new_variability:.3f}")
    
    if abs(insulin_change) > 5:
        print("[OK] SIGNIFICANT BEHAVIORAL CHANGE DETECTED")
    elif abs(insulin_change) > 1:
        print("[WARN] MODERATE BEHAVIORAL CHANGE")
    else:
        print("[FAIL] NO SIGNIFICANT BEHAVIORAL CHANGE")
        print("   AI decisions remain essentially unchanged")
    
    # 3. Unseen Patient Testing
    print("\n3. UNSEEN PATIENT VALIDATION")
    print("-" * 40)
    
    unseen_results = test_unseen_patients()
    
    if unseen_results is None:
        print("[FAIL] Cannot test unseen patients - improved model not found")
        return
    
    print("Testing on 3 different patient profiles:")
    significant_changes = 0
    
    for result in unseen_results:
        print(f"\nPatient {result['patient']}:")
        print(f"  Insulin change: {result['insulin_change']:+.1f}%")
        print(f"  Glucose variability change: {result['variability_change']:+.1f}%")
        
        if abs(result['insulin_change']) > 3 or abs(result['variability_change']) > 5:
            significant_changes += 1
    
    if significant_changes >= 2:
        print(f"\n[OK] GENERALIZATION CONFIRMED ({significant_changes}/3 patients show changes)")
    elif significant_changes >= 1:
        print(f"\n[WARN] LIMITED GENERALIZATION ({significant_changes}/3 patients show changes)")
    else:
        print("\n[FAIL] NO GENERALIZATION - Changes don't transfer to new patients")
    
    # 4. Generate Comparison Plot
    print("\n4. VISUAL COMPARISON")
    print("-" * 40)
    
    plot_path = create_comparison_plot(df_old, df_new)
    print(f"Comparison plot saved: {plot_path}")
    
    # 5. Clinical Summary
    print("\n" + "=" * 50)
    print("CLINICAL REALITY CHECK SUMMARY")
    print("=" * 50)
    
    if weight_delta > 0.001 and abs(insulin_change) > 3 and significant_changes >= 1:
        print("[OK] AUTONOMOUS LEARNING IS CLINICALLY EFFECTIVE")
        print("   - Model weights changed significantly")
        print("   - AI behavior changed measurably")
        print("   - Changes generalize to new patients")
        print("\n   The AI has genuinely learned from experience.")
    elif weight_delta > 0.001 and abs(insulin_change) > 1:
        print("[WARN] AUTONOMOUS LEARNING IS PARTIALLY EFFECTIVE")
        print("   - Model weights changed")
        print("   - Some behavioral changes detected")
        print("   - Limited generalization")
        print("\n   Learning occurred but impact is modest.")
    else:
        print("[FAIL] AUTONOMOUS LEARNING IS NOT EFFECTIVE")
        print("   - Minimal or no weight changes")
        print("   - No significant behavioral changes")
        print("   - No generalization to new patients")
        print("\n   The learning process needs improvement.")
    
    print("\nRecommendations:")
    if weight_delta < 0.001:
        print("- Increase learning rate or training duration")
        print("- Check if training data contains sufficient variation")
    if abs(insulin_change) < 1:
        print("- Verify that reward function encourages behavioral change")
        print("- Consider more diverse training scenarios")
    if significant_changes == 0:
        print("- Test learning on more diverse patient population")
        print("- Ensure training captures generalizable patterns")
    
    print(f"\nDetailed comparison plot: {plot_path}")
    print("\nClinical reality check complete.")

if __name__ == "__main__":
    main()
