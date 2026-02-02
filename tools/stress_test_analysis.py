#!/usr/bin/env python3
import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Dict

# Ensure src is in path so we can import the SDK modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from iints.core.simulator import Simulator
from iints.api.template_algorithm import TemplateAlgorithm
from iints.api.base_algorithm import AlgorithmInput, AlgorithmMetadata

# --- 1. Mock Components for Stress Testing ---

class MockPatientModel:
    """
    A lightweight patient model for high-speed stress testing.
    Simulates basic glucose dynamics without heavy dependencies.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.insulin_on_board = 2.0 # Start with some IOB
        self.carbs_on_board = 0.0
        self.current_glucose = 180.0 # Start high to provoke insulin response
        self.time = 0

    def get_current_glucose(self) -> float:
        return self.current_glucose

    def get_patient_state(self) -> Dict:
        return {}

    def update(self, time_step, delivered_insulin, carb_intake):
        self.time += time_step
        
        # Simple decay
        self.insulin_on_board = max(0, self.insulin_on_board * 0.95 + delivered_insulin)
        
        # Random Walk / Noise to simulate sensor noise
        noise = np.random.normal(0, 2)
        
        # Dynamics: Insulin drops glucose
        # We keep glucose high to force the algorithm to keep dosing, 
        # eventually hitting the IOB limit of the Supervisor.
        glucose_change = - (self.insulin_on_board * 1.5) + 5 + noise 
        
        self.current_glucose += glucose_change
        # Clamp to realistic values
        self.current_glucose = max(40, min(400, self.current_glucose))

    def start_exercise(self, intensity): pass
    def stop_exercise(self): pass


class AggressiveTestAlgorithm(TemplateAlgorithm):
    """
    An 'unsafe' algorithm that ignores IOB, designed to trigger the Safety Supervisor.
    """
    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Aggressive Stress Test Algo",
            version="0.0.1",
            author="StressTest",
            description="Intentionally aggressive to trigger safety limits.",
            algorithm_type="rule_based"
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, float]:
        # Always try to correct to 100 mg/dL, ignoring IOB
        target = 100
        isf = 30 # Aggressive ISF
        
        dose = 0.0
        if data.current_glucose > target:
            dose = (data.current_glucose - target) / isf
            
        # We intentionally DO NOT subtract IOB here, 
        # relying on the SafetySupervisor to catch it.
        
        return {
            'total_insulin_delivered': dose,
            'basal_insulin': 0.0,
            'meal_bolus': 0.0,
            'correction_bolus': dose
        }

# --- 2. Analysis Script ---

def run_stress_test():
    print("\n" + "="*60)
    print("   IINTS-AF SDK: HIGH-SPEED SAFETY STRESS TEST")
    print("="*60 + "\n")
    
    # Setup
    patient = MockPatientModel()
    algorithm = AggressiveTestAlgorithm()
    
    # Initialize Simulator
    # Note: SafetySupervisor is initialized inside Simulator
    simulator = Simulator(
        patient_model=patient,
        algorithm=algorithm,
        time_step=5,
        seed=1337
    )
    
    duration_days = 7
    duration_minutes = duration_days * 24 * 60
    total_steps = duration_minutes // 5
    
    print(f"[*] Configuration:")
    print(f"    - Duration: {duration_days} days ({total_steps} simulation steps)")
    print(f"    - Patient:  MockPatientModel (High Glucose Scenario)")
    print(f"    - Algo:     AggressiveTestAlgorithm (No internal safety)")
    print(f"    - Hardware: Emulated Edge Environment")
    
    print(f"\n[*] Starting Simulation...")
    start_time = time.perf_counter()
    
    # Run Batch
    results_df, audit_log = simulator.run_batch(duration_minutes)
    
    end_time = time.perf_counter()
    total_time_sec = end_time - start_time
    steps_per_sec = total_steps / total_time_sec
    
    print(f"[*] Simulation Complete.")
    print(f"    - Real-time elapsed: {total_time_sec:.4f} seconds")
    print(f"    - Speed: {steps_per_sec:.0f} steps/sec")
    
    # --- Analysis ---
    
    print("\n" + "-"*30)
    print("   RESULTS ANALYSIS")
    print("-"*30)
    
    # 1. Latency Analysis
    latencies = results_df['supervisor_latency_ms']
    avg_latency = latencies.mean()
    max_latency = latencies.max()
    p99_latency = latencies.quantile(0.99)
    
    print(f"\n[1] LATENCY (Real-time Capability)")
    print(f"    - Average Safety Check: {avg_latency:.4f} ms")
    print(f"    - 99th Percentile:      {p99_latency:.4f} ms")
    print(f"    - Max Latency:          {max_latency:.4f} ms")
    
    if avg_latency < 1.0:
        print("    >>> STATUS: PASS (Suitable for <1ms Edge Control Loops)")
    else:
        print("    >>> STATUS: WARNING (High Latency)")

    # 2. Safety Interventions
    num_interventions = len(audit_log)
    intervention_rate = (num_interventions / total_steps) * 100
    
    print(f"\n[2] SAFETY AUDIT (The 'Safety Guard')")
    print(f"    - Total Steps:          {total_steps}")
    print(f"    - Interventions:        {num_interventions}")
    print(f"    - Intervention Rate:    {intervention_rate:.2f}%")
    
    if num_interventions > 0:
        print("\n    [!] Intervention Reasons Breakdown:")
        reasons = [entry['reason'] for entry in audit_log]
        from collections import Counter
        for reason, count in Counter(reasons).items():
            print(f"        - {reason}: {count} times")
            
        print("\n    [!] Sample Audit Log Entries (Last 3):")
        print(f"        {'Timestamp':<10} | {'AI Proposed':<12} | {'Final Dose':<12} | {'Reason'}")
        print("        " + "-"*60)
        for entry in audit_log[-3:]:
            print(f"        {entry['timestamp']:<10.0f} | {entry['ai_proposed']:<12.2f} | {entry['final_dose']:<12.2f} | {entry['reason']}")
    
    print("\n" + "="*60)
    print("   END OF REPORT")
    print("="*60 + "\n")

    # --- Export Data ---
    print("[*] Exporting data for analysis...")
    
    results_dir = "resultaten"
    os.makedirs(results_dir, exist_ok=True)
    print(f"    - Saving results to '{results_dir}/' directory.")

    # 1. Save Audit Log (The "Evidence")
    if audit_log:
        audit_df = pd.DataFrame(audit_log)
        audit_csv = os.path.join(results_dir, "stress_test_audit.csv")
        audit_df.to_csv(audit_csv, index=False)
        print(f"    - Audit Log saved to: {audit_csv}")
    
    # 2. Save Full Simulation Data (For graphs)
    results_csv = os.path.join(results_dir, "stress_test_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"    - Full Simulation Data saved to: {results_csv}")

if __name__ == "__main__":
    run_stress_test()