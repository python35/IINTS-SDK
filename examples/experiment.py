#!/usr/bin/env python3
"""
IINTS-AF Experiment Runner
Research-focused data collection and hypothesis testing framework
"""

import sys
import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from iints.core.patient.models import PatientModel
from iints.core.algorithms.correction_bolus import CorrectionBolus
from iints.core.algorithms.lstm_algorithm import LSTMInsulinAlgorithm
from iints.core.algorithms.hybrid_algorithm import HybridInsulinAlgorithm
from iints.core.simulator import Simulator, StressEvent
from iints.core.supervisor import IndependentSupervisor

class ExperimentRunner:
    """Research-focused experiment execution and data collection"""
    
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_experiment(self, config):
        """Create new experiment with unique ID and metadata"""
        exp_id = f"EXP-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir()
        
        # Save experiment configuration
        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        return exp_id, exp_dir
    
    def run_experiment(self, config):
        """Execute single experiment run with comprehensive logging"""
        exp_id, exp_dir = self.create_experiment(config)
        
        print(f"Experiment ID: {exp_id}")
        print(f"Hypothesis: {config.get('hypothesis', 'Not specified')}")
        print(f"Seed: {config['seed']}")
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        
        # Initialize components
        patient = PatientModel(initial_glucose=config.get('initial_glucose', 120))
        
        algorithms = {
            'rule_based': CorrectionBolus(),
            'lstm': LSTMInsulinAlgorithm(),
            'hybrid': HybridInsulinAlgorithm()
        }
        
        algorithm = algorithms[config['algorithm']]
        simulator = Simulator(patient, algorithm)
        supervisor = IndependentSupervisor() if config.get('enable_safety', True) else None
        
        # Add stress events
        for event_config in config.get('events', []):
            event = StressEvent(
                start_time=event_config['time'],
                event_type=event_config['type'],
                value=event_config['value']
            )
            simulator.add_stress_event(event)
        
        # Data collection structures
        decision_log = []
        safety_log = []
        internal_states = []
        
        # Run simulation with detailed logging
        print("Status:")
        duration = config.get('duration_minutes', 480)
        
        for step in range(0, duration + 1, 5):
            progress = step / duration
            bar_length = 50
            filled = int(bar_length * progress)
            bar = "" * filled + "" * (bar_length - filled)
            print(f"\r[{bar}] {progress:.1%}", end="", flush=True)
            
            if step >= duration:
                break
                
            # Get current state
            glucose = patient.get_current_glucose()
            
            # Algorithm decision
            insulin_output = algorithm.calculate_insulin(
                current_glucose=glucose,
                time_step=5,
                carb_intake=0  # Simplified for now
            )
            
            proposed_insulin = insulin_output.get("total_insulin_delivered", 0.0)
            
            # Safety supervision
            if supervisor:
                safety_result = supervisor.evaluate_safety(
                    current_glucose=glucose,
                    proposed_insulin=proposed_insulin,
                    current_time=step,
                    current_iob=patient.insulin_on_board
                )
                delivered_insulin = safety_result["approved_insulin"]
                safety_override = proposed_insulin != delivered_insulin
                safety_actions = safety_result["actions_taken"]
            else:
                delivered_insulin = proposed_insulin
                safety_override = False
                safety_actions = []
            
            # Log decision data
            decision_entry = {
                "t": step,
                "glucose": float(glucose),
                "delta_glucose": float(glucose - (decision_log[-1]["glucose"] if decision_log else glucose)),
                "iob": float(patient.insulin_on_board),
                "algorithm_used": config['algorithm'],
                "confidence": float(getattr(algorithm, 'last_confidence', 1.0) if hasattr(algorithm, 'last_confidence') else 1.0),
                "insulin_proposed": float(proposed_insulin),
                "insulin_delivered": float(delivered_insulin),
                "safety_override": bool(safety_override),
                "override_reason": safety_actions[0] if safety_actions else None
            }
            decision_log.append(decision_entry)
            
            # Log safety events
            if safety_override:
                safety_entry = {
                    "t": step,
                    "glucose": glucose,
                    "proposed": proposed_insulin,
                    "delivered": delivered_insulin,
                    "reason": safety_actions[0] if safety_actions else "unknown",
                    "severity": "override"
                }
                safety_log.append(safety_entry)
            
            # Log internal AI states (if available)
            if hasattr(algorithm, 'get_internal_state'):
                internal_state = algorithm.get_internal_state()
                internal_state['t'] = step
                internal_states.append(internal_state)
            
            # Update patient
            patient.update(5, delivered_insulin, 0)
        
        print("\n")
        
        # Save all data
        self._save_experiment_data(exp_dir, {
            'decision_log': decision_log,
            'safety_log': safety_log,
            'internal_states': internal_states,
            'config': config,
            'metadata': {
                'exp_id': exp_id,
                'timestamp': datetime.now().isoformat(),
                'duration_actual': len(decision_log) * 5,
                'total_decisions': len(decision_log),
                'safety_overrides': len(safety_log)
            }
        })
        
        print(f"\nArtifacts generated:")
        print(f"- decision_log.jsonl")
        print(f"- safety_log.jsonl") 
        print(f"- metadata.json")
        print(f"- raw_timeseries.csv")
        
        return exp_id, exp_dir
    
    def _save_experiment_data(self, exp_dir, data):
        """Save experiment data in research-friendly formats"""
        
        # Decision log as JSONL for streaming analysis
        with open(exp_dir / "decision_log.jsonl", "w") as f:
            for entry in data['decision_log']:
                f.write(json.dumps(entry) + "\n")
        
        # Safety log
        with open(exp_dir / "safety_log.jsonl", "w") as f:
            for entry in data['safety_log']:
                f.write(json.dumps(entry) + "\n")
        
        # Metadata
        with open(exp_dir / "metadata.json", "w") as f:
            json.dump(data['metadata'], f, indent=2)
        
        # CSV for easy analysis
        import pandas as pd
        df = pd.DataFrame(data['decision_log'])
        df.to_csv(exp_dir / "raw_timeseries.csv", index=False)
        
        # Internal states if available
        if data['internal_states']:
            with open(exp_dir / "internal_states.jsonl", "w") as f:
                for entry in data['internal_states']:
                    f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="IINTS-AF Experiment Runner")
    parser.add_argument("--scenario", required=True, help="Experiment scenario")
    parser.add_argument("--algorithm", required=True, choices=['rule_based', 'lstm', 'hybrid'])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=int, default=480, help="Duration in minutes")
    parser.add_argument("--hypothesis", help="Research hypothesis")
    parser.add_argument("--noise", choices=['none', 'sensor', 'meal'], default='none')
    parser.add_argument("--initial-glucose", type=float, default=120)
    
    args = parser.parse_args()
    
    # Define scenario configurations
    scenarios = {
        "standard_meal": {
            "events": [{"time": 60, "type": "meal", "value": 60}],
            "description": "Standard 60g carb meal at 1 hour"
        },
        "unannounced_meal": {
            "events": [{"time": 60, "type": "missed_meal", "value": 60}],
            "description": "Unannounced 60g carb meal"
        },
        "hyperglycemia": {
            "events": [],
            "initial_glucose": 250,
            "description": "Hyperglycemia correction from 250 mg/dL"
        },
        "dawn_phenomenon": {
            "events": [{"time": 360, "type": "dawn", "value": 30}],
            "description": "Dawn phenomenon glucose rise"
        }
    }
    
    if args.scenario not in scenarios:
        print(f"Unknown scenario: {args.scenario}")
        print(f"Available: {list(scenarios.keys())}")
        return
    
    scenario_config = scenarios[args.scenario]
    
    # Build experiment configuration
    config = {
        "scenario": args.scenario,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "duration_minutes": args.duration,
        "initial_glucose": args.initial_glucose,
        "events": scenario_config["events"],
        "enable_safety": True,
        "noise_type": args.noise,
        "hypothesis": args.hypothesis or f"Investigating {args.algorithm} behavior in {args.scenario} scenario",
        "description": scenario_config["description"]
    }
    
    # Run experiment
    runner = ExperimentRunner()
    exp_id, exp_dir = runner.run_experiment(config)
    
    print(f"\nExperiment complete: {exp_id}")
    print(f"Data saved to: {exp_dir}")

if __name__ == "__main__":
    main()
