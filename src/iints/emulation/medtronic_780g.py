#!/usr/bin/env python3
"""
Medtronic 780G Emulator - IINTS-AF
Emulates the Medtronic MiniMed 780G with SmartGuard algorithm.

Based on:
- FDA 510(k) clearance documentation
- Clinical studies (Bergenstal et al.)
- User manual and technical specifications

Part of the #WeAreNotWaiting movement for transparent diabetes tech.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .legacy_base import (
    LegacyEmulator, PumpBehavior, PIDParameters, SafetyLimits,
    SafetyLevel, EmulatorDecision
)


@dataclass
class Medtronic780GBehavior(PumpBehavior):
    """Medtronic 780G specific behavior profile"""
    # Additional 780G-specific parameters
    auto_basal_enabled: bool = True
    target_range_low: float = 100.0  # mg/dL (configurable 100-120)
    target_range_high: float = 120.0  # mg/dL
    max_auto_basal: float = 6.0  # units/hour
    predictive_low_suspend: bool = True
    plsg_window: int = 30  # minutes
    plsg_threshold: float = 70.0  # mg/dL
    time_step_minutes: float = 5.0 # Explicitly define the decision interval
    
    def __init__(self):
        super().__init__(
            pump_name="Medtronic MiniMed 780G",
            manufacturer="Medtronic",
            safety_limits=SafetyLimits(
                low_suspend_threshold=70.0,
                high_suspend_threshold=250.0,
                max_bolus=10.0,
                max_basal_rate=6.0,
                max_daily_total=80.0,
                auto_off_duration=60.0,
                low_suspend_duration=5.0,
                target_glucose=120.0
            ),
            pid_parameters=PIDParameters(
                kp=0.02,  # Proportional gain (tuned for 780G)
                ki=0.005,  # Integral gain
                kd=0.03,  # Derivative gain
                target_glucose=120.0,
                integral_limit=5.0
            ),
            correction_factor=50.0,
            carb_ratio=10.0,
            insulin_sensitivity_factor=50.0,
            active_insulin_duration=4.0,
            safety_level=SafetyLevel.CONSERVATIVE
        )


class Medtronic780GEmulator(LegacyEmulator):
    """
    Emulates Medtronic 780G SmartGuard algorithm.
    
    Based on Medtronic 780G Clinical User Guide, Section 4.2 (SmartGuard).
    
    The 780G uses a hybrid closed-loop system with:
    - Automatic basal delivery (auto-basal)
    - Automatic correction boluses
    - Predictive Low Glucose Suspend (PLGS)
    - Target glucose range (configurable 100-120 mg/dL)
    
    Key characteristics:
    - Conservative approach to avoid hypoglycemia
    - Frequent insulin delivery (every 5 minutes)
    - Automatic correction for elevated glucose
    - Meal detection triggers additional insulin
    
    Sources:
    - Bergenstal et al. "New Closed-Loop Insulin Delivery System"
    - FDA 510(k) K193510
    - Medtronic 780G User Guide
    """
    
    def __init__(self):
        """Initialize the 780G emulator"""
        super().__init__()
        self.behavior = Medtronic780GBehavior()
        
    def _get_default_behavior(self) -> Medtronic780GBehavior:
        """Get 780G default behavior"""
        return Medtronic780GBehavior()
    
    def get_sources(self) -> List[Dict[str, str]]:
        """Get sources for 780G emulation logic"""
        return [
            {
                'title': 'Bergenstal et al. - Hybrid Closed-Loop Therapy',
                'type': 'clinical_study',
                'year': '2020',
                'url': 'https://doi.org/10.1056/NEJMoa2003479'
            },
            {
                'title': 'FDA 510(k) K193510 - MiniMed 780G System',
                'type': 'regulatory',
                'year': '2020',
                'url': 'https://www.accessdata.fda.gov/'
            },
            {
                'title': 'Medtronic 780G User Guide',
                'type': 'technical_manual',
                'year': '2020',
                'url': 'https://www.medtronicdiabetes.com/'
            }
        ]
    
    def emulate_decision(self,
                         glucose: float,
                         velocity: float,
                         insulin_on_board: float,
                         carbs: float,
                         current_time: float = 0) -> EmulatorDecision:
        """
        Emulate 780G decision-making for current conditions.
        
        The 780G makes decisions every 5 minutes:
        1. Check for automatic correction bolus (micro-bolus)
        2. Check for auto-basal adjustment
        3. Check for PLGS (Predictive Low Glucose Suspend)
        4. Apply safety constraints
        
        Args:
            glucose: Current glucose (mg/dL)
            velocity: Rate of change (mg/dL/min)
            insulin_on_board: Current IOB (units)
            carbs: Carbs consumed (grams)
            current_time: Simulation time (minutes)
            
        Returns:
            EmulatorDecision with insulin delivery details
        """
        reasoning = []
        safety_overrides = []
        action = 'deliver'
        insulin_delivered = 0.0
        pid = self.behavior.pid_parameters
        
        # --- Update internal state ---
        time_step_minutes = self.behavior.time_step_minutes 

        if self.state['previous_glucose'] is not None:
            # Re-calculate velocity based on actual time step
            self.state['velocity'] = (glucose - self.state['previous_glucose']) / time_step_minutes
        else:
            self.state['velocity'] = velocity # Use provided velocity if no previous data
        self.state['previous_glucose'] = glucose
        
        # --- 1. Initial Safety Check & PLGS (Predictive Low Glucose Suspend) ---
        should_suspend, suspend_reasons, safety_adjustment_factor = self._check_safety_constraints(
            glucose, self.state['velocity'], insulin_on_board, carbs, current_time
        )
        reasoning.extend(suspend_reasons)
        
        if self.behavior.predictive_low_suspend and self.state['velocity'] < 0: # Only suspend if glucose is falling
            predicted_glucose = glucose + self.state['velocity'] * self.behavior.plsg_window
            if predicted_glucose < self.behavior.plsg_threshold:
                action = 'suspend_insulin'
                insulin_delivered = 0.0
                safety_overrides.append(
                    f"PLGS activated: predicted glucose {predicted_glucose:.0f} mg/dL in "
                    f"{self.behavior.plsg_window} min. Insulin suspended."
                )
                reasoning.append(safety_overrides[-1])
                
        if action != 'suspend_insulin': # If not suspended by PLGS or initial safety check

            # --- 2. Calculate PID correction (Auto-Correction Bolus) ---
            error = glucose - pid.target_glucose
            
            # Update integral term (cumulative error over time)
            self.state['integral_term'] += error * time_step_minutes 
            self.state['integral_term'] = max(-pid.integral_limit, 
                                              min(pid.integral_limit, 
                                                  self.state['integral_term']))
            
            # Derivative term is based on glucose velocity (trend)
            derivative_term = self.state['velocity']
            
            # Calculate total PID output (insulin units per time_step)
            pid_insulin_correction = (
                pid.kp * error +
                pid.ki * self.state['integral_term'] +
                pid.kd * derivative_term
            ) / self.behavior.correction_factor
            
            # --- 3. Add Meal Bolus ---
            meal_bolus = 0.0
            if carbs > 0:
                meal_bolus = carbs / self.behavior.carb_ratio
                reasoning.append(
                    f"Meal detected: {carbs:.0f}g -> {meal_bolus:.2f} U meal bolus"
                )
            
            # --- 4. Combine and apply internal limits ---
            total_calculated_insulin = max(0, pid_insulin_correction + meal_bolus)
            
            # Apply max bolus limit
            total_calculated_insulin = min(total_calculated_insulin, self.behavior.safety_limits.max_bolus)
            
            # Apply individual safety adjustment factor from _check_safety_constraints
            insulin_delivered = total_calculated_insulin * safety_adjustment_factor
            
            # The 780G delivers frequent small automatic correction boluses.
            # This is primarily the pid_insulin_correction.
            if insulin_delivered > 0:
                if meal_bolus > 0:
                    reasoning.append(f"Insulin delivered for meal and auto-correction: {insulin_delivered:.2f} U")
                elif pid_insulin_correction > 0:
                    reasoning.append(f"Auto-correction micro-bolus: {insulin_delivered:.2f} U (PID output based on glucose error and trend).")
            else:
                reasoning.append("No insulin delivered. Glucose at or near target, or PLGS active.")

        # Create decision
        decision = EmulatorDecision(
            insulin_delivered=insulin_delivered,
            action=action,
            reasoning=reasoning,
            safety_overrides=safety_overrides,
            predicted_glucose=glucose + self.state['velocity'] * 30,  # 30-min prediction (simplified)
            confidence=0.90 # Higher confidence for a well-established pump
        )
        
        self._decision_history.append(decision)
        
        return decision
    
    def get_algorithm_personality(self) -> Dict:
        """
        Get the algorithm's "personality" characteristics.
        
        Returns:
            Dictionary with personality traits
        """
        return {
            'name': 'Medtronic 780G SmartGuard',
            'type': 'Hybrid Closed-Loop',
            'personality': {
                'aggressiveness': 'Conservative',
                'hypo_aversion': 'High',
                'response_speed': 'Moderate',
                'correction_aggressiveness': 'Moderate',
                'meal_handling': 'Automatic correction bolus',
                'predictive_features': 'PLGS, meal detection'
            },
            'key_differences': [
                'Target glucose 100-120 mg/dL (configurable)',
                'Automatic correction boluses every 5 minutes',
                'Conservative approach to avoid hypoglycemia',
                'Strong predictive low glucose suspend'
            ],
            'limitations': [
                'Requires meal announcements for best results',
                'Conservative tuning may lead to higher glucose',
                'No exercise mode integration'
            ]
        }


def demo_medtronic_780g():
    """Demonstrate Medtronic 780G emulator"""
    print("=" * 70)
    print("MEDTRONIC 780G EMULATOR DEMONSTRATION")
    print("=" * 70)
    
    emulator = Medtronic780GEmulator()
    
    # Print behavior profile
    print("\nAlgorithm Personality:")
    personality = emulator.get_algorithm_personality()
    print(f"  Name: {personality['name']}")
    print(f"  Type: {personality['type']}")
    print(f"  Aggressiveness: {personality['personality']['aggressiveness']}")
    print(f"  Hypo Aversion: {personality['personality']['hypo_aversion']}")
    
    # Print sources
    print("\nSources:")
    for source in emulator.get_sources():
        print(f"  - [{source['type']}] {source['title']} ({source['year']})")
    
    # Simulate scenarios
    print("\nScenario Simulation:")
    print("-" * 50)
    
    scenarios = [
        {'glucose': 180, 'velocity': 1.0, 'iob': 1.0, 'carbs': 0, 'desc': 'High glucose rising'},
        {'glucose': 120, 'velocity': 0.5, 'iob': 2.0, 'carbs': 0, 'desc': 'At target, moderate IOB'},
        {'glucose': 70, 'velocity': -1.5, 'iob': 1.0, 'carbs': 0, 'desc': 'Falling toward low'},
        {'glucose': 250, 'velocity': 2.0, 'iob': 0.5, 'carbs': 30, 'desc': 'High with meal'},
        {'glucose': 60, 'velocity': -2.5, 'iob': 0.5, 'carbs': 0, 'desc': 'Low with rapid fall'},
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        decision = emulator.emulate_decision(
            glucose=scenario['glucose'],
            velocity=scenario['velocity'],
            insulin_on_board=scenario['iob'],
            carbs=scenario['carbs'],
            current_time=i * 5
        )
        
        print(f"\n  Scenario {i}: {scenario['desc']}")
        print(f"    Glucose: {scenario['glucose']} mg/dL, Velocity: {scenario['velocity']} mg/dL/min")
        print(f"    IOB: {scenario['iob']} U, Carbs: {scenario['carbs']}g")
        print(f"    -> Action: {decision.action}, Insulin: {decision.insulin_delivered:.2f} U")
        print(f"    Reasoning: {', '.join(decision.reasoning[:2])}")
        
        if decision.safety_overrides:
            print(f"    [WARN] Safety: {', '.join(decision.safety_overrides)}")
    
    # Export behavior report
    print("\nBehavior Report:")
    report = emulator.export_behavior_report()
    print(f"  Pump: {report['pump_name']}")
    print(f"  Decisions: {report['decision_count']}")
    
    print("\n" + "=" * 70)
    print("MEDTRONIC 780G EMULATOR DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_medtronic_780g()