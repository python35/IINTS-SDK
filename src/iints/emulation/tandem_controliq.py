#!/usr/bin/env python3
"""
Tandem Control-IQ Emulator - IINTS-AF
Emulates the Tandem t:slim X2 with Control-IQ algorithm.

Based on:
- Brown et al. (2019) Diabetes Technology & Therapeutics
- FDA 510(k) clearance documentation
- Clinical trial results (iDCL Trial)

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
class TandemControlIQBehavior(PumpBehavior):
    """Tandem Control-IQ specific behavior profile"""
    # Control-IQ specific parameters
    exercise_mode: bool = False
    target_glucose_day: float = 112.5  # mg/dL
    target_glucose_exercise: float = 140.0  # mg/dL (when exercise mode on)
    max_delivery: float = 3.0  # units/hour maximum auto delivery
    correction_divisor: float = 30.0  # mg/dL per unit for corrections
    basal_limit: float = 2.0  # units/hour max basal override
    prediction_horizon: int = 30  # minutes
    prediction_window: int = 60  # minutes for activity
    low_limit: float = 70.0  # mg/dL low glucose limit
    high_limit: float = 180.0  # mg/dL high glucose limit
    time_step_minutes: float = 5.0 # Explicitly define the decision interval
    
    def __init__(self):
        super().__init__(
            pump_name="Tandem t:slim X2 Control-IQ",
            manufacturer="Tandem Diabetes Care",
            safety_limits=SafetyLimits(
                low_suspend_threshold=70.0,
                high_suspend_threshold=250.0,
                max_bolus=15.0,
                max_basal_rate=6.0,
                max_daily_total=100.0,
                auto_off_duration=60.0,
                low_suspend_duration=5.0,
                target_glucose=112.5
            ),
            pid_parameters=PIDParameters(
                kp=0.03,  # Proportional gain (Control-IQ tuned)
                ki=0.002,  # Integral gain (lower = less accumulation)
                kd=0.04,  # Derivative gain
                target_glucose=112.5,
                integral_limit=3.0
            ),
            correction_factor=30.0,
            carb_ratio=10.0,
            insulin_sensitivity_factor=45.0,
            active_insulin_duration=6.0,  # Control-IQ uses 6-hour DIA
            safety_level=SafetyLevel.MODERATE
        )


class TandemControlIQEmulator(LegacyEmulator):
    """
    Emulates Tandem Control-IQ algorithm.
    
    Control-IQ is a hybrid closed-loop system that features:
    - Predictive Low Glucose Suspend (PLGS)
    - Predictive High Glucose Assist
    - Exercise Mode (higher target)
    - Automatic correction boluses
    - Basal rate override
    
    Key characteristics:
    - More aggressive than Medtronic 780G
    - Uses 6-hour insulin duration (vs 4-hour typical)
    - Targets 112.5 mg/dL (lower than 780G's 120)
    - Automatic correction every hour
    
    Sources:
    - Brown et al. (2019) "Control-IQ Technology"
    - FDA 510(k) K191289
    - iDCL Trial Results
    """
    
    def __init__(self):
        """Initialize the Control-IQ emulator"""
        super().__init__()
        self.behavior = TandemControlIQBehavior()
        
    def _get_default_behavior(self) -> TandemControlIQBehavior:
        """Get Control-IQ default behavior"""
        return TandemControlIQBehavior()
    
    def get_sources(self) -> List[Dict[str, str]]:
        """Get sources for Control-IQ emulation logic"""
        return [
            {
                'title': 'Brown et al. - Control-IQ Technology',
                'type': 'clinical_study',
                'year': '2019',
                'url': 'https://doi.org/10.1089/dia.2019.0226'
            },
            {
                'title': 'FDA 510(k) K191289 - Control-IQ System',
                'type': 'regulatory',
                'year': '2019',
                'url': 'https://www.accessdata.fda.gov/'
            },
            {
                'title': 'iDCL Trial - International Diabetes Closed Loop',
                'type': 'clinical_trial',
                'year': '2019-2020',
                'url': 'https://clinicaltrials.gov/ct2/show/NCT03563313'
            },
            {
                'title': 'Control-IQ User Guide',
                'type': 'technical_manual',
                'year': '2020',
                'url': 'https://www.tandemdiabetes.com/'
            }
        ]
    
    def set_exercise_mode(self, enabled: bool):
        """Enable/disable exercise mode (higher target)"""
        self.behavior.exercise_mode = enabled
        self.behavior.target_glucose_day = (
            140.0 if enabled else 112.5
        )
        if self.behavior.pid_parameters:
            self.behavior.pid_parameters.target_glucose = (
                self.behavior.target_glucose_day
            )
        print(f"Exercise mode: {'ON' if enabled else 'OFF'}")
    
    def emulate_decision(self,
                         glucose: float,
                         velocity: float,
                         insulin_on_board: float,
                         carbs: float,
                         current_time: float = 0) -> EmulatorDecision:
        """
        Emulate Control-IQ decision-making for current conditions.
        
        Control-IQ makes decisions:
        1. Predict glucose 30-60 minutes ahead
        2. Adjust delivery based on prediction
        3. Apply PLGS if predicted < 70 mg/dL
        4. Apply PHGS (predictive high) if predicted > 180 mg/dL
        5. Automatic corrections every hour (or every 5 mins in our sim for continuous control)
        
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
        safety = self.behavior.safety_limits
        
        # --- Update internal state ---
        time_step_minutes = self.behavior.time_step_minutes
        if self.state['previous_glucose'] is not None:
            self.state['velocity'] = (glucose - self.state['previous_glucose']) / time_step_minutes
        else:
            self.state['velocity'] = velocity 
        self.state['previous_glucose'] = glucose

        # Get target based on exercise mode
        target = (
            self.behavior.target_glucose_exercise 
            if self.behavior.exercise_mode 
            else self.behavior.target_glucose_day
        )
        
        # Predict glucose
        predicted_glucose_30min = glucose + self.state['velocity'] * 30
        predicted_glucose_60min = glucose + self.state['velocity'] * 60
        
        # Check safety constraints first (from LegacyEmulator base class)
        should_suspend_base, suspend_reasons_base, safety_adjustment_factor_base = self._check_safety_constraints(
            glucose, self.state['velocity'], insulin_on_board, carbs, current_time
        )
        reasoning.extend(suspend_reasons_base)
        
        # --- Tandem Control-IQ Specific Logic ---

        # 1. PLGS: Predictive Low Glucose Suspend
        if predicted_glucose_30min < self.behavior.low_limit and self.state['velocity'] < 0: # Only suspend if falling
            action = 'suspend_insulin'
            insulin_delivered = 0.0
            safety_overrides.append(
                f"PLGS activated: predicted {predicted_glucose_30min:.0f} mg/dL in 30 min. Insulin suspended."
            )
            reasoning.append(safety_overrides[-1])
        
        # If not suspended, proceed with insulin calculation
        if action != 'suspend_insulin' and not should_suspend_base: # Ensure not suspended by base safety either

            # --- Range-based Target Adjustment ---
            # Control-IQ dynamically adjusts its "target" based on current and predicted glucose.
            # For simplicity in emulation, we'll primarily use the fixed target (112.5 or 140 for exercise), 
            # but then layer on range-specific corrections.
            
            current_target = target # Base target
            
            # --- Auto-Correction Bolus (PID based) ---
            error = glucose - current_target # Error relative to current target

            # Update integral term (cumulative error over time)
            self.state['integral_term'] += error * time_step_minutes 
            self.state['integral_term'] = max(-pid.integral_limit, 
                                              min(pid.integral_limit, 
                                                  self.state['integral_term']))
            
            derivative_term = self.state['velocity']
            
            pid_insulin_correction = (
                pid.kp * error + 
                pid.ki * self.state['integral_term'] + 
                pid.kd * derivative_term
            ) / self.behavior.correction_factor
            
            # --- Meal Bolus ---
            meal_bolus = 0.0
            if carbs > 0:
                meal_bolus = carbs / self.behavior.carb_ratio
                reasoning.append(
                    f"Meal bolus: {carbs:.0f}g -> {meal_bolus:.2f} U"
                )
            
            # --- Apply Predictive High Glucose Assist (PHGS) ---
            phgs_multiplier = 1.0
            if predicted_glucose_60min > self.behavior.high_limit and predicted_glucose_60min > glucose: # Only if predicted high AND rising
                # If predicted to be high and rising, be more aggressive
                phgs_multiplier = 1.5 # Increase correction aggressiveness
                reasoning.append(
                    f"PHGS activated: predicted {predicted_glucose_60min:.0f} mg/dL in 60 min. Increasing correction aggressiveness."
                )
            
            # --- Combine and apply internal limits ---
            total_calculated_insulin = max(0, (pid_insulin_correction + meal_bolus) * phgs_multiplier)
            
            # Apply individual safety adjustment factor from _check_safety_constraints (base class)
            insulin_delivered = total_calculated_insulin * safety_adjustment_factor_base
            
            # Control-IQ has max delivery limit for automated insulin
            insulin_delivered = min(insulin_delivered, self.behavior.max_delivery)
            
            if insulin_delivered > 0:
                if meal_bolus > 0:
                    reasoning.append(f"Insulin delivered for meal and auto-correction: {insulin_delivered:.2f} U")
                elif pid_insulin_correction > 0:
                    reasoning.append(f"Auto-correction bolus: {insulin_delivered:.2f} U (PID output based on glucose error and trend).")
            else:
                reasoning.append("No insulin delivered. Glucose at or near target, or PLGS active.")
        
        # Check cumulative insulin limit (from LegacyEmulator base class)
        self.state['cumulative_insulin'] += insulin_delivered # This should be cumulative sum of delivered_insulin
        # The base class _check_safety_constraints should handle max_daily_total
        
        # Create decision
        decision = EmulatorDecision(
            insulin_delivered=insulin_delivered,
            action=action,
            reasoning=reasoning,
            safety_overrides=safety_overrides,
            predicted_glucose=predicted_glucose_30min,
            confidence=0.88
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
            'name': 'Tandem Control-IQ',
            'type': 'Hybrid Closed-Loop',
            'personality': {
                'aggressiveness': 'Moderate-Aggressive',
                'hypo_aversion': 'Moderate',
                'response_speed': 'Fast',
                'correction_aggressiveness': 'Aggressive',
                'meal_handling': 'Automatic corrections',
                'predictive_features': 'PLGS, PHGS, Exercise Mode'
            },
            'key_differences': [
                'Lower target glucose (112.5 mg/dL)',
                '6-hour insulin duration (vs 4-hour typical)',
                'Predictive High Glucose Assist (PHGS)',
                'Exercise mode with 140 mg/dL target',
                'More aggressive corrections than 780G'
            ],
            'limitations': [
                'No auto-bolus feature',
                'Less conservative than 780G',
                'Exercise mode requires manual activation'
            ]
        }


def demo_tandem_controliq():
    """Demonstrate Tandem Control-IQ emulator"""
    print("=" * 70)
    print("TANDEM CONTROL-IQ EMULATOR DEMONSTRATION")
    print("=" * 70)
    
    emulator = TandemControlIQEmulator()
    
    # Print behavior profile
    print("\nAlgorithm Personality:")
    personality = emulator.get_algorithm_personality()
    print(f"  Name: {personality['name']}")
    print(f"  Type: {personality['type']}")
    print(f"  Aggressiveness: {personality['personality']['aggressiveness']}")
    print(f"  Response Speed: {personality['personality']['response_speed']}")
    
    # Print sources
    print("\nSources:")
    for source in emulator.get_sources():
        print(f"  - [{source['type']}] {source['title']} ({source['year']})")
    
    # Simulate scenarios
    print("\nScenario Simulation:")
    print("-" * 50)
    
    scenarios = [
        {'glucose': 180, 'velocity': 1.5, 'iob': 1.0, 'carbs': 0, 'desc': 'High glucose rising fast'},
        {'glucose': 112, 'velocity': 0.3, 'iob': 1.5, 'carbs': 0, 'desc': 'At target with IOB'},
        {'glucose': 70, 'velocity': -1.0, 'iob': 1.0, 'carbs': 0, 'desc': 'At low limit, stable'},
        {'glucose': 200, 'velocity': 2.5, 'iob': 0.5, 'carbs': 40, 'desc': 'High with meal'},
        {'glucose': 75, 'velocity': -2.0, 'iob': 0.5, 'carbs': 0, 'desc': 'Predicted low (PLGS)'},
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
        print(f"    -> Action: {decision.action}, Insulin: {decision.insulin_delivered:.2f} U")
        print(f"    Reasoning: {', '.join(decision.reasoning[:2])}")
        
        if decision.predicted_glucose:
            print(f"    Predicted (30min): {decision.predicted_glucose:.0f} mg/dL")
    
    # Test exercise mode
    print("\nExercise Mode Test:")
    print("-" * 50)
    emulator.set_exercise_mode(True)
    
    decision = emulator.emulate_decision(
        glucose=150, velocity=0.5, insulin_on_board=1.0, carbs=0, current_time=0
    )
    print(f"  Exercise mode: glucose=150 -> {decision.insulin_delivered:.2f} U")
    print(f"  Reasoning: {', '.join(decision.reasoning)}")
    
    emulator.set_exercise_mode(False)
    
    print("\n" + "=" * 70)
    print("TANDEM CONTROL-IQ EMULATOR DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_tandem_controliq()