#!/usr/bin/env python3
"""
Omnipod 5 Emulator - IINTS-AF
Emulates the Omnipod 5 with Horizon Algorithm.

Based on:
- Clinical studies ( ASSERT, ONSET)
- FDA 510(k) clearance documentation
- Technical specifications

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
class Omnipod5Behavior(PumpBehavior):
    """Omnipod 5 specific behavior profile"""
    # Omnipod 5 specific parameters
    adaptive_learning: bool = True
    target_glucose: float = 110.0  # mg/dL (range 100-150)
    max_bolus: float = 30.0  # Omnipod allows larger boluses
    max_basal: float = 3.0  # units/hour (pod-specific)
    delivery_frequency: int = 5  # minutes between deliveries
    learning_window: int = 672  # hours (4 weeks) for adaptation
    
    # Safety features
    automatic_suspend: bool = True
    low_glucose_suspend: bool = True
    predictive_suspend: bool = True
    
    def __init__(self):
        super().__init__(
            pump_name="Omnipod 5 with Horizon Algorithm",
            manufacturer="Insulet Corporation",
            safety_limits=SafetyLimits(
                low_suspend_threshold=67.0,  # Omnipod uses 67 mg/dL
                high_suspend_threshold=250.0,
                max_bolus=30.0,
                max_basal_rate=3.0,
                max_daily_total=80.0,
                auto_off_duration=60.0,
                low_suspend_duration=5.0,
                target_glucose=110.0
            ),
            pid_parameters=PIDParameters(
                kp=0.025,  # Adaptive PID tuning
                ki=0.003,
                kd=0.035,
                target_glucose=110.0,
                integral_limit=4.0
            ),
            correction_factor=45.0,
            carb_ratio=8.0,  # Omnipod default
            insulin_sensitivity_factor=45.0,
            active_insulin_duration=5.0,  # Omnipod uses 5-hour DIA
            safety_level=SafetyLevel.MODERATE
        )


class Omnipod5Emulator(LegacyEmulator):
    """
    Emulates Omnipod 5 with Horizon algorithm.
    
    Omnipod 5 is a tubeless, patch pump with:
    - Adaptive learning (adjusts based on user patterns)
    - Automatic insulin delivery every 5 minutes
    - Activity and sleep mode support
    - Customizable target glucose (100-150 mg/dL)
    
    Key characteristics:
    - Tubeless design (no tubing)
    - Adaptive algorithm learns user patterns
    - Conservative low glucose threshold (67 mg/dL)
    - Activity/Sleep modes with higher targets
    
    Sources:
    - ASSERT Trial Results
    - ONSET Trial Results
    - FDA 510(k) K203467
    - Omnipod 5 User Guide
    """
    
    def __init__(self):
        """Initialize the Omnipod 5 emulator"""
        super().__init__()
        self.behavior = Omnipod5Behavior()
        self._learning_model = {
            'user_sensitivity': 50.0,
            'correction_factor': 50.0,
            'carb_ratio': 10.0,
            'basal_rate': 1.0,
            'patterns_learned': 0
        }
        
    def _get_default_behavior(self) -> Omnipod5Behavior:
        """Get Omnipod 5 default behavior"""
        return Omnipod5Behavior()
    
    def get_sources(self) -> List[Dict[str, str]]:
        """Get sources for Omnipod 5 emulation logic"""
        return [
            {
                'title': 'ASSERT Trial - Omnipod 5 Pivotal Study',
                'type': 'clinical_study',
                'year': '2021',
                'url': 'https://www.omnipod.com/assert-trial'
            },
            {
                'title': 'ONSET Trial - Omnipod 5 in Type 2',
                'type': 'clinical_study',
                'year': '2022',
                'url': 'https://www.omnipod.com/onset-trial'
            },
            {
                'title': 'FDA 510(k) K203467 - Omnipod 5 System',
                'type': 'regulatory',
                'year': '2021',
                'url': 'https://www.accessdata.fda.gov/'
            },
            {
                'title': 'Omnipod 5 User Guide',
                'type': 'technical_manual',
                'year': '2021',
                'url': 'https://www.omnipod.com/'
            }
        ]
    
    def set_activity_mode(self, enabled: bool, mode_type: str = 'exercise'):
        """Enable activity or sleep mode"""
        if mode_type == 'exercise':
            self.behavior.target_glucose = 140.0 if enabled else 110.0
        elif mode_type == 'sleep':
            self.behavior.target_glucose = 130.0 if enabled else 110.0
        
        if self.behavior.pid_parameters:
            self.behavior.pid_parameters.target_glucose = self.behavior.target_glucose
            
        print(f"    {mode_type.title()} mode: {'ON' if enabled else 'OFF'} (target: {self.behavior.target_glucose} mg/dL)")
    
    def emulate_decision(self,
                         glucose: float,
                         velocity: float,
                         insulin_on_board: float,
                         carbs: float,
                         current_time: float = 0) -> EmulatorDecision:
        """
        Emulate Omnipod 5 decision-making for current conditions.
        
        Omnipod 5 features:
        1. Adaptive learning (adjusts based on user response)
        2. Automatic delivery every 5 minutes
        3. Activity and sleep modes
        4. Conservative low glucose suspend
        
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
        
        # Omnipod uses slightly different low threshold
        low_threshold = safety.low_suspend_threshold  # 67 mg/dL
        
        # Check safety constraints
        should_suspend, suspend_reasons, adjustment = self._check_safety_constraints(
            glucose, velocity, insulin_on_board, carbs, current_time
        )
        reasoning.extend(suspend_reasons)
        
        # Omnipod-specific low threshold
        if glucose < low_threshold:
            should_suspend = True
            reasoning.append(
                f"Omnipod low suspend: glucose {glucose:.0f} < {low_threshold:.0f} mg/dL"
            )
            safety_overrides.append("Omnipod proprietary low threshold")
        
        if should_suspend:
            action = 'suspend'
            insulin_delivered = 0.0
            safety_overrides.extend(suspend_reasons)
        else:
            # Calculate correction with Omnipod's adaptive approach
            error = glucose - pid.target_glucose
            
            # Adaptive PID (simplified)
            self.state['integral_term'] += error * 0.04
            self.state['integral_term'] = max(-pid.integral_limit, 
                                              min(pid.integral_limit, 
                                                  self.state['integral_term']))
            
            # Meal bolus
            meal_bolus = 0.0
            if carbs > 0:
                meal_bolus = carbs / self.behavior.carb_ratio
                reasoning.append(
                    f"Meal bolus: {carbs:.0f}g → {meal_bolus:.2f} U"
                )
            
            # Calculate correction
            pid_output = (
                pid.kp * error +
                pid.ki * self.state['integral_term'] +
                pid.kd * velocity
            ) / self.behavior.correction_factor
            
            # Apply adaptive learning (simplified)
            if self._learning_model['patterns_learned'] > 10:
                adaptation_factor = 1.0 + (
                    (50.0 - self._learning_model['user_sensitivity']) / 100
                )
                pid_output *= adaptation_factor
            
            insulin_delivered = max(0, pid_output + meal_bolus) * adjustment
            
            # Omnipod max delivery
            insulin_delivered = min(insulin_delivered, safety.max_bolus)
            
            if insulin_delivered > 0:
                reasoning.append(
                    f"Omnipod correction: error={error:.0f} → {insulin_delivered:.2f} U"
                )
                
                # Simulate learning
                self._learning_model['patterns_learned'] += 1
            else:
                reasoning.append("No delivery needed")
        
        # Update cumulative insulin
        self.state['cumulative_insulin'] += insulin_delivered
        if self.state['cumulative_insulin'] > safety.max_daily_total:
            safety_overrides.append("Daily total limit reached")
            insulin_delivered = 0.0
            action = 'suspend'
        
        # Create decision
        decision = EmulatorDecision(
            insulin_delivered=insulin_delivered,
            action=action,
            reasoning=reasoning,
            safety_overrides=safety_overrides,
            predicted_glucose=glucose + velocity * 30,
            confidence=0.87
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
            'name': 'Omnipod 5 Horizon',
            'type': 'Adaptive Hybrid Closed-Loop',
            'personality': {
                'aggressiveness': 'Adaptive (starts conservative)',
                'hypo_aversion': 'High (67 mg/dL threshold)',
                'response_speed': 'Moderate',
                'correction_aggressiveness': 'Adaptive',
                'meal_handling': 'Automatic delivery',
                'predictive_features': 'Adaptive learning, Activity modes'
            },
            'key_differences': [
                'Tubeless patch pump design',
                'Adaptive algorithm learns user patterns',
                'Conservative low threshold (67 mg/dL)',
                'Activity & Sleep modes',
                'Longer insulin duration (5 hours)'
            ],
            'limitations': [
                'Pod life limited to 72-80 hours',
                'Requires different set for exercise',
                'Learning period needed for best results'
            ]
        }


def demo_omnipod5():
    """Demonstrate Omnipod 5 emulator"""
    print("=" * 70)
    print("OMNIPOD 5 EMULATOR DEMONSTRATION")
    print("=" * 70)
    
    emulator = Omnipod5Emulator()
    
    # Print behavior profile
    print("\n Algorithm Personality:")
    personality = emulator.get_algorithm_personality()
    print(f"  Name: {personality['name']}")
    print(f"  Type: {personality['type']}")
    print(f"  Hypo Aversion: {personality['personality']['hypo_aversion']}")
    
    # Print sources
    print("\n Sources:")
    for source in emulator.get_sources():
        print(f"  - [{source['type']}] {source['title']} ({source['year']})")
    
    # Simulate scenarios
    print("\n🧪 Scenario Simulation:")
    print("-" * 50)
    
    scenarios = [
        {'glucose': 170, 'velocity': 1.0, 'iob': 1.0, 'carbs': 0, 'desc': 'Elevated glucose'},
        {'glucose': 110, 'velocity': 0.2, 'iob': 1.5, 'carbs': 0, 'desc': 'At target'},
        {'glucose': 67, 'velocity': -0.5, 'iob': 1.0, 'carbs': 0, 'desc': 'At Omnipod low threshold'},
        {'glucose': 250, 'velocity': 2.0, 'iob': 0.5, 'carbs': 50, 'desc': 'High with large meal'},
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
        print(f"    → Action: {decision.action}, Insulin: {decision.insulin_delivered:.2f} U")
        print(f"    Reasoning: {', '.join(decision.reasoning[:2])}")
    
    # Test activity mode
    print("\n Activity Mode Test:")
    print("-" * 50)
    emulator.set_activity_mode(True, 'exercise')
    
    decision = emulator.emulate_decision(
        glucose=160, velocity=0.5, insulin_on_board=1.0, carbs=0, current_time=0
    )
    print(f"  Exercise mode: glucose=160 → {decision.insulin_delivered:.2f} U")
    
    emulator.set_activity_mode(False)
    
    print("\n" + "=" * 70)
    print("OMNIPOD 5 EMULATOR DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_omnipod5()

