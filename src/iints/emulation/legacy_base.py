#!/usr/bin/env python3
"""
Legacy Emulator Base Class - IINTS-AF
Base class for commercial insulin pump emulation.

This module provides the foundation for emulating commercial pump behaviors
based on published clinical studies and regulatory documentation.

Part of the #WeAreNotWaiting movement for transparent diabetes tech.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety level classification for pump behaviors"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class SafetyLimits:
    """Pump-specific safety constraints"""
    low_suspend_threshold: float = 70.0  # mg/dL
    high_suspend_threshold: float = 250.0  # mg/dL
    max_bolus: float = 10.0  # units
    max_basal_rate: float = 5.0  # units/hour
    max_daily_total: float = 100.0  # units
    auto_off_duration: float = 60.0  # minutes
    low_suspend_duration: float = 5.0  # minutes below threshold before suspend
    target_glucose: float = 120.0  # mg/dL


@dataclass
class PIDParameters:
    """PID controller parameters"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    target_glucose: float  # Target glucose setpoint
    integral_limit: float = 10.0  # Integral windup protection


@dataclass
class PumpBehavior:
    """Complete pump behavior profile"""
    pump_name: str
    manufacturer: str
    safety_limits: SafetyLimits
    pid_parameters: Optional[PIDParameters] = None
    correction_factor: float = 50.0  # mg/dL per unit
    carb_ratio: float = 10.0  # grams per unit
    insulin_sensitivity_factor: float = 50.0  # mg/dL per unit
    active_insulin_duration: float = 4.0  # hours
    safety_level: SafetyLevel = SafetyLevel.MODERATE
    
    def to_dict(self) -> Dict:
        return {
            'pump_name': self.pump_name,
            'manufacturer': self.manufacturer,
            'safety_limits': {
                'low_suspend_threshold': self.safety_limits.low_suspend_threshold,
                'high_suspend_threshold': self.safety_limits.high_suspend_threshold,
                'max_bolus': self.safety_limits.max_bolus,
                'max_basal_rate': self.safety_limits.max_basal_rate,
                'max_daily_total': self.safety_limits.max_daily_total,
                'auto_off_duration': self.safety_limits.auto_off_duration,
                'target_glucose': self.safety_limits.target_glucose
            },
            'correction_factor': self.correction_factor,
            'carb_ratio': self.carb_ratio,
            'insulin_sensitivity_factor': self.insulin_sensitivity_factor,
            'safety_level': self.safety_level.value
        }


@dataclass
class EmulatorDecision:
    """Decision output from a pump emulator"""
    insulin_delivered: float
    action: str  # 'deliver', 'suspend', 'reduce_basal', 'no_action'
    reasoning: List[str]
    safety_overrides: List[str]
    predicted_glucose: Optional[float] = None
    confidence: float = 0.9
    
    def to_dict(self) -> Dict:
        return {
            'insulin_delivered': self.insulin_delivered,
            'action': self.action,
            'reasoning': self.reasoning,
            'safety_overrides': self.safety_overrides,
            'predicted_glucose': self.predicted_glucose,
            'confidence': self.confidence
        }


from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput


class LegacyEmulator(InsulinAlgorithm):
    """
    Abstract base class for commercial insulin pump emulation.
    
    This class provides the interface for implementing emulators that
    replicate the behavior of commercial insulin pumps based on:
    - Published clinical studies
    - Regulatory documentation
    - User manuals and technical specifications
    
    Implementations should cite their sources in the get_sources() method.
    """
    
    def __init__(self):
        """Initialize the emulator with default settings"""
        self.behavior = self._get_default_behavior()
        self.state = self._create_initial_state()
        self._decision_history = []
        
    @abstractmethod
    def _get_default_behavior(self) -> PumpBehavior:
        """
        Get the default behavior profile for this pump.
        
        Returns:
            PumpBehavior: Complete behavior profile
        """
        pass
    
    @abstractmethod
    def get_sources(self) -> List[Dict[str, str]]:
        """
        Get citation information for the emulation logic.
        
        Returns:
            List of dictionaries with 'title', 'url', and 'type' keys
        """
        pass
    
    def _create_initial_state(self) -> Dict[str, Any]:
        """Create initial emulator state"""
        return {
            'cumulative_insulin': 0.0,
            'suspended_until': 0,  # timestamp
            'mode': 'auto',  # 'auto', 'manual', 'safe_mode'
            'last_decision_time': 0,
            'integral_term': 0.0,
            'previous_glucose': None
        }
    
    def reset(self):
        """Reset emulator state for new simulation"""
        self.state = self._create_initial_state()
        self._decision_history = []
        print(f"   {self.behavior.pump_name} emulator reset")
    
    def set_safety_mode(self, level: SafetyLevel):
        """Adjust safety level for testing"""
        self.behavior.safety_level = level
        print(f"    {self.behavior.pump_name} safety level set to {level.value}")
    
    def get_behavior_profile(self) -> PumpBehavior:
        """Get the complete behavior profile"""
        return self.behavior
    
    def _check_safety_constraints(self, 
                                  glucose: float,
                                  velocity: float,
                                  insulin_on_board: float,
                                  carbs: float,
                                  current_time: float) -> tuple:
        """
        Check safety constraints and return overrides.
        
        Returns:
            tuple: (should_suspend, reasoning_list, insulin_adjustment)
        """
        reasoning = []
        should_suspend = False
        insulin_adjustment = 1.0
        safety_limits = self.behavior.safety_limits
        
        # Low glucose suspend check
        if glucose < safety_limits.low_suspend_threshold:
            should_suspend = True
            reasoning.append(
                f"Low glucose suspend: glucose {glucose:.0f} < {safety_limits.low_suspend_threshold:.0f} mg/dL"
            )
            insulin_adjustment = 0.0
        
        # Rapid fall check
        elif velocity < -2.0 and glucose < 100:
            should_suspend = True
            reasoning.append(
                f"Rapid fall detected: {velocity:.1f} mg/dL/min with glucose {glucose:.0f}"
            )
            insulin_adjustment = 0.0
        
        # High glucose check (may increase insulin)
        elif glucose > safety_limits.high_suspend_threshold:
            reasoning.append(
                f"High glucose detected: {glucose:.0f} > {safety_limits.high_suspend_threshold:.0f} mg/dL"
            )
            # Some pumps increase delivery for high glucose
        
        # Insulin stacking check
        if insulin_on_board > 3.0:
            insulin_adjustment *= 0.5
            reasoning.append(
                f"Insulin stacking reduction: {insulin_on_board:.1f} U IOB"
            )
        
        return should_suspend, reasoning, insulin_adjustment
    
    def _calculate_pid_correction(self,
                                  glucose: float,
                                  velocity: float,
                                  target: float) -> float:
        """
        Calculate insulin correction using PID.
        
        Args:
            glucose: Current glucose
            velocity: Rate of change
            target: Target glucose
            
        Returns:
            Insulin correction in units
        """
        pid = self.behavior.pid_parameters
        if pid is None:
            # Fallback to proportional only
            error = glucose - target
            return max(0, error / self.behavior.correction_factor)
        
        # PID calculation
        error = glucose - target
        self.state['integral_term'] += error * 0.1  # Simplified integral
        self.state['integral_term'] = max(-pid.integral_limit, 
                                          min(pid.integral_limit, 
                                              self.state['integral_term']))
        derivative = velocity
        
        # PID formula
        correction = (
            pid.kp * error +
            pid.ki * self.state['integral_term'] +
            pid.kd * derivative
        ) / self.behavior.correction_factor
        
        return max(0, correction)
    
    @abstractmethod
    def emulate_decision(self,
                         glucose: float,
                         velocity: float,
                         insulin_on_board: float,
                         carbs: float,
                         current_time: float = 0) -> EmulatorDecision:
        """
        Emulate a pump decision for the current conditions.
        
        This is the main method to implement for each pump type.
        
        Args:
            glucose: Current glucose reading (mg/dL)
            velocity: Rate of glucose change (mg/dL/min)
            insulin_on_board: Current insulin on board (units)
            carbs: Carbs consumed in last period (grams)
            current_time: Current simulation time (minutes)
            
        Returns:
            EmulatorDecision: The pump's decision
        """
        pass

    def predict_insulin(self, algo_input: AlgorithmInput) -> Dict[str, Any]:
        """
        Adapts the InsulinAlgorithm interface to the LegacyEmulator's emulate_decision.
        """
        self.why_log = [] # Clear why_log for this prediction cycle

        # Velocity calculation, considering previous glucose state
        previous_glucose = self.state.get('previous_glucose', algo_input.current_glucose)
        
        # Avoid division by zero if time_step is 0 or if called at current_time=0 with no prev glucose
        if algo_input.time_step > 0:
            velocity_calc = (algo_input.current_glucose - previous_glucose) / algo_input.time_step
        else:
            velocity_calc = 0.0 # Or handle as error
        
        # If the first step, velocity might be zero, or need external init
        # Use initial velocity as 0 if it's the very first step of the simulation
        if self.state.get('last_decision_time') is None:
            velocity_for_emulator = 0.0 
        else:
            velocity_for_emulator = velocity_calc

        emulator_decision = self.emulate_decision(
            glucose=algo_input.current_glucose,
            velocity=velocity_for_emulator,
            insulin_on_board=algo_input.insulin_on_board,
            carbs=algo_input.carb_intake,
            current_time=algo_input.current_time # Use the correct current_time
        )
        self.state['previous_glucose'] = algo_input.current_glucose # Update state
        self.state['last_decision_time'] = algo_input.current_time # Update last decision time


        # Transfer emulator_decision.reasoning to why_log
        for reason_str in emulator_decision.reasoning:
            self._log_reason(reason_str, "emulator_logic")
        for override_str in emulator_decision.safety_overrides:
            self._log_reason(override_str, "emulator_safety", clinical_impact="Safety Override")


        # Convert EmulatorDecision to the format expected by Simulator
        return {
            "total_insulin_delivered": emulator_decision.insulin_delivered,
            "basal_insulin": emulator_decision.insulin_delivered, # Simplified, assume all is basal/correction
            "bolus_insulin": 0.0,
            "correction_bolus": emulator_decision.insulin_delivered, # Simplified
            "uncertainty": 1.0 - emulator_decision.confidence, # Higher uncertainty for lower confidence
            "fallback_triggered": True if emulator_decision.safety_overrides else False, # Re-using this flag
        }
    
    def get_decision_history(self) -> List[EmulatorDecision]:
        """Get history of all decisions"""
        return self._decision_history
    
    def export_behavior_report(self) -> Dict:
        """Export complete behavior profile for analysis"""
        return {
            'pump_name': self.behavior.pump_name,
            'manufacturer': self.behavior.manufacturer,
            'behavior': self.behavior.to_dict(),
            'sources': self.get_sources(),
            'decision_count': len(self._decision_history)
        }
    
    def compare_with_new_ai(self, new_ai_decisions: List[Dict]) -> Dict:
        """
        Compare this pump's behavior with a new AI algorithm.
        
        Args:
            new_ai_decisions: List of decisions from new algorithm
            
        Returns:
            Comparison analysis
        """
        if not self._decision_history:
            return {'error': 'No decisions to compare'}
        
        pump_insulin = [d.insulin_delivered for d in self._decision_history]
        ai_insulin = [d.get('total_insulin_delivered', 0) for d in new_ai_decisions]
        
        # Basic comparison
        comparison = {
            'pump_name': self.behavior.pump_name,
            'pump_total_insulin': sum(pump_insulin),
            'ai_total_insulin': sum(ai_insulin) if ai_insulin else 0,
            'insulin_difference_percent': (
                (sum(pump_insulin) - sum(ai_insulin)) / sum(pump_insulin) * 100
            ) if sum(pump_insulin) > 0 else 0,
            'pump_suspend_count': len([
                d for d in self._decision_history if d.action == 'suspend'
            ]),
            'ai_suspend_count': len([
                d for d in new_ai_decisions 
                if d.get('safety_override', False)
            ])
        }
        
        return comparison


def demo_legacy_emulator():
    """Demonstrate legacy emulator functionality"""
    print("=" * 70)
    print("LEGACY EMULATOR BASE CLASS DEMONSTRATION")
    print("=" * 70)
    
    # This shows the interface - actual emulators inherit from this
    print("\n Legacy Emulator Features:")
    print("  - Abstract base class for pump emulation")
    print("  - PID parameter support")
    print("  - Safety constraint checking")
    print("  - Decision history tracking")
    print("  - Behavior profile export")
    print("  - Comparison with new AI algorithms")
    
    print("\n Implementations Available:")
    print("  - Medtronic 780G Emulator")
    print("  - Tandem Control-IQ Emulator")
    print("  - Omnipod 5 Emulator")
    
    print("\n Sources:")
    print("  - Regulatory documentation")
    print("  - Clinical studies")
    print("  - User manuals")
    
    print("\n" + "=" * 70)
    print("LEGACY EMULATOR BASE CLASS DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    demo_legacy_emulator()

