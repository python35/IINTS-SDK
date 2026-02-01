import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SafetyViolation:
    level: SafetyLevel
    message: str
    glucose_value: float
    insulin_dose: float
    timestamp: float
    action_taken: str
    original_proposed_insulin: float

class IndependentSupervisor:
    """
    Independent safety supervisor that operates separately from algorithms.
    Implements hard safety limits and emergency overrides.
    """
    
    def __init__(self, 
                 hypoglycemia_threshold=70,      # mg/dL
                 severe_hypoglycemia_threshold=54,  # mg/dL
                 hyperglycemia_threshold=250,    # mg/dL
                 max_insulin_per_bolus=5,        # Units
                 glucose_rate_alarm=5):          # mg/dL per minute
        
        self.hypoglycemia_threshold = hypoglycemia_threshold
        self.severe_hypoglycemia_threshold = severe_hypoglycemia_threshold
        self.hyperglycemia_threshold = hyperglycemia_threshold
        self.max_insulin_per_bolus = max_insulin_per_bolus
        self.glucose_rate_alarm = glucose_rate_alarm
        
        # State tracking
        self.glucose_history = []
        self.violations = []
        self.emergency_mode = False
        self.last_iob = 0.0
        
    def evaluate_safety(self, current_glucose: float, proposed_insulin: float, 
                       current_time: float, current_iob: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate safety of proposed insulin dose based on current glucose and IOB.
        Returns modified insulin dose and safety status.
        """
        original_insulin = proposed_insulin
        self.last_iob = current_iob
        safety_status = SafetyLevel.SAFE
        actions_taken = []
        
        # Update glucose history
        self.glucose_history.append((current_time, current_glucose))
        if len(self.glucose_history) > 20:  # Keep last 20 readings
            self.glucose_history.pop(0)
        
        # 1. Glucose Level Checks
        if current_glucose <= self.severe_hypoglycemia_threshold:
            safety_status = SafetyLevel.EMERGENCY
            proposed_insulin = 0  # No insulin during severe hypoglycemia
            actions_taken.append("EMERGENCY_STOP: Severe hypoglycemia detected")
            self.emergency_mode = True
            
        elif current_glucose <= self.hypoglycemia_threshold:
            safety_status = SafetyLevel.CRITICAL
            proposed_insulin = min(proposed_insulin, 0.1)  # Minimal insulin only
            actions_taken.append("CRITICAL: Hypoglycemia - insulin limited")
            
        elif current_glucose >= self.hyperglycemia_threshold:
            safety_status = SafetyLevel.WARNING
            actions_taken.append("WARNING: Hyperglycemia detected")
        
        # 2. Insulin Dose Limits
        if proposed_insulin > self.max_insulin_per_bolus:
            proposed_insulin = self.max_insulin_per_bolus
            actions_taken.append(f"LIMIT: Bolus capped at {self.max_insulin_per_bolus}U")
            safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)
        
        # 3. Insulin Stacking Check (using IOB)
        # If IOB is high, be more conservative with additional insulin.
        if current_iob > self.max_insulin_per_bolus: # Use max_bolus as proxy for "high IOB"
            if current_glucose < 150: # Don't stack if not significantly high
                reduction_factor = max(0, (150 - current_glucose) / 100) # Reduce more aggressively closer to 100
                proposed_insulin *= (1 - reduction_factor)
                actions_taken.append(f"CAUTION: High IOB ({current_iob:.2f}U) - insulin reduced to prevent stacking")
                safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)

        # 4. Glucose Rate of Change
        if len(self.glucose_history) >= 2:
            glucose_rate = self._calculate_glucose_rate()
            if abs(glucose_rate) > self.glucose_rate_alarm:
                if glucose_rate < -self.glucose_rate_alarm and proposed_insulin > 0:
                    proposed_insulin *= 0.5  # Reduce insulin if glucose dropping fast
                    actions_taken.append("CAUTION: Fast glucose drop - insulin reduced")
                    safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)
        
        # 5. Emergency Mode Recovery
        if self.emergency_mode and current_glucose > self.hypoglycemia_threshold + 20:
            self.emergency_mode = False
            actions_taken.append("RECOVERY: Emergency mode cleared")
        
        # Log violation if any action was taken
        if actions_taken:
            violation = SafetyViolation(
                level=safety_status,
                message="; ".join(actions_taken),
                glucose_value=current_glucose,
                insulin_dose=proposed_insulin,
                timestamp=current_time,
                action_taken=f"Insulin: {original_insulin:.2f} → {proposed_insulin:.2f}U",
                original_proposed_insulin=original_insulin
            )
            self.violations.append(violation)
        
        return {
            "approved_insulin": proposed_insulin,
            "safety_level": safety_status,
            "actions_taken": actions_taken,
            "original_insulin": original_insulin,
            "insulin_reduction": original_insulin - proposed_insulin,
            "emergency_mode": self.emergency_mode
        }
    
    def _calculate_glucose_rate(self) -> float:
        """Calculate glucose rate of change in mg/dL per minute."""
        if len(self.glucose_history) < 2:
            return 0
        
        recent_points = self.glucose_history[-3:]  # Use last 3 points for stability
        if len(recent_points) < 2:
            return 0
        
        time_diff = recent_points[-1][0] - recent_points[0][0]
        glucose_diff = recent_points[-1][1] - recent_points[0][1]
        
        return glucose_diff / time_diff if time_diff > 0 else 0
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        violation_counts = {}
        for level in SafetyLevel:
            violation_counts[level.value] = sum(1 for v in self.violations if v.level == level)
        
        bolus_interventions_count = sum(1 for v in self.violations if v.original_proposed_insulin != v.insulin_dose)
        
        return {
            "total_violations": len(self.violations),
            "violation_breakdown": violation_counts,
            "bolus_interventions_count": bolus_interventions_count,
            "emergency_mode_active": self.emergency_mode,
            "current_iob": self.last_iob,
            "recent_violations": [
                {
                    "level": v.level.value,
                    "message": v.message,
                    "glucose": v.glucose_value,
                    "time": v.timestamp,
                    "original_proposed_insulin": v.original_proposed_insulin,
                    "approved_insulin": v.insulin_dose
                } for v in self.violations[-5:]  # Last 5 violations
            ]
        }
    
    def reset(self):
        """Reset supervisor state."""
        self.glucose_history.clear()
        self.violations.clear()
        self.emergency_mode = False
        self.last_iob = 0.0