import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from iints.core.safety.config import SafetyConfig
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


@dataclass
class SafetyDecision:
    original_dose: float
    final_dose: float
    reason: str
    triggered: bool

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
                 glucose_rate_alarm=5,           # mg/dL per minute
                 max_60min=3.0,                  # Units per 60 minutes
                 max_iob=4.0,                    # Units
                 trend_stop=-2.0,                # mg/dL per minute
                 hypo_cutoff=70.0,               # mg/dL
                 predicted_hypoglycemia_threshold=60.0,  # mg/dL
                 predicted_hypoglycemia_horizon_minutes=30,  # minutes
                 safety_config: Optional["SafetyConfig"] = None):
        
        if safety_config is not None:
            hypoglycemia_threshold = safety_config.hypoglycemia_threshold
            severe_hypoglycemia_threshold = safety_config.severe_hypoglycemia_threshold
            hyperglycemia_threshold = safety_config.hyperglycemia_threshold
            max_insulin_per_bolus = safety_config.max_insulin_per_bolus
            glucose_rate_alarm = safety_config.glucose_rate_alarm
            max_60min = safety_config.max_insulin_per_hour
            max_iob = safety_config.max_iob
            trend_stop = safety_config.trend_stop
            hypo_cutoff = safety_config.hypo_cutoff
            predicted_hypoglycemia_threshold = safety_config.predicted_hypoglycemia_threshold
            predicted_hypoglycemia_horizon_minutes = safety_config.predicted_hypoglycemia_horizon_minutes

        self.hypoglycemia_threshold = hypoglycemia_threshold
        self.severe_hypoglycemia_threshold = severe_hypoglycemia_threshold
        self.hyperglycemia_threshold = hyperglycemia_threshold
        self.max_insulin_per_bolus = max_insulin_per_bolus
        self.glucose_rate_alarm = glucose_rate_alarm
        self.max_60min = max_60min
        self.max_iob = max_iob
        self.trend_stop = trend_stop
        self.hypo_cutoff = hypo_cutoff
        self.predicted_hypoglycemia_threshold = predicted_hypoglycemia_threshold
        self.predicted_hypoglycemia_horizon_minutes = predicted_hypoglycemia_horizon_minutes
        
        # State tracking
        self.glucose_history: List[Tuple[float, float]] = []
        self.violations: List[SafetyViolation] = []
        self.emergency_mode = False
        self.last_iob = 0.0
        self.dose_history: List[tuple] = []
        
    def evaluate_safety(
        self,
        current_glucose: float,
        proposed_insulin: float,
        current_time: float,
        current_iob: float = 0.0,
        predicted_glucose_30min: Optional[float] = None,
        basal_insulin_units: Optional[float] = None,
        basal_limit_units: Optional[float] = None,
    ) -> Dict[str, Any]:
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

        # Predictive hypo guard (30-min horizon)
        if predicted_glucose_30min is not None:
            if predicted_glucose_30min <= self.predicted_hypoglycemia_threshold:
                safety_status = SafetyLevel.EMERGENCY
                proposed_insulin = 0
                actions_taken.append(
                    f"PREDICTED_HYPO: {predicted_glucose_30min:.1f} mg/dL in "
                    f"{self.predicted_hypoglycemia_horizon_minutes} min"
                )
                self.emergency_mode = True

        # Basal rate limit (relative to patient basal)
        if basal_insulin_units is not None and basal_limit_units is not None:
            if basal_insulin_units > basal_limit_units:
                excess = basal_insulin_units - basal_limit_units
                proposed_insulin = max(0.0, proposed_insulin - excess)
                actions_taken.append(
                    f"BASAL_LIMIT: basal {basal_insulin_units:.2f}U exceeds "
                    f"limit {basal_limit_units:.2f}U"
                )
                safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)
        
        # 1. Hard Hypo Cutoff (absolute stop)
        if current_glucose <= self.hypo_cutoff:
            safety_status = SafetyLevel.EMERGENCY
            proposed_insulin = 0
            actions_taken.append("HYPO_CUTOFF: Glucose below safety cutoff")
            self.emergency_mode = True

        # 2. Glucose Level Checks
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
        
        # 3. Rate-of-Change Protection (fast drop)
        glucose_rate = self._calculate_glucose_rate()
        if glucose_rate <= self.trend_stop:
            proposed_insulin = 0
            actions_taken.append(f"NEGATIVE_TREND_LIMIT: Glucose dropping at {glucose_rate:.2f} mg/dL/min")
            safety_status = max(safety_status, SafetyLevel.CRITICAL, key=lambda x: x.value)

        # 4. Dynamic IOB Clamp
        if current_iob >= self.max_iob:
            proposed_insulin = 0
            actions_taken.append(f"MAX_IOB_REACHED: IOB {current_iob:.2f}U exceeds {self.max_iob:.2f}U")
            safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)

        # 5. Insulin Dose Limits
        if proposed_insulin > self.max_insulin_per_bolus:
            proposed_insulin = self.max_insulin_per_bolus
            actions_taken.append(f"LIMIT: Bolus capped at {self.max_insulin_per_bolus}U")
            safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)
        
        # 6. Insulin Stacking Check (using IOB)
        # If IOB is high, be more conservative with additional insulin.
        if current_iob > self.max_insulin_per_bolus: # Use max_bolus as proxy for "high IOB"
            if current_glucose < 150: # Don't stack if not significantly high
                reduction_factor = max(0, (150 - current_glucose) / 100) # Reduce more aggressively closer to 100
                proposed_insulin *= (1 - reduction_factor)
                actions_taken.append(f"CAUTION: High IOB ({current_iob:.2f}U) - insulin reduced to prevent stacking")
                safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)

        # 7. Glucose Rate of Change (legacy alarm)
        if len(self.glucose_history) >= 2:
            if abs(glucose_rate) > self.glucose_rate_alarm:
                if glucose_rate < -self.glucose_rate_alarm and proposed_insulin > 0:
                    proposed_insulin *= 0.5  # Reduce insulin if glucose dropping fast
                    actions_taken.append("CAUTION: Fast glucose drop - insulin reduced")
                    safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)

        # 8. Hard Time-Window Cap (last 60 minutes)
        if proposed_insulin > 0:
            window_start = current_time - 60
            self.dose_history = [(t, d) for (t, d) in self.dose_history if t >= window_start]
            total_last_60 = sum(d for (_, d) in self.dose_history)
            if total_last_60 + proposed_insulin > self.max_60min:
                allowed = max(0.0, self.max_60min - total_last_60)
                if allowed < proposed_insulin:
                    proposed_insulin = allowed
                    actions_taken.append(
                        f"WINDOW_CAP_EXCEEDED: 60min cap {self.max_60min:.2f}U reached"
                    )
                    safety_status = max(safety_status, SafetyLevel.WARNING, key=lambda x: x.value)
            self.dose_history.append((current_time, proposed_insulin))
        
        # 9. Emergency Mode Recovery
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
        
        decision = SafetyDecision(
            original_dose=original_insulin,
            final_dose=proposed_insulin,
            reason="APPROVED" if not actions_taken else "; ".join(actions_taken),
            triggered=bool(actions_taken),
        )

        return {
            "approved_insulin": proposed_insulin,
            "safety_level": safety_status,
            "actions_taken": actions_taken,
            "original_insulin": original_insulin,
            "insulin_reduction": original_insulin - proposed_insulin,
            "emergency_mode": self.emergency_mode,
            "safety_decision": decision,
            "safety_reason": decision.reason,
            "safety_triggered": decision.triggered,
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
        self.dose_history = []

    def get_state(self) -> Dict[str, Any]:
        return {
            "glucose_history": self.glucose_history,
            "violations": [
                {
                    "level": v.level.value,
                    "message": v.message,
                    "glucose_value": v.glucose_value,
                    "insulin_dose": v.insulin_dose,
                    "timestamp": v.timestamp,
                    "action_taken": v.action_taken,
                    "original_proposed_insulin": v.original_proposed_insulin,
                }
                for v in self.violations
            ],
            "emergency_mode": self.emergency_mode,
            "last_iob": self.last_iob,
            "dose_history": self.dose_history,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.glucose_history = state.get("glucose_history", [])
        self.violations = [
            SafetyViolation(
                level=SafetyLevel(v["level"]),
                message=v["message"],
                glucose_value=v["glucose_value"],
                insulin_dose=v["insulin_dose"],
                timestamp=v["timestamp"],
                action_taken=v["action_taken"],
                original_proposed_insulin=v["original_proposed_insulin"],
            )
            for v in state.get("violations", [])
        ]
        self.emergency_mode = state.get("emergency_mode", False)
        self.last_iob = state.get("last_iob", 0.0)
        self.dose_history = state.get("dose_history", [])
