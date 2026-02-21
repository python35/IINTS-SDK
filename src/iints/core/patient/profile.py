from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PatientProfile:
    """
    User-facing patient profile that maps to the simulator config.
    """
    isf: float = 50.0  # Insulin Sensitivity Factor (mg/dL per unit)
    icr: float = 10.0  # Insulin-to-carb ratio (grams per unit)
    basal_rate: float = 0.8  # U/hr
    initial_glucose: float = 120.0
    dawn_phenomenon_strength: float = 0.0  # mg/dL per hour
    dawn_start_hour: float = 4.0
    dawn_end_hour: float = 8.0

    # Advanced knobs (optional)
    glucose_decay_rate: float = 0.002
    glucose_absorption_rate: float = 0.03
    insulin_action_duration: float = 300.0
    insulin_peak_time: float = 75.0
    meal_mismatch_epsilon: float = 1.0

    def to_patient_config(self) -> Dict[str, Any]:
        return {
            "basal_insulin_rate": self.basal_rate,
            "insulin_sensitivity": self.isf,
            "carb_factor": self.icr,
            "initial_glucose": self.initial_glucose,
            "dawn_phenomenon_strength": self.dawn_phenomenon_strength,
            "dawn_start_hour": self.dawn_start_hour,
            "dawn_end_hour": self.dawn_end_hour,
            "glucose_decay_rate": self.glucose_decay_rate,
            "glucose_absorption_rate": self.glucose_absorption_rate,
            "insulin_action_duration": self.insulin_action_duration,
            "insulin_peak_time": self.insulin_peak_time,
            "meal_mismatch_epsilon": self.meal_mismatch_epsilon,
        }
