from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyConfig:
    """
    Central safety configuration for simulator, input validation, and supervisor.
    """
    # Input validation limits
    min_glucose: float = 20.0
    max_glucose: float = 600.0
    max_glucose_delta_per_5_min: float = 35.0

    # Supervisor thresholds
    hypoglycemia_threshold: float = 70.0
    severe_hypoglycemia_threshold: float = 54.0
    hyperglycemia_threshold: float = 250.0
    max_insulin_per_bolus: float = 5.0
    glucose_rate_alarm: float = 5.0
    max_insulin_per_hour: float = 3.0
    max_iob: float = 4.0
    trend_stop: float = -2.0
    hypo_cutoff: float = 70.0
    max_basal_multiplier: float = 3.0
    predicted_hypoglycemia_threshold: float = 60.0
    predicted_hypoglycemia_horizon_minutes: int = 30

    # Simulation termination limits
    critical_glucose_threshold: float = 40.0
    critical_glucose_duration_minutes: int = 30
