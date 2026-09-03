from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


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
    dawn_insulin_resistance_fraction: float = 0.0  # peak fractional loss of insulin sensitivity
    dawn_start_hour: float = 4.0
    dawn_end_hour: float = 8.0

    # Advanced knobs (optional)
    glucose_decay_rate: float = 0.001
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
            "dawn_insulin_resistance_fraction": self.dawn_insulin_resistance_fraction,
            "dawn_start_hour": self.dawn_start_hour,
            "dawn_end_hour": self.dawn_end_hour,
            "glucose_decay_rate": self.glucose_decay_rate,
            "glucose_absorption_rate": self.glucose_absorption_rate,
            "insulin_action_duration": self.insulin_action_duration,
            "insulin_peak_time": self.insulin_peak_time,
            "meal_mismatch_epsilon": self.meal_mismatch_epsilon,
        }


@dataclass(frozen=True)
class PatientProfilePreset:
    """Named starter profile with a concise purpose statement."""

    name: str
    description: str
    expected_behavior: str
    profile_kwargs: Dict[str, float]

    def build_profile(self) -> PatientProfile:
        return PatientProfile(**self.profile_kwargs)


PATIENT_PROFILE_PRESETS: Dict[str, PatientProfilePreset] = {
    "stable-demo": PatientProfilePreset(
        name="stable-demo",
        description="Clinic-safe starter profile for smoke tests and teaching demos.",
        expected_behavior="Designed to complete calm baseline smoke runs without critical hypoglycemia.",
        profile_kwargs={
            "initial_glucose": 130.0,
            "basal_rate": 0.2,
            "isf": 40.0,
            "icr": 15.0,
            "glucose_decay_rate": 0.001,
        },
    ),
    "stress-test": PatientProfilePreset(
        name="stress-test",
        description="More reactive profile for meal, exercise, and safety stress testing.",
        expected_behavior="Designed to expose algorithm and supervisor behavior under stronger disturbances.",
        profile_kwargs={
            "initial_glucose": 120.0,
            "basal_rate": 0.5,
            "isf": 50.0,
            "icr": 10.0,
            "glucose_decay_rate": 0.003,
        },
    ),
    "endurance": PatientProfilePreset(
        name="endurance",
        description="Stable long-run profile for software endurance and reproducibility studies.",
        expected_behavior="Designed for long unattended runs where completion and artifact integrity matter most.",
        profile_kwargs={
            "initial_glucose": 140.0,
            "basal_rate": 0.0,
            "isf": 20.0,
            "icr": 25.0,
            "glucose_decay_rate": 0.0,
        },
    ),
}


def get_patient_profile_preset(name: str) -> PatientProfilePreset:
    try:
        return PATIENT_PROFILE_PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PATIENT_PROFILE_PRESETS))
        raise KeyError(f"Unknown patient profile preset '{name}'. Available: {available}") from exc
