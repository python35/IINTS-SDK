from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

SENSOR_GLUCOSE_MIN_MGDL = 40.0
SENSOR_GLUCOSE_MAX_MGDL = 500.0
SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL = 20.0
SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL = 4.0
SIMULATION_GLUCOSE_FLOOR_MGDL = 20.0
SIMULATION_GLUCOSE_CEILING_MGDL = 600.0
CONTROLLER_HYPO_GUARD_MGDL = 90.0
CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN = -1.0
CONTROLLER_HIGH_IOB_GUARD_UNITS = 4.0
ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS = 1.0
SAFETY_FORMULA_VERSION = "iints-safety-formulas-v1"


@dataclass
class SafetyConfig:
    """
    Central safety configuration for simulator, input validation, and supervisor.
    """
    # Broad CGM/sensor plausibility limits.
    # These are intentionally device-aware fail-soft bounds, not a claim about
    # the full physiologic envelope of blood glucose inside the simulator.
    min_glucose: float = SENSOR_GLUCOSE_MIN_MGDL
    max_glucose: float = SENSOR_GLUCOSE_MAX_MGDL
    max_glucose_delta_per_5_min: float = SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL

    # Supervisor thresholds
    hypoglycemia_threshold: float = 70.0
    severe_hypoglycemia_threshold: float = 54.0
    hyperglycemia_threshold: float = 250.0
    max_insulin_per_bolus: float = 5.0
    # Research-only glucagon safety rails for bi-hormonal simulations.
    # These caps prevent algorithm outputs from becoming unbounded actuator
    # commands. They are not clinical dosing advice.
    max_glucagon_per_step_mg: float = 1.0
    max_glucagon_per_hour_mg: float = 2.0
    glucagon_allowed_above_glucose_mgdl: float = 110.0
    glucose_rate_alarm: float = 5.0
    max_insulin_per_hour: float = 3.0
    max_iob: float = 4.0
    trend_stop: float = -2.0
    hypo_cutoff: float = 70.0
    max_basal_multiplier: float = 3.0
    predicted_hypoglycemia_threshold: float = 60.0
    predicted_hypoglycemia_horizon_minutes: int = 30
    predictor_mc_dropout_samples: int = 30
    predictor_uncertainty_gate_enabled: bool = True
    predictor_uncertainty_max_std_mgdl: float = 20.0
    predictor_ood_gate_enabled: bool = True
    predictor_ood_zscore_threshold: float = 4.0
    predictor_ood_max_feature_fraction: float = 0.30

    # Shared deterministic controller guards. Keep these centralized so MPC,
    # ML fallbacks, and the AI boundary cannot silently drift apart.
    controller_hypo_guard_mgdl: float = CONTROLLER_HYPO_GUARD_MGDL
    controller_falling_trend_guard_mgdl_min: float = CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN
    controller_high_iob_guard_units: float = CONTROLLER_HIGH_IOB_GUARD_UNITS
    ml_max_insulin_candidate_per_step_units: float = ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS

    # Formal safety contract (logic validation)
    contract_enabled: bool = True
    contract_glucose_threshold: float = 90.0
    contract_trend_threshold_mgdl_min: float = -1.0  # -5 mg/dL per 5 minutes

    # Simulation termination limits
    critical_glucose_threshold: float = 40.0
    critical_glucose_duration_minutes: int = 30

    def to_versioned_dict(self) -> dict[str, Any]:
        return {
            "formula_version": SAFETY_FORMULA_VERSION,
            "units": {
                "glucose": "mg/dL",
                "glucose_rate": "mg/dL/min",
                "insulin": "U",
                "glucagon": "mg",
                "time": "min",
            },
            "values": asdict(self),
        }

    def fingerprint_sha256(self) -> str:
        canonical = json.dumps(self.to_versioned_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
