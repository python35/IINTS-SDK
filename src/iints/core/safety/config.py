from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

SENSOR_GLUCOSE_MIN_MGDL = 40.0
SENSOR_GLUCOSE_MAX_MGDL = 500.0
# Sensor rate-of-change ceilings.
#
# These two constants state the same physical claim in two units and are used to
# reject an implausible CGM step (core/safety/input_validator.py) and to flag
# implausible rows in real data (data/quality_checker.py, analysis/validator.py).
#
# They were 20.0 mg/dL per 5 min (4.0 mg/dL/min), which is below the real step
# distribution and therefore rejected genuine physiology. Measured on the prepared
# cohorts over exact 5-minute steps (absolute step, per subject):
#
#   cohort      n         p99    p99.9   max     share > 20 mg/dL
#   AZT1D       283,921   21.0   34.1    163.0   1.03%
#   HUPA-UCM    309,342   17.0   54.0    259.0   0.64%
#   OhioT1DM    188,959   17.6   34.8     85.0   0.61%
#
# A 20 mg/dL ceiling discards 0.61-1.03% of real steps, and it discards them where
# it matters most: a fall steeper than 20 mg/dL per 5 min is exactly the situation
# the supervisor exists for, and the validator raises instead of acting on it.
# The ceiling is therefore set above the highest measured p99.9 (54.0), which leaves
# 0.009-0.094% of steps rejected as sensor artifact rather than 1 in 100.
#
# This is a plausibility ceiling on a measured signal, not a physiologic limit on
# blood glucose: the simulator's own envelope is SIMULATION_GLUCOSE_* below.
SENSOR_MAX_GLUCOSE_DELTA_PER_5_MIN_MGDL = 55.0
SENSOR_MAX_GLUCOSE_RATE_PER_MIN_MGDL = 11.0

# How fast the simulator's reported value may follow a reading the validator has
# already rejected (core/simulator.py:_validate_glucose_fail_soft).
#
# This used to be the same constant as the plausibility ceiling above, which
# conflated two opposite requirements: the ceiling should be permissive enough not
# to reject real physiology, while the damping applied to an implausible reading
# should stay tight, because that path exists to keep injected sensor corruption
# from reaching the algorithm in one step. Raising the ceiling would otherwise have
# made the simulator follow a corrupted sensor faster. They are separate now.
SENSOR_FAIL_SOFT_MAX_FOLLOW_PER_5_MIN_MGDL = 20.0
SIMULATION_GLUCOSE_FLOOR_MGDL = 20.0
SIMULATION_GLUCOSE_CEILING_MGDL = 600.0
CONTROLLER_HYPO_GUARD_MGDL = 90.0
CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN = -1.0
CONTROLLER_HIGH_IOB_GUARD_UNITS = 4.0
ML_MAX_INSULIN_CANDIDATE_PER_STEP_UNITS = 1.0
SAFETY_FORMULA_VERSION = "iints-safety-formulas-v2"


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
    # Damping applied after a reading has been rejected; deliberately tighter than
    # the plausibility ceiling above. See the comment on the constant.
    fail_soft_max_follow_per_5_min: float = SENSOR_FAIL_SOFT_MAX_FOLLOW_PER_5_MIN_MGDL

    # Supervisor thresholds
    hypoglycemia_threshold: float = 70.0
    severe_hypoglycemia_threshold: float = 54.0
    hyperglycemia_threshold: float = 250.0
    # Configurable engineering envelopes, not patient-specific prescriptions.
    # Defaults must permit ordinary meal coverage in the bundled ICR profiles;
    # low-glucose, falling-trend, prediction and basal guards remain separate.
    max_insulin_per_bolus: float = 15.0
    # Research-only glucagon safety rails for bi-hormonal simulations.
    # These caps prevent algorithm outputs from becoming unbounded actuator
    # commands. They are not clinical dosing advice.
    max_glucagon_per_step_mg: float = 1.0
    max_glucagon_per_hour_mg: float = 2.0
    glucagon_allowed_above_glucose_mgdl: float = 110.0
    glucose_rate_alarm: float = 5.0
    max_insulin_per_hour: float = 20.0
    max_iob: float = 20.0
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

    def __post_init__(self) -> None:
        numeric_fields = {
            name: value
            for name, value in asdict(self).items()
            if not isinstance(value, bool)
        }
        for name, value in numeric_fields.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"SafetyConfig.{name} must be finite")

        if not 0.0 <= self.min_glucose < self.max_glucose:
            raise ValueError("min_glucose must be non-negative and below max_glucose")
        if self.max_glucose_delta_per_5_min <= 0.0:
            raise ValueError("max_glucose_delta_per_5_min must be positive")
        if not (
            0.0
            < self.severe_hypoglycemia_threshold
            < self.hypoglycemia_threshold
            < self.hyperglycemia_threshold
        ):
            raise ValueError(
                "glucose thresholds must satisfy 0 < severe < hypo < hyper"
            )
        if not 0.0 < self.hypo_cutoff <= self.hyperglycemia_threshold:
            raise ValueError("hypo_cutoff must be positive and below hyper threshold")
        if self.predicted_hypoglycemia_threshold <= 0.0:
            raise ValueError("predicted_hypoglycemia_threshold must be positive")
        if self.critical_glucose_threshold <= 0.0:
            raise ValueError("critical_glucose_threshold must be positive")

        strictly_positive = (
            "glucose_rate_alarm",
            "max_basal_multiplier",
            "predicted_hypoglycemia_horizon_minutes",
            "predictor_mc_dropout_samples",
            "predictor_uncertainty_max_std_mgdl",
            "predictor_ood_zscore_threshold",
            "controller_hypo_guard_mgdl",
            "contract_glucose_threshold",
            "critical_glucose_duration_minutes",
        )
        for name in strictly_positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"SafetyConfig.{name} must be positive")
        non_negative = (
            "max_insulin_per_bolus",
            "max_glucagon_per_step_mg",
            "max_glucagon_per_hour_mg",
            "max_insulin_per_hour",
            "max_iob",
            "controller_high_iob_guard_units",
            "ml_max_insulin_candidate_per_step_units",
        )
        for name in non_negative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"SafetyConfig.{name} must be non-negative")
        if self.glucagon_allowed_above_glucose_mgdl <= 0.0:
            raise ValueError("glucagon_allowed_above_glucose_mgdl must be positive")
        if self.controller_falling_trend_guard_mgdl_min >= 0.0:
            raise ValueError(
                "controller_falling_trend_guard_mgdl_min must be negative"
            )
        if self.trend_stop >= 0.0 or self.contract_trend_threshold_mgdl_min >= 0.0:
            raise ValueError("falling-trend thresholds must be negative")
        if not 0.0 <= self.predictor_ood_max_feature_fraction <= 1.0:
            raise ValueError(
                "predictor_ood_max_feature_fraction must be between 0 and 1"
            )
        for name in (
            "predictor_uncertainty_gate_enabled",
            "predictor_ood_gate_enabled",
            "contract_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"SafetyConfig.{name} must be a boolean")
        for name in (
            "predicted_hypoglycemia_horizon_minutes",
            "predictor_mc_dropout_samples",
            "critical_glucose_duration_minutes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"SafetyConfig.{name} must be an integer")

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
