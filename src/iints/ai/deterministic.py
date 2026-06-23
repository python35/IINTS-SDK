"""Deterministic numerical boundaries for the optional AI assistant.

The language model may explain SDK results, but it must never calculate or
alter physiological metrics, insulin candidates, or glucagon candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from iints.core.safety.config import (
    CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN,
    CONTROLLER_HIGH_IOB_GUARD_UNITS,
    CONTROLLER_HYPO_GUARD_MGDL,
)
from iints.core.units import finite_value


DETERMINISTIC_DOSE_VERSION = "iints-deterministic-dose-v1"
_INPUT_UNITS = {
    "mpc_recommended_units": "U/step",
    "hard_max_insulin_units": "U/step",
    "deterministic_glucagon_candidate_mg": "mg/step",
    "hard_max_glucagon_mg": "mg/step",
    "current_glucose": "mg/dL",
    "current_glucose_mgdl": "mg/dL",
    "predicted_glucose_30min": "mg/dL",
    "predicted_glucose_30min_mgdl": "mg/dL",
    "glucose_trend_mgdl_min": "mg/dL/min",
    "glucose_trend": "mg/dL/min",
    "insulin_on_board": "U",
    "active_insulin": "U",
}


@dataclass(frozen=True)
class DoseSafetyLimits:
    """Fixed research-sandbox limits applied after deterministic MPC output."""

    low_glucose_mgdl: float = CONTROLLER_HYPO_GUARD_MGDL
    falling_trend_mgdl_min: float = CONTROLLER_FALLING_TREND_GUARD_MGDL_MIN
    high_iob_units: float = CONTROLLER_HIGH_IOB_GUARD_UNITS


@dataclass(frozen=True)
class DeterministicDoseResult:
    """Auditable result produced without invoking a language model."""

    final_insulin_units: float
    final_glucagon_mg: float
    mpc_recommended_units: float
    deterministic_glucagon_candidate_mg: float
    safety_hold: bool
    reasons: tuple[str, ...]
    calculation_version: str
    input_fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_insulin_units": self.final_insulin_units,
            "final_glucagon_mg": self.final_glucagon_mg,
            "mpc_recommended_units": self.mpc_recommended_units,
            "deterministic_glucagon_candidate_mg": self.deterministic_glucagon_candidate_mg,
            "safety_hold": self.safety_hold,
            "reasons": list(self.reasons),
            "calculation_version": self.calculation_version,
            "input_fingerprint_sha256": self.input_fingerprint_sha256,
            "ai_numeric_authority": False,
        }


def _finite_float(payload: Mapping[str, Any], key: str, *, default: float | None = None) -> float:
    raw = payload.get(key, default)
    if raw is None:
        raise ValueError(f"Missing deterministic numeric input: {key}")
    return finite_value(raw, name=f"Deterministic numeric input '{key}'", unit=_INPUT_UNITS[key])


def _optional_finite_float(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return _finite_float(payload, key)
    return None


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def calculate_deterministic_dose(
    payload: Mapping[str, Any],
    *,
    limits: DoseSafetyLimits = DoseSafetyLimits(),
) -> DeterministicDoseResult:
    """Resolve fixed simulation doses from deterministic controller outputs.

    Insulin starts from ``mpc_recommended_units`` and can only be clamped or
    reduced to zero by transparent safety rules. Glucagon must be supplied by a
    deterministic controller; this function never invents a rescue dose.
    """

    mpc_units = max(0.0, _finite_float(payload, "mpc_recommended_units"))
    insulin_cap = max(
        0.0,
        _finite_float(payload, "hard_max_insulin_units", default=mpc_units),
    )
    deterministic_glucagon = max(
        0.0,
        _finite_float(payload, "deterministic_glucagon_candidate_mg", default=0.0),
    )
    glucagon_cap = max(
        0.0,
        _finite_float(payload, "hard_max_glucagon_mg", default=deterministic_glucagon),
    )

    current_glucose = _optional_finite_float(payload, "current_glucose", "current_glucose_mgdl")
    predicted_glucose = _optional_finite_float(
        payload,
        "predicted_glucose_30min",
        "predicted_glucose_30min_mgdl",
    )
    trend = _optional_finite_float(payload, "glucose_trend_mgdl_min", "glucose_trend")
    iob = _optional_finite_float(payload, "insulin_on_board", "active_insulin")

    reasons: list[str] = []
    safety_hold = False
    if current_glucose is not None and current_glucose <= limits.low_glucose_mgdl:
        safety_hold = True
        reasons.append("current_glucose_at_or_below_fixed_guard")
    if predicted_glucose is not None and predicted_glucose <= limits.low_glucose_mgdl:
        safety_hold = True
        reasons.append("predicted_glucose_at_or_below_fixed_guard")
    if trend is not None and trend <= limits.falling_trend_mgdl_min:
        safety_hold = True
        reasons.append("falling_trend_at_or_below_fixed_guard")
    if iob is not None and iob >= limits.high_iob_units:
        safety_hold = True
        reasons.append("iob_at_or_above_fixed_guard")

    clamped_insulin = min(mpc_units, insulin_cap)
    if clamped_insulin < mpc_units:
        reasons.append("insulin_clamped_to_hard_cap")
    final_insulin = 0.0 if safety_hold else clamped_insulin

    final_glucagon = min(deterministic_glucagon, glucagon_cap)
    if final_glucagon < deterministic_glucagon:
        reasons.append("glucagon_clamped_to_hard_cap")
    if deterministic_glucagon == 0.0:
        reasons.append("no_deterministic_glucagon_candidate")
    if not reasons:
        reasons.append("deterministic_controller_candidate_accepted")

    return DeterministicDoseResult(
        final_insulin_units=final_insulin,
        final_glucagon_mg=final_glucagon,
        mpc_recommended_units=mpc_units,
        deterministic_glucagon_candidate_mg=deterministic_glucagon,
        safety_hold=safety_hold,
        reasons=tuple(reasons),
        calculation_version=DETERMINISTIC_DOSE_VERSION,
        input_fingerprint_sha256=_fingerprint(payload),
    )
