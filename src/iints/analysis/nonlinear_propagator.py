"""Vectorised nonlinear Hovorka ODE propagator for batch multi-origin forecasting.

Integrates the control-relevant Hovorka equations (Q1, Q2, S1, S2, I, x1, x2, x3,
and meal gut compartments) across all forecast origins simultaneously via RK4.
Preserves all state-dependent nonlinearities discarded by a fixed linearisation:
- renal clearance above 162 mg/dL,
- insulin-independent uptake roll-off below 81 mg/dL,
- counter-regulatory EGP lift below 70 mg/dL,
- bilinear glucose disposal ``x1 * Q1``.

Also supports testable physiological meal-absorption extensions:
- ``"hovorka_2comp"``: standard 2-compartment Hovorka gut (D1 -> D2 -> Ra).
- ``"biphasic"``: fast simple-carb branch (``fast_carb_fraction``, ``fast_meal_tau_min``)
  plus slow complex/fat-protein branch (``meal_tau_min``).
- ``"glucose_gated"``: hyperglycemia-slowed gastric emptying (physiological
  glucose-dependent emptying multiplier around 140 mg/dL).

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PlantParameters", "STATE_NAMES", "from_patient", "propagate"]

_KNEE_MGDL = 81.0
_RENAL_THRESHOLD_MGDL = 162.0
_RENAL_SPLAY_MGDL = 10.0
_RENAL_RATE_PER_MIN = 0.003
_RESCUE_THRESHOLD_MGDL = 70.0
_RESCUE_HALF_MGDL = 16.0
_RESCUE_MAX_FRACTION = 1.0

STATE_NAMES = ("Q1", "Q2", "S1", "S2", "I", "x1", "x2", "x3", "D1", "D2")


@dataclass(frozen=True)
class PlantParameters:
    volume_dl: float
    volume_insulin_l: float
    k_e: float
    k_12: float
    k_a: np.ndarray          # (3,)
    k_b: np.ndarray          # (3,) already multiplied by sigma
    t_max_i: float
    egp_0: float
    f_01: float
    meal_tau_min: float
    bioavailability: float
    basal_u_per_h: float
    counterregulation: bool = True
    meal_model: str = "hovorka_2comp"     # "hovorka_2comp" | "biphasic" | "glucose_gated"
    fast_carb_fraction: float = 0.35      # used when meal_model == "biphasic"
    fast_meal_tau_min: float = 25.0       # used when meal_model == "biphasic"
    gastric_glucose_gain: float = 0.35    # used when meal_model == "glucose_gated"


def from_patient(
    patient,
    *,
    meal_tau_min: float,
    bioavailability: float,
    counterregulation: bool = True,
    meal_model: str = "hovorka_2comp",
    fast_carb_fraction: float = 0.35,
    fast_meal_tau_min: float = 25.0,
    gastric_glucose_gain: float = 0.35,
) -> PlantParameters:
    """Extract nonlinear plant parameters from a configured ``HovorkaPatientModel``."""
    p = patient.params
    sensitivity = (
        float(patient._basal_parameter_scale)
        * float(patient._clinical_sensitivity_scale)
        * float(patient.molecular_affinity_scalar)
    )
    k_a = np.array([p.k_a1, p.k_a2, p.k_a3], dtype=float)
    k_b = np.array(
        [
            p.S_IT * p.k_a1 * sensitivity,
            p.S_ID * p.k_a2 * sensitivity * float(patient.muscle_sensitivity_scalar),
            p.S_IE * p.k_a3 * sensitivity * float(patient.liver_sensitivity_scalar),
        ],
        dtype=float,
    )
    return PlantParameters(
        volume_dl=float(patient._glucose_volume_dl()),
        volume_insulin_l=float(p.V_I_per_kg * p.body_weight_kg),
        k_e=float(p.k_e),
        k_12=float(p.k_12),
        k_a=k_a,
        k_b=k_b,
        t_max_i=float(patient._insulin_tmax_minutes()),
        egp_0=float(p.EGP_0_per_kg * p.body_weight_kg),
        f_01=float(p.F_01c_per_kg * p.body_weight_kg),
        meal_tau_min=float(meal_tau_min),
        bioavailability=float(bioavailability),
        basal_u_per_h=float(patient.basal_insulin_rate),
        counterregulation=bool(counterregulation),
        meal_model=str(meal_model),
        fast_carb_fraction=float(np.clip(fast_carb_fraction, 0.0, 1.0)),
        fast_meal_tau_min=float(max(5.0, fast_meal_tau_min)),
        gastric_glucose_gain=float(max(0.0, gastric_glucose_gain)),
    )


def _softplus_excess(value: np.ndarray, threshold: float, splay: float) -> np.ndarray:
    z = (value - threshold) / splay
    return splay * np.where(z > 50.0, z, np.log1p(np.exp(np.clip(z, -50.0, 50.0))))


def _rescue_multiplier(glucose: np.ndarray) -> np.ndarray:
    """Saturating counter-regulatory lift of endogenous production below 70 mg/dL."""
    deficit = np.maximum(0.0, _RESCUE_THRESHOLD_MGDL - glucose)
    return 1.0 + _RESCUE_MAX_FRACTION * deficit / (_RESCUE_HALF_MGDL + deficit)


def _derivatives(
    state: np.ndarray,
    insulin_mu_per_min: np.ndarray,
    params: PlantParameters,
) -> np.ndarray:
    """``d/dt`` for an ``(n, 10)`` or ``(n, 12)`` block of states."""
    q1, q2, s1, s2, insulin = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
    x1, x2, x3 = state[:, 5], state[:, 6], state[:, 7]
    d1, d2 = state[:, 8], state[:, 9]

    glucose = q1 / params.volume_dl
    uptake = params.f_01 * np.clip(glucose / _KNEE_MGDL, 0.0, 1.0)
    renal = _RENAL_RATE_PER_MIN * params.volume_dl * _softplus_excess(
        glucose, _RENAL_THRESHOLD_MGDL, _RENAL_SPLAY_MGDL
    )
    egp = params.egp_0 * np.maximum(0.0, 1.0 - x3)
    if params.counterregulation:
        egp = egp * _rescue_multiplier(glucose)

    out = np.empty_like(state)
    if params.meal_model == "biphasic" and state.shape[1] >= 12:
        d1_fast, d2_fast = state[:, 10], state[:, 11]
        appearance = 1000.0 * params.bioavailability * (
            d2 / params.meal_tau_min + d2_fast / params.fast_meal_tau_min
        )
        out[:, 8] = -d1 / params.meal_tau_min
        out[:, 9] = (d1 - d2) / params.meal_tau_min
        out[:, 10] = -d1_fast / params.fast_meal_tau_min
        out[:, 11] = (d1_fast - d2_fast) / params.fast_meal_tau_min
    elif params.meal_model == "glucose_gated":
        # Physiological gastric-emptying feedback: lower/normal glucose empties faster,
        # hyperglycemia slows gastric emptying smoothly (bounded in [0.55, 1.45]).
        gate = np.clip(1.0 - params.gastric_glucose_gain * (glucose - 140.0) / 140.0, 0.55, 1.45)
        rate_gut = gate / params.meal_tau_min
        appearance = 1000.0 * params.bioavailability * d2 * rate_gut
        out[:, 8] = -d1 * rate_gut
        out[:, 9] = (d1 - d2) * rate_gut
    else:
        appearance = 1000.0 * params.bioavailability * d2 / params.meal_tau_min
        out[:, 8] = -d1 / params.meal_tau_min
        out[:, 9] = (d1 - d2) / params.meal_tau_min

    out[:, 0] = -(uptake + renal) - x1 * q1 + params.k_12 * q2 + egp + appearance
    out[:, 1] = x1 * q1 - (params.k_12 + x2) * q2
    out[:, 2] = insulin_mu_per_min - s1 / params.t_max_i
    out[:, 3] = (s1 - s2) / params.t_max_i
    out[:, 4] = s2 / (params.t_max_i * params.volume_insulin_l) - params.k_e * insulin
    for j in range(3):
        out[:, 5 + j] = -params.k_a[j] * state[:, 5 + j] + params.k_b[j] * insulin
    return out


def propagate(
    state0: np.ndarray,
    insulin_u_per_h: np.ndarray,
    carbs_g: np.ndarray,
    params: PlantParameters,
    *,
    step_minutes: float = 5.0,
    substeps: int = 5,
    feedback_gain: float = 0.0,
    feedback_target_mgdl: float = 120.0,
    max_rate_u_per_h: float = 10.0,
) -> np.ndarray:
    """Integrate every origin forward via RK4 and return glucose, shape ``(n, steps)``."""
    base_state = np.array(state0, dtype=float, copy=True)
    n, steps = insulin_u_per_h.shape

    if params.meal_model == "biphasic":
        # Split existing Kalman gut mass (D1, D2) between slow and fast branches.
        f_fast = params.fast_carb_fraction
        state = np.zeros((n, 12), dtype=float)
        state[:, :8] = base_state[:, :8]
        state[:, 8] = base_state[:, 8] * (1.0 - f_fast)
        state[:, 9] = base_state[:, 9] * (1.0 - f_fast)
        state[:, 10] = base_state[:, 8] * f_fast
        state[:, 11] = base_state[:, 9] * f_fast
    else:
        state = base_state

    glucose_out = np.empty((n, steps), dtype=float)
    dt = step_minutes / substeps

    for k in range(steps):
        if params.meal_model == "biphasic":
            f_fast = params.fast_carb_fraction
            state[:, 8] += carbs_g[:, k] * (1.0 - f_fast)
            state[:, 10] += carbs_g[:, k] * f_fast
        else:
            state[:, 8] += carbs_g[:, k]

        rate = insulin_u_per_h[:, k]
        if feedback_gain > 0.0:
            glucose = state[:, 0] / params.volume_dl
            rate = np.clip(
                rate + feedback_gain * (glucose - feedback_target_mgdl),
                0.0,
                max_rate_u_per_h,
            )
        drive = rate * 1000.0 / 60.0  # U/h -> mU/min
        for _ in range(substeps):
            k1 = _derivatives(state, drive, params)
            k2 = _derivatives(state + 0.5 * dt * k1, drive, params)
            k3 = _derivatives(state + 0.5 * dt * k2, drive, params)
            k4 = _derivatives(state + dt * k3, drive, params)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            np.maximum(state, 0.0, out=state)
        glucose_out[:, k] = state[:, 0] / params.volume_dl
    return np.clip(glucose_out, 20.0, 500.0)
