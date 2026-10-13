"""Physiological calibration and measurement utilities for virtual patients.

Why this module exists
----------------------
A virtual patient is configured with *clinical* knobs (basal rate, insulin
sensitivity factor, carbohydrate ratio) while the ODE backends are defined by
*physiological* parameters (insulin sensitivities, volumes, rate constants).
Those two descriptions are not independent:

* a fasting equilibrium at a requested glucose, for a requested basal rate, and
* a requested insulin sensitivity factor (ISF)

are two constraints on one quantity — insulin sensitivity. Satisfying both with
a single scaling factor is impossible: tightening one loosens the other. The
shipped Hovorka backend resolves this by fitting its sensitivity scale to the
configured basal rate, which makes the fasting state an equilibrium but leaves
the realised ISF and the basal rate unidentifiable.

This module resolves it the other way around, which is also the physiologically
honest direction: **insulin sensitivity is the patient property, and the basal
requirement follows from it.** Given a requested ISF and a target fasting
glucose it solves for

* ``insulin_sensitivity_scale`` — the multiplier on the published S_IT/S_ID/S_IE
  set that reproduces the requested ISF, and
* ``basal_insulin_rate`` — the basal rate that makes the target glucose a true
  fasting equilibrium for that sensitivity.

Everything here is *measurement-based*: the reported ISF is obtained from a
paired simulation (one run with a bolus, one without) rather than asserted, so
a calibrated patient can always be re-audited with the same functions.

Nothing in this module is clinically validated. It calibrates a research model
against its own definitions; it does not fit patient data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .physiology import smooth_threshold_excess

__all__ = [
    "HovorkaCalibration",
    "calibrate_carb_factor",
    "calibrate_hovorka",
    "hovorka_euglycaemic_basal_rate",
    "hovorka_euglycaemic_insulin",
    "hovorka_fasting_balance",
    "measure_clinical_parameters",
    "measure_fasting_drift",
    "measure_insulin_sensitivity_factor",
    "measure_plasma_insulin_tmax",
    "plasma_insulin_mu_per_l",
    "run_protocol",
]

# --------------------------------------------------------------------------
# generic driving helpers
# --------------------------------------------------------------------------


def run_protocol(
    model: Any,
    minutes: float,
    basal_u_per_h: float,
    *,
    dt: float = 5.0,
    boluses: Optional[Dict[float, float]] = None,
    meals: Optional[Dict[float, float]] = None,
    t0: float = 0.0,
) -> Dict[str, Any]:
    """Drive any patient backend over a protocol and return its trajectory.

    ``boluses`` and ``meals`` map a protocol minute (relative to ``t0``) to
    insulin units respectively grams of carbohydrate. A backend that leaves its
    declared domain is recorded as ``aborted`` with the minute it stopped; no
    values are extrapolated.
    """

    if minutes <= 0.0 or dt <= 0.0:
        raise ValueError("minutes and dt must be positive")
    boluses = {round(float(k), 6): float(v) for k, v in (boluses or {}).items()}
    meals = {round(float(k), 6): float(v) for k, v in (meals or {}).items()}
    steps = int(round(minutes / dt))
    times = np.zeros(steps + 1)
    glucose = np.zeros(steps + 1)
    insulin = np.full(steps + 1, np.nan)
    times[0] = t0
    glucose[0] = float(model.get_current_glucose())
    insulin[0] = plasma_insulin_mu_per_l(model)
    aborted, abort_t, error = False, float("nan"), ""
    for index in range(steps):
        now = t0 + index * dt
        key = round(index * dt, 6)
        dose = basal_u_per_h * dt / 60.0 + boluses.get(key, 0.0)
        carbs = meals.get(key, 0.0)
        try:
            glucose[index + 1] = model.update(dt, dose, carbs, current_time=now)
        except Exception as exc:  # the backend refuses to continue
            aborted, abort_t, error = True, now + dt, f"{type(exc).__name__}: {exc}"
            times, glucose, insulin = times[: index + 1], glucose[: index + 1], insulin[: index + 1]
            break
        times[index + 1] = now + dt
        insulin[index + 1] = plasma_insulin_mu_per_l(model)
    return {
        "time": times,
        "glucose": glucose,
        "plasma_insulin": insulin,
        "aborted": aborted,
        "abort_minute": abort_t,
        "error": error,
    }


def plasma_insulin_mu_per_l(model: Any) -> float:
    """Plasma insulin in mU/L, or NaN for backends without an insulin state."""
    state = getattr(model, "_state", None)
    if state is None:
        return float("nan")
    state = np.asarray(state, dtype=float)
    if state.size == 19:          # Hovorka state vector
        return float(state[4])
    if state.size >= 3:           # Bergman / Advanced state vector
        return float(state[2])
    return float("nan")


# --------------------------------------------------------------------------
# measurement — what the patient actually does
# --------------------------------------------------------------------------


def measure_insulin_sensitivity_factor(
    factory: Callable[[], Any],
    basal_u_per_h: float,
    *,
    bolus_units: float = 1.0,
    hours: float = 8.0,
    dt: float = 5.0,
) -> float:
    """Realised ISF in mg/dL per unit, as a paired difference.

    Two identical patients are driven over the same protocol; one receives the
    bolus at t = 0 and one does not. The reported value is the largest glucose
    separation between them, divided by the bolus size. Pairing removes any
    drift the patient would have shown anyway, which is what makes the number
    comparable between backends that are and are not self-stabilising.

    Returns NaN when either run leaves its domain.
    """

    if bolus_units <= 0.0:
        raise ValueError("bolus_units must be positive")
    with_bolus = run_protocol(factory(), hours * 60.0, basal_u_per_h, dt=dt,
                              boluses={0.0: bolus_units})
    without = run_protocol(factory(), hours * 60.0, basal_u_per_h, dt=dt)
    if with_bolus["aborted"] or without["aborted"]:
        return float("nan")
    separation = without["glucose"] - with_bolus["glucose"]
    return float(np.max(separation) / bolus_units)


def measure_fasting_drift(
    factory: Callable[[], Any],
    basal_u_per_h: float,
    *,
    hours: float = 24.0,
    dt: float = 5.0,
) -> Dict[str, float]:
    """Largest absolute deviation from the starting glucose while fasting."""
    run = run_protocol(factory(), hours * 60.0, basal_u_per_h, dt=dt)
    glucose = run["glucose"]
    return {
        "max_abs_drift_mgdl": float(np.max(np.abs(glucose - glucose[0]))),
        "final_glucose_mgdl": float(glucose[-1]),
        "min_glucose_mgdl": float(np.min(glucose)),
        "aborted_minute": float(run["abort_minute"]),
    }


def measure_plasma_insulin_tmax(
    factory: Callable[[], Any],
    basal_u_per_h: float,
    *,
    bolus_units: float = 1.0,
    hours: float = 6.0,
    dt: float = 1.0,
) -> float:
    """Minutes from bolus to the plasma-insulin peak (NaN if not exposed)."""
    run = run_protocol(factory(), hours * 60.0, basal_u_per_h, dt=dt,
                       boluses={0.0: bolus_units})
    insulin = run["plasma_insulin"]
    if not np.isfinite(insulin).any():
        return float("nan")
    return float(run["time"][int(np.nanargmax(insulin))] - run["time"][0])


def measure_clinical_parameters(
    factory: Callable[[], Any],
    basal_u_per_h: float,
    *,
    carb_factor: Optional[float] = None,
    meal_grams: float = 60.0,
    dt: float = 5.0,
) -> Dict[str, float]:
    """Audit sheet: what the virtual patient does, not what it was told.

    Returns the realised ISF, the plasma-insulin peak time, the 24 h fasting
    drift, and — when ``carb_factor`` is given — the glucose deviation six
    hours after a carbohydrate-ratio-matched meal. Put these next to the
    configured values; a large gap means the configuration is decorative.
    """

    report: Dict[str, float] = {
        "realised_isf_mgdl_per_u": measure_insulin_sensitivity_factor(
            factory, basal_u_per_h, dt=dt),
        "plasma_insulin_tmax_min": measure_plasma_insulin_tmax(factory, basal_u_per_h),
    }
    report.update({f"fasting_{k}": v for k, v in
                   measure_fasting_drift(factory, basal_u_per_h, dt=dt).items()})
    if carb_factor is not None and carb_factor > 0.0:
        run = run_protocol(factory(), 6 * 60.0, basal_u_per_h, dt=dt,
                           meals={0.0: meal_grams},
                           boluses={0.0: meal_grams / carb_factor})
        glucose = run["glucose"]
        report["meal_peak_rise_mgdl"] = float(np.max(glucose) - glucose[0])
        report["meal_minutes_to_peak"] = float(
            run["time"][int(np.argmax(glucose))] - run["time"][0])
        report["meal_deviation_at_6h_mgdl"] = (
            float("nan") if run["aborted"] else float(glucose[-1] - glucose[0]))
        report["meal_min_glucose_mgdl"] = float(np.min(glucose))
    return report


# --------------------------------------------------------------------------
# Hovorka fasting balance (analytic)
# --------------------------------------------------------------------------


def hovorka_fasting_balance(
    params: Any,
    glucose_mgdl: float,
    plasma_insulin_mu_l: float,
    *,
    transport_scale: float = 1.0,
    disposal_scale: float = 1.0,
    egp_scale: float = 1.0,
) -> float:
    """dQ1/dt (mg/min) for a fasting Hovorka patient at a steady insulin level.

    Mirrors ``HovorkaPatientModel._ode`` with no meal, no exercise, no stress
    and no dawn effect, and with Q2 at its own steady state. The three scales
    are the *effective* multipliers the ODE applies to S_IT, S_ID and S_IE —
    that is, the sensitivity scale times the clinical ISF scale, the molecular
    affinity and the muscle/liver scalars. Below roughly 70 mg/dL the model's
    counter-regulatory rescue term also contributes; this balance deliberately
    ignores it, so use it for targets in the euglycaemic range and above.
    """

    glucose = max(float(glucose_mgdl), 1.0)
    volume_dl = params.V_G_per_kg * params.body_weight_kg * 10.0
    q1 = glucose * volume_dl
    x1 = params.S_IT * plasma_insulin_mu_l * transport_scale
    x2 = params.S_ID * plasma_insulin_mu_l * disposal_scale
    x3 = params.S_IE * plasma_insulin_mu_l * egp_scale
    q2 = x1 * q1 / max(params.k_12 + x2, 1e-9)
    f01 = params.F_01c_per_kg * params.body_weight_kg
    f01c = f01 * min(1.0, max(0.0, glucose / 81.0))
    renal = 0.003 * volume_dl * smooth_threshold_excess(glucose, threshold=162.0, splay=10.0)
    egp = params.EGP_0_per_kg * params.body_weight_kg * max(0.0, 1.0 - x3)
    return -(f01c + renal) - x1 * q1 + params.k_12 * q2 + egp


def hovorka_euglycaemic_insulin(
    params: Any,
    glucose_mgdl: float,
    *,
    transport_scale: float = 1.0,
    disposal_scale: float = 1.0,
    egp_scale: float = 1.0,
    max_insulin_mu_l: float = 1.0e4,
) -> float:
    """Plasma insulin (mU/L) that makes ``glucose_mgdl`` a fasting equilibrium."""

    def residual(insulin: float) -> float:
        return hovorka_fasting_balance(
            params, glucose_mgdl, insulin,
            transport_scale=transport_scale, disposal_scale=disposal_scale,
            egp_scale=egp_scale)

    low, high = 0.0, 1.0
    while residual(high) > 0.0:
        high *= 2.0
        if high > max_insulin_mu_l:
            raise ValueError(
                "no fasting equilibrium below the insulin bracket; the "
                "requested glucose is unreachable for this parameter set")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if residual(mid) > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def hovorka_euglycaemic_basal_rate(
    params: Any,
    glucose_mgdl: float,
    *,
    transport_scale: float = 1.0,
    disposal_scale: float = 1.0,
    egp_scale: float = 1.0,
) -> float:
    """Basal rate (U/h) that holds ``glucose_mgdl`` for this parameter set."""
    insulin = hovorka_euglycaemic_insulin(
        params, glucose_mgdl, transport_scale=transport_scale,
        disposal_scale=disposal_scale, egp_scale=egp_scale)
    volume_l = params.V_I_per_kg * params.body_weight_kg
    return insulin * volume_l * params.k_e * 60.0 / 1000.0


# --------------------------------------------------------------------------
# joint calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HovorkaCalibration:
    """A Hovorka patient whose clinical knobs are physiologically true."""

    insulin_sensitivity: float               # requested ISF, mg/dL per unit
    target_glucose_mgdl: float
    insulin_sensitivity_scale: float         # multiplier on published S_I set
    basal_insulin_rate: float                # derived, U/h
    realised_isf_mgdl_per_u: float           # measured, paired simulation
    fasting_drift_mgdl: float                # measured over 24 h
    carb_factor: Optional[float] = None
    body_weight_kg: float = 70.0
    iterations: int = 0
    converged: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def build(self, **overrides: Any) -> Any:
        """Instantiate the calibrated HovorkaPatientModel."""
        from .hovorka_model import HovorkaPatientModel
        kwargs = dict(self.model_kwargs)
        kwargs.update(overrides)
        return HovorkaPatientModel(**kwargs)


def _hovorka_factory(model_kwargs: Dict[str, Any], initial_glucose: float
                     ) -> Callable[[], Any]:
    from .hovorka_model import HovorkaPatientModel
    kwargs = dict(model_kwargs)
    kwargs["initial_glucose"] = initial_glucose

    def factory() -> Any:
        return HovorkaPatientModel(**kwargs)

    return factory


def calibrate_hovorka(
    insulin_sensitivity: float = 50.0,
    target_glucose_mgdl: float = 120.0,
    *,
    carb_factor: Optional[float] = 10.0,
    body_weight_kg: float = 70.0,
    measurement_glucose_mgdl: float = 180.0,
    tolerance_mgdl: float = 0.5,
    max_iterations: int = 30,
    dt: float = 5.0,
    scale_bracket: Tuple[float, float] = (0.02, 20.0),
    extra_model_kwargs: Optional[Dict[str, Any]] = None,
) -> HovorkaCalibration:
    """Solve for the sensitivity scale and basal rate of a Hovorka patient.

    The sensitivity scale is found by bisection on the *measured* ISF (paired
    simulation, so the number is the real correction response, not an assumed
    one). For every candidate scale the basal rate is recomputed analytically
    from the fasting balance, so the fasting equilibrium holds at every step of
    the search rather than only at the end.

    The measurement starts from ``measurement_glucose_mgdl`` because a
    correction factor is a property of the hyperglycaemic range; starting at
    the target would push the patient into the counter-regulatory region and
    measure something else.
    """

    from .hovorka_model import HovorkaPatientModel

    if insulin_sensitivity <= 0.0:
        raise ValueError("insulin_sensitivity must be positive")
    if target_glucose_mgdl < 70.0:
        raise ValueError(
            "target_glucose_mgdl below 70 mg/dL is outside the range where the "
            "fasting balance is valid (counter-regulation is not modelled here)")
    base_kwargs: Dict[str, Any] = {
        "insulin_sensitivity": float(insulin_sensitivity),
        "carb_factor": float(carb_factor) if carb_factor else 10.0,
    }
    base_kwargs.update(extra_model_kwargs or {})

    def build_kwargs(scale: float, basal: float, initial_glucose: float) -> Dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs.update({
            "basal_insulin_rate": basal,
            "initial_glucose": initial_glucose,
            "insulin_sensitivity_scale": scale,
        })
        return kwargs

    def basal_for(scale: float) -> float:
        probe = HovorkaPatientModel(**build_kwargs(scale, 0.8, target_glucose_mgdl))
        clinical = probe._clinical_sensitivity_scale
        affinity = probe.molecular_affinity_scalar
        return hovorka_euglycaemic_basal_rate(
            probe.params, target_glucose_mgdl,
            transport_scale=scale * clinical * affinity,
            disposal_scale=scale * clinical * affinity * probe.muscle_sensitivity_scalar,
            egp_scale=scale * clinical * affinity * probe.liver_sensitivity_scalar)

    def realised_isf(scale: float) -> Tuple[float, float]:
        basal = basal_for(scale)
        factory = _hovorka_factory(
            build_kwargs(scale, basal, measurement_glucose_mgdl),
            measurement_glucose_mgdl)
        return measure_insulin_sensitivity_factor(factory, basal, dt=dt), basal

    low, high = float(scale_bracket[0]), float(scale_bracket[1])
    iterations = 0
    converged = False
    scale = 0.5 * (low + high)
    isf_value, basal = realised_isf(scale)
    for iterations in range(1, max_iterations + 1):
        scale = 0.5 * (low + high)
        isf_value, basal = realised_isf(scale)
        if np.isfinite(isf_value) and abs(isf_value - insulin_sensitivity) <= tolerance_mgdl:
            converged = True
            break
        if not np.isfinite(isf_value) or isf_value > insulin_sensitivity:
            high = scale
        else:
            low = scale

    drift = measure_fasting_drift(
        _hovorka_factory(build_kwargs(scale, basal, target_glucose_mgdl),
                         target_glucose_mgdl),
        basal, dt=dt)
    model_kwargs = build_kwargs(scale, basal, target_glucose_mgdl)
    return HovorkaCalibration(
        insulin_sensitivity=float(insulin_sensitivity),
        target_glucose_mgdl=float(target_glucose_mgdl),
        insulin_sensitivity_scale=float(scale),
        basal_insulin_rate=float(basal),
        realised_isf_mgdl_per_u=float(isf_value),
        fasting_drift_mgdl=float(drift["max_abs_drift_mgdl"]),
        carb_factor=float(carb_factor) if carb_factor else None,
        body_weight_kg=float(body_weight_kg),
        iterations=int(iterations),
        converged=bool(converged),
        model_kwargs=model_kwargs,
    )


def calibrate_carb_factor(
    calibration: HovorkaCalibration,
    *,
    meal_grams: float = 60.0,
    hours: float = 6.0,
    tolerance_mgdl: float = 2.0,
    max_iterations: int = 24,
    bracket: Tuple[float, float] = (3.0, 40.0),
    dt: float = 5.0,
) -> float:
    """Carbohydrate ratio (g/U) that returns the patient to baseline after a meal.

    Bisects the ratio until the glucose deviation ``hours`` after a meal plus
    its matching bolus is within ``tolerance_mgdl`` of zero. A ratio that is too
    small over-doses (deviation negative), one that is too large under-doses.
    """

    from .hovorka_model import HovorkaPatientModel

    def deviation(ratio: float) -> float:
        kwargs = dict(calibration.model_kwargs)
        kwargs["carb_factor"] = ratio
        model = HovorkaPatientModel(**kwargs)
        run = run_protocol(model, hours * 60.0, calibration.basal_insulin_rate, dt=dt,
                           meals={0.0: meal_grams},
                           boluses={0.0: meal_grams / ratio})
        if run["aborted"]:
            return -1.0e6
        return float(run["glucose"][-1] - run["glucose"][0])

    low, high = float(bracket[0]), float(bracket[1])
    ratio = 0.5 * (low + high)
    for _ in range(max_iterations):
        ratio = 0.5 * (low + high)
        value = deviation(ratio)
        if abs(value) <= tolerance_mgdl:
            return float(ratio)
        if value < 0.0:      # over-dosed -> need a larger ratio (less insulin)
            low = ratio
        else:
            high = ratio
    return float(ratio)
