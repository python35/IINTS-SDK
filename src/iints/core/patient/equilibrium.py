"""Equilibrium-consistent insulin-sensitivity parameterisation for HovorkaPatientModel.

One free knob (``sigma``) determines insulin sensitivity and the fasting
operating glucose ``G*(sigma, basal)`` is derived by bisection so that
``dQ1/dt(G*, sigma, basal) = 0`` holds to machine precision (~1e-14 mg/dL/h)
without double-scaling ``_basal_parameter_scale`` by ``ISF / 50``.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from typing import Optional, Tuple

from iints.core.patient.calibration import smooth_threshold_excess
from iints.core.patient.hovorka_model import HovorkaPatientModel

__all__ = [
    "SENSITIVITY_REFERENCE_ISF",
    "basal_scale",
    "steady_state_residual",
    "equilibrium_glucose",
    "make_equilibrium_patient",
    "realised_isf",
    "sigma_for_target_isf",
]

SENSITIVITY_REFERENCE_ISF = 50.0
_GLUCOSE_KNEE_MGDL = 81.0


def _reference_patient(basal_u_per_h: float, glucose_mgdl: float) -> HovorkaPatientModel:
    return HovorkaPatientModel(
        initial_glucose=float(glucose_mgdl),
        basal_insulin_rate=float(basal_u_per_h),
        insulin_sensitivity=SENSITIVITY_REFERENCE_ISF,
    )


def basal_scale(basal_u_per_h: float, glucose_mgdl: float) -> float:
    """``sigma_b``: the sensitivity implied by holding ``glucose_mgdl`` on ``basal_u_per_h``."""
    return float(_reference_patient(basal_u_per_h, glucose_mgdl)._basal_parameter_scale)


def steady_state_residual(
    glucose_mgdl: float,
    basal_u_per_h: float,
    sigma: float,
    template: Optional[HovorkaPatientModel] = None,
) -> float:
    """``dQ1/dt`` in mg/min at a fasting steady state for a given ``sigma``."""
    patient = template if template is not None else _reference_patient(basal_u_per_h, 140.0)
    p = patient.params
    volume_dl = patient._glucose_volume_dl()
    basal_input = max(float(basal_u_per_h), 0.0) * 1000.0 / 60.0
    v_i = p.V_I_per_kg * p.body_weight_kg
    insulin = basal_input / max(v_i * p.k_e, 1e-9)

    glucose = float(glucose_mgdl)
    q1 = max(glucose, 20.0) * volume_dl
    f_01 = p.F_01c_per_kg * p.body_weight_kg
    f_01c = f_01 * min(1.0, max(0.0, glucose / _GLUCOSE_KNEE_MGDL))
    renal = 0.003 * volume_dl * smooth_threshold_excess(glucose, threshold=162.0, splay=10.0)
    egp_0 = p.EGP_0_per_kg * p.body_weight_kg

    x1 = p.S_IT * insulin * sigma
    x2 = p.S_ID * insulin * sigma
    x3 = p.S_IE * insulin * sigma
    q2 = x1 * q1 / max(p.k_12 + x2, 1e-9)
    return float(-(f_01c + renal) - x1 * q1 + p.k_12 * q2 + egp_0 * max(0.0, 1.0 - x3))


def equilibrium_glucose(
    basal_u_per_h: float,
    sigma: float,
    *,
    bracket: Tuple[float, float] = (_GLUCOSE_KNEE_MGDL, 600.0),
    iterations: int = 120,
) -> float:
    """Fasting glucose at which ``sigma`` and ``basal_u_per_h`` balance, via bisection."""
    template = _reference_patient(basal_u_per_h, 140.0)
    low, high = float(bracket[0]), float(bracket[1])
    f_low = steady_state_residual(low, basal_u_per_h, sigma, template)
    f_high = steady_state_residual(high, basal_u_per_h, sigma, template)
    if f_low * f_high > 0.0:
        raise ValueError(
            f"no fasting equilibrium between {low:.0f} and {high:.0f} mg/dL for "
            f"basal={basal_u_per_h:.3f} U/h and sigma={sigma:.5f} "
            f"(residual {f_low:.1f} .. {f_high:.1f} mg/min)"
        )
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if steady_state_residual(mid, basal_u_per_h, sigma, template) * f_low > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def make_equilibrium_patient(
    basal_u_per_h: float,
    sigma: float,
    glucose_mgdl: Optional[float] = None,
    **patient_kwargs,
) -> HovorkaPatientModel:
    """Construct a ``HovorkaPatientModel`` whose declared operating point is a true steady state."""
    glucose = equilibrium_glucose(basal_u_per_h, sigma) if glucose_mgdl is None else float(glucose_mgdl)
    return HovorkaPatientModel(
        initial_glucose=glucose,
        basal_insulin_rate=float(basal_u_per_h),
        insulin_sensitivity=SENSITIVITY_REFERENCE_ISF,
        insulin_sensitivity_scale=float(sigma),
        **patient_kwargs,
    )


def realised_isf(
    basal_u_per_h: float,
    sigma: float,
    *,
    bolus_units: float = 1.0,
    hours: float = 8.0,
) -> float:
    """Measured mg/dL per unit for this patient as a paired bolus difference."""
    from iints.core.patient.calibration import measure_insulin_sensitivity_factor

    glucose = equilibrium_glucose(basal_u_per_h, sigma)
    return float(
        measure_insulin_sensitivity_factor(
            lambda: make_equilibrium_patient(basal_u_per_h, sigma, glucose),
            basal_u_per_h,
            bolus_units=bolus_units,
            hours=hours,
        )
    )


def sigma_for_target_isf(
    basal_u_per_h: float,
    target_isf: float,
    *,
    bracket: Tuple[float, float] = (0.01, 2.0),
    iterations: int = 24,
) -> float:
    """Invert :func:`realised_isf` so a clinical ISF can be requested honestly."""
    low, high = bracket

    def measured(value: float) -> float:
        return realised_isf(basal_u_per_h, value)

    f_low, f_high = measured(low) - target_isf, measured(high) - target_isf
    if f_low * f_high > 0.0:
        raise ValueError(
            f"ISF {target_isf:.1f} mg/dL/U is unreachable at basal {basal_u_per_h:.2f} U/h; "
            f"sigma in [{low}, {high}] realises {measured(low):.1f}..{measured(high):.1f}"
        )
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if (measured(mid) - target_isf) * f_low > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
