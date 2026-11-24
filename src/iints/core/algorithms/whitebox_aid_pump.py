"""Universal White-Box Mathematical Model of Automated Insulin Delivery (AID) Pumps.

This module implements a pure, observable-only mathematical formulation of
modern Automated Insulin Delivery (AID) smart insulin pump algorithms, reverse-
engineered from peer-reviewed control-engineering and clinical literature.
Unlike simplified rule-based baselines or oracle controllers that read
unobservable simulator compartments (`data.patient_state`), every algorithm in
this module operates strictly on signals available to a physical insulin pump:

1. Sampled CGM glucose $G_k$ (mg/dL),
2. The pump's own historical insulin delivery $u_{0:k-1}$ (U),
3. Announced carbohydrate intakes $D_{\\mathrm{carb}, 0:k}$ (g) (or unannounced
   meal detection via glucose residuals),
4. Configured clinical pump parameters ($\\mathrm{ISF}$, $\\mathrm{ICR}$,
   $\\mathrm{DIA}$, $t_p$, $u_{\\mathrm{basal}}$, $G_{\\mathrm{target}}$) or
   solely body weight $W$ (kg) in autonomous Bionic Pancreas mode.

Published Smart Pump Architectures Implemented
----------------------------------------------
1. **Law 1 (`pid_ifb`) — Steil PID with Pharmacokinetic Insulin Feedback**
   (*Steil et al., Diabetes 2006; Medtronic MiniMed 670G/780G SmartGuard*):
   Continuous-to-discrete PID controller coupled with exact matrix-exponential
   two-compartment subcutaneous/plasma insulin feedback $(\\hat{I}_{\\mathrm{sc}},
   \\hat{I}_{\\mathrm{p}})$ and 780G-style IOB-guarded auto-correction pulses.
2. **Law 2 (`trajectory_impulse`) — TypeZero / Control-IQ Modular Range & Kovatchev Brakes**
   (*Kovatchev et al., 1997; Patek et al., 2012; Breton et al., NEJM 2020*):
   Multi-curve impulse-response trajectory forecast paired with Kovatchev's
   non-linear symmetrizing risk transformation $f(G) = 1.509((\\ln G)^{1.084} - 5.381)$,
   Low Blood Glucose Index ($\\mathrm{LBGI}$) power-law safety brakes, and
   60% hourly auto-correction boluses.
3. **Law 3 (`kalman_zone_mpc`) — Omnipod 5 / Doyle-Gondhalekar Velocity-Weighted Zone-MPC**
   (*Gondhalekar et al., Automatica 2016; Huyett et al., 2021*):
   Observable-only Kalman state observer $\\hat{x}_{k|k}$ paired with an
   asymmetric target-zone cost function, velocity-dependent insulin penalty, and
   Total Daily Dose ($\\mathrm{TDD}$) personalization.
4. **Law 4 (`oref1_smb`) — OpenAPS `oref1` / AndroidAPS / Trio with Logarithmic Dynamic ISF, UAM & SMB**
   (*OpenAPS oref1; Wilson Logarithmic Dynamic ISF; Trio/AAPS*):
   Computes four distinct prediction curves (`IOB`, `COB`, `UAM`, `ZT`), scales
   sensitivity in real time via the logarithmic Dynamic ISF equation
   $\\mathrm{ISF}_{\\mathrm{dyn}}(G) = 1800 / (\\mathrm{TDD}\\cdot \\kappa \\cdot \\ln(G/d_{\\mathrm{ins}} + 1))$,
   detects Unannounced Meals (`UAM`) from positive glucose residuals, and doses
   Super Micro Boluses (`SMB`) paired with Zero-Temp (`ZT`) safety guards.
5. **Law 5 (`camaps_bayesian_mpc`) — Cambridge / Hovorka Bayesian Adaptive MPC**
   (*Hovorka et al., Physiol Meas 2004; Lancet 2010; CamAPS FX*):
   Maintains a recursive Bayesian posterior over time-varying insulin
   sensitivity $\\hat{\\theta}_{\\mathrm{ISF},k}$ from CGM prediction innovations
   and optimizes receding-horizon infusion over the compartmental action state.
6. **Law 6 (`bionic_autonomous`) — El-Khatib / Damiano Autonomous Body-Weight Bionic Pancreas**
   (*El-Khatib et al., Sci Transl Med 2010; Lancet 2017; Beta Bionics iLet*):
   Initializes all dosing gains solely from body weight $W$ (kg)
   ($\\widehat{\\mathrm{TDD}}_0 = 0.55 W$) and continuously adapts its reference
   basal rate and sensitivity offset over a rolling horizon.
7. **Unified Mode (`unified`)**:
   Evaluates all canonical mathematical laws simultaneously, exposes their full
   equations and trajectories in `mathematical_telemetry`, and arbitrates
   dosing through a Kovatchev-brake-guarded convex combination.

Research and clinical safety boundary
-------------------------------------
This mathematical pump model is for pre-clinical in-silico research and
algorithm reverse-engineering only. It is not a medical device and must never
be used to calculate insulin doses for a real person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from iints.api.base_algorithm import AlgorithmInput, AlgorithmMetadata, InsulinAlgorithm
from iints.core.formula_registry import FormulaSpec


ControlLawType = Literal[
    "unified",
    "pid_ifb",
    "trajectory_impulse",
    "kalman_zone_mpc",
    "oref1_smb",
    "camaps_bayesian_mpc",
    "bionic_autonomous",
]


AID_PUMP_FORMULAS: tuple[FormulaSpec, ...] = (
    FormulaSpec(
        formula_id="P01_BIEXPONENTIAL_INSULIN_PK_KERNEL",
        title="Bi-exponential insulin activity kernel and survival curve (Walsh / Loop / oref1)",
        category="physiology",
        canonical_expression=(
            "tau_I = tp*(1 - tp/D)/(1 - 2*tp/D); a = 2*tau_I/D; S = 1/(1 - a + (1 + a)*exp(-D/tau_I)); "
            "IOB(t) = 1 - S*(1 - a)*((t^2/(tau_I*D*(1 - a)) - t/tau_I - 1)*exp(-t/tau_I) + 1)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\tau_I &= \frac{t_p(1 - t_p/D)}{1 - 2t_p/D},\quad a = \frac{2\tau_I}{D},\quad "
            r"S = \frac{1}{1 - a + (1 + a)e^{-D/\tau_I}} \\"
            r"S_{\mathrm{IOB}}(t) &= 1 - S(1 - a)\!\left[\left(\frac{t^2}{\tau_I D(1 - a)} "
            r"- \frac{t}{\tau_I} - 1\right)e^{-t/\tau_I} + 1\right]"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Evaluated analytically over historical pump deliveries u(t_i) for t in [0, DIA], "
            "paired with exact matrix-exponential integration of the two-compartment PK state."
        ),
        state_variables=("I_sc", "I_p", "IOB", "a_I"),
        parameters=("tp", "D", "tau_sc", "tau_p"),
        units="t, tp, D in min; IOB in U; a_I in U/min",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:biexponential_iob_fraction",),
        literature_basis=(
            "https://doi.org/10.1177/1932296814526474",
            "https://loopkit.github.io/loopdocs/operation/algorithm/prediction/",
        ),
        validation_note=(
            "Closed-form bi-exponential survival and activity kernel used in modern open-source "
            "and commercial AID trajectory engines."
        ),
        evidence_class="canonical",
    ),
    FormulaSpec(
        formula_id="P02_GLUCOSE_RESIDUAL_KALMAN_OBSERVER",
        title="Observable-only Kalman state observer and dynamic glucose residual",
        category="physiology",
        canonical_expression=(
            "x_hat_{k|k-1} = A*x_hat_{k-1|k-1} + B_u*du_{k-1} + B_d*d_meal_{k-1}; "
            "x_hat_{k|k} = x_hat_{k|k-1} + L*(y_k - C*x_hat_{k|k-1}); "
            "r_k = dG_hat_dt - (-ISF * a_I)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\hat{x}_{k|k-1} &= A(\Delta t)\hat{x}_{k-1|k-1} + B_u(\Delta t)\Delta u_{k-1} + B_d(\Delta t)d_{\mathrm{meal},k-1} \\"
            r"\hat{x}_{k|k} &= \hat{x}_{k|k-1} + L\!\left(y_k - C\hat{x}_{k|k-1}\right), \qquad "
            r"r_k = \dot{\hat{G}}_k + \mathrm{ISF}\cdot a_{I,k}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Discrete-time steady-state Kalman update driven solely by CGM glucose y_k, "
            "delivered insulin history, and announced carbohydrates."
        ),
        state_variables=("dG_hat", "G_dot_hat", "X_hat", "r_k"),
        parameters=("A", "B_u", "B_d", "L", "ISF"),
        units="dG_hat in mg/dL, G_dot_hat and r_k in mg/dL/min",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:WhiteBoxPumpStateEstimator",),
        literature_basis=(
            "https://doi.org/10.1109/TBME.2012.2202661",
        ),
        validation_note=(
            "Eliminates oracle state peeking by reconstructing velocity, remote insulin action, "
            "and unexplained glucose residuals from CGM alone."
        ),
        evidence_class="adapted",
    ),
    FormulaSpec(
        formula_id="P03_STEIL_PID_IFB_LAW",
        title="PID with pharmacokinetic insulin feedback (Steil / Medtronic 670G/780G architecture)",
        category="physiology",
        canonical_expression=(
            "u_PID = u_basal + Kp*(G - G_ref) + Ki*I_int + Kd*dG_dt; "
            "u_IFB = max(0, (1 + gamma_1 + gamma_2)*u_PID - gamma_1*I_sc - gamma_2*I_p)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"u_{\mathrm{PID}}(t) &= u_{\mathrm{basal}} + K_p\!\left(G(t)-G_{\mathrm{ref}}\right) "
            r"+ K_i\int_0^t\!\left(G(\tau)-G_{\mathrm{ref}}\right)d\tau + K_d\dot{\hat{G}}(t) \\"
            r"u_{\mathrm{IFB}}(t) &= \max\!\Big(0,\;(1+\gamma_1+\gamma_2)u_{\mathrm{PID}}(t) "
            r"- \gamma_1\hat{I}_{\mathrm{sc}}(t) - \gamma_2\hat{I}_{\mathrm{p}}(t)\Big)"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Exact matrix-exponential integration of (I_sc, I_p) per control step with "
            "trapezoidal anti-windup and IOB-headroom-limited auto-correction micro-bolus."
        ),
        state_variables=("I_int", "I_sc", "I_p"),
        parameters=("Kp", "Ki", "Kd", "gamma_1", "gamma_2", "G_ref"),
        units="u in U/min or U/step; I_sc, I_p normalized to basal-equivalent rate",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:WhiteBoxAIDPumpAlgorithm._compute_pid_ifb_law",),
        literature_basis=(
            "https://doi.org/10.2337/db06-0220",
            "https://doi.org/10.1177/193229681100500321",
        ),
        validation_note="Published Steil et al. PID-IFB equations with 780G-style auto-correction extension.",
        evidence_class="canonical",
    ),
    FormulaSpec(
        formula_id="P04_TRAJECTORY_IMPULSE_LAW",
        title="Four-curve trajectory superposition and Kovatchev LBGI safety brake (Control-IQ / TypeZero)",
        category="physiology",
        canonical_expression=(
            "f(G) = 1.509*((ln G)^1.084 - 5.381); r_L(G) = 10*f(G)^2 if f(G) < 0 else 0; "
            "phi_brake = 1 / (1 + alpha_brake * r_L(G_guard)); "
            "G_hat(t+tau) = G(t) + dG_IOB(tau) + dG_COB(tau) + dG_mom(tau) + dG_RC(tau)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"f(G) &= 1.509\!\left((\ln G)^{1.084} - 5.381\right), \qquad "
            r"r_L(G) = 10\,\min(0, f(G))^2 \\"
            r"\phi_{\mathrm{brake}} &= \frac{1}{1 + \alpha_{\mathrm{brake}}\,r_L(G_{\mathrm{guard}})}, \qquad "
            r"\hat{G}(t+\tau) = G(t) + \Delta G_{\mathrm{IOB}} + \Delta G_{\mathrm{COB}} + \Delta G_{\mathrm{mom}} + \Delta G_{\mathrm{RC}}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Evaluates the 4 impulse curves on a 5-minute grid up to DIA, applies Kovatchev LBGI "
            "power-law braking, and computes 60%-scaled hourly auto-correction boluses."
        ),
        state_variables=("G_hat_curve", "G_ev", "G_guard", "r_L", "phi_brake"),
        parameters=("ISF", "ICR", "DIA", "alpha_brake"),
        units="G_hat in mg/dL, r_L dimensionless risk (0-100)",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:WhiteBoxAIDPumpAlgorithm._compute_trajectory_impulse_law",),
        literature_basis=(
            "https://doi.org/10.2337/diacare.20.11.1655",
            "https://doi.org/10.1056/NEJMoa1907863",
        ),
        validation_note="Combines Kovatchev-Patek TypeZero LBGI safety brakes with 4-curve trajectory superposition.",
        evidence_class="canonical",
    ),
    FormulaSpec(
        formula_id="P05_KALMAN_ZONE_MPC_LAW",
        title="Asymmetric velocity-weighted Zone-MPC objective with adaptive TDD (Omnipod 5 / Doyle style)",
        category="physiology",
        canonical_expression=(
            "J(du) = sum_j [ q_high*max(0, G_hat_j - G_high)^2 + q_low*max(0, G_low - G_hat_j)^2 "
            "+ q_ref*(G_hat_j - G_target)^2 ] + sum_j R_0*(1 + kappa_v*max(0, -dG_hat_j))*du_j^2"
        ),
        latex_expression=(
            r"J(\Delta\mathbf{u}) = \sum_{j=1}^{N_p}\!\Big["
            r"q_{\mathrm{high}}\max\!\left(0,\hat{G}_{k+j}-G_{\mathrm{high}}\right)^2 "
            r"+ q_{\mathrm{low}}\max\!\left(0,G_{\mathrm{low}}-\hat{G}_{k+j}\right)^2 "
            r"+ q_{\mathrm{ref}}\!\left(\hat{G}_{k+j}-G_{\mathrm{target}}\right)^2\Big] "
            r"+ \sum_{j=0}^{N_u-1} R_0\!\left(1+\kappa_v\max(0,-\dot{\hat{G}}_{k+j})\right)\Delta u_j^2"
        ),
        solved_or_runtime_form=(
            "Solved via bounded L-BFGS-B over linear state-space prediction matrices "
            "anchored to adaptive Total Daily Dose (TDD)."
        ),
        state_variables=("x_hat", "G_hat_j", "du_opt", "TDD_hat"),
        parameters=("G_low", "G_high", "q_low", "q_high", "R_0", "kappa_v"),
        units="cost J dimensionless; du_j in U/step",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:WhiteBoxAIDPumpAlgorithm._compute_kalman_zone_mpc_law",),
        literature_basis=(
            "https://doi.org/10.1016/j.automatica.2016.04.015",
            "https://doi.org/10.2337/dc21-0172",
        ),
        validation_note="Observable-only state-space Zone-MPC with velocity-dependent insulin cost and TDD personalization.",
        evidence_class="canonical",
    ),
    FormulaSpec(
        formula_id="P06_OREF1_LOGARITHMIC_DYNISF_SMB_LAW",
        title="Logarithmic Dynamic ISF, UAM triangular forecast, and Super Micro Bolus (OpenAPS oref1 / AndroidAPS / Trio)",
        category="physiology",
        canonical_expression=(
            "ISF_dyn(G, TDD) = 1800 / (TDD * kappa_adj * ln(G / d_ins + 1)); "
            "u_SMB = min(du_req / 2, u_basal_hr * (T_max_smb_min / 60))"
        ),
        latex_expression=(
            r"\mathrm{ISF}_{\mathrm{dyn}}(G, \mathrm{TDD}) = "
            r"\frac{1800}{\mathrm{TDD}\cdot \kappa_{\mathrm{adj}}\cdot \ln\!\left(\frac{G}{d_{\mathrm{ins}}} + 1\right)}, \qquad "
            r"u_{\mathrm{SMB}} = \min\!\left(\frac{\Delta u_{\mathrm{req}}}{2},\; u_{\mathrm{basal}}\frac{T_{\max,\mathrm{SMB}}}{60}\right)"
        ),
        solved_or_runtime_form=(
            "Evaluates IOB, COB, UAM (Unannounced Meal), and ZT (Zero-Temp) curves using "
            "logarithmic Dynamic ISF and caps 5-minute SMB pulses."
        ),
        state_variables=("ISF_dyn", "G_ZT", "G_UAM", "u_SMB"),
        parameters=("TDD", "kappa_adj", "d_ins", "T_max_smb_min"),
        units="ISF_dyn in mg/dL/U; u_SMB in U",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:logarithmic_dynamic_isf",),
        literature_basis=(
            "https://openaps.readthedocs.io/en/latest/docs/While%20You%20Wait%20For%20Gear/understanding-insulin-on-board-calculations.html",
            "https://androidaps.readthedocs.io/en/latest/Usage/DynamicISF.html",
        ),
        validation_note="Exact logarithmic Dynamic ISF and oref1 SMB/UAM/ZT equations from AndroidAPS/Trio.",
        evidence_class="canonical",
    ),
    FormulaSpec(
        formula_id="P07_CAMBRIDGE_BAYESIAN_ADAPTIVE_ISF_LAW",
        title="Recursive Bayesian posterior adaptation of insulin sensitivity (Hovorka / CamAPS FX architecture)",
        category="physiology",
        canonical_expression=(
            "theta_post = (sigma_obs^2 * theta_prior + sigma_prior^2 * theta_lik) / (sigma_obs^2 + sigma_prior^2); "
            "ISF_eff = ISF_0 * theta_post"
        ),
        latex_expression=(
            r"\hat{\theta}_{k} = \frac{\sigma_{\mathrm{obs}}^2\,\hat{\theta}_{k-1} + \sigma_{\mathrm{prior}}^2\,\theta_{\mathrm{lik},k}}"
            r"{\sigma_{\mathrm{obs}}^2 + \sigma_{\mathrm{prior}}^2}, \qquad "
            r"\mathrm{ISF}_{\mathrm{eff},k} = \mathrm{ISF}_0 \cdot \hat{\theta}_{k}"
        ),
        solved_or_runtime_form=(
            "Conjugate Gaussian Bayesian update of the insulin sensitivity multiplier from "
            "the ratio of observed vs. expected glucose change."
        ),
        state_variables=("theta_post", "ISF_eff"),
        parameters=("sigma_obs", "sigma_prior", "ISF_0"),
        units="theta_post dimensionless multiplier; ISF_eff in mg/dL/U",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:bayesian_isf_multiplier_update",),
        literature_basis=(
            "https://doi.org/10.1088/0967-3334/25/4/010",
            "https://doi.org/10.1016/S0140-6736(10)60129-7",
        ),
        validation_note="Recursive Bayesian parameter adaptation inspired by the Cambridge Hovorka MPC.",
        evidence_class="adapted",
    ),
    FormulaSpec(
        formula_id="P08_BIONIC_PANCREAS_BODY_WEIGHT_AUTONOMOUS_LAW",
        title="Body-weight-initialized autonomous adaptive basal and PD/MPC dosing (El-Khatib / Damiano iLet style)",
        category="physiology",
        canonical_expression=(
            "TDD_0 = 0.55 * W_kg; u_basal_0 = 0.5 * TDD_0 / 24; ISF_0 = 1800 / TDD_0; "
            "u_bionic(t) = u_basal_adapt(t) + Kp*(G - G_target) + Kd*dG_dt"
        ),
        latex_expression=(
            r"\widehat{\mathrm{TDD}}_0 = 0.55\,W_{\mathrm{kg}}, \qquad "
            r"u_{\mathrm{basal},0} = \frac{0.5\,\widehat{\mathrm{TDD}}_0}{24}, \qquad "
            r"\mathrm{ISF}_0 = \frac{1800}{\widehat{\mathrm{TDD}}_0}"
        ),
        solved_or_runtime_form=(
            "Requires only body weight W_kg at startup and adapts basal infusion rate "
            "autonomously via rolling glycemic tracking error."
        ),
        state_variables=("TDD_hat", "u_basal_adapt"),
        parameters=("W_kg",),
        units="W_kg in kg; TDD_hat in U/day; u_basal_adapt in U/hr",
        implementation_paths=("src/iints/core/algorithms/whitebox_aid_pump.py:body_weight_autonomous_profile",),
        literature_basis=(
            "https://doi.org/10.1126/scitranslmed.3000673",
            "https://doi.org/10.1016/S0140-6736(16)32567-3",
        ),
        validation_note="Autonomous body-weight-initialized scaling and rolling basal adaptation.",
        evidence_class="canonical",
    ),
)


def biexponential_iob_fraction(
    elapsed_minutes: float,
    *,
    peak_minutes: float = 65.0,
    dia_minutes: float = 300.0,
) -> Tuple[float, float]:
    """Return `(iob_fraction, activity_per_min)` using the analytical Walsh/Loop bi-exponential kernel.

    For $t \\in [0, D]$ with peak time $t_p < D/2$:
    $$\\tau = \\frac{t_p(1 - t_p/D)}{1 - 2 t_p/D}, \\qquad a = \\frac{2\\tau}{D}, \\qquad
      S = \\frac{1}{1 - a + (1 + a)e^{-D/\\tau}}$$
    """

    if elapsed_minutes <= 0.0:
        return 1.0, 0.0
    dia = max(120.0, float(dia_minutes))
    if elapsed_minutes >= dia:
        return 0.0, 0.0
    tp = min(max(30.0, float(peak_minutes)), 0.45 * dia)
    tau = tp * (1.0 - tp / dia) / max(1e-6, 1.0 - 2.0 * tp / dia)
    a = 2.0 * tau / dia
    exp_d = math.exp(-dia / tau)
    scale = 1.0 / max(1e-9, 1.0 - a + (1.0 + a) * exp_d)

    t = float(elapsed_minutes)
    exp_t = math.exp(-t / tau)
    # Algebraically cancel (1 - a) to eliminate the removable 0/0 singularity
    # at a = 1 (i.e. tp / dia = 1 - 1/sqrt(2) ~= 0.292893).
    iob = 1.0 - scale * (
        ((t * t) / (tau * dia) + (a - 1.0) * (t / tau + 1.0)) * exp_t + (1.0 - a)
    )
    activity = scale * (t / (tau * tau)) * (1.0 - t / dia) * exp_t
    return max(0.0, min(1.0, iob)), max(0.0, activity)


def kovatchev_symmetrized_risk(
    glucose_mgdl: float,
    *,
    brake_aggressiveness: float = 0.08,
) -> Tuple[float, float, float, float]:
    """Return `(f_sym, rl_low_risk, rh_high_risk, brake_factor)` using Kovatchev's risk transformation.

    Published in Kovatchev et al. (*Diabetes Care* 1997) and Patek et al. (*JDST* 2012,
    UVA / TypeZero / Tandem Control-IQ Safety Supervision Module):
    $$f(G) = 1.509\\left((\\ln G)^{1.084} - 5.381\\right), \\qquad r_L(G) = 10\\min(0, f(G))^2$$
    """

    g_clamped = max(20.0, min(600.0, float(glucose_mgdl)))
    f_val = 1.509 * ((math.log(g_clamped) ** 1.084) - 5.381)
    rl = 10.0 * (f_val * f_val) if f_val < 0.0 else 0.0
    rh = 10.0 * (f_val * f_val) if f_val > 0.0 else 0.0
    brake_factor = 1.0 / (1.0 + max(0.0, float(brake_aggressiveness)) * rl)
    return f_val, rl, rh, max(0.0, min(1.0, brake_factor))


def logarithmic_dynamic_isf(
    glucose_mgdl: float,
    *,
    tdd_units: float = 40.0,
    adjustment_factor: float = 0.85,
    insulin_divisor: float = 65.0,
    nominal_isf: float = 50.0,
) -> float:
    """Compute real-time Logarithmic Dynamic ISF (AndroidAPS / Trio / Wilson DynISF formula).

    $$\\mathrm{ISF}_{\\mathrm{dyn}}(G) = \\frac{1800}{\\mathrm{TDD} \\cdot \\kappa_{\\mathrm{adj}} \\cdot \\ln\\!\\left(\\frac{G}{d_{\\mathrm{ins}}} + 1\\right)}$$
    Clamped to $[0.35\\,\\mathrm{ISF}_0,\\; 2.20\\,\\mathrm{ISF}_0]$ for numerical and physiological safety.
    """

    g_safe = max(40.0, float(glucose_mgdl))
    tdd_safe = max(8.0, float(tdd_units))
    adj = max(0.2, float(adjustment_factor))
    div = max(15.0, float(insulin_divisor))
    denom = tdd_safe * adj * math.log((g_safe / div) + 1.0)
    raw_isf = 1800.0 / max(1e-3, denom)
    nom = max(5.0, float(nominal_isf))
    return max(0.35 * nom, min(2.20 * nom, raw_isf))


def bayesian_isf_multiplier_update(
    prior_multiplier: float,
    *,
    glucose_residual_mgdl_min: float,
    expected_insulin_drop_mgdl_min: float,
    sigma_prior: float = 0.12,
    sigma_obs: float = 0.45,
) -> float:
    """Conjugate Gaussian Bayesian update for time-varying insulin sensitivity $\\hat{\\theta}_k$ (CamAPS FX style).

    When glucose falls slower than expected ($r_k > 0$), sensitivity multiplier $\\hat{\\theta}_k < 1$
    (indicating higher insulin resistance / lower effective ISF). When glucose falls faster than
    expected ($r_k < 0$), $\\hat{\\theta}_k > 1$.
    """

    prior = max(0.45, min(1.85, float(prior_multiplier)))
    # Implied sensitivity ratio from observed vs predicted insulin drop
    denom = max(0.15, abs(float(expected_insulin_drop_mgdl_min)) + 0.25)
    likelihood_theta = max(
        0.45,
        min(1.85, prior - 0.35 * (float(glucose_residual_mgdl_min) / denom)),
    )
    var_p = max(1e-6, float(sigma_prior) ** 2)
    var_o = max(1e-6, float(sigma_obs) ** 2)
    posterior = (var_o * prior + var_p * likelihood_theta) / (var_o + var_p)
    # Gentle reversion toward 1.0 prior mean
    reverted = 0.96 * posterior + 0.04 * 1.0
    return max(0.50, min(1.75, reverted))


def body_weight_autonomous_profile(body_weight_kg: float) -> Dict[str, float]:
    """Initialize TDD, basal rate, ISF, and ICR solely from body weight $W$ (El-Khatib / Damiano Bionic Pancreas)."""

    w_kg = max(15.0, min(250.0, float(body_weight_kg)))
    tdd_u = 0.55 * w_kg
    basal_u_hr = (0.50 * tdd_u) / 24.0
    isf_mgdl_u = 1800.0 / tdd_u
    icr_g_u = 500.0 / tdd_u
    return {
        "body_weight_kg": w_kg,
        "estimated_tdd_units": tdd_u,
        "basal_rate_u_per_hr": basal_u_hr,
        "isf_mgdl_per_u": isf_mgdl_u,
        "icr_g_per_u": icr_g_u,
    }


@dataclass
class PumpObserverSnapshot:
    """Observable-only mathematical state reconstructed by the pump at step $k$."""

    time_minutes: float
    cgm_glucose_mgdl: float
    filtered_glucose_mgdl: float
    glucose_velocity_mgdl_min: float
    remote_insulin_action_mgdl_min: float
    iob_units: float
    excess_iob_units: float
    insulin_activity_u_min: float
    cob_grams: float
    subcut_insulin_norm_u_hr: float
    plasma_insulin_norm_u_hr: float
    glucose_residual_mgdl_min: float
    retrospective_correction_mgdl_min: float
    kalman_innovation_mgdl: float
    dynamic_isf_mgdl_u: float
    bayesian_isf_multiplier: float
    bayesian_isf_mgdl_u: float
    adaptive_tdd_units: float
    kovatchev_lbgi_risk: float
    kovatchev_hbgi_risk: float
    kovatchev_brake_factor: float
    uam_detected: bool


class WhiteBoxPumpStateEstimator:
    """Reconstructs PK/PD, COB, residuals, Bayesian ISF, and Kalman states strictly from pump observables."""

    def __init__(
        self,
        *,
        target_glucose: float = 110.0,
        isf: float = 50.0,
        icr: float = 10.0,
        basal_rate_u_hr: float = 0.8,
        dia_minutes: float = 300.0,
        peak_insulin_minutes: float = 65.0,
        carb_absorption_minutes: float = 180.0,
        tau_sc_minutes: float = 50.0,
        tau_p_minutes: float = 65.0,
    ) -> None:
        self.target_glucose = float(target_glucose)
        self.isf = max(5.0, float(isf))
        self.icr = max(1.0, float(icr))
        self.basal_rate_u_hr = max(0.0, float(basal_rate_u_hr))
        self.dia_minutes = max(120.0, float(dia_minutes))
        self.peak_insulin_minutes = max(30.0, float(peak_insulin_minutes))
        self.carb_absorption_minutes = max(60.0, float(carb_absorption_minutes))
        self.tau_sc_minutes = max(15.0, float(tau_sc_minutes))
        self.tau_p_minutes = max(20.0, float(tau_p_minutes))
        self.reset()

    def reset(self) -> None:
        self.elapsed_minutes: float = 0.0
        self.initialized: bool = False
        # Delivered insulin history: list of (time_min, total_units, excess_units_over_basal)
        self.insulin_history: List[Tuple[float, float, float]] = []
        # Announced meal history: list of [time_min, grams_remaining, initial_grams]
        self.carb_history: List[List[float]] = []
        # Two-depot normalized insulin rates (in U/hr equivalent so steady-state basal = u_basal)
        self.i_sc_u_hr: float = self.basal_rate_u_hr
        self.i_p_u_hr: float = self.basal_rate_u_hr
        # Kalman state vector: x_hat = [dG (mg/dL), dG/dt (mg/dL/min), X_action (mg/dL/min)]
        self.x_hat = np.zeros(3, dtype=float)
        self.retrospective_residual_mgdl_min: float = 0.0
        self.bayesian_isf_multiplier: float = 1.0
        self.adaptive_tdd_units: float = max(10.0, 2.0 * 24.0 * max(0.3, self.basal_rate_u_hr))
        self.last_delivered_units: float = 0.0
        self.last_announced_carbs: float = 0.0

    def state_space_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return discrete-time $(A, B_u, B_d, C)$ for sample interval `dt` (minutes)."""

        dt_val = max(0.5, float(dt))
        alpha_v = math.exp(-dt_val / 25.0)
        alpha_x = math.exp(-dt_val / 55.0)
        a_mat = np.array(
            [
                [1.0, dt_val, -dt_val],
                [0.0, alpha_v, -0.18 * (1.0 - alpha_v)],
                [0.0, 0.0, alpha_x],
            ],
            dtype=float,
        )
        gain_u = (self.isf / 65.0) * (1.0 - alpha_x)
        b_u = np.array([0.0, -0.08 * gain_u * dt_val, gain_u], dtype=float)
        csf = self.isf / self.icr
        gain_d = (csf / self.carb_absorption_minutes) * (1.0 - alpha_v)
        b_d = np.array([0.5 * gain_d * dt_val, gain_d, 0.0], dtype=float)
        c_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        return a_mat, b_u, b_d, c_vec

    def advance_pk_compartments(self, u_rate_u_hr: float, dt: float) -> Tuple[float, float]:
        """Exact matrix-exponential integration of $(\\hat{I}_{\\mathrm{sc}}, \\hat{I}_{\\mathrm{p}})$ over `dt`."""

        dt_val = max(1e-6, float(dt))
        k1 = 1.0 / self.tau_sc_minutes
        k2 = 1.0 / self.tau_p_minutes
        e1 = math.exp(-k1 * dt_val)
        e2 = math.exp(-k2 * dt_val)

        i_sc_prev = self.i_sc_u_hr
        i_p_prev = self.i_p_u_hr
        i_sc_next = i_sc_prev * e1 + u_rate_u_hr * (1.0 - e1)

        if abs(k1 - k2) < 1e-6:
            homogeneous = (i_p_prev + k2 * dt_val * i_sc_prev) * e2
            particular = u_rate_u_hr * (1.0 - (1.0 + k2 * dt_val) * e2)
        else:
            homogeneous = i_p_prev * e2 + (k2 / (k2 - k1)) * i_sc_prev * (e1 - e2)
            particular = u_rate_u_hr * (
                1.0 - (k2 * e1 - k1 * e2) / (k2 - k1)
            )
        i_p_next = max(0.0, homogeneous + particular)
        self.i_sc_u_hr = max(0.0, i_sc_next)
        self.i_p_u_hr = i_p_next
        return self.i_sc_u_hr, self.i_p_u_hr

    def record_delivery(self, delivered_units: float, dt: float) -> None:
        """Record the actual insulin dose delivered at the end of step $k$ and update adaptive TDD."""

        dt_val = max(0.5, float(dt))
        total_u = max(0.0, float(delivered_units))
        basal_step_u = (self.basal_rate_u_hr / 60.0) * dt_val
        excess_u = total_u - basal_step_u
        self.insulin_history.append((self.elapsed_minutes, total_u, excess_u))
        cutoff = self.elapsed_minutes - self.dia_minutes - 5.0
        self.insulin_history = [item for item in self.insulin_history if item[0] >= cutoff]
        self.advance_pk_compartments((total_u / dt_val) * 60.0, dt_val)
        # Omnipod 5 / Horizon style rolling TDD EWMA update
        instantaneous_daily_u = (total_u / dt_val) * 1440.0
        alpha_tdd = min(0.05, dt_val / 1440.0)
        self.adaptive_tdd_units = max(
            8.0,
            min(180.0, (1.0 - alpha_tdd) * self.adaptive_tdd_units + alpha_tdd * instantaneous_daily_u),
        )
        self.last_delivered_units = total_u

    def compute_iob_and_activity(self, future_offset_min: float = 0.0) -> Tuple[float, float, float]:
        """Compute `(total_iob_u, excess_iob_u, excess_activity_u_min)` at $t + \\tau$."""

        query_time = self.elapsed_minutes + max(0.0, float(future_offset_min))
        total_iob = 0.0
        excess_iob = 0.0
        excess_activity = 0.0
        for t_deliv, total_u, excess_u in self.insulin_history:
            age = query_time - t_deliv
            if 0.0 <= age < self.dia_minutes:
                iob_frac, act_rate = biexponential_iob_fraction(
                    age,
                    peak_minutes=self.peak_insulin_minutes,
                    dia_minutes=self.dia_minutes,
                )
                if total_u > 0.0:
                    total_iob += total_u * iob_frac
                excess_iob += excess_u * iob_frac
                excess_activity += excess_u * act_rate
        return max(0.0, total_iob), excess_iob, excess_activity

    def observe(self, data: AlgorithmInput) -> PumpObserverSnapshot:
        """Assimilate a new CGM sample and update the observable-only pump state."""

        dt = max(0.5, float(data.time_step or 5.0))
        if data.isf is not None and float(data.isf) > 0:
            self.isf = float(data.isf)
        if data.icr is not None and float(data.icr) > 0:
            self.icr = float(data.icr)
        if data.basal_rate_u_per_hr is not None and float(data.basal_rate_u_per_hr) >= 0:
            self.basal_rate_u_hr = float(data.basal_rate_u_per_hr)
        if data.dia_minutes is not None and float(data.dia_minutes) >= 120.0:
            self.dia_minutes = float(data.dia_minutes)

        cgm = float(data.current_glucose)
        if not self.initialized:
            self.x_hat = np.array([cgm - self.target_glucose, 0.0, 0.0], dtype=float)
            self.i_sc_u_hr = self.basal_rate_u_hr
            self.i_p_u_hr = self.basal_rate_u_hr
            self.adaptive_tdd_units = max(10.0, 2.0 * 24.0 * max(0.3, self.basal_rate_u_hr))
            self.initialized = True
        else:
            self.elapsed_minutes += dt

        if float(data.carb_intake or 0.0) > 0.0:
            carbs_g = float(data.carb_intake)
            self.carb_history.append([self.elapsed_minutes, carbs_g, carbs_g])
            self.last_announced_carbs = carbs_g
        else:
            self.last_announced_carbs = 0.0

        a_mat, b_u, b_d, c_vec = self.state_space_matrices(dt)
        basal_step_u = (self.basal_rate_u_hr / 60.0) * dt
        prev_excess_u = self.last_delivered_units - basal_step_u
        x_pred = a_mat @ self.x_hat + b_u * prev_excess_u + b_d * self.last_announced_carbs

        y_meas = cgm - self.target_glucose
        innovation = float(y_meas - float(c_vec @ x_pred))
        kalman_gain = np.array([0.68, 0.16 / max(1.0, dt / 5.0), -0.03], dtype=float)
        self.x_hat = x_pred + kalman_gain * innovation

        if data.glucose_trend_mgdl_min is not None:
            self.x_hat[1] = 0.65 * float(self.x_hat[1]) + 0.35 * float(data.glucose_trend_mgdl_min)

        calc_iob, excess_iob, excess_act = self.compute_iob_and_activity(0.0)
        effective_iob = max(calc_iob, float(data.insulin_on_board or 0.0))
        if effective_iob > calc_iob and calc_iob < 1e-6:
            excess_iob = effective_iob
            excess_act = effective_iob / self.dia_minutes

        expected_insulin_vel = -self.isf * excess_act
        residual = float(self.x_hat[1]) - expected_insulin_vel
        self.retrospective_residual_mgdl_min = (
            0.70 * self.retrospective_residual_mgdl_min + 0.30 * residual
        )

        # Bayesian ISF posterior update (CamAPS FX style)
        self.bayesian_isf_multiplier = bayesian_isf_multiplier_update(
            self.bayesian_isf_multiplier,
            glucose_residual_mgdl_min=residual,
            expected_insulin_drop_mgdl_min=expected_insulin_vel,
        )
        bayesian_isf = self.isf * self.bayesian_isf_multiplier

        # Logarithmic Dynamic ISF (AndroidAPS / Trio / Wilson DynISF)
        filtered_g = float(self.x_hat[0] + self.target_glucose)
        dyn_isf = logarithmic_dynamic_isf(
            filtered_g,
            tdd_units=self.adaptive_tdd_units,
            nominal_isf=self.isf,
        )

        # Kovatchev symmetrized risk & brake factor
        _, rl_risk, rh_risk, brake_factor = kovatchev_symmetrized_risk(filtered_g)

        # Dynamic carbohydrate absorption
        csf = max(1e-3, self.isf / self.icr)
        min_carb_rate_g_min = 0.12
        dynamic_carb_rate_g_min = max(
            min_carb_rate_g_min,
            max(0.0, residual) / csf,
        )
        remaining_carbs_total = 0.0
        updated_carbs: List[List[float]] = []
        for entry in self.carb_history:
            t_meal, rem_g, init_g = entry
            if self.elapsed_minutes > t_meal:
                base_rate = init_g / self.carb_absorption_minutes
                absorbed = max(base_rate, dynamic_carb_rate_g_min) * dt
                rem_g = max(0.0, rem_g - absorbed)
            if rem_g > 0.1 and (self.elapsed_minutes - t_meal) <= self.carb_absorption_minutes * 1.5:
                updated_carbs.append([t_meal, rem_g, init_g])
                remaining_carbs_total += rem_g
        self.carb_history = updated_carbs
        effective_cob = max(remaining_carbs_total, float(data.carbs_on_board or 0.0))

        # Unannounced Meal (UAM) detection when residual or velocity is positive without COB
        uam_detected = bool(
            (self.retrospective_residual_mgdl_min > 0.45 or float(self.x_hat[1]) > 0.8)
            and effective_cob < 1.0
        )

        return PumpObserverSnapshot(
            time_minutes=self.elapsed_minutes,
            cgm_glucose_mgdl=cgm,
            filtered_glucose_mgdl=filtered_g,
            glucose_velocity_mgdl_min=float(self.x_hat[1]),
            remote_insulin_action_mgdl_min=float(self.x_hat[2]),
            iob_units=effective_iob,
            excess_iob_units=excess_iob,
            insulin_activity_u_min=excess_act,
            cob_grams=effective_cob,
            subcut_insulin_norm_u_hr=self.i_sc_u_hr,
            plasma_insulin_norm_u_hr=self.i_p_u_hr,
            glucose_residual_mgdl_min=residual,
            retrospective_correction_mgdl_min=self.retrospective_residual_mgdl_min,
            kalman_innovation_mgdl=innovation,
            dynamic_isf_mgdl_u=dyn_isf,
            bayesian_isf_multiplier=self.bayesian_isf_multiplier,
            bayesian_isf_mgdl_u=bayesian_isf,
            adaptive_tdd_units=self.adaptive_tdd_units,
            kovatchev_lbgi_risk=rl_risk,
            kovatchev_hbgi_risk=rh_risk,
            kovatchev_brake_factor=brake_factor,
            uam_detected=uam_detected,
        )


class WhiteBoxAIDPumpAlgorithm(InsulinAlgorithm):
    """Universal White-Box Mathematical AID Pump Algorithm.

    Supports all six canonical mathematical pump control laws (`pid_ifb`,
    `trajectory_impulse`, `kalman_zone_mpc`, `oref1_smb`, `camaps_bayesian_mpc`,
    `bionic_autonomous`) as well as `unified` mode.
    """

    DEFAULT_SETTINGS: Dict[str, Any] = {
        "control_law": "unified",
        "target_glucose": 110.0,
        "isf": 50.0,
        "icr": 10.0,
        "basal_rate_u_per_hr": 0.80,
        "body_weight_kg": 70.0,
        "dia_minutes": 300.0,
        "peak_insulin_minutes": 65.0,
        "carb_absorption_minutes": 180.0,
        # Law 1: Steil PID-IFB parameters (Medtronic 670G/780G style)
        "kp_u_hr_per_mgdl": 0.022,
        "ti_minutes": 150.0,
        "td_minutes": 32.0,
        "gamma_sc": 0.35,
        "gamma_p": 0.25,
        # Law 2: Trajectory-Impulse & Kovatchev Brakes (Control-IQ / TypeZero style)
        "auto_correction_threshold_mgdl": 160.0,
        "auto_correction_target_mgdl": 110.0,
        "auto_correction_factor": 0.60,
        "auto_correction_cooldown_minutes": 60.0,
        "hypo_suspend_threshold_mgdl": 75.0,
        "hypo_guard_threshold_mgdl": 90.0,
        # Law 3: Kalman State-Space Zone-MPC (Omnipod 5 / Doyle style)
        "zone_low_mgdl": 95.0,
        "zone_high_mgdl": 135.0,
        "mpc_horizon_steps": 12,
        "mpc_control_steps": 6,
        "q_low": 45.0,
        "q_high": 2.2,
        "q_target": 0.25,
        "r_insulin": 35.0,
        "velocity_penalty_kappa": 2.5,
        # Law 4: OpenAPS oref1 / AndroidAPS / Trio SMB & DynISF parameters
        "max_smb_basal_minutes": 30.0,
        "dyn_isf_adjustment_factor": 0.85,
        "dyn_isf_insulin_divisor": 65.0,
        # Pump hardware / safety envelope
        "max_basal_multiplier": 4.0,
        "max_bolus_u_per_step": 6.0,
        "max_total_u_per_step": 8.0,
        "iob_ceiling_units": 5.5,
    }

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**self.DEFAULT_SETTINGS, **(settings or {})}
        super().__init__(merged)
        self.control_law: ControlLawType = str(merged.get("control_law", "unified"))  # type: ignore[assignment]
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box AID Pump (Unified)",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference=(
                    "Steil 2006; Kovatchev 1997/Breton 2020; Gondhalekar 2016; "
                    "Hovorka 2004; El-Khatib 2010; OpenAPS oref1"
                ),
                description=(
                    "Observable-only white-box mathematical model of an AID insulin pump "
                    "unifying PID-IFB, Control-IQ Trajectory+Brakes, Omnipod Zone-MPC, "
                    "oref1 DynISF/SMB, CamAPS Bayesian MPC, and Bionic Pancreas."
                ),
                algorithm_type="hybrid",
            )
        )
        self._init_mathematical_state()

    def _init_mathematical_state(self) -> None:
        self.observer = WhiteBoxPumpStateEstimator(
            target_glucose=float(self.settings["target_glucose"]),
            isf=float(self.settings["isf"]),
            icr=float(self.settings["icr"]),
            basal_rate_u_hr=float(self.settings["basal_rate_u_per_hr"]),
            dia_minutes=float(self.settings["dia_minutes"]),
            peak_insulin_minutes=float(self.settings["peak_insulin_minutes"]),
            carb_absorption_minutes=float(self.settings["carb_absorption_minutes"]),
        )
        self.pid_integral_u_hr: float = 0.0
        self.prev_error_mgdl: float = 0.0
        self.last_auto_correction_time_min: float = -1e6
        bionic = body_weight_autonomous_profile(float(self.settings["body_weight_kg"]))
        self.bionic_adaptive_basal_u_hr: float = float(bionic["basal_rate_u_per_hr"])

    def reset(self) -> None:
        super().reset()
        self._init_mathematical_state()

    def _compute_pid_ifb_law(
        self,
        snap: PumpObserverSnapshot,
        dt: float,
    ) -> Dict[str, float]:
        """Law 1: Steil et al. PID with Two-Compartment Pharmacokinetic Insulin Feedback (Medtronic 670G/780G)."""

        target = float(self.settings["target_glucose"])
        kp = float(self.settings["kp_u_hr_per_mgdl"])
        ti = max(30.0, float(self.settings["ti_minutes"]))
        td = max(0.0, float(self.settings["td_minutes"]))
        gamma_1 = max(0.0, float(self.settings["gamma_sc"]))
        gamma_2 = max(0.0, float(self.settings["gamma_p"]))

        error = snap.filtered_glucose_mgdl - target
        ki = kp / ti
        kd = kp * td

        asymmetry = 1.0 if error >= 0.0 else 2.2
        mean_error = 0.5 * (error + self.prev_error_mgdl)
        max_int_u_hr = 1.8 * self.observer.basal_rate_u_hr
        self.pid_integral_u_hr = max(
            -self.observer.basal_rate_u_hr,
            min(
                self.pid_integral_u_hr + ki * mean_error * dt * asymmetry,
                max_int_u_hr,
            ),
        )
        self.prev_error_mgdl = error

        u_pid_u_hr = (
            self.observer.basal_rate_u_hr
            + kp * error
            + self.pid_integral_u_hr
            + kd * snap.glucose_velocity_mgdl_min
        )
        u_ifb_u_hr = (
            (1.0 + gamma_1 + gamma_2) * u_pid_u_hr
            - gamma_1 * snap.subcut_insulin_norm_u_hr
            - gamma_2 * snap.plasma_insulin_norm_u_hr
        )
        max_rate_u_hr = float(self.settings["max_basal_multiplier"]) * max(
            0.4, self.observer.basal_rate_u_hr
        )
        clamped_rate_u_hr = max(0.0, min(u_ifb_u_hr, max_rate_u_hr))
        basal_step_u = (clamped_rate_u_hr / 60.0) * dt

        # 780G SmartGuard style every-5-min auto-correction pulse when basal is saturated
        auto_corr_u = 0.0
        auto_thresh = float(self.settings["auto_correction_threshold_mgdl"])
        iob_ceiling = float(self.settings["iob_ceiling_units"])
        if (
            snap.filtered_glucose_mgdl >= auto_thresh
            and snap.glucose_velocity_mgdl_min >= -0.3
            and snap.iob_units < 0.75 * iob_ceiling
        ):
            raw_deficit = (snap.filtered_glucose_mgdl - target) / self.observer.isf - snap.excess_iob_units
            if raw_deficit > 0.0:
                auto_corr_u = min(0.35 * raw_deficit, 0.8)

        return {
            "basal_units": basal_step_u,
            "auto_correction_units": auto_corr_u,
            "total_Before_Meal_Units": basal_step_u + auto_corr_u,
            "u_pid_u_hr": u_pid_u_hr,
            "u_ifb_u_hr": clamped_rate_u_hr,
        }

    def _compute_trajectory_impulse_law(
        self,
        snap: PumpObserverSnapshot,
        dt: float,
    ) -> Dict[str, Any]:
        """Law 2: 4-Curve Analytical Trajectory Superposition + Kovatchev LBGI Brakes (Control-IQ / TypeZero)."""

        target = float(self.settings["target_glucose"])
        dia = self.observer.dia_minutes
        isf = self.observer.isf
        icr = self.observer.icr
        csf = isf / icr

        tau_mom = 18.0
        tau_rc = 45.0
        carb_abs_rate_g_min = max(0.15, snap.cob_grams / 120.0)

        horizons = list(range(5, int(dia) + 5, 5))
        trajectory: List[float] = []
        iob_trajectory: List[float] = []
        zt_trajectory: List[float] = []
        uam_trajectory: List[float] = []

        basal_u_min = self.observer.basal_rate_u_hr / 60.0
        cum_missing_basal_effect_u = 0.0

        for tau in horizons:
            _, excess_iob_tau, _ = self.observer.compute_iob_and_activity(float(tau))
            iob_frac_tau, _ = biexponential_iob_fraction(
                float(tau),
                peak_minutes=self.observer.peak_insulin_minutes,
                dia_minutes=dia,
            )
            # Convolution integral of withheld basal insulin over [0, tau]:
            # int_0^tau (1 - S_IOB(s)) ds * u_basal_min
            cum_missing_basal_effect_u += basal_u_min * (1.0 - iob_frac_tau) * 5.0

            # 1. Insulin impulse decay curve (excess over scheduled basal)
            dg_iob = -isf * (snap.excess_iob_units - excess_iob_tau)
            # Zero-Temp (ZT) curve: existing excess IOB decays, PLUS future basal is withheld (u=0)
            dg_zt = dg_iob + isf * cum_missing_basal_effect_u
            # 2. Carbohydrate appearance curve
            absorbed_g = min(snap.cob_grams, carb_abs_rate_g_min * float(tau))
            dg_cob = csf * absorbed_g
            # 3. Short-term velocity momentum curve
            dg_mom = snap.glucose_velocity_mgdl_min * tau_mom * (1.0 - math.exp(-float(tau) / tau_mom))
            # 4. Retrospective / UAM glucose-residual curve
            dg_rc = (
                snap.retrospective_correction_mgdl_min
                * tau_rc
                * (1.0 - math.exp(-float(tau) / tau_rc))
            )
            # Triangular UAM decay over 120 minutes when UAM is active
            uam_slope = max(0.0, snap.retrospective_correction_mgdl_min)
            tau_uam = min(float(tau), 120.0)
            dg_uam = uam_slope * (tau_uam - (tau_uam * tau_uam) / 240.0)

            g_iob_only = max(35.0, min(450.0, snap.filtered_glucose_mgdl + dg_iob + dg_mom))
            g_tau = max(35.0, min(450.0, snap.filtered_glucose_mgdl + dg_iob + dg_cob + dg_mom + dg_rc))
            g_zt = max(35.0, min(450.0, snap.filtered_glucose_mgdl + dg_zt + 0.5 * dg_mom))
            g_uam = max(35.0, min(450.0, snap.filtered_glucose_mgdl + dg_iob + dg_uam + dg_mom))
            iob_trajectory.append(g_iob_only)
            trajectory.append(g_tau)
            zt_trajectory.append(g_zt)
            uam_trajectory.append(g_uam)

        g_pred_30 = trajectory[min(5, len(trajectory) - 1)]
        g_eventual = trajectory[-1]
        guard_slice = trajectory[2:] if len(trajectory) > 2 else trajectory
        g_min_guard = min(guard_slice)
        g_zt_min = min(zt_trajectory[:12]) if zt_trajectory else g_min_guard
        g_uam_eventual = uam_trajectory[-1] if uam_trajectory else g_eventual

        # Kovatchev LBGI safety brake evaluated at G_min_guard
        _, rl_guard, rh_pred30, brake_phi = kovatchev_symmetrized_risk(g_min_guard)

        effective_future_g = (
            g_min_guard
            if g_min_guard < float(self.settings["hypo_guard_threshold_mgdl"])
            else 0.35 * g_min_guard + 0.65 * g_eventual
        )
        insulin_req_u = (effective_future_g - target) / isf

        scheduled_basal_u_hr = self.observer.basal_rate_u_hr
        tbr_u_hr = (scheduled_basal_u_hr + (insulin_req_u / 1.5)) * brake_phi
        max_rate_u_hr = float(self.settings["max_basal_multiplier"]) * max(0.4, scheduled_basal_u_hr)
        clamped_tbr_u_hr = max(0.0, min(tbr_u_hr, max_rate_u_hr))
        basal_step_u = (clamped_tbr_u_hr / 60.0) * dt

        # Control-IQ 60% auto-correction bolus (at most once per cooldown window)
        auto_corr_u = 0.0
        cooldown_min = float(self.settings["auto_correction_cooldown_minutes"])
        cooldown_elapsed = (snap.time_minutes - self.last_auto_correction_time_min) >= cooldown_min - 1e-6
        if (
            g_pred_30 >= float(self.settings["auto_correction_threshold_mgdl"])
            and g_min_guard >= float(self.settings["hypo_guard_threshold_mgdl"])
            and snap.iob_units < float(self.settings["iob_ceiling_units"])
            and cooldown_elapsed
        ):
            net_corr_req = (
                (g_pred_30 - float(self.settings["auto_correction_target_mgdl"])) / isf
                - max(0.0, snap.excess_iob_units)
            )
            if net_corr_req > 0.0:
                auto_corr_u = min(
                    float(self.settings["auto_correction_factor"]) * net_corr_req * (dt / 30.0),
                    1.2,
                )

        return {
            "basal_units": basal_step_u,
            "auto_correction_units": auto_corr_u,
            "total_Before_Meal_Units": basal_step_u + auto_corr_u,
            "g_pred_30_mgdl": g_pred_30,
            "g_eventual_mgdl": g_eventual,
            "g_min_guard_mgdl": g_min_guard,
            "g_zt_min_mgdl": g_zt_min,
            "g_uam_eventual_mgdl": g_uam_eventual,
            "kovatchev_lbgi_guard": rl_guard,
            "kovatchev_hbgi_pred30": rh_pred30,
            "kovatchev_brake_factor": brake_phi,
            "trajectory_preview_mgdl": [round(val, 2) for val in trajectory[:12]],
        }

    def _compute_kalman_zone_mpc_law(
        self,
        snap: PumpObserverSnapshot,
        dt: float,
        *,
        isf_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """Law 3: Observable-Only State-Space Asymmetric Zone-MPC with Velocity Penalty (Omnipod 5 / Doyle)."""

        n_p = max(4, int(self.settings["mpc_horizon_steps"]))
        n_u = max(2, min(n_p, int(self.settings["mpc_control_steps"])))
        target = float(self.settings["target_glucose"])
        g_low = float(self.settings["zone_low_mgdl"])
        g_high = float(self.settings["zone_high_mgdl"])
        q_low = float(self.settings["q_low"])
        q_high = float(self.settings["q_high"])
        q_target = float(self.settings["q_target"])
        r_base = float(self.settings["r_insulin"])
        kappa_v = float(self.settings["velocity_penalty_kappa"])

        saved_isf = self.observer.isf
        if isf_override is not None and isf_override > 0.0:
            self.observer.isf = float(isf_override)
        try:
            a_mat, b_u, b_d, c_vec = self.observer.state_space_matrices(dt)
        finally:
            self.observer.isf = saved_isf

        basal_step_u = (self.observer.basal_rate_u_hr / 60.0) * dt
        max_step_u = min(
            float(self.settings["max_total_u_per_step"]),
            max(0.4, float(self.settings["max_basal_multiplier"]) * basal_step_u + 0.6),
        )
        min_du = -basal_step_u
        max_du = max(0.0, max_step_u - basal_step_u)

        step_carb_g = min(snap.cob_grams / max(1.0, float(n_p)), 1.2 * dt)
        x0 = self.observer.x_hat.copy()

        def objective(du_vec: np.ndarray) -> float:
            x = x0.copy()
            total_cost = 0.0
            for j in range(n_p):
                du_j = float(du_vec[j]) if j < n_u else 0.0
                x = a_mat @ x + b_u * du_j + b_d * step_carb_g
                g_j = float(c_vec @ x) + target
                vel_j = float(x[1])
                if g_j < g_low:
                    # Asymmetric exponential hypo penalty (Gondhalekar et al.)
                    hypo_depth = min(40.0, g_low - g_j)
                    total_cost += q_low * (hypo_depth ** 2) * math.exp(0.04 * hypo_depth)
                elif g_j > g_high:
                    total_cost += q_high * ((g_j - g_high) ** 2)
                total_cost += q_target * ((g_j - target) ** 2)
                if j < n_u:
                    vel_factor = (
                        1.0 + kappa_v * max(0.0, -vel_j)
                        if du_j > 0.0
                        else 0.25
                    )
                    total_cost += r_base * vel_factor * (du_j ** 2)
            return total_cost

        init_du = np.zeros(n_u, dtype=float)
        bounds = [(min_du, max_du) for _ in range(n_u)]
        res = minimize(
            objective,
            init_du,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 25},
        )
        du_opt = float(res.x[0]) if res.success else 0.0
        total_pre_meal_u = max(0.0, basal_step_u + du_opt)
        basal_delivered_u = min(total_pre_meal_u, basal_step_u)
        corr_delivered_u = max(0.0, total_pre_meal_u - basal_delivered_u)
        return {
            "basal_units": basal_delivered_u,
            "auto_correction_units": corr_delivered_u,
            "total_Before_Meal_Units": total_pre_meal_u,
            "du_optimal_units": du_opt,
            "mpc_objective_cost": float(res.fun) if res.success else 0.0,
        }

    def _compute_oref1_smb_law(
        self,
        snap: PumpObserverSnapshot,
        traj_res: Dict[str, Any],
        dt: float,
    ) -> Dict[str, float]:
        """Law 4: OpenAPS `oref1` / AndroidAPS / Trio with Logarithmic Dynamic ISF, UAM, and SMB."""

        target = float(self.settings["target_glucose"])
        dyn_isf = snap.dynamic_isf_mgdl_u
        g_min_guard = float(traj_res["g_min_guard_mgdl"])
        g_ev = (
            float(traj_res["g_uam_eventual_mgdl"])
            if snap.uam_detected
            else float(traj_res["g_eventual_mgdl"])
        )
        g_zt_min = float(traj_res["g_zt_min_mgdl"])

        effective_g = min(g_ev, max(g_min_guard, g_zt_min))
        insulin_req_u = (effective_g - target) / max(5.0, dyn_isf)

        scheduled_basal_u_hr = self.observer.basal_rate_u_hr
        max_smb_minutes = float(self.settings["max_smb_basal_minutes"])
        max_smb_u = scheduled_basal_u_hr * (max_smb_minutes / 60.0)

        # When SMB is triggered (positive insulin_req and safe ZT/guard), oref1 doses an SMB
        # capped at min(insulin_req / 2, max_smb_u) and sets a low/zero temporary basal (ZT guard)
        if (
            insulin_req_u > 0.05
            and g_min_guard >= float(self.settings["hypo_guard_threshold_mgdl"])
            and snap.iob_units < float(self.settings["iob_ceiling_units"])
        ):
            smb_u = min(0.5 * insulin_req_u, max_smb_u)
            # Zero-temp / reduced basal pairing during active SMB
            basal_u = 0.25 * (scheduled_basal_u_hr / 60.0) * dt
        else:
            smb_u = 0.0
            tbr_u_hr = max(0.0, scheduled_basal_u_hr + (insulin_req_u / 1.5))
            max_rate_u_hr = float(self.settings["max_basal_multiplier"]) * max(0.4, scheduled_basal_u_hr)
            basal_u = (min(tbr_u_hr, max_rate_u_hr) / 60.0) * dt

        return {
            "basal_units": basal_u,
            "auto_correction_units": smb_u,
            "total_Before_Meal_Units": basal_u + smb_u,
            "dynamic_isf_mgdl_u": dyn_isf,
            "insulin_req_units": insulin_req_u,
        }

    def _compute_bionic_autonomous_law(
        self,
        snap: PumpObserverSnapshot,
        dt: float,
    ) -> Dict[str, float]:
        """Law 6: El-Khatib / Damiano Body-Weight Autonomous Bionic Pancreas."""

        target = float(self.settings["target_glucose"])
        profile = body_weight_autonomous_profile(float(self.settings["body_weight_kg"]))
        isf_bionic = float(profile["isf_mgdl_per_u"])
        error = snap.filtered_glucose_mgdl - target

        # Autonomous rolling adaptation of reference basal rate
        adapt_gain = 0.00015 * (dt / 5.0)
        self.bionic_adaptive_basal_u_hr = max(
            0.35 * float(profile["basal_rate_u_per_hr"]),
            min(
                2.50 * float(profile["basal_rate_u_per_hr"]),
                self.bionic_adaptive_basal_u_hr + adapt_gain * error,
            ),
        )
        basal_u = (self.bionic_adaptive_basal_u_hr / 60.0) * dt * snap.kovatchev_brake_factor

        # PD + IOB-subtracted autonomous correction micro-dose
        pd_glucose_Projection = error + 22.0 * snap.glucose_velocity_mgdl_min
        raw_corr = (pd_glucose_Projection / isf_bionic) - max(0.0, snap.excess_iob_units)
        corr_u = max(0.0, min(0.30 * raw_corr, 0.75)) if pd_glucose_Projection > 25.0 else 0.0
        return {
            "basal_units": basal_u,
            "auto_correction_units": corr_u,
            "total_Before_Meal_Units": basal_u + corr_u,
            "bionic_adaptive_basal_u_hr": self.bionic_adaptive_basal_u_hr,
            "bionic_isf_mgdl_u": isf_bionic,
        }

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = []
        dt = max(0.5, float(data.time_step or 5.0))
        snap = self.observer.observe(data)

        pid_res = self._compute_pid_ifb_law(snap, dt)
        traj_res = self._compute_trajectory_impulse_law(snap, dt)
        mpc_res = self._compute_kalman_zone_mpc_law(snap, dt)
        oref1_res = self._compute_oref1_smb_law(snap, traj_res, dt)
        camaps_res = self._compute_kalman_zone_mpc_law(
            snap,
            dt,
            isf_override=snap.bayesian_isf_mgdl_u,
        )
        bionic_res = self._compute_bionic_autonomous_law(snap, dt)

        # Announced meal bolus with IOB headroom and falling-trend attenuation
        meal_bolus_u = 0.0
        if float(data.carb_intake or 0.0) > 0.0:
            raw_meal_u = float(data.carb_intake) / self.observer.icr
            if snap.filtered_glucose_mgdl < float(self.settings["target_glucose"]):
                reverse_corr = (
                    float(self.settings["target_glucose"]) - snap.filtered_glucose_mgdl
                ) / self.observer.isf
                raw_meal_u = max(0.0, raw_meal_u - reverse_corr)
            meal_bolus_u = min(raw_meal_u, float(self.settings["max_bolus_u_per_step"]))
            self._log_reason(
                "White-box meal feedforward bolus calculated",
                "meal_response",
                meal_bolus_u,
                f"Carbs {float(data.carb_intake):.0f}g / ICR {self.observer.icr:.1f} g/U.",
            )

        law = self.control_law
        if law == "pid_ifb":
            basal_u = float(pid_res["basal_units"])
            corr_u = float(pid_res["auto_correction_units"])
        elif law == "trajectory_impulse":
            basal_u = float(traj_res["basal_units"])
            corr_u = float(traj_res["auto_correction_units"])
        elif law == "kalman_zone_mpc":
            basal_u = float(mpc_res["basal_units"])
            corr_u = float(mpc_res["auto_correction_units"])
        elif law == "oref1_smb":
            basal_u = float(oref1_res["basal_units"])
            corr_u = float(oref1_res["auto_correction_units"])
        elif law == "camaps_bayesian_mpc":
            basal_u = float(camaps_res["basal_units"])
            corr_u = float(camaps_res["auto_correction_units"])
        elif law == "bionic_autonomous":
            basal_u = float(bionic_res["basal_units"])
            corr_u = float(bionic_res["auto_correction_units"])
        else:
            basal_u = (
                0.20 * float(pid_res["basal_units"])
                + 0.25 * float(traj_res["basal_units"])
                + 0.25 * float(mpc_res["basal_units"])
                + 0.15 * float(oref1_res["basal_units"])
                + 0.15 * float(camaps_res["basal_units"])
            )
            corr_u = min(
                max(float(traj_res["auto_correction_units"]), float(oref1_res["auto_correction_units"])),
                0.5 * (float(pid_res["auto_correction_units"]) + float(mpc_res["auto_correction_units"]))
                + 0.25,
            )

        # Mathematical predictive low-glucose suspend (PLGS) & IOB ceiling guard
        suspend_thresh = float(self.settings["hypo_suspend_threshold_mgdl"])
        guard_thresh = float(self.settings["hypo_guard_threshold_mgdl"])
        g_min_guard = float(traj_res["g_min_guard_mgdl"])
        pred_30 = float(data.predicted_glucose_30min or traj_res["g_pred_30_mgdl"])

        if (
            snap.cgm_glucose_mgdl <= suspend_thresh
            or pred_30 <= suspend_thresh
            or g_min_guard <= suspend_thresh
        ):
            basal_u = 0.0
            corr_u = 0.0
            meal_bolus_u *= 0.5
            self._log_reason(
                "Predictive Low-Glucose Suspend (PLGS) zeroed basal and correction",
                "safety_constraint",
                0.0,
                f"CGM={snap.cgm_glucose_mgdl:.1f}, G_pred30={pred_30:.1f}, G_guard={g_min_guard:.1f} mg/dL.",
            )
        elif g_min_guard < guard_thresh or snap.glucose_velocity_mgdl_min < -1.2:
            attenuation = max(
                0.0,
                min(
                    float(traj_res["kovatchev_brake_factor"]),
                    (g_min_guard - suspend_thresh) / max(1.0, guard_thresh - suspend_thresh),
                ),
            )
            basal_u *= attenuation
            corr_u = 0.0
            self._log_reason(
                "Kovatchev LBGI & minimum-guard trajectory attenuated insulin delivery",
                "safety_constraint",
                basal_u,
                f"Attenuation={attenuation:.2f} (G_guard={g_min_guard:.1f} mg/dL).",
            )

        iob_headroom = max(0.0, float(self.settings["iob_ceiling_units"]) - snap.iob_units)
        if corr_u > iob_headroom:
            corr_u = iob_headroom

        if corr_u > 1e-4:
            self.last_auto_correction_time_min = snap.time_minutes

        total_u = min(
            float(self.settings["max_total_u_per_step"]),
            max(0.0, basal_u + corr_u + meal_bolus_u),
        )
        self.observer.record_delivery(total_u, dt)

        self._log_reason(
            f"White-Box AID ({law}) delivery computed",
            "final_decision",
            total_u,
            (
                f"basal={basal_u:.3f}U, corr={corr_u:.3f}U, meal={meal_bolus_u:.3f}U, "
                f"IOB={snap.iob_units:.2f}U, DynISF={snap.dynamic_isf_mgdl_u:.1f}, "
                f"G_ev={float(traj_res['g_eventual_mgdl']):.1f} mg/dL"
            ),
        )

        return {
            "total_insulin_delivered": total_u,
            "basal_insulin": basal_u,
            "correction_bolus": corr_u,
            "meal_bolus": meal_bolus_u,
            "bolus_insulin": corr_u + meal_bolus_u,
            "control_law": law,
            "observable_only": True,
            "peeks_patient_state": False,
            "mathematical_telemetry": {
                "observer": asdict(snap),
                "pid_ifb": pid_res,
                "trajectory_impulse": traj_res,
                "kalman_zone_mpc": mpc_res,
                "oref1_smb": oref1_res,
                "camaps_bayesian_mpc": camaps_res,
                "bionic_autonomous": bionic_res,
            },
        }


class MedtronicPIDIFBPumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """Steil et al. PID with Pharmacokinetic Insulin Feedback (Medtronic 670G/780G style)."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "pid_ifb"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box PID-IFB Pump (Medtronic Style)",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="Steil et al., Diabetes 2006 (PID-IFB)",
                description=(
                    "Continuous-to-discrete PID controller coupled with two-compartment "
                    "subcutaneous/plasma insulin feedback and safe auto-correction boluses."
                ),
                algorithm_type="hybrid",
            )
        )


class TrajectoryImpulsePumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """4-Curve Analytical Trajectory + Kovatchev LBGI Brakes (Control-IQ / TypeZero style)."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "trajectory_impulse"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box Trajectory-Impulse Pump (Control-IQ / Loop Style)",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="Kovatchev 1997; Patek 2012; Breton et al., NEJM 2020",
                description=(
                    "Analytical 4-curve glucose forecast decomposition with Kovatchev LBGI "
                    "power-law safety brakes and 60% auto-correction."
                ),
                algorithm_type="hybrid",
            )
        )


class KalmanZoneMPCPumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """Observable-Only Kalman State-Space Asymmetric Zone-MPC (Omnipod 5 / Doyle style)."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "kalman_zone_mpc"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box Kalman Zone-MPC Pump (Omnipod Style)",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="Gondhalekar et al., Automatica 2016 (Velocity-Weighting Zone-MPC)",
                description=(
                    "Observable-only Kalman state observer paired with an asymmetric "
                    "velocity-weighted Zone-MPC optimization and adaptive TDD."
                ),
                algorithm_type="hybrid",
            )
        )


class OpenAPSOREF1SMBPumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """OpenAPS `oref1` / AndroidAPS / Trio with Logarithmic Dynamic ISF, UAM, and Super Micro Bolus."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "oref1_smb"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box OpenAPS oref1 SMB + DynISF Pump",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="OpenAPS oref1; Wilson Logarithmic Dynamic ISF (AndroidAPS/Trio)",
                description=(
                    "4-curve (IOB, COB, UAM, ZT) trajectory engine with Logarithmic Dynamic ISF "
                    "and Super Micro Boluses (SMB) paired with Zero-Temp safety."
                ),
                algorithm_type="hybrid",
            )
        )


class CambridgeBayesianMPCPumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """Hovorka / CamAPS FX Style Recursive Bayesian Adaptive MPC."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "camaps_bayesian_mpc"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box Cambridge Bayesian Adaptive MPC Pump",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="Hovorka et al., Physiol Meas 2004; Lancet 2010 (CamAPS FX)",
                description=(
                    "Recursive Bayesian posterior estimation of time-varying insulin sensitivity "
                    "coupled with receding-horizon MPC."
                ),
                algorithm_type="hybrid",
            )
        )


class BionicPancreasAutonomousPumpAlgorithm(WhiteBoxAIDPumpAlgorithm):
    """El-Khatib / Damiano Body-Weight-Initialized Autonomous Bionic Pancreas (iLet style)."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        merged = {**(settings or {}), "control_law": "bionic_autonomous"}
        super().__init__(merged)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="White-Box Autonomous Bionic Pancreas Pump (iLet Style)",
                version="1.1.0",
                author="IINTS-AF Mathematical Control Lab",
                paper_reference="El-Khatib et al., Sci Transl Med 2010; Lancet 2017",
                description=(
                    "Autonomous body-weight-initialized controller with rolling basal adaptation "
                    "and PD + IOB-feedback micro-dosing."
                ),
                algorithm_type="hybrid",
            )
        )
