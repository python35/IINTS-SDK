"""Executable mathematical proof suite for the IINTS-SDK.

Verifies that the analytical derivations in the SDK mathematical workbook
match the deterministic runtime implementation to numerical precision:
  1. Kovatchev symmetrization center and convex noise bias (Jensen/Ito).
  2. Softplus renal clearance C^inf derivative and strict convexity.
  3. HAAF memory closed-form solution, [0, 1] invariance, and 13x hysteresis.
  4. Hovorka 8x8 block-upper-triangular Jacobian and non-oscillation theorem.
  5. AdvancedMetabolicModel FFA/Ketone minute-scale closed-form ODE solution.
  6. IndependentSupervisor linear Control Barrier Function alignment.
  7. Nyquist Gain Margin (> 6 dB) of Medtronic 780G and Tandem Control-IQ.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from iints.core.algorithms.whitebox_aid_pump import biexponential_iob_fraction
from iints.core.digital_twin import DigitalTwinCalibrator
from iints.core.glycemic_risk import (
    EUGLYCEMIC_CENTER_MGDL,
    bg_risk,
    bg_risk_transform,
    bgri,
    hbgi,
    lbgi,
)
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.patient.physiology import (
    antecedent_hypoglycemia_memory_derivative,
    smooth_threshold_excess,
)
from iints.core.supervisor import IndependentSupervisor
from iints.emulation.medtronic_780g import Medtronic780GBehavior
from iints.emulation.tandem_controliq import TandemControlIQBehavior


def test_proof_1_kovatchev_center_and_jensen_noise_bias() -> None:
    expected_center = math.exp(5.381 ** (1.0 / 1.084))
    assert math.isclose(EUGLYCEMIC_CENTER_MGDL, expected_center, rel_tol=1e-12)
    assert math.isclose(float(bg_risk_transform([EUGLYCEMIC_CENTER_MGDL])[0]), 0.0, abs_tol=1e-12)

    # Zero-mean Gaussian CGM noise must strictly inflate expected risk (r'' > 0)
    rng = np.random.default_rng(42)
    true_bg = 112.5
    noisy = true_bg + rng.normal(0.0, 6.0, size=20_000)
    assert float(np.mean(bg_risk(noisy))) > float(bg_risk([true_bg])[0])


def test_proof_2_softplus_renal_clearance_derivative_and_convexity() -> None:
    threshold = 162.0
    splay = 10.0
    eps = 1e-5

    # Exact threshold value: s * ln(2)
    at_threshold = smooth_threshold_excess(threshold, threshold=threshold, splay=splay)
    assert math.isclose(at_threshold, splay * math.log(2.0), rel_tol=1e-12)

    # First derivative equals logistic sigmoid; second derivative strictly positive
    eps_d2 = 1e-3  # Optimal float64 step (~eps_mach^(1/4)) to avoid 2nd-diff cancellation
    for g in (100.0, 140.0, 162.0, 185.0, 240.0):
        f_plus = smooth_threshold_excess(g + eps, threshold=threshold, splay=splay)
        f_minus = smooth_threshold_excess(g - eps, threshold=threshold, splay=splay)
        num_d1 = (f_plus - f_minus) / (2.0 * eps)

        f_p2 = smooth_threshold_excess(g + eps_d2, threshold=threshold, splay=splay)
        f_mid = smooth_threshold_excess(g, threshold=threshold, splay=splay)
        f_m2 = smooth_threshold_excess(g - eps_d2, threshold=threshold, splay=splay)
        num_d2 = (f_p2 - 2.0 * f_mid + f_m2) / (eps_d2**2)

        z = (g - threshold) / splay
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        exact_d2 = (sigmoid * (1.0 - sigmoid)) / splay

        assert math.isclose(num_d1, sigmoid, abs_tol=1e-7)
        assert math.isclose(num_d2, exact_d2, abs_tol=1e-7)
        assert num_d2 > 0.0


def test_proof_3_haaf_closed_form_solution_and_hysteresis() -> None:
    # Constant severe hypo at G = 54 mg/dL -> severity s = 1.0
    # Exact solution from H(0) = 0: H(t) = (12/13) * (1 - exp(-13*t / 4320))
    sol = solve_ivp(
        fun=lambda t, y: [antecedent_hypoglycemia_memory_derivative(54.0, float(y[0]))],
        t_span=(0.0, 180.0),
        y0=[0.0],
        t_eval=[60.0, 120.0, 180.0],
        rtol=1e-9,
        atol=1e-11,
    )
    assert sol.success
    for t_val, h_num in zip(sol.t, sol.y[0]):
        h_exact = (12.0 / 13.0) * (1.0 - math.exp(-13.0 * t_val / 4320.0))
        assert math.isclose(float(h_num), h_exact, rel_tol=1e-6)
        assert 0.0 <= float(h_num) <= 1.0

    # Boundary invariance: dH/dt >= 0 at H = 0, dH/dt <= 0 at H = 1
    assert antecedent_hypoglycemia_memory_derivative(50.0, 0.0) >= 0.0
    assert antecedent_hypoglycemia_memory_derivative(50.0, 1.0) <= 0.0


def test_proof_4_hovorka_block_jacobian_and_non_oscillation_theorem() -> None:
    patient = HovorkaPatientModel(initial_glucose=110.0, basal_insulin_rate=0.8)
    y0 = patient._state.copy()
    u_basal_mu_min = patient.basal_insulin_rate * 1000.0 / 60.0

    # Permuted 8-state core order: [Q1, Q2 | x1, x2, x3 | I, S2, S1]
    order = [0, 1, 5, 6, 7, 4, 3, 2]
    n = len(order)
    J = np.zeros((n, n), dtype=float)
    eps = 1e-6

    for col, idx in enumerate(order):
        step = eps * max(1.0, abs(float(y0[idx])))
        y_plus = y0.copy()
        y_minus = y0.copy()
        y_plus[idx] += step
        y_minus[idx] -= step
        f_plus = patient._ode(0.0, y_plus, u_basal_mu_min, 0.0, 0.0)
        f_minus = patient._ode(0.0, y_minus, u_basal_mu_min, 0.0, 0.0)
        deriv = (f_plus - f_minus) / (2.0 * step)
        J[:, col] = deriv[order]

    # Exact block-upper-triangular structure: lower blocks must vanish
    assert float(np.max(np.abs(J[2:5, 0:2]))) < 1e-10
    assert float(np.max(np.abs(J[5:8, 0:5]))) < 1e-10

    # 2x2 Glucose block J_QQ: trace < 0, det > 0, strictly real negative eigenvalues
    J_QQ = J[0:2, 0:2]
    tr = float(np.trace(J_QQ))
    det = float(np.linalg.det(J_QQ))
    discriminant = tr**2 - 4.0 * det
    eigs = np.linalg.eigvals(J_QQ)

    assert tr < 0.0
    assert det > 0.0
    assert discriminant > 0.0
    assert np.all(np.isclose(eigs.imag, 0.0, atol=1e-12))
    assert np.all(eigs.real < 0.0)


def test_proof_5_advanced_ffa_ketone_minute_scale_closed_form() -> None:
    patient = AdvancedMetabolicModel(
        initial_glucose=120.0,
        basal_insulin_rate=0.0,
        initial_ffa=0.4,
        initial_ketones=0.1,
    )
    # Zero insulin occlusion: I(t) = 0 -> exact linear 2-state subsystem
    y0 = patient._state.copy()
    y0[2] = 0.0  # I = 0
    y0[6] = 0.0  # S1 = 0
    y0[7] = 0.0  # S2 = 0

    def rhs_fk(t: float, fk: np.ndarray) -> list[float]:
        y = y0.copy()
        y[13] = fk[0]
        y[14] = fk[1]
        dy = patient._ode(t, y, 0.0, 0.0, t)
        return [float(dy[13]), float(dy[14])]

    sol = solve_ivp(
        fun=rhs_fk,
        t_span=(0.0, 360.0),
        y0=[0.4, 0.1],
        t_eval=[30.0, 120.0, 360.0],
        rtol=1e-9,
        atol=1e-11,
    )
    assert sol.success
    for t_val, f_num, k_num in zip(sol.t, sol.y[0], sol.y[1]):
        f_exact = 2.0 - 1.6 * math.exp(-t_val / 600.0)
        k_exact = 5.0 + 4.0 * math.exp(-t_val / 600.0) - 8.9 * math.exp(-t_val / 1200.0)
        assert math.isclose(float(f_num), f_exact, rel_tol=1e-6)
        assert math.isclose(float(k_num), k_exact, rel_tol=1e-6)

    # Hour-scale realism: 30 minutes of occlusion must NOT already trigger full DKA (3.0 mmol/L)
    assert float(sol.y[1][0]) < 0.5


def test_proof_6_supervisor_linear_control_barrier_function() -> None:
    sup = IndependentSupervisor()
    # Linear CBF: dG/dt >= -(1/20) * (G - 70)
    # Verify configured thresholds lie exactly on this boundary
    assert math.isclose(-(110.0 - sup.hypo_cutoff) / 20.0, sup.trend_stop, abs_tol=1e-12)
    assert math.isclose(
        -(sup.contract_glucose_threshold - sup.hypo_cutoff) / 20.0,
        sup.contract_trend_threshold_mgdl_min,
        abs_tol=1e-12,
    )
    assert math.isclose(-(70.0 - sup.hypo_cutoff) / 20.0, 0.0, abs_tol=1e-12)

    # At G = 70 mg/dL, supervisor must block insulin delivery
    decision = sup.evaluate_safety(
        current_glucose=70.0,
        proposed_insulin=1.0,
        current_time=5.0,
    )
    assert decision["approved_insulin"] == 0.0


def test_proof_7_emulator_gains_satisfy_nyquist_stability_margin() -> None:
    # Analytical Nyquist critical gain for tau_I=55, tau_G=100, tau_ISF=10, delta=2, ISF=50
    tau_i, tau_g, tau_isf, delta, isf = 55.0, 100.0, 10.0, 2.0, 50.0

    # Solve phase(w) = -pi via bisection
    lo, hi = 0.005, 0.1
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        phase = (
            -2.0 * math.atan(mid * tau_i)
            - math.atan(mid * tau_isf)
            - math.atan(mid * tau_g)
            - mid * delta
        )
        if phase > -math.pi:
            lo = mid
        else:
            hi = mid
    w_180 = 0.5 * (lo + hi)
    mag_norm = 1.0 / (
        (1.0 + (w_180 * tau_i) ** 2)
        * math.sqrt(1.0 + (w_180 * tau_isf) ** 2)
        * math.sqrt(1.0 + (w_180 * tau_g) ** 2)
    )
    k_p_crit_u_per_hour = (1.0 / (mag_norm * isf * tau_g)) * 60.0

    gm_780g_db = 20.0 * math.log10(k_p_crit_u_per_hour / Medtronic780GBehavior().pid_parameters.kp)
    gm_ciq_db = 20.0 * math.log10(k_p_crit_u_per_hour / TandemControlIQBehavior().pid_parameters.kp)

    assert gm_780g_db > 6.0
    assert gm_ciq_db > 6.0
    assert gm_780g_db > gm_ciq_db


def test_proof_8_biexponential_iob_no_singularity_at_a_equals_1() -> None:
    # Removable 0/0 singularity occurs when tp / dia = 1 - 1/sqrt(2) ~= 0.2928932
    dia = 300.0
    tp_singular = dia * (1.0 - 1.0 / math.sqrt(2.0))

    iob_mid, act_mid = biexponential_iob_fraction(120.0, peak_minutes=tp_singular, dia_minutes=dia)
    assert math.isfinite(iob_mid) and 0.0 < iob_mid < 1.0
    assert math.isfinite(act_mid) and act_mid > 0.0

    # Verify smoothness across the singularity and normalization integral_0^D a_I(t) dt == 1
    grid = np.linspace(0.0, dia, 601)
    activities = np.array(
        [biexponential_iob_fraction(float(t), peak_minutes=tp_singular, dia_minutes=dia)[1] for t in grid]
    )
    integral = float(np.trapezoid(activities, grid)) if hasattr(np, "trapezoid") else float(np.trapz(activities, grid))
    assert math.isclose(integral, 1.0, rel_tol=1e-4)

    # Verify -d(IOB)/dt == activity at t = 120 min
    dt = 1e-3
    iob_plus, _ = biexponential_iob_fraction(120.0 + dt, peak_minutes=tp_singular, dia_minutes=dia)
    iob_minus, _ = biexponential_iob_fraction(120.0 - dt, peak_minutes=tp_singular, dia_minutes=dia)
    assert math.isclose(-(iob_plus - iob_minus) / (2.0 * dt), act_mid, abs_tol=1e-6)


def test_proof_9_glycemic_risk_no_nan_on_sub_unit_values() -> None:
    readings = [0.2, 0.5, 0.99, 100.0, 140.0]
    risks = bg_risk(readings)
    assert np.all(np.isfinite(risks))
    assert risks.size == 2
    assert math.isfinite(lbgi(readings))
    assert math.isfinite(hbgi(readings))
    assert math.isfinite(bgri(readings))
    assert math.isclose(lbgi(readings) + hbgi(readings), bgri(readings), rel_tol=1e-12)


def test_proof_10_digital_twin_sensitivity_scale_is_not_cancelled() -> None:
    # Generate synthetic meal + bolus trace from a known Hovorka patient
    patient = HovorkaPatientModel(initial_glucose=115.0, basal_insulin_rate=0.8)
    history = []
    dt = 5.0
    for step in range(30):
        t = step * dt
        carbs = 40.0 if step == 4 else 0.0
        bolus = 3.0 if step == 4 else 0.0
        history.append({"time_min": t, "cgm": patient.get_current_glucose(), "carbs": carbs, "insulin": bolus})
        patient.update(
            time_step=dt,
            delivered_insulin=bolus + 0.8 * dt / 60.0,
            carb_intake=carbs,
            current_time=t,
        )

    calibrator = DigitalTwinCalibrator(base_weight_kg=70.0, basal_insulin_rate_u_per_hour=0.8)
    calibrator.fit(history)
    cond = calibrator.last_report["local_sensitivity_condition_number"]
    # Before the fix, x[1] was cancelled out by auto-basal scaling, forcing cond == inf
    assert math.isfinite(cond)
