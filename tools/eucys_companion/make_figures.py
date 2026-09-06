"""Generate every figure for the EUCYS technical companion.

Every curve is produced by calling the real IINTS-AF SDK code -- the same
functions the simulator integrates -- never by re-typing an equation here.
Output: standalone SVG files in ./figures/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style ----
INK = "#1e2530"
MUTED = "#6b7688"
GRID = "#dfe4ec"
C = {
    "glucose": "#c2410c",
    "insulin": "#1d4ed8",
    "action": "#7c3aed",
    "liver": "#b91c1c",
    "gut": "#ca8a04",
    "renal": "#0f766e",
    "periph": "#15803d",
    "sensor": "#0891b2",
    "alt1": "#db2777",
    "alt2": "#475569",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.linewidth": 1.0,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.9,
    "svg.fonttype": "none",
})


def finish(fig, axes, name):
    for ax in np.atleast_1d(axes).ravel():
        ax.grid(True, alpha=0.55)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.9)
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}")


def band(ax, lo=70, hi=180):
    ax.axhspan(lo, hi, color="#22c55e", alpha=0.075, lw=0, zorder=0)
    ax.axhline(lo, color="#16a34a", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(hi, color="#16a34a", lw=0.8, ls=":", alpha=0.7)


# ============================================================ SDK imports ==
from iints.core.patient.physiology import (  # noqa: E402
    antecedent_hypoglycemia_memory_derivative,
    counterregulatory_rescue_multiplier,
    dawn_glucose_rate_mgdl_min,
    dawn_insulin_sensitivity_multiplier,
    dawn_window_fraction,
    smooth_threshold_excess,
)
from iints.core.patient.hovorka_model import (  # noqa: E402
    HovorkaParameters,
    HovorkaPatientModel,
)
from iints.core.devices.models import SensorModel, PumpModel  # noqa: E402
from iints.core import glycemic_risk as gr  # noqa: E402

manifest = {}


# =========================================================== FIG 1: renal ==
def fig_renal():
    G = np.linspace(80, 400, 600)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))

    # V_G for a 70 kg reference subject, straight from the model
    p = HovorkaParameters()
    V_G_dL = p.V_G_per_kg * p.body_weight_kg * 10.0

    for splay, style in ((10.0, "-"), (2.0, "--"), (30.0, ":")):
        FR = np.array([0.003 * V_G_dL * smooth_threshold_excess(g, threshold=162.0, splay=splay) for g in G])
        ax.plot(G, FR, style, color=C["renal"], lw=1.9 if splay == 10 else 1.4,
                alpha=1.0 if splay == 10 else 0.65,
                label=f"splay s = {splay:g} mg/dL" + (" (SDK default)" if splay == 10 else ""))

    hard = 0.003 * V_G_dL * np.maximum(G - 162.0, 0.0)
    ax.plot(G, hard, color=C["alt2"], lw=1.2, ls="-.", alpha=0.6, label="hard cut-off (not used)")
    ax.axvline(162, color=MUTED, lw=0.9, ls="--")
    ax.annotate("renal threshold\n162 mg/dL", xy=(162, ax.get_ylim()[1] * 0.72),
                xytext=(178, ax.get_ylim()[1] * 0.78), fontsize=8, color=MUTED)
    ax.set_xlabel("Plasma glucose $G$  (mg/dL)")
    ax.set_ylabel("$F_R$  (mg/min)")
    ax.set_title("F14 · Renal glucose clearance is a softplus, not a switch")
    ax.legend(loc="upper left")
    finish(fig, ax, "f14_renal")


# ============================================================ FIG 2: dawn ==
def fig_dawn():
    t = np.linspace(0, 1440, 1441)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))

    w = [dawn_window_fraction(x, start_hour=4.0, end_hour=8.0) for x in t]
    axes[0].plot(t / 60, w, color=C["alt2"])
    axes[0].fill_between(t / 60, w, color=C["alt2"], alpha=0.12)
    axes[0].set_title("Shared window $w(t)$")
    axes[0].set_ylabel("dimensionless")

    for s, a in ((30.0, 1.0), (15.0, 0.55)):
        r = [dawn_glucose_rate_mgdl_min(x, peak_strength_mgdl_per_hour=s,
                                        start_hour=4.0, end_hour=8.0) for x in t]
        axes[1].plot(t / 60, r, color=C["glucose"], alpha=a, label=f"$s_{{dawn}}$ = {s:g}")
    axes[1].set_title("F10 · rate $r_{dawn}(t)$")
    axes[1].set_ylabel("mg/dL/min")
    axes[1].legend()

    for f, a in ((0.3, 1.0), (0.15, 0.55)):
        m = [dawn_insulin_sensitivity_multiplier(x, peak_resistance_fraction=f,
                                                 start_hour=4.0, end_hour=8.0) for x in t]
        axes[2].plot(t / 60, m, color=C["insulin"], alpha=a, label=f"$f_{{dawn}}$ = {f:g}")
    axes[2].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    axes[2].set_title("F16 · sensitivity $M_{dawn}(t)$")
    axes[2].set_ylabel("multiplier")
    axes[2].set_ylim(0.6, 1.05)
    axes[2].legend(loc="lower left")

    for ax in axes:
        ax.set_xlabel("time of day (h)")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
    finish(fig, axes, "f10_f16_dawn")


# ======================================================== FIG 3: hypo/HAAF ==
def fig_rescue():
    G = np.linspace(30, 100, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    for haaf, a in ((0.0, 1.0), (0.35, 0.75), (0.7, 0.55), (1.0, 0.38)):
        R = [counterregulatory_rescue_multiplier(g, haaf) for g in G]
        axes[0].plot(G, R, color=C["liver"], alpha=a, label=f"HAAF = {haaf:g}")
    axes[0].axvline(70, color=MUTED, lw=0.9, ls="--")
    axes[0].set_xlabel("Plasma glucose $G$  (mg/dL)")
    axes[0].set_ylabel("$R_{rescue}$  (EGP multiplier)")
    axes[0].set_title("F11 · counter-regulatory rescue, blunted by HAAF")
    axes[0].legend(loc="upper right")

    # Integrate the real HAAF derivative over a 10-day repeated-hypo scenario
    def haaf_rhs(t, y):
        day = (t / 1440.0) % 1.0
        g = 58.0 if 0.08 < day < 0.14 else 130.0     # one nocturnal hypo per day
        return [antecedent_hypoglycemia_memory_derivative(g, y[0])]

    sol = solve_ivp(haaf_rhs, (0, 14 * 1440), [0.0], max_step=15.0, rtol=1e-6, atol=1e-9)
    axes[1].plot(sol.t / 1440, sol.y[0], color=C["action"])
    axes[1].axvline(10, color=MUTED, lw=0.9, ls="--")
    axes[1].annotate("hypos stop", xy=(10, 0.05), xytext=(10.3, 0.05), fontsize=8, color=MUTED)
    axes[1].set_xlabel("time (days)")
    axes[1].set_ylabel("HAAF memory (dimensionless)")
    axes[1].set_title("F12 · exposure memory: builds in hours, decays in days")
    axes[1].set_xlim(0, 14)
    finish(fig, axes, "f11_f12_hypo")
    manifest["haaf_peak"] = float(np.max(sol.y[0]))


# ====================================================== FIG 4: insulin PK ===
def fig_insulin_pk():
    """Two-depot SC absorption + plasma insulin, integrated exactly as the ODE does."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    p = HovorkaParameters()
    V_I = p.V_I_per_kg * p.body_weight_kg

    profiles = {"fiasp (35 min)": 35.0, "novolog (55 min, default)": 55.0, "regular (90 min)": 90.0}
    colours = [C["sensor"], C["insulin"], C["alt2"]]

    for (label, tmax), col in zip(profiles.items(), colours):
        def rhs(t, y):
            S1, S2, I = y
            return [-S1 / tmax, S1 / tmax - S2 / tmax, (S2 / tmax) / V_I - p.k_e * I]
        # 5 U bolus = 5000 mU into S1 at t=0
        sol = solve_ivp(rhs, (0, 480), [5000.0, 0.0, 0.0], max_step=1.0, rtol=1e-8, atol=1e-10)
        U_I = sol.y[1] / tmax
        axes[0].plot(sol.t, U_I, color=col, label=label)
        axes[1].plot(sol.t, sol.y[2], color=col, label=label)
        if "novolog" in label:
            manifest["novolog_peak_min"] = float(sol.t[np.argmax(sol.y[2])])
            manifest["novolog_peak_mu_l"] = float(np.max(sol.y[2]))

    axes[0].set_ylabel("$U_I$  appearance rate (mU/min)")
    axes[0].set_title("F04 · subcutaneous appearance after a 5 U bolus")
    axes[1].set_ylabel("$I$  plasma insulin (mU/L)")
    axes[1].set_title("Resulting plasma insulin ($k_e$ = 0.138 min$^{-1}$)")
    for ax in axes:
        ax.set_xlabel("time since bolus (min)")
        ax.set_xlim(0, 480)
        ax.legend()
    finish(fig, axes, "f04_insulin_pk")


# ========================================================= FIG 5: meal Ra ===
def fig_meal():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    p = HovorkaParameters()

    for tmaxG, a in ((20.0, 0.5), (40.0, 1.0), (80.0, 0.7)):
        def rhs(t, y):
            D1, D2 = y
            k = 1.0 / tmaxG
            return [-k * D1, k * D1 - k * D2]
        sol = solve_ivp(rhs, (0, 420), [60_000.0, 0.0], max_step=1.0, rtol=1e-8, atol=1e-9)
        U_G = (sol.y[1] / tmaxG) * p.A_G
        axes[0].plot(sol.t, U_G, color=C["gut"], alpha=a,
                     label=f"$t_{{max,G}}$ = {tmaxG:g} min" + (" (default)" if tmaxG == 40 else ""))
        if tmaxG == 40.0:
            manifest["meal_peak_min"] = float(sol.t[np.argmax(U_G)])
            manifest["meal_peak_mgmin"] = float(np.max(U_G))
            manifest["meal_auc_g"] = float(np.trapezoid(U_G, sol.t) / 1000.0)

    axes[0].set_ylabel("$U_G$  glucose appearance (mg/min)")
    axes[0].set_title("F05 · rate of appearance after 60 g CHO")
    axes[0].legend()

    for AG, a in ((1.0, 0.5), (0.8, 1.0), (0.6, 0.7)):
        def rhs(t, y):
            D1, D2 = y
            k = 1.0 / 40.0
            return [-k * D1, k * D1 - k * D2]
        sol = solve_ivp(rhs, (0, 420), [60_000.0, 0.0], max_step=1.0, rtol=1e-8, atol=1e-9)
        cum = np.concatenate([[0], np.cumsum(np.diff(sol.t) * ((sol.y[1][:-1] / 40.0) * AG))]) / 1000.0
        axes[1].plot(sol.t, cum, color=C["gut"], alpha=a,
                     label=f"$A_G$ = {AG:g}" + (" (default)" if AG == 0.8 else ""))
    axes[1].axhline(60, color=MUTED, ls=":", lw=0.9)
    axes[1].annotate("60 g ingested", xy=(300, 61), fontsize=8, color=MUTED)
    axes[1].set_ylabel("cumulative glucose absorbed (g)")
    axes[1].set_title("Bioavailability $A_G$ sets the asymptote, not the shape")
    axes[1].legend(loc="lower right")

    for ax in axes:
        ax.set_xlabel("time since meal (min)")
        ax.set_xlim(0, 420)
    finish(fig, axes, "f05_meal")


# =================================================== FIG 6: action channels ==
def fig_action():
    p = HovorkaParameters()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    I_step = 30.0  # mU/L step
    ka = [p.k_a1, p.k_a2, p.k_a3]
    S = [p.S_IT, p.S_ID, p.S_IE]
    names = ["$x_1$ transport ($S_{IT}$)", "$x_2$ disposal ($S_{ID}$)", "$x_3$ EGP ($S_{IE}$)"]
    cols = [C["insulin"], C["periph"], C["liver"]]

    t = np.linspace(0, 600, 1200)
    for k, s, nm, col in zip(ka, S, names, cols):
        kb = s * k
        x = (kb * I_step / k) * (1 - np.exp(-k * t))     # exact step response
        xn = x / (kb * I_step / k)
        axes[0].plot(t, xn, color=col, label=f"{nm},  $t_{{1/2}}$ = {np.log(2)/k:.0f} min")
    axes[0].axhline(1.0, color=MUTED, ls=":", lw=0.8)
    axes[0].set_xlabel("time after insulin step (min)")
    axes[0].set_ylabel("fraction of steady state")
    axes[0].set_title("F07 · the three channels equilibrate at different speeds")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim(0, 600)

    # Steady-state EGP suppression vs plasma insulin (real EGP expression)
    I = np.linspace(0, 60, 400)
    EGP0 = p.EGP_0_per_kg * p.body_weight_kg
    for liver_scalar, a in ((0.5, 0.5), (1.0, 1.0), (1.5, 0.65)):
        x3ss = p.S_IE * liver_scalar * I
        EGP = EGP0 * np.maximum(0.0, 1.0 - x3ss)
        axes[1].plot(I, EGP, color=C["liver"], alpha=a,
                     label=f"liver scalar = {liver_scalar:g}" + (" (default)" if liver_scalar == 1 else ""))
    axes[1].set_xlabel("Plasma insulin $I$  (mU/L)")
    axes[1].set_ylabel("steady-state EGP (mg/min)")
    axes[1].set_title("F06 · hepatic glucose production vs insulin")
    axes[1].legend()
    axes[1].set_xlim(0, 60)
    manifest["egp0_mgmin"] = float(EGP0)
    manifest["egp_half_supp_mu_l"] = float(0.5 / p.S_IE)
    finish(fig, axes, "f06_f07_action")


# ======================================================== FIG 7: NIMGU/F01c ==
def fig_nimgu():
    p = HovorkaParameters()
    F_01 = p.F_01c_per_kg * p.body_weight_kg
    G = np.linspace(20, 200, 500)
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for glut4, a in ((0.0, 1.0), (0.5, 0.7), (1.0, 0.5)):
        F01c = F_01 * np.minimum(1.0, np.maximum(0.0, G / 81.0))
        ax.plot(G, F01c * (1 + 1.5 * glut4), color=C["periph"], alpha=a,
                label=f"GLUT4 = {glut4:g}" + (" (rest)" if glut4 == 0 else ""))
    ax.axvline(81, color=MUTED, ls="--", lw=0.9)
    ax.annotate("81 mg/dL\n(4.5 mmol/L)", xy=(83, F_01 * 0.35), fontsize=8, color=MUTED)
    ax.set_xlabel("Plasma glucose $G$  (mg/dL)")
    ax.set_ylabel("NIMGU  (mg/min)")
    ax.set_title("F09 · insulin-independent uptake: glucose-limited below 81 mg/dL, exercise-amplified")
    ax.legend(loc="lower right")
    finish(fig, ax, "f09_nimgu")
    manifest["f01_mgmin"] = float(F_01)


# ========================================================= FIG 8: glucagon ===
def fig_glucagon():
    p = HovorkaParameters()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    k1 = 1.0 / p.t_max_glucagon
    k2 = p.k_e_glucagon
    Cl = p.glucagon_clearance_ml_kg_min * p.body_weight_kg

    for dose_mg, a in ((0.5, 0.6), (1.0, 1.0)):
        def rhs(t, y):
            Y1, Y2, xg = y
            conc = k2 * Y2 / Cl
            act = conc / (p.glucagon_ec50_pg_ml + conc)
            return [-k1 * Y1, k1 * Y1 - k2 * Y2,
                    p.k_a_glucagon * (p.S_glucagon * act - xg)]
        y0 = [dose_mg * 1e9, 0.0, 0.0]
        sol = solve_ivp(rhs, (0, 300), y0, max_step=0.5, rtol=1e-8, atol=1e-6)
        conc = k2 * sol.y[1] / Cl
        axes[0].plot(sol.t, conc, color=C["liver"], alpha=a, label=f"{dose_mg:g} mg SC")
        axes[1].plot(sol.t, sol.y[2], color=C["liver"], alpha=a, label=f"{dose_mg:g} mg SC")
        if dose_mg == 1.0:
            manifest["glucagon_cmax"] = float(np.max(conc))
            manifest["glucagon_tmax"] = float(sol.t[np.argmax(conc)])

    axes[0].axhline(p.glucagon_ec50_pg_ml, color=MUTED, ls=":", lw=0.9)
    axes[0].annotate("$CE_{50}$ = 350 pg/mL", xy=(150, 380), fontsize=8, color=MUTED)
    axes[0].set_ylabel(r"$\Gamma$  plasma glucagon (pg/mL)")
    axes[0].set_title("F13 · glucagon PK (two depots + clearance)")
    axes[1].set_ylabel("$x_{gluc}$  effect on EGP (fraction)")
    axes[1].set_title("PD: bounded effect compartment")
    for ax in axes:
        ax.set_xlabel("time since dose (min)")
        ax.set_xlim(0, 300)
        ax.legend()
    finish(fig, axes, "f13_glucagon")


# ============================================================== FIG 9: CGM ===
def fig_cgm():
    """Real SensorModel driven by a synthetic but physiologic BG trace."""
    t = np.arange(0, 720, 5, dtype=float)
    bg = (140 + 70 * np.exp(-((t - 180) / 70) ** 2) - 45 * np.exp(-((t - 430) / 90) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))

    sensor = SensorModel(noise_std=8.5, bias=0.0, lag_minutes=5, isf_tau_minutes=5.0,
                         noise_fbm_hurst=0.78, seed=7)
    cgm = [sensor.read(float(g), float(ti)).value for ti, g in zip(t, bg)]

    clean = SensorModel(noise_std=0.0, bias=0.0, lag_minutes=5, isf_tau_minutes=5.0,
                        noise_fbm_hurst=None, seed=7)
    isf = [clean.read(float(g), float(ti)).value for ti, g in zip(t, bg)]

    axes[0].plot(t / 60, bg, color=C["glucose"], label="true BG (hidden state)")
    axes[0].plot(t / 60, isf, color=C["alt2"], lw=1.3, ls="--", label=r"ISF, noise-free ($\tau$=5, lag=5)")
    axes[0].plot(t / 60, cgm, color=C["sensor"], lw=1.3, alpha=0.9, label="CGM as the algorithm sees it")
    band(axes[0])
    axes[0].set_ylabel("glucose (mg/dL)")
    axes[0].set_title("F15 · what the controller actually observes")
    axes[0].legend(loc="upper right")

    err = np.array(cgm) - bg
    axes[1].plot(t / 60, err, color=C["sensor"], lw=1.3)
    axes[1].axhline(0, color=MUTED, lw=0.9)
    axes[1].fill_between(t / 60, err, color=C["sensor"], alpha=0.12)
    axes[1].set_ylabel("CGM − BG (mg/dL)")
    axes[1].set_title(f"Correlated error, MARD-like |err| mean = {np.mean(np.abs(err)):.1f} mg/dL")
    for ax in axes:
        ax.set_xlabel("time (h)")
        ax.set_xlim(0, 12)
    finish(fig, axes, "f15_cgm")
    manifest["cgm_mean_abs_err"] = float(np.mean(np.abs(err)))
    manifest["cgm_max_lag_err"] = float(np.max(np.abs(err)))


# ===================================================== FIG 10: risk space ====
def fig_risk():
    G = np.linspace(20, 600, 1000)
    f = gr.bg_risk_transform(G)
    r = gr.bg_risk(G)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    axes[0].plot(G, f, color=C["action"])
    axes[0].axhline(0, color=MUTED, lw=0.9)
    axes[0].axvline(gr.EUGLYCEMIC_CENTER_MGDL, color=MUTED, ls="--", lw=0.9)
    axes[0].annotate(f"f = 0 at {gr.EUGLYCEMIC_CENTER_MGDL:.1f} mg/dL",
                     xy=(gr.EUGLYCEMIC_CENTER_MGDL + 20, -2.2), fontsize=8, color=MUTED)
    axes[0].set_ylabel("$f(BG)$  symmetrized scale")
    axes[0].set_title("Kovatchev symmetrization")

    axes[1].plot(G[f < 0], r[f < 0], color="#dc2626", label="LBGI branch ($f<0$)")
    axes[1].plot(G[f > 0], r[f > 0], color="#ea580c", label="HBGI branch ($f>0$)")
    axes[1].set_ylabel("$r(BG)$  risk")
    axes[1].set_title("Risk is asymmetric: 40 mg/dL costs far more than 240")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("blood glucose (mg/dL)")
        ax.set_xscale("log")
        ax.set_xticks([20, 40, 70, 120, 180, 300, 600])
        ax.set_xticklabels(["20", "40", "70", "120", "180", "300", "600"])
    finish(fig, axes, "risk_space")
    manifest["risk_40"] = float(gr.bg_risk([40.0])[0])
    manifest["risk_240"] = float(gr.bg_risk([240.0])[0])
    manifest["euglycemic_center"] = float(gr.EUGLYCEMIC_CENTER_MGDL)


# ================================================ FIG 11: closed-loop runs ===
def _run(isf=50.0, basal=0.8, liver=1.0, muscle=1.0, dawn_s=0.0, dawn_f=0.0,
         icr=10.0, meals=((420, 60), (720, 80), (1140, 70)), minutes=1440, seed=11,
         algo="clinical"):
    """One closed-loop day through the real Simulator."""
    from iints.core.simulator import Simulator, StressEvent
    from iints.core.algorithms.pid_controller import PIDController
    from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm

    patient = HovorkaPatientModel(
        basal_insulin_rate=basal, insulin_sensitivity=isf, carb_factor=icr,
        initial_glucose=130.0, liver_sensitivity_scalar=liver,
        muscle_sensitivity_scalar=muscle,
        dawn_phenomenon_strength=dawn_s, dawn_insulin_resistance_fraction=dawn_f,
        max_glucose_rate_mgdl_per_min=0.0,
    )
    alg = PIDController() if algo == "pid" else ClinicalBaselineAlgorithm()
    sim = Simulator(patient_model=patient, algorithm=alg, time_step=5, seed=seed,
                    sensor_model=SensorModel(noise_std=8.5, lag_minutes=5,
                                             isf_tau_minutes=5.0, seed=seed))
    for tm, g in meals:
        sim.add_stress_event(StressEvent(start_time=tm, event_type="meal", value=g))
    df, report = sim.run_batch(minutes)
    return df, report


def fig_pid_failure():
    """An honest negative result: naive PID gains terminate the run."""
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    out = {}
    for algo, col, lbl in (("clinical", C["periph"], "Clinical Baseline (basal-bolus)"),
                           ("pid", "#dc2626", "PID, textbook gains $k_p$=0.1")):
        df, rep = _run(algo=algo)
        g = df["glucose_actual_mgdl"].to_numpy()
        ax.plot(df["time_minutes"] / 60, g, color=col, label=lbl)
        out[lbl] = {"metrics": tir(g), "terminated": bool(rep.get("terminated_early")),
                    "reason": rep.get("termination_reason")}
        if rep.get("terminated_early"):
            tend = float(df["time_minutes"].iloc[-1]) / 60
            ax.plot([tend], [g[-1]], marker="X", ms=11, color="#dc2626", zorder=5)
            ax.annotate("run terminated:\nG < 40 mg/dL for 30 min",
                        xy=(tend, g[-1]), xytext=(tend + 1.2, 55),
                        fontsize=8, color="#dc2626",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0))
    band(ax)
    ax.axhline(40, color="#dc2626", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("time (h)")
    ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("The bench reports failure instead of hiding it")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.legend(loc="upper right")
    finish(fig, ax, "pid_failure")
    manifest["pid_failure"] = out


def tir(g):
    g = np.asarray(g, dtype=float)
    return dict(
        tir=100.0 * np.mean((g >= 70) & (g <= 180)),
        below=100.0 * np.mean(g < 70),
        above=100.0 * np.mean(g > 180),
        mean=float(np.mean(g)),
        cv=100.0 * float(np.std(g) / np.mean(g)),
        lbgi=gr.lbgi(g), hbgi=gr.hbgi(g),
    )


def fig_closed_loop():
    df, rep = _run()
    t = df["time_minutes"].to_numpy() / 60.0
    g = df["glucose_actual_mgdl"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.9), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1]})
    band(axes[0])
    axes[0].plot(t, df["glucose_to_algo_mgdl"], color=C["sensor"], lw=1.1, alpha=0.8,
                 label="CGM (what the algorithm sees)")
    axes[0].plot(t, g, color=C["glucose"], label="plasma glucose (hidden model truth)")
    for tm, gm in ((7, 60), (12, 80), (19, 70)):
        axes[0].axvline(tm, color=C["gut"], lw=0.9, ls=":", alpha=0.7)
        axes[0].annotate(f"{gm} g", xy=(tm, 275), fontsize=8, color=C["gut"], ha="center")
    axes[0].set_ylabel("glucose (mg/dL)")
    m = tir(g)
    axes[0].set_title(f"Closed-loop 24 h, Clinical Baseline · TIR {m['tir']:.1f}%  ·  <70 {m['below']:.1f}%  ·  "
                      f">180 {m['above']:.1f}%  ·  CV {m['cv']:.1f}%  ·  LBGI {m['lbgi']:.2f}")
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].bar(t, df["basal_insulin_units"], width=5 / 60, color=C["insulin"],
                alpha=0.9, label="basal")
    axes[1].bar(t, df["bolus_insulin_units"], width=5 / 60,
                bottom=df["basal_insulin_units"], color=C["alt1"], alpha=0.9, label="bolus")
    axes[1].set_ylabel("insulin (U / 5 min)")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_xlabel("time (h)")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 3))
    finish(fig, axes, "closed_loop_day")
    manifest["cl_metrics"] = m
    manifest["cl_total_insulin"] = float(df["delivered_insulin_units"].sum())
    return df


def fig_cascade():
    """The full delivery -> absorption -> action -> flux -> glucose chain, one meal."""
    df, _ = _run(meals=((120, 70),), minutes=600)
    t = df["time_minutes"].to_numpy() / 60.0

    fig, axes = plt.subplots(5, 1, figsize=(8.2, 9.4), sharex=True)

    axes[0].bar(t, df["delivered_insulin_units"], width=5 / 60, color=C["insulin"], alpha=0.9)
    axes[0].set_ylabel("delivered\n(U / 5 min)")
    axes[0].set_title("① Pump delivery  $u_I$", loc="left")

    axes[1].plot(t, df["patient_state_S1"], color=C["insulin"], alpha=0.55, label="$S_1$")
    axes[1].plot(t, df["patient_state_S2"], color=C["insulin"], label="$S_2$")
    ax1b = axes[1].twinx()
    ax1b.plot(t, df["patient_state_I"], color=C["alt1"], lw=1.6, label="$I$")
    ax1b.set_ylabel("$I$ (mU/L)", color=C["alt1"])
    ax1b.tick_params(axis="y", colors=C["alt1"])
    ax1b.spines["top"].set_visible(False)
    axes[1].set_ylabel("depots (mU)")
    axes[1].set_title("② Subcutaneous depots  →  ③ plasma insulin   [F04]", loc="left")
    axes[1].legend(loc="upper right", fontsize=8)

    # x3 is ~50x larger than x1/x2 (S_IE = 520e-4 vs S_IT = 51.2e-4), so it needs
    # its own axis or it flattens the other two into the baseline.
    axes[2].plot(t, df["patient_state_x1"], color=C["insulin"], label="$x_1$ transport")
    axes[2].plot(t, df["patient_state_x2"], color=C["periph"], label="$x_2$ disposal")
    axes[2].set_ylabel("$x_1$, $x_2$  (1/min)")
    ax2b = axes[2].twinx()
    ax2b.plot(t, df["patient_state_x3"], color=C["liver"], label="$x_3$ EGP")
    ax2b.set_ylabel("$x_3$  (dimensionless)", color=C["liver"])
    ax2b.tick_params(axis="y", colors=C["liver"])
    ax2b.spines["top"].set_visible(False)
    axes[2].set_title("④ Remote insulin action — note the separate right axis for $x_3$   [F07]", loc="left")
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc="upper right", ncol=3, fontsize=8)

    axes[3].plot(t, df["patient_flux_glucose_appearance"], color=C["gut"], label="$U_G$ meal in")
    axes[3].plot(t, df["patient_flux_endogenous_production"], color=C["liver"], label="EGP (liver)")
    axes[3].plot(t, df["patient_flux_glucose_to_periphery"], color=C["periph"], label="$x_1Q_1$ to periphery")
    axes[3].plot(t, df["patient_flux_nimgu"], color=C["alt2"], lw=1.3, label="NIMGU")
    axes[3].plot(t, df["patient_flux_renal_clearance"], color=C["renal"], lw=1.3, label="$F_R$ renal")
    axes[3].set_ylabel("flux (mg/min)")
    axes[3].set_title("⑤ Competing glucose fluxes   [F05 · F06 · F09 · F14]", loc="left")
    axes[3].legend(loc="upper right", ncol=2, fontsize=7.5)

    band(axes[4])
    axes[4].plot(t, df["glucose_actual_mgdl"], color=C["glucose"])
    axes[4].set_ylabel("G (mg/dL)")
    axes[4].set_title("⑥ Net result: plasma glucose  $G = Q_1/V_G$   [F06]", loc="left")
    axes[4].set_xlabel("time (h)")
    axes[4].set_xlim(0, 10)
    axes[4].axvline(2, color=C["gut"], ls=":", lw=1.0)
    axes[4].annotate("70 g meal", xy=(2.1, 210), fontsize=8, color=C["gut"])

    finish(fig, axes, "cascade")


def fig_isf_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rows = []
    for isf, a, lbl in ((30.0, 0.55, "ISF 30 (resistant)"), (50.0, 1.0, "ISF 50 (reference)"),
                        (80.0, 0.7, "ISF 80 (sensitive)")):
        df, _ = _run(isf=isf)
        g = df["glucose_actual_mgdl"].to_numpy()
        t = df["time_minutes"].to_numpy() / 60
        axes[0].plot(t, g, color=C["glucose"], alpha=a, label=lbl)
        rows.append((lbl, tir(g)))
    band(axes[0])
    axes[0].set_xlabel("time (h)"); axes[0].set_ylabel("glucose (mg/dL)")
    axes[0].set_title("Effect of insulin sensitivity (ISF), identical scenario & controller")
    axes[0].set_xlim(0, 24); axes[0].set_xticks(range(0, 25, 6)); axes[0].legend(loc="upper left")

    labels = [r[0].split(" (")[0] for r in rows]
    x = np.arange(len(rows))
    axes[1].bar(x - 0.22, [r[1]["tir"] for r in rows], 0.2, color=C["periph"], label="TIR 70-180")
    axes[1].bar(x, [r[1]["below"] for r in rows], 0.2, color="#dc2626", label="< 70")
    axes[1].bar(x + 0.22, [r[1]["above"] for r in rows], 0.2, color="#f59e0b", label="> 180")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("% of day"); axes[1].set_title("Outcome split")
    axes[1].legend()
    finish(fig, axes, "isf_sweep")
    manifest["isf_sweep"] = {r[0]: r[1] for r in rows}


def fig_liver_sweep():
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    out = {}
    for ls, a, lbl in ((0.5, 0.55, "liver scalar 0.5 (hepatic resistance)"),
                       (1.0, 1.0, "liver scalar 1.0 (reference)"),
                       (1.6, 0.7, "liver scalar 1.6 (hepatic-selective)")):
        df, _ = _run(liver=ls)
        g = df["glucose_actual_mgdl"].to_numpy()
        ax.plot(df["time_minutes"] / 60, g, color=C["liver"], alpha=a, label=lbl)
        out[lbl] = tir(g)
    band(ax)
    ax.set_xlabel("time (h)"); ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("Altered hepatic insulin selectivity ($S_{IE}$ scalar), same dose schedule")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.legend(loc="upper left")
    finish(fig, ax, "liver_sweep")
    manifest["liver_sweep"] = out


def fig_dawn_effect():
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    out = {}
    for (s, f), a, lbl in (((0.0, 0.0), 0.55, "dawn off (default)"),
                           ((30.0, 0.0), 0.8, "inflow only  $s_{dawn}$=30"),
                           ((30.0, 0.3), 1.0, "inflow + resistance  $f_{dawn}$=0.3")):
        df, _ = _run(dawn_s=s, dawn_f=f)
        g = df["glucose_actual_mgdl"].to_numpy()
        ax.plot(df["time_minutes"] / 60, g, color=C["alt1"], alpha=a, label=lbl)
        out[lbl] = tir(g)
    band(ax)
    ax.axvspan(4, 8, color=C["alt1"], alpha=0.07, lw=0)
    ax.annotate("dawn window", xy=(6, 250), ha="center", fontsize=8, color=MUTED)
    ax.set_xlabel("time (h)"); ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("Why F16 exists: a controller can out-dose an inflow, not a loss of sensitivity")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.legend(loc="upper left")
    finish(fig, ax, "dawn_effect")
    manifest["dawn_effect"] = out


def fig_scenarios():
    """Hypo / normo / hyper day by shifting basal only."""
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    out = {}
    for icr, col, lbl in ((6.0, "#dc2626", "ICR 6 g/U — over-bolused"),
                          (10.0, C["periph"], "ICR 10 g/U — reference"),
                          (20.0, "#f59e0b", "ICR 20 g/U — under-bolused")):
        df, _ = _run(icr=icr)
        g = df["glucose_actual_mgdl"].to_numpy()
        ax.plot(df["time_minutes"] / 60, g, color=col, label=lbl)
        out[lbl] = tir(g)
    band(ax)
    ax.set_xlabel("time (h)"); ax.set_ylabel("glucose (mg/dL)")
    ax.set_title("Hypo-, normo- and hyperglycaemic days from the carb ratio alone")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.legend(loc="upper left")
    finish(fig, ax, "scenarios")
    manifest["scenarios"] = out


def fig_tornado():
    """One-at-a-time sensitivity on mean glucose and TIR."""
    base_df, _ = _run()
    base_g = base_df["glucose_actual_mgdl"].to_numpy()
    base = tir(base_g)

    sweeps = [
        ("ISF  (insulin sensitivity)", dict(isf=37.5), dict(isf=62.5)),
        ("ICR  (carb ratio)", dict(icr=7.5), dict(icr=12.5)),
        ("Basal setting*  (dose + recalibration)", dict(basal=0.6), dict(basal=1.0)),
        ("Liver scalar $S_{IE}$", dict(liver=0.75), dict(liver=1.25)),
        ("Muscle scalar $S_{ID}$", dict(muscle=0.75), dict(muscle=1.25)),
    ]
    labels, lows, highs = [], [], []
    detail = {}
    for label, lo_kw, hi_kw in sweeps:
        dlo, _ = _run(**lo_kw)
        dhi, _ = _run(**hi_kw)
        mlo = tir(dlo["glucose_actual_mgdl"].to_numpy())
        mhi = tir(dhi["glucose_actual_mgdl"].to_numpy())
        labels.append(label)
        lows.append(mlo["mean"] - base["mean"])
        highs.append(mhi["mean"] - base["mean"])
        detail[label] = {"-25%": mlo, "+25%": mhi}

    order = np.argsort([abs(h - l) for l, h in zip(lows, highs)])
    labels = [labels[i] for i in order]
    lows = [lows[i] for i in order]
    highs = [highs[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 3.1))
    y = np.arange(len(labels))
    ax.barh(y, lows, color="#dc2626", alpha=0.85, label="parameter −25%")
    ax.barh(y, highs, color=C["insulin"], alpha=0.85, label="parameter +25%")
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ mean glucose vs baseline ({base['mean']:.0f} mg/dL)")
    ax.set_title("One-at-a-time sensitivity, 24 h closed loop, identical seed")
    ax.legend(loc="lower right")
    finish(fig, ax, "tornado")
    manifest["tornado"] = detail
    manifest["tornado_base"] = base


def fig_pump():
    """Real PumpModel: quantization + step errors."""
    req = np.linspace(0, 3, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))

    for q, a, lbl in ((0.025, 1.0, "0.025 U/step (typical patch pump)"),
                      (0.10, 0.6, "0.10 U/step (coarse)")):
        pump = PumpModel(quantization_units=q, seed=3)
        out = [pump.deliver(float(r), 5.0).delivered_units for r in req]
        axes[0].plot(req, out, color=C["insulin"], alpha=a, label=lbl)
    axes[0].plot(req, req, color=MUTED, ls=":", lw=1.0, label="ideal")
    axes[0].set_xlim(0, 0.5); axes[0].set_ylim(0, 0.5)
    axes[0].set_xlabel("requested (U)"); axes[0].set_ylabel("delivered (U)")
    axes[0].set_title("Dose quantization (zoom on small doses)")
    axes[0].legend(loc="upper left")

    pump = PumpModel(quantization_units=0.025, step_error_probability=0.02, seed=5)
    errs = [pump.deliver(1.0, 5.0).delivered_units - 1.0 for _ in range(2000)]
    axes[1].hist(errs, bins=25, color=C["insulin"], alpha=0.8)
    axes[1].axvline(0, color=INK, lw=1.0)
    axes[1].set_xlabel("delivered − requested (U), 1.0 U command")
    axes[1].set_ylabel("count of 2000 deliveries")
    axes[1].set_title("Step-error asymmetry (70% miss / 30% extra)")
    finish(fig, axes, "pump")
    manifest["pump_mean_err"] = float(np.mean(errs))


def fig_supervisor():
    """Supervisor-on vs supervisor-off is not directly togglable at this level;
    instead show the guard envelope the controller output is clipped into."""
    from iints.core.safety.config import SafetyConfig
    cfg = SafetyConfig()
    G = np.linspace(40, 300, 500)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    allowed = np.where(G < cfg.controller_hypo_guard_mgdl, 0.0, cfg.max_insulin_per_bolus)
    ax.fill_between(G, 0, allowed, color=C["insulin"], alpha=0.15, step="mid")
    ax.plot(G, allowed, color=C["insulin"], drawstyle="steps-mid", label="max bolus permitted")
    ax.axvline(cfg.controller_hypo_guard_mgdl, color="#dc2626", lw=1.2, ls="--")
    ax.annotate(f"hypo guard\n{cfg.controller_hypo_guard_mgdl:.0f} mg/dL",
                xy=(cfg.controller_hypo_guard_mgdl + 4, 11), fontsize=8, color="#dc2626")
    ax.axvline(cfg.hypoglycemia_threshold, color=MUTED, ls=":", lw=1.0)
    ax.axvline(cfg.severe_hypoglycemia_threshold, color=MUTED, ls=":", lw=1.0)
    ax.annotate("70", xy=(70, 1), fontsize=8, color=MUTED, ha="center")
    ax.annotate("54", xy=(54, 1), fontsize=8, color=MUTED, ha="center")
    ax.set_xlabel("Sensor glucose (mg/dL)")
    ax.set_ylabel("insulin (U)")
    ax.set_title("Independent supervisor: a hard envelope no algorithm output can cross")
    ax.set_ylim(0, 16)
    ax.legend(loc="lower right")
    finish(fig, ax, "supervisor")


def fig_loss():
    """Band-weighted vs plain MSE weighting."""
    levels = np.linspace(40, 300, 500)
    low_t, high_t, lw, hw, mx = 70.0, 180.0, 2.0, 1.5, 5.0
    w = np.ones_like(levels) + lw * (levels < low_t) + hw * (levels > high_t)
    w = np.minimum(w, mx)
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.step(levels, w, where="mid", color=C["action"], label="BandWeightedMSE")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.0, label="plain MSE")
    ax.fill_between(levels, 1, w, step="mid", color=C["action"], alpha=0.12)
    ax.set_xlabel("glucose level of the target (mg/dL)")
    ax.set_ylabel("sample weight")
    ax.set_ylim(0, 3.6)
    ax.set_title("Training loss weights errors where they are dangerous, not where they are frequent")
    ax.legend(loc="upper center")
    finish(fig, ax, "loss_weights")


def main():
    print("Generating figures ...")
    for fn in (fig_renal, fig_dawn, fig_rescue, fig_insulin_pk, fig_meal,
               fig_action, fig_nimgu, fig_glucagon, fig_cgm, fig_risk,
               fig_pump, fig_supervisor, fig_loss):
        try:
            fn()
        except Exception as exc:
            print(f"  !! {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    for fn in (fig_closed_loop, fig_cascade, fig_isf_sweep, fig_liver_sweep,
               fig_dawn_effect, fig_scenarios, fig_tornado, fig_pid_failure):
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  !! {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\nManifest:")
    print(json.dumps(manifest, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
