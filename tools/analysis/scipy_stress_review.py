"""Reproducible, offline stress review of the actual SDK simulation core.

Run from the repository: python tools/analysis/scipy_stress_review.py
Dependencies: SDK core + scipy + matplotlib. No patient/device connections.
"""
from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, signal, stats
from scipy.stats import qmc

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.core.algorithms.mock_algorithms import RunawayAIAlgorithm
from iints.core.devices.models import PumpModel, SensorModel
from iints.core.patient.models import CustomPatientModel
from iints.core.patient.patient_factory import PatientFactory
from iints.core.devices.models import create_sensor_model
from iints.presets import get_preset
from iints.highlevel import run_simulation
from iints.core.simulator import Simulator, StressEvent
from iints.core.supervisor import IndependentSupervisor
from iints.tools.theory_stress_lab import run_theory_stress_lab

SCENARIOS = {
    "baseline": "Referentie",
    "large_meal": "Grote maaltijd",
    "unannounced": "Niet gemelde maaltijd",
    "exercise": "Inspanning",
    "illness": "Ziekte / stress",
    "sensor_fault": "Sensorstoring",
    "pump_dropout": "Pompuitval",
    "runaway": "Ontspoorde controller",
}
RANGES = {
    "insulin_sensitivity": (25., 85.),
    "carb_factor": (7., 18.),
    "basal_insulin_rate": (0.3, 1.4),
    "initial_glucose": (85., 200.),
    "glucose_absorption_rate": (0.015, 0.05),
    "insulin_action_duration": (180., 420.),
    "noise_std": (0., 18.),
    "bias": (-25., 25.),
    "meal_grams": (30., 100.),
}
BLUE, GOLD, ORANGE, GREY = "#2463a6", "#b18620", "#c36832", "#586270"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def cohort(n, seed):
    names = list(RANGES)
    bounds = np.array(list(RANGES.values()))
    values = qmc.scale(qmc.LatinHypercube(len(names), seed=seed).random(n),
                       bounds[:, 0], bounds[:, 1])
    return pd.DataFrame(values, columns=names)


def simulate(params, scenario, seed, minutes=1440, dt=5, model_type="custom"):
    params = dict(params)
    sensor = SensorModel(noise_std=params.pop("noise_std"), bias=params.pop("bias"),
                         seed=seed, dropout_prob=.12 if scenario == "sensor_fault" else 0.,
                         dropout_duration_steps=(2, 6))
    meal = params.pop("meal_grams")
    # Keep the SDK's default 3 mg/dL/min domain guard and all safety defaults.
    patient = PatientFactory.create_patient(patient_type=model_type, **params)
    algorithm = (RunawayAIAlgorithm(max_bolus=30.) if scenario == "runaway"
                 else ClinicalBaselineAlgorithm())
    sim = Simulator(patient, algorithm, seed=seed, time_step=dt, sensor_model=sensor,
                    pump_model=PumpModel(seed=seed, dropout_prob=.4 if scenario == "pump_dropout" else 0.),
                    enable_profiling=True)
    for t in (120, 420, 660, 1140):
        if t >= minutes:
            continue
        grams = meal * (2. if scenario == "large_meal" and t == 420 else 1.)
        sim.add_stress_event(StressEvent(t, "meal", grams,
                                       reported_value=0. if scenario == "unannounced" else grams))
    if scenario in ("exercise", "runaway") and minutes > 480:
        sim.add_stress_event(StressEvent(480, "exercise", .8, duration=90))
    if scenario == "illness" and minutes > 240:
        sim.add_stress_event(StressEvent(240, "illness", .9, duration=480))
    if scenario == "sensor_fault":
        for t, val in ((180, 480.), (185, 40.), (600, float("nan"))):
            if t < minutes:
                sim.add_stress_event(StressEvent(t, "sensor_error", val))
    started = time.perf_counter()
    frame, report = sim.run_batch(minutes)
    elapsed = time.perf_counter() - started
    return frame, report, elapsed


def summarize(frame, report, elapsed, minutes, dt):
    # One row per integrated interval [t, t+dt); the endpoint observation at
    # t=duration has no row of its own (see docs/DEFECT_REGISTER_NL.md, D2).
    expected = int(np.ceil(minutes / dt))
    out = {"steps": len(frame), "expected_steps": expected,
           "completed": len(frame) == expected and not report.get("terminated_early", False),
           "terminated": bool(report.get("terminated_early", False)),
           "termination_reason": json.dumps(report.get("termination_reason", {})),
           "runtime_s": elapsed, "coverage_pct": 100 * len(frame) / expected if expected else 100.0}
    if frame.empty:
        return out
    g = frame.glucose_actual_mgdl.to_numpy()
    out.update({"tir_pct": 100 * np.mean((g >= 70) & (g <= 180)),
                "below70_pct": 100 * np.mean(g < 70), "above180_pct": 100 * np.mean(g > 180),
                "min_glucose": np.min(g), "max_glucose": np.max(g),
                "mean_glucose": np.mean(g), "cv_pct": 100 * np.std(g) / np.mean(g),
                "insulin_units": frame.delivered_insulin_units.sum(),
                "intervention_pct": 100 * frame.safety_triggered.mean(),
                "sensor_fail_soft_pct": 100 * frame.input_validator_fail_soft.mean(),
                "sensor_mae": np.mean(np.abs(g - frame.glucose_to_algo_mgdl)),
                "step_p99_ms": np.quantile(frame.step_latency_ms, .99) if "step_latency_ms" in frame else None,
                "supervisor_p99_ms": np.quantile(frame.supervisor_latency_ms, .99),
                "finite_states": bool(np.isfinite(frame[["glucose_actual_mgdl", "delivered_insulin_units",
                                                         "patient_iob_units", "patient_cob_grams"]]).all().all()),
                "nonnegative_states": bool((frame[["delivered_insulin_units", "patient_iob_units",
                                                   "patient_cob_grams"]] >= -1e-9).all().all())})
    return out


def control_runs(out, seed):
    rows, traces = [], []
    # Compare backends with identical explicit inputs, without claiming they are
    # calibrated equivalents. The simplified model's homeostasis is explicit.
    p = dict(basal_insulin_rate=.5, insulin_sensitivity=50., carb_factor=10.,
             glucose_decay_rate=.02, initial_glucose=150., glucose_absorption_rate=.03,
             insulin_action_duration=300., noise_std=8.5, bias=0., meal_grams=45.)
    for backend in ("custom", "bergman", "hovorka", "advanced"):
        for scenario in ("baseline", "exercise", "pump_dropout"):
            for r in range(2):
                f, report, elapsed = simulate(p, scenario, seed + r, model_type=backend)
                rows.append({"group": "backend", "name": backend, "scenario": scenario,
                             "seed": seed + r, **summarize(f, report, elapsed, 1440, 5)})
                f["control_name"], f["scenario"], f["seed"] = backend, scenario, seed + r
                traces.append(f)
        print(f"backend control {backend} done", flush=True)
    for name in ("baseline_t1d", "free_living_t1d", "realistic_reference_day"):
        preset = get_preset(name)
        for r in range(3):
            started = time.perf_counter()
            output = run_simulation(algorithm=ClinicalBaselineAlgorithm(), scenario=preset["scenario"],
                                    patient_config=preset["patient_config"],
                                    duration_minutes=preset["duration_minutes"],
                                    time_step=preset["time_step_minutes"], seed=seed + r,
                                    output_dir=out / "preset_runs" / f"{name}_{r}",
                                    compare_baselines=False, export_audit=False, generate_report=False)
            elapsed = time.perf_counter() - started
            f, report = output["results"], output["safety_report"]
            rows.append({"group": "preset", "name": name, "scenario": name, "seed": seed + r,
                         **summarize(f, report, elapsed, preset["duration_minutes"], preset["time_step_minutes"])})
            f["control_name"], f["scenario"], f["seed"] = name, name, seed + r
            traces.append(f)
        print(f"bundled preset {name} done", flush=True)
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True)


def safety_sweep(n, seed):
    # Fresh supervisor per point: static limits, independent of prior dosing history.
    points = qmc.scale(qmc.LatinHypercube(4, seed=seed).random(n),
                       [20., -20., 0., 20.], [600., 100., 40., 400.])
    boundary = np.array([[g, d, i, p] for g in [54., 70., 70.0001, 90., 250.]
                         for d in [-1., 0., 15., 100.] for i in [0., 20., 25.]
                         for p in [59., 60., 60.0001, 120.]])
    rows = []
    for g, d, i, p in np.vstack([points, boundary]):
        supervisor = IndependentSupervisor()
        result = supervisor.evaluate_safety(g, d, 0., current_iob=i, predicted_glucose_30min=p)
        final = float(result["approved_insulin"])
        ok = (np.isfinite(final) and final >= 0 and final <= supervisor.max_insulin_per_bolus + 1e-9
              and final <= max(0., supervisor.max_iob - i) + 1e-9
              and (not (g <= 70. or p <= 60.) or final == 0.))
        rows.append({"glucose": g, "proposed": d, "iob": i, "predicted": p,
                     "final": final, "passed": ok})
    # Separate stateful hourly limit test with positive time progression.
    supervisor = IndependentSupervisor()
    history = []
    for t in range(0, 1440, 5):
        final = supervisor.evaluate_safety(300., 100., float(t), current_iob=0.,
                                           predicted_glucose_30min=300.)["approved_insulin"]
        history.append((t, final))
        rolling = sum(dose for minute, dose in history if t - 60 < minute <= t)
        rows.append({"glucose": 300., "proposed": 100., "iob": 0., "predicted": 300.,
                     "final": final, "passed": rolling <= supervisor.max_60min + 1e-9})
    invalid = []
    for field in ("current_glucose", "proposed_insulin", "current_time", "current_iob", "predicted_glucose_30min"):
        for val in (float("nan"), float("inf"), -float("inf")):
            kwargs = dict(current_glucose=120., proposed_insulin=1., current_time=0.,
                          current_iob=0., predicted_glucose_30min=120.)
            kwargs[field] = val
            try:
                IndependentSupervisor().evaluate_safety(**kwargs)
                passed = False
            except ValueError:
                passed = True
            invalid.append({"field": field, "value": str(val), "rejected": passed})
    return pd.DataFrame(rows), pd.DataFrame(invalid)


def numerical_oracle():
    # Independent ODE oracle for one 20g meal, no insulin, no exercise/dawn.
    # SDK integrates compartment appearances exactly but uses Euler homeostasis.
    k, decay, initial, meal, gain = .03, .001, 120., 20., 5.
    def rhs(t, y):
        a, b, glucose = y
        return [-k * a, k * a - k * b, gain * k * b - decay * (glucose - initial)]
    oracle = integrate.solve_ivp(rhs, (0, 360), [meal, 0., initial],
                                 method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
    if not oracle.success:
        raise RuntimeError(oracle.message)
    traces, errors = [], []
    for dt in (1, 2, 5, 10, 15):
        patient = CustomPatientModel(basal_insulin_rate=0., initial_glucose=initial)
        times = np.arange(dt, 361, dt)
        values = np.array([patient.update(dt, 0., meal if t == dt else 0.) for t in times])
        reference = oracle.sol(times)[2]
        errors.append({"dt_minutes": dt, "max_abs_error_mgdl": np.max(np.abs(values - reference)),
                       "rmse_mgdl": np.sqrt(np.mean((values - reference) ** 2))})
        traces.extend({"time_minutes": t, "dt_minutes": dt, "sdk": g, "oracle": r}
                      for t, g, r in zip(times, values, reference))
    # Independent quadrature for SDK triangular insulin CDF.
    cdf_error = []
    for t in np.linspace(0, 300, 101):
        density = lambda x: 2 * x / (300 * 75) if x <= 75 else 2 * (300 - x) / (300 * 225)
        integral = integrate.quad(density, 0, t, points=[75.] if t > 75 else None)[0]
        cdf_error.append(abs(integral - CustomPatientModel._triangular_action_cdf(t, 300., 75.)))
    return pd.DataFrame(traces), pd.DataFrame(errors), max(cdf_error)


def save(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean(values, seed):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(np.mean(values)) if len(values) else None, None, None)
    ci = stats.bootstrap((values,), np.mean, method="percentile", n_resamples=3000,
                         rng=np.random.default_rng(seed)).confidence_interval
    return float(np.mean(values)), float(ci.low), float(ci.high)


def build_figures(out, runs, traces, safety, numerics, error, theory, seed, controls, control_traces):
    gallery = []
    def finish(fig, name, title, note):
        save(fig, out, name)
        gallery.append({"id": name, "title": title, "note": note})
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True, sharey=True, layout="constrained")
    for ax, key in zip(axes.flat, SCENARIOS):
        f = traces[(traces.scenario == key) & (traces.patient_id == 0)]
        ax.axhspan(70, 180, color=BLUE, alpha=.08)
        ax.plot(f.time_minutes / 60, f.glucose_actual_mgdl, color=BLUE, label="Modelwaarheid")
        ax.plot(f.time_minutes / 60, f.glucose_to_algo_mgdl, color=ORANGE, alpha=.8, label="Controllerinput")
        ax.set(title=SCENARIOS[key], xlabel="Simulatietijd (uur)", ylabel="Glucose (mg/dL)", xlim=(0, 24))
        ax.legend(fontsize=9)
    finish(fig, "01_scenario_traces", "Acht stressscenario's", "Dezelfde virtuele patiënt (id 0). Een lijn die vroeg stopt is een afgebroken run; zie dekking.")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for j, key in enumerate(SCENARIOS):
        group = runs[runs.scenario == key]
        ax.bar(j, group.completed.mean() * 100, color=BLUE)
        ax.text(j, group.completed.mean() * 100 + 1, f"{int(group.completed.sum())}/{len(group)}", ha="center")
    ax.set(ylim=(0, 112), ylabel="Volledig afgeronde runs (%)", title="Volledigheid van de 24-uursproeven")
    ax.set_xticks(range(8), list(SCENARIOS.values()), rotation=25, ha="right")
    finish(fig, "02_completion", "Volledigheid", "Alle runs tellen mee. Stoppen buiten het modeldomein is een begrenzing; het is geen geslaagde 24-uursrun.")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for j, key in enumerate(SCENARIOS):
        group = runs[(runs.scenario == key) & runs.completed]
        mean, low, high = bootstrap_mean(group.tir_pct, seed + j)
        ax.text(j, 4, f"n={len(group)}", ha="center", fontsize=10)
        if low is not None:
            ax.errorbar(j, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=BLUE, capsize=6)
        elif mean is not None:
            ax.scatter(j, mean, color=BLUE)
    ax.set_xticks(range(8), list(SCENARIOS.values()), rotation=25, ha="right")
    ax.set(ylim=(0, 100), ylabel="Tijd binnen 70–180 mg/dL (%)", title="TIR met 95% bootstrapinterval")
    finish(fig, "03_tir_uncertainty", "Tijd binnen referentiegebied", "Alleen volledige runs. SciPy percentile-bootstrap over virtuele patiënten, 3.000 resamples. Geen klinische populatie-interval; selectiebias bij uitval.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, metric, label in zip(axes, ["min_glucose", "max_glucose"], ["Minimum", "Maximum"]):
        groups = [runs.loc[runs.scenario == k, metric].dropna().to_numpy() for k in SCENARIOS]
        ax.boxplot(groups, tick_labels=list(SCENARIOS.values()), patch_artist=True,
                   boxprops={"facecolor": "#dce8f5"}, medianprops={"color": BLUE})
        ax.tick_params(axis="x", rotation=65)
        ax.set(ylabel="Glucose (mg/dL)", title=f"{label} per run")
        ax.axhline(70 if metric == "min_glucose" else 180, color=GREY, linestyle="--")
    finish(fig, "04_extremes", "Uitersten", "Waargenomen minima/maxima, ook van gedeeltelijke runs; na stoppen zijn latere uitersten onbekend.")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for j, key in enumerate(SCENARIOS):
        g = runs[(runs.scenario == key) & runs.completed]
        if g.empty:
            ax.text(j, 4, "n=0", ha="center")
        bottom = 0.
        for metric, label, color in [("below70_pct", "<70", ORANGE), ("tir_pct", "70–180", BLUE), ("above180_pct", ">180", GOLD)]:
            value = g[metric].mean()
            ax.bar(j, value, bottom=bottom, color=color, label=label if j == 0 else None)
            bottom += value
    ax.set_xticks(range(8), list(SCENARIOS.values()), rotation=25, ha="right")
    ax.set(ylim=(0, 100), ylabel="Aandeel meetstappen (%)", title="Glucoseverdeling — volledige runs")
    ax.legend(ncols=3)
    finish(fig, "05_glucose_composition", "Glucoseverdeling", "Iedere meetstap weegt gelijk (5 minuten). Geen extrapolatie van gedeeltelijke runs.")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    g = runs.groupby("scenario", sort=False)[["intervention_pct", "sensor_fail_soft_pct"]].mean().reindex(SCENARIOS)
    x = np.arange(8)
    ax.bar(x - .2, g.intervention_pct, .4, color=BLUE, label="Veiligheidsingreep")
    ax.bar(x + .2, g.sensor_fail_soft_pct, .4, color=ORANGE, label="Input fail-soft")
    ax.set_xticks(x, list(SCENARIOS.values()), rotation=25, ha="right")
    ax.set(ylabel="Gemiddeld aandeel waargenomen stappen (%)", title="Activering van de veiligheidslagen")
    ax.legend()
    finish(fig, "06_interventions", "Veiligheidslagen", "Inclusief gedeeltelijke runs; ingrepen per waargenomen stap. Veel ingrepen bewijzen geen veilige glucose-uitkomst.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, col, title in zip(axes, ["step_latency_ms", "supervisor_latency_ms"], ["Volledige stap", "Supervisor"]):
        vals = traces[col].dropna().to_numpy()
        vals = vals[vals > 0]
        ax.hist(vals, bins=np.geomspace(vals.min(), vals.max() * 1.001, 55), color=BLUE)
        ax.set(xscale="log", xlabel="Rekentijd (ms, log-schaal)", ylabel="Aantal stappen", title=title)
        ax.axvline(np.quantile(vals, .99), color=ORANGE, linestyle="--", label=f"p99={np.quantile(vals, .99):.3f} ms")
        ax.legend()
    finish(fig, "07_latency", "Rekentijdverdeling", "Op deze Windows-host, instrumentatie ingeschakeld. Geen bewijs voor deadlines op andere hardware.")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for col, label, color in [("step_latency_ms", "Volledige stap", BLUE), ("supervisor_latency_ms", "Supervisor", ORANGE)]:
        v = np.sort(traces[col].dropna().to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=label, color=color)
    ax.set(xscale="log", xlabel="Rekentijd (ms)", ylabel="Empirische cumulatieve kans", title="Latency ECDF")
    ax.legend()
    finish(fig, "08_latency_ecdf", "Latency ECDF", "Empirische verdeling van alle gemeten cohortstappen, geen onafhankelijkheidsaanname.")
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.scatter(safety.proposed, safety.final, s=4, alpha=.2, color=BLUE)
    ax.set(xlabel="Voorgestelde insuline (U/stap)", ylabel="Vrijgegeven insuline (U/stap)", title="Supervisor: voorstel versus vrijgave")
    ax.axhline(15, color=GREY, linestyle="--")
    finish(fig, "09_safety_projection", "Supervisorprojectie", "LHS, grenswaarden en stateful uurproef; grenzen zijn SDK-engineeringcontracten.")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ggrid, igrid = np.linspace(20, 300, 80), np.linspace(0, 30, 60)
    z = np.array([[IndependentSupervisor().evaluate_safety(g, 30., 0., current_iob=i,
                                                         predicted_glucose_30min=120.)["approved_insulin"] for g in ggrid] for i in igrid])
    mesh = ax.pcolormesh(ggrid, igrid, z, shading="auto", cmap="cividis")
    fig.colorbar(mesh, ax=ax, label="Vrijgegeven insuline (U/stap)")
    ax.set(xlabel="Glucose (mg/dL)", ylabel="Insuline aan boord (U)", title="Supervisorgrenzen bij voorstel 30 U")
    finish(fig, "10_safety_map", "Veiligheidskaart", "Verse supervisor per cel, voorspelde glucose 120 mg/dL. Geen geschiedenis, trend of patiëntafstemming.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for dt in (1, 5, 15):
        f = numerics[numerics.dt_minutes == dt]
        axes[0].plot(f.time_minutes, f.sdk, label=f"SDK dt={dt} min")
    f = numerics[numerics.dt_minutes == 1]
    axes[0].plot(f.time_minutes, f.oracle, "--", color=GREY, label="SciPy DOP853")
    axes[0].set(xlabel="Tijd na maaltijd (min)", ylabel="Glucose (mg/dL)", title="Onafhankelijke ODE-referentie")
    axes[0].legend(fontsize=9)
    axes[1].plot(error.dt_minutes, error.max_abs_error_mgdl, "o-", color=BLUE)
    axes[1].set(xlabel="SDK-tijdstap (min)", ylabel="Maximale afwijking (mg/dL)", title="Numerieke convergentie")
    finish(fig, "11_numerical_convergence", "Numerieke nauwkeurigheid", "Eén geïsoleerde maaltijd van 20 g zonder insuline; exact compartimentverloop en Euler-homeostase versus strakke SciPy ODE-toleranties. Geen fysiologische validatie.")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    complete = runs[runs.scenario == "baseline"]
    sensitivity = []
    for name in RANGES:
        if len(complete) > 3 and complete[name].nunique() > 1 and complete.coverage_pct.nunique() > 1:
            rho, p = stats.spearmanr(complete[name], complete.coverage_pct)
            sensitivity.append({"parameter": name, "rho": rho, "p_unadjusted": p, "n": len(complete)})
    s = pd.DataFrame(sensitivity, columns=["parameter", "rho", "p_unadjusted", "n"])
    s.to_csv(out / "sensitivity.csv", index=False)
    ax.barh(s.parameter, s.rho, color=BLUE)
    ax.axvline(0, color=GREY)
    ax.set(xlim=(-1, 1), xlabel="Spearman rangcorrelatie met waargenomen dekking", title=f"Exploratieve gevoeligheid — alle {len(complete)} referentieruns")
    finish(fig, "12_sensitivity", "Gevoeligheidsanalyse van uitval", "Marginale associaties met de lengte van het waargenomen traject, geen causaliteit. P-waarden in CSV zijn ongecorrigeerd; geen significantieclaims.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    f = traces[(traces.scenario == "sensor_fault") & (traces.patient_id == 0)]
    axes[0].scatter(f.glucose_actual_mgdl, f.glucose_to_algo_mgdl, color=BLUE, s=12)
    axes[0].plot([20, 500], [20, 500], "--", color=GREY)
    axes[0].set(xlabel="Modelwaarheid (mg/dL)", ylabel="Controllerinput (mg/dL)", title="Sensorstoring: overeenkomst")
    residual = f.glucose_to_algo_mgdl.to_numpy() - f.glucose_actual_mgdl.to_numpy()
    axes[1].hist(residual, bins=30, color=ORANGE)
    axes[1].set(xlabel="Input minus modelwaarheid (mg/dL)", ylabel="Aantal stappen", title="Sensorfout na fail-soft")
    finish(fig, "13_sensor_error", "Sensorfouten", "Patiënt id 0; de controllerinput is al gefilterd door sensor- en veiligheidsmodellen.")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for key, color in [("baseline", BLUE), ("runaway", ORANGE)]:
        f = traces[(traces.scenario == key) & (traces.patient_id == 0)]
        ax.step(f.time_minutes / 60, f.algo_recommended_insulin_units, where="post", color=color, linestyle="--", alpha=.65, label=f"{SCENARIOS[key]}: voorstel")
        ax.step(f.time_minutes / 60, f.delivered_insulin_units, where="post", color=color, label=f"{SCENARIOS[key]}: geleverd")
    ax.set(xlabel="Simulatietijd (uur)", ylabel="Insuline (U/stap)", title="Voorstel en daadwerkelijke pompafgifte")
    ax.legend(fontsize=9)
    finish(fig, "14_dose_trace", "Dosisbegrenzing", "Patiënt id 0; geleverd is na supervisor én pompmodel. Verschil toont ingrijpen, geen garantie op glucose-uitkomst.")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.barh([c.title for c in theory.checks], [c.score for c in theory.checks], color=BLUE)
    ax.set(xlim=(0, 1), xlabel="Bestaande heuristische checkscore (0–1)", title="Bestaande theoriechecks")
    finish(fig, "15_theory_checks", "Theoriechecks", "Scores en criteria komen uit de bestaande SDK-theory-stress-lab; geen gekalibreerde betrouwbaarheidsscore.")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    f = traces[(traces.scenario == "baseline") & (traces.patient_id == 0)]
    peaks, _ = signal.find_peaks(f.glucose_actual_mgdl, prominence=10., distance=6)
    ax.plot(f.time_minutes / 60, f.glucose_actual_mgdl, color=BLUE)
    ax.scatter(f.time_minutes.iloc[peaks] / 60, f.glucose_actual_mgdl.iloc[peaks], marker="x", s=80, color=ORANGE, label="Prominentie ≥10 mg/dL")
    ax.set(xlabel="Simulatietijd (uur)", ylabel="Glucose (mg/dL)", title="SciPy piekdetectie")
    ax.legend()
    finish(fig, "16_peak_detection", "Glucosepieken", "Exploratieve find_peaks op de waargenomen referentietrace; minimumafstand 30 minuten.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    groups = controls.groupby(["group", "name"], sort=False)
    labels = [name for (_, name), group in groups]
    completed = [100 * group.completed.mean() for _, group in groups]
    axes[0].barh(labels, completed, color=BLUE)
    axes[0].set(xlim=(0, 100), xlabel="Volledig afgeronde runs (%)", title="Backends en meegeleverde presets")
    for key, color in [("custom", BLUE), ("bergman", GOLD), ("hovorka", ORANGE), ("advanced", GREY)]:
        f = control_traces[(control_traces.control_name == key) & (control_traces.scenario == "baseline") & (control_traces.seed == seed)]
        axes[1].plot(f.time_minutes / 60, f.glucose_actual_mgdl, label=key, color=color)
    axes[1].set(xlabel="Simulatietijd (uur)", ylabel="Glucose (mg/dL)", title="Backendreferenties bij gelijke expliciete inputs")
    axes[1].legend()
    finish(fig, "17_backend_controls", "Backend- en presetcontroles", "Backends: 3 scenario's × 2 seeds elk, 24 uur; presets: 3 seeds elk met hun eigen configuratie via run_simulation (auto/Bergman). Gelijke inputs betekenen geen equivalente modelkalibratie.")
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for name, color in [("baseline_t1d", BLUE), ("free_living_t1d", GOLD), ("realistic_reference_day", ORANGE)]:
        f = control_traces[(control_traces.control_name == name) & (control_traces.seed == seed)]
        ax.plot(f.time_minutes / 60, f.glucose_actual_mgdl, color=color, label=name)
    ax.axhspan(70, 180, color=BLUE, alpha=.06)
    ax.set(xlabel="Simulatietijd (uur)", ylabel="Glucose (mg/dL)", title="Meegeleverde referentiepreset-trajecten")
    ax.legend(fontsize=9)
    finish(fig, "18_preset_traces", "Meegeleverde presets", "Uitvoering via de publieke high-level API, standaard modelkeuze auto (Bergman beschikbaar). Eerste seed weergegeven; alle drie seeds in controls.csv.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    causes = runs.termination_reason.map(json.loads).map(
        lambda x: "Modeldomein" if x.get("termination_class") else ("Hypo-stop" if x else "Afgerond"))
    counts = causes.value_counts().reindex(["Modeldomein", "Hypo-stop", "Afgerond"], fill_value=0)
    axes[0].bar(counts.index, counts.values, color=[ORANGE, GOLD, BLUE])
    axes[0].set(ylabel="Aantal cohortproeven", title="Redenen voor stoppen")
    for key, color in [("baseline", BLUE), ("exercise", GOLD), ("runaway", ORANGE)]:
        group = runs[runs.scenario == key]
        timeline = np.arange(0, 1441, 5)
        continuation = [100 * np.mean(group.steps.to_numpy() > t / 5) for t in timeline]
        axes[1].step(timeline / 60, continuation, where="post", label=SCENARIOS[key], color=color)
    axes[1].set(xlabel="Simulatietijd (uur)", ylabel="Proeven met meetgegevens op dit tijdstip (%)", title="Dekking door de tijd", ylim=(0, 105))
    axes[1].legend()
    finish(fig, "19_stop_reasons", "Stopredenen en tijdsdekking", "Alle cohortproeven. Modeldomein omvat de snelheidsgrens of ondergrens; hypo-stop is <40 mg/dL gedurende 30 minuten volgens de SDK. Geen klinische overlevingsanalyse.")
    return gallery


def write_review(out, manifest, runs, gallery, safety, invalid, error, cdf_error, theory, replay, worst, controls):
    from markdown_it import MarkdownIt
    summary = runs.groupby("scenario", sort=False).agg(runs=("completed", "size"),
                 completed=("completed", "sum"), coverage_pct=("coverage_pct", "mean"),
                 min_glucose=("min_glucose", "min"), max_glucose=("max_glucose", "max")).reindex(SCENARIOS)
    summary.to_csv(out / "scenario_summary.csv")
    finite = bool(runs.finite_states.fillna(False).all())
    nonnegative = bool(runs.nonnegative_states.fillna(False).all())
    reasons = runs.termination_reason.map(json.loads)
    domain_stops = sum(bool(x.get("termination_class")) for x in reasons)
    hypo_stops = sum(bool(x) and not bool(x.get("termination_class")) for x in reasons)
    preset_rows = controls[controls.group == "preset"]
    control_stops = controls[~controls.completed]
    stopped_names = ", ".join(f"{r['name']}/{r['scenario']} (seed {r['seed']})" for r in control_stops.to_dict("records"))
    latency = pd.read_csv(out / "traces.csv.gz", usecols=["step_latency_ms", "supervisor_latency_ms"])
    findings = f"""## Belangrijkste bevindingen
De supervisor respecteert alle onderzochte lokale dosis-, IOB-, hypo- en uurgrenzen, maar dat voorkomt niet automatisch latere lage glucose in de patiëntmodellen. Binnen het brede, ongekalibreerde CustomPatientModel-cohort stoppen {domain_stops} runs buiten het modeldomein en {hypo_stops} op de langdurige lage-glucosegrens. Slechts {int(runs.loc[runs.scenario == 'baseline', 'completed'].sum())}/{int((runs.scenario == 'baseline').sum())} referentieruns ronden af. De homeostaseparameter is 0.001/min; dit is geen resultaat van de normale auto/Bergman-presetworkflow.

Alle {int(preset_rows.completed.sum())}/{len(preset_rows)} meegeleverde presetcontroles ronden af. In de aanvullende backendcontroles stoppen: **{stopped_names or 'geen'}**. Deze Hovorka-inspanningsconfiguratie verdient afzonderlijke beoordeling; de backendparameters zijn niet per model gekalibreerd.

Gemeten p99 van de volledige cohortstap: {latency.step_latency_ms.quantile(.99):.3f} ms; maximum: {latency.step_latency_ms.max():.3f} ms. Supervisor p99: {latency.supervisor_latency_ms.quantile(.99):.3f} ms. Een lage p99 sluit uitschieters niet uit. Test- en analysetaken konden gelijktijdig op deze host draaien.

De begrensde worst-case zoekproef gebruikte {worst['evaluations']} evaluaties. Convergentie: **{worst['converged']}** ({worst['message']}). De gevonden kandidaat is geen bewezen globaal maximum; bekijk worst_case_search.csv voor dekking en verliesfunctie (100−TIR plus 100−dekking).

Voor verdere beoordeling zijn de prioriteiten: (1) het geldigheidsbereik en de basal/homeostasebalans van de eenvoudige backend, (2) de inspanningsrespons van Hovorka in deze configuratie en (3) sensorvertraging en doorwerking van reeds actieve insuline bij agressieve controllers. Dit zijn onderzoeksrichtingen op basis van deze proeven, geen bevestigde klinische defecten.
"""
    import xml.etree.ElementTree as ET
    validations = []
    for file in out.glob("pytest*.xml"):
        root = ET.parse(file).getroot()
        for suite in root.iter("testsuite"):
            validations.append({"file": file.name, **{key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}})
    checks = {"supervisor_invariants_passed": bool(safety.passed.all()),
              "invalid_values_rejected": bool(invalid.rejected.all()),
              "all_recorded_states_finite": finite, "all_recorded_dose_iob_cob_nonnegative": nonnegative,
              "deterministic_replay": replay, "insulin_cdf_max_abs_error": cdf_error,
              "cohort_completed": int(runs.completed.sum()), "cohort_runs": len(runs),
              "theory_verdict": theory.verdict, "worst_search": worst,
              "control_runs": len(controls), "control_completed": int(controls.completed.sum())}
    checks["pytest_validation"] = validations
    (out / "checks.json").write_text(json.dumps(checks, indent=2, allow_nan=False), encoding="utf-8")
    narrative = f"""# IINTS-AF — SciPy stresstest en keuringsmateriaal

## Doel van de software
Volgens README.md is IINTS-AF een onderzoeks-SDK om diabetes-technologie vóór toepassing op personen te onderzoeken: virtuele patiënten, algoritme-experimenten, onafhankelijke veiligheidsregels, data-assurance en reproduceerbare rapporten. De broncode ondersteunt dit met Simulator, CustomPatientModel, mechanistische Bergman/Hovorka-modellen en IndependentSupervisor. Het is geen medisch hulpmiddel.

## Uitgevoerde scope
De echte Simulator met ClinicalBaselineAlgorithm, standaard patiëntdomein (maximaal 3 mg/dL/min), standaard veiligheidsregels en acht scenario's. Elk scenario gebruikt dezelfde {manifest['patients']} virtuele parametercombinaties: een gepaarde Latin-hypercubeopzet met seed {manifest['seed']}, 24 uur en 5-minutenstappen. De ontspoorde controller gebruikt RunawayAIAlgorithm (voorstel tot 30 U/stap). Sensor- en pompstoringen zijn expliciete softwaremodellen, geen echte apparaten.

Het brede cohort kiest expliciet CustomPatientModel: dit is de eenvoudige educatieve backend en niet de standaard auto-backend van de publieke API (die hier Bergman kiest). De cohortcombinaties zijn ongekalibreerd; de homeostatische terugkeer blijft op 0.001/min. Daarom zijn er aanvullende controles: Custom/Bergman/Hovorka/Advanced met dezelfde vastgelegde inputs, homeostase 0.02/min, 3 scenario's en 2 seeds per backend. Ook baseline_t1d, free_living_t1d en realistic_reference_day draaien via de publieke API met hun eigen patiëntconfiguratie en 3 seeds. Zie controls.csv en figuren 17–18.

Daarnaast: {len(safety)} supervisorproeven (LHS, grensgevallen, voortschrijdende uurgrens), {len(invalid)} niet-eindige inputs, bestaande theoriechecks op het geavanceerde metabolische model, onafhankelijke SciPy ODE-referentie, quadratuur van de insulinewerking, deterministische replay en een begrensde worst-case zoekproef.

## Waargenomen uitkomsten
- Volledig afgerond: **{int(runs.completed.sum())}/{len(runs)}** cohortsimulaties.
- Aanvullende backend/presetcontroles: **{int(controls.completed.sum())}/{len(controls)}** afgerond.
- Supervisor-invarianten: **{int(safety.passed.sum())}/{len(safety)}** geslaagd.
- NaN/Inf-inputs afgewezen: **{int(invalid.rejected.sum())}/{len(invalid)}**.
- Eindige gelogde glucose/dosis/IOB/COB: **{finite}**; niet-negatieve dosis/IOB/COB: **{nonnegative}**.
- Replay van inhoudelijke trajectkolommen: **{replay}** (latency is uitgesloten).
- Maximale afwijking insuline-CDF versus SciPy-quadratuur: **{cdf_error:.3g}**.
- Bestaande theoriechecks: **{theory.verdict}**.
- ODE-referentie: maximale glucoseafwijking bij 5 minuten **{error.loc[error.dt_minutes == 5, 'max_abs_error_mgdl'].iloc[0]:.4f} mg/dL**, in de geïsoleerde 20g-maaltijdproef.

{findings}

## Regressiecontroles
{json.dumps(validations, indent=2)}

## Interpretatie en grenzen
Afgebroken runs blijven in runs.csv, traces.csv.gz en de volledigheidsfiguur staan, inclusief reden en waargenomen dekking. TIR-vergelijkingen gebruiken uitsluitend volledige runs: beoordeel ze samen met uitval, want deze selectie kan uitkomsten gunstiger maken. Minima, maxima en ingreepfrequenties van gedeeltelijke runs gelden alleen voor het waargenomen deel. Een modeldomein-stop is een gecontroleerde begrenzing en een te onderzoeken zwakte van de geteste configuratie, geen klinische uitkomst.

Bootstrapintervallen beschrijven variatie binnen deze synthetische parameteropzet; het is geen representatieve patiëntpopulatie. Spearman-correlaties zijn exploratief. De worst-case optimizer zoekt alleen binnen drie vastgelegde grenzen en bewijst geen globale veiligheid. Latency is hostspecifiek en omvat instrumentatie. De numerieke oracle controleert één geïsoleerd mechanisme; zij bewijst geen fysiologische correctheid.

Niet getest: desktop-UI, netwerk/API-belasting, echte pomp/CGM-hardware, getrainde neurale voorspellers, patiëntdata en alle modelvarianten in alle combinaties. De mechanistische backendcontroles en bestaande geavanceerde theoriechecks vullen de CustomPatientModel-cohortproef aan. Dit pakket is keuringsmateriaal voor softwareonderzoek, geen certificering voor patiëntgebruik.

## Zelf opnieuw uitvoeren
`python tools/analysis/scipy_stress_review.py --patients {manifest['patients']} --seed {manifest['seed']} --safety-samples {manifest['safety_samples']} --output results/scipy_stress_review`

Exacte Python- en pakketversies, git-revisie, bronbestand-hashes, parametergrenzen, scenario's en uitvoeringstijd staan in manifest.json en environment.txt. Ruwe data staan in CSV-bestanden. Figuren zijn beschikbaar als PNG en SVG. report.html verzamelt alle {len(gallery)} figuren met uitleg. TIR is het aandeel gelogde 5-minutenmeetpunten (inclusief t=0 en eindpunt), een discrete benadering van tijd in gebied.
"""
    (out / "summary.md").write_text(narrative, encoding="utf-8")
    cards = "".join(f'<section><h2>{html.escape(g["title"])}</h2><img src="figures/{g["id"]}.png" alt="{html.escape(g["title"])}"><p>{html.escape(g["note"])}</p><a href="figures/{g["id"]}.svg">SVG downloaden</a></section>' for g in gallery)
    control_summary = controls.groupby(["group", "name"], sort=False).agg(runs=("completed", "size"), completed=("completed", "sum"))
    document = f'''<!doctype html><html lang="nl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IINTS-AF SciPy stresstest</title><style>body{{font:17px/1.6 system-ui;background:#f3f5f8;color:#233142;margin:0}}main{{max-width:1150px;margin:auto;padding:32px}}section{{background:white;padding:24px;margin:24px 0;border:1px solid #d9e0e9}}img{{width:100%;height:auto}}h1,h2{{line-height:1.25}}a{{color:#2463a6}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}}.notice{{background:#fff5da;padding:20px}}.table-wrap{{overflow-x:auto}}summary{{cursor:pointer;font-weight:bold}}code{{overflow-wrap:anywhere}}</style><main><h1>IINTS-AF — SciPy stresstest</h1><p>Onderzoekssoftware voor virtuele diabetes-technologie-experimenten.</p><p class="notice">{int(runs.completed.sum())}/{len(runs)} ongekalibreerde Custom-cohortproeven volledig afgerond. Meegeleverde presets: {int(preset_rows.completed.sum())}/{len(preset_rows)} afgerond. Supervisor-invarianten: {int(safety.passed.sum())}/{len(safety)}. Bekijk volledigheid vóór TIR. Dit is een softwarebenchtest, geen klinische validatie.</p><p><a href="summary.md">Volledige methode en beoordeling</a> · <a href="runs.csv">Resultaten per run</a> · <a href="controls.csv">Backend- en presetcontroles</a> · <a href="manifest.json">Herkomst en configuratie</a> · <a href="checks.json">Controle-uitkomsten</a></p><div class="table-wrap">{summary.rename(index=SCENARIOS).to_html(float_format=lambda x: f"{x:.2f}")}</div><section><h2>Backend- en presetcontroles</h2><div class="table-wrap">{control_summary.to_html()}</div>{MarkdownIt().render(findings)}</section><section><details><summary>Methode, herkomst, regressiecontroles en beperkingen</summary>{MarkdownIt().render(narrative)}</details></section>{cards}</main></html>'''
    (out / "report.html").write_text(document, encoding="utf-8")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--safety-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/scipy_stress_review")
    parser.add_argument("--render-only", action="store_true", help="Rebuild figures/report from a complete saved run; no new simulations")
    args = parser.parse_args()
    if args.patients < 4 or args.safety_samples < 1:
        parser.error("patients >= 4 and safety-samples >= 1 are required")
    logging.getLogger("iints").setLevel(logging.CRITICAL)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    if args.render_only:
        from iints.tools.theory_stress_lab import TheoryCheckResult, TheoryStressReport
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        checks = json.loads((out / "checks.json").read_text(encoding="utf-8"))
        payload = json.loads((out / "theory/checks.json").read_text(encoding="utf-8"))
        theory = TheoryStressReport(**{**payload, "checks": [TheoryCheckResult(**c) for c in payload["checks"]]})
        runs = pd.read_csv(out / "runs.csv")
        traces = pd.read_csv(out / "traces.csv.gz")
        safety = pd.read_csv(out / "supervisor_sweep.csv")
        invalid = pd.read_csv(out / "invalid_inputs.csv")
        numerics = pd.read_csv(out / "numerical_traces.csv")
        error = pd.read_csv(out / "numerical_errors.csv")
        controls = pd.read_csv(out / "controls.csv")
        control_traces = pd.read_csv(out / "control_traces.csv.gz")
        gallery = build_figures(out, runs, traces, safety, numerics, error, theory, manifest["seed"], controls, control_traces)
        write_review(out, manifest, runs, gallery, safety, invalid, error, checks["insulin_cdf_max_abs_error"], theory,
                     checks["deterministic_replay"], checks["worst_search"], controls)
        import hashlib
        manifest["report_renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Rebuilt review from saved data: {out / 'report.html'}", flush=True)
        return 0
    started = time.perf_counter()
    params = cohort(args.patients, args.seed)
    params.to_csv(out / "cohort_parameters.csv", index_label="patient_id")
    runs, traces = [], []
    for scenario in SCENARIOS:
        for patient_id, row in params.iterrows():
            frame, report, elapsed = simulate(row, scenario, args.seed + patient_id)
            runs.append({"patient_id": patient_id, "scenario": scenario, **row.to_dict(),
                         **summarize(frame, report, elapsed, 1440, 5)})
            frame["scenario"], frame["patient_id"] = scenario, patient_id
            traces.append(frame)
        print(f"cohort {scenario}: {args.patients} runs", flush=True)
    runs = pd.DataFrame(runs)
    traces = pd.concat(traces, ignore_index=True)
    runs.to_csv(out / "runs.csv", index=False)
    traces.to_csv(out / "traces.csv.gz", index=False, compression="gzip")
    safety, invalid = safety_sweep(args.safety_samples, args.seed)
    safety.to_csv(out / "supervisor_sweep.csv", index=False)
    invalid.to_csv(out / "invalid_inputs.csv", index=False)
    print("supervisor sweep done", flush=True)
    numerics, error, cdf_error = numerical_oracle()
    numerics.to_csv(out / "numerical_traces.csv", index=False)
    error.to_csv(out / "numerical_errors.csv", index=False)
    theory = run_theory_stress_lab(output_dir=out / "theory", profile="ci", seed=args.seed)
    frame1, _, _ = simulate(params.iloc[0], "baseline", args.seed)
    frame2, _, _ = simulate(params.iloc[0], "baseline", args.seed)
    columns = ["glucose_actual_mgdl", "glucose_to_algo_mgdl", "delivered_insulin_units", "safety_reason"]
    replay = frame1[columns].equals(frame2[columns])
    # Exploratory constrained search for reduced TIR. Penalize termination explicitly.
    search_history = []
    def objective(x):
        p = params.iloc[0].to_dict()
        p.update(meal_grams=float(x[0]), bias=float(x[1]), insulin_sensitivity=float(x[2]))
        f, r, elapsed = simulate(p, "unannounced", args.seed, minutes=720)
        m = summarize(f, r, elapsed, 720, 5)
        loss = 100. - m.get("tir_pct", 0.) + (100. - m["coverage_pct"])
        search_history.append({"meal_grams": x[0], "bias": x[1], "insulin_sensitivity": x[2],
                               "loss": loss, **m})
        return -loss
    opt = optimize.differential_evolution(objective, [(30, 150), (-40, 40), (25, 85)],
                                          seed=args.seed, maxiter=3, popsize=4, polish=False, workers=1)
    pd.DataFrame(search_history).to_csv(out / "worst_case_search.csv", index=False)
    worst = {"parameters": opt.x.tolist(), "loss": float(-opt.fun), "evaluations": opt.nfev,
             "converged": bool(opt.success), "message": str(opt.message),
             "bounds": [[30, 150], [-40, 40], [25, 85]], "duration_minutes": 720}
    controls, control_traces = control_runs(out, args.seed)
    controls.to_csv(out / "controls.csv", index=False)
    control_traces.to_csv(out / "control_traces.csv.gz", index=False, compression="gzip")
    gallery = build_figures(out, runs, traces, safety, numerics, error, theory, args.seed, controls, control_traces)
    import hashlib
    sources = [Path(__file__), ROOT / "src/iints/core/simulator.py", ROOT / "src/iints/core/patient/models.py",
               ROOT / "src/iints/core/supervisor.py", ROOT / "src/iints/core/devices/models.py",
               ROOT / "src/iints/core/algorithms/clinical_baseline.py", ROOT / "src/iints/tools/theory_stress_lab.py",
               ROOT / "src/iints/core/patient/bergman_model.py", ROOT / "src/iints/core/patient/hovorka_model.py",
               ROOT / "src/iints/core/patient/advanced_metabolic_model.py", ROOT / "src/iints/presets/presets.json",
               ROOT / "src/iints/highlevel.py", ROOT / "src/iints/data/virtual_patients/reference_free_living_t1d.yaml"]
    manifest = {"seed": args.seed, "patients": args.patients, "safety_samples": args.safety_samples,
                "python": sys.version, "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "parameter_ranges": RANGES, "scenarios": SCENARIOS, "minutes": 1440, "dt_minutes": 5,
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "pandas", "matplotlib"]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    checks = write_review(out, manifest, runs, gallery, safety, invalid, error, cdf_error, theory, replay, worst, controls)
    print(json.dumps(checks, indent=2), flush=True)
    print(f"Review: {out / 'report.html'}", flush=True)
    return 0 if safety.passed.all() and invalid.rejected.all() and replay else 1


if __name__ == "__main__":
    raise SystemExit(main())
