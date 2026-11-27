"""Module B — Sensor & Scenario Simulation Engine for IINTS Simulation SDK 2.0.

Explicitly separates:
1. ``bg_true_mgdl``      — true simulated plasma glucose,
2. ``ig_mgdl``           — blood-to-interstitial delayed glucose,
3. ``cgm_clean_mgdl``    — sensor reading with noise/bias/drift before dropouts,
4. ``cgm_observed_mgdl`` — NaN when missing (under ``nan_blank``) or stale value (under ``stale_hold``),
5. ``cgm_displayed_mgdl``— exact value rendered to an algorithm/interface, with
                           ``is_dropout``, ``is_stale_displayed``, and ``steps_since_recovery``.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from iints.sdk2.contracts import MealEventSpec, PatientSpec, ScenarioSpec, SensorFaultSpec
from iints.sdk2.patients import create_virtual_patient

__all__ = ["build_meal_events", "simulate_patient_trajectory", "apply_sensor_fault_model"]


def _schedule_template(name: str, duration_hours: float) -> List[Tuple[float, float]]:
    """Return list of ``(hour_of_day, base_carbs_g)`` templates across ``duration_hours``."""
    n_days = int(np.ceil(duration_hours / 24.0))
    events: List[Tuple[float, float]] = []
    for day in range(n_days):
        offset = day * 24.0
        if name == "standard_day":
            events.extend([(offset + 7.5, 45.0), (offset + 12.5, 65.0), (offset + 18.5, 70.0)])
        elif name == "high_carb_stress":
            events.extend([(offset + 7.5, 75.0), (offset + 13.0, 95.0), (offset + 19.5, 105.0)])
        elif name == "late_fat_protein":
            events.extend([(offset + 8.0, 40.0), (offset + 13.0, 55.0), (offset + 21.0, 90.0)])
        elif name == "nocturnal_descent":
            events.extend([(offset + 8.0, 45.0), (offset + 13.0, 55.0), (offset + 19.0, 50.0)])
    return [(h, g) for h, g in events if h * 60.0 < duration_hours * 60.0 - 30.0]


def build_meal_events(
    scenario: ScenarioSpec,
    rng: np.random.Generator,
) -> List[MealEventSpec]:
    if scenario.meal_schedule == "custom" or scenario.meals:
        return list(scenario.meals)

    generated: List[MealEventSpec] = []
    for base_hour, base_grams in _schedule_template(scenario.meal_schedule, scenario.duration_hours):
        jitter = float(rng.normal(0.0, scenario.meal_time_jitter_min)) if scenario.meal_time_jitter_min > 0 else 0.0
        minute = float(np.clip(base_hour * 60.0 + jitter, 15.0, scenario.duration_hours * 60.0 - 30.0))
        grams = float(np.clip(base_grams * rng.normal(1.0, 0.12), 10.0, 180.0))
        if float(rng.random()) < scenario.unannounced_meal_fraction:
            announced = 0.0
        elif scenario.carb_announcement_error_sd > 0:
            announced = float(
                np.clip(grams * np.exp(rng.normal(0.0, scenario.carb_announcement_error_sd)), 0.0, 250.0)
            )
        else:
            announced = grams
        generated.append(
            MealEventSpec(
                minute=minute,
                carbs_g=round(grams, 2),
                announced_carbs_g=round(announced, 2),
            )
        )
    return generated


def simulate_patient_trajectory(
    patient_spec: PatientSpec,
    scenario: ScenarioSpec,
    seed: int,
) -> pd.DataFrame:
    """Simulate true physiological blood glucose, insulin delivery, and carb events."""
    rng = np.random.default_rng(seed)
    patient = create_virtual_patient(patient_spec)
    patient.reset()

    step = float(scenario.step_minutes)
    total_hours = float(scenario.warmup_hours + scenario.duration_hours)
    n_steps = int(round(total_hours * 60.0 / step))

    # Build meals for warmup + main window so state is realistically active at evaluation start
    full_scenario = scenario.model_copy(update={"duration_hours": total_hours})
    meals = build_meal_events(full_scenario, rng)

    carbs_physical = np.zeros(n_steps, dtype=float)
    carbs_announced = np.zeros(n_steps, dtype=float)
    bolus_u = np.zeros(n_steps, dtype=float)

    for meal in meals:
        idx = int(round(meal.minute / step))
        if not 0 <= idx < n_steps:
            continue
        carbs_physical[idx] += float(meal.carbs_g)
        ann = float(meal.carbs_g if meal.announced_carbs_g is None else meal.announced_carbs_g)
        carbs_announced[idx] += ann
        if ann > 0.0:
            err = (
                float(np.exp(rng.normal(0.0, scenario.bolus_relative_error_sd)))
                if scenario.bolus_relative_error_sd > 0
                else 1.0
            )
            # Nocturnal descent stress scenario adds mild over-bolus on evening meal
            if scenario.meal_schedule == "nocturnal_descent" and (meal.minute % 1440.0) >= 18.0 * 60.0:
                err *= 1.25
            dose = (ann / patient_spec.carb_factor_g_per_u) * err
            b_idx = int(np.clip(idx + int(round(meal.bolus_delay_min / step)), 0, n_steps - 1))
            bolus_u[b_idx] += dose

    basal_per_step = float(patient_spec.basal_u_per_h) * step / 60.0
    basal_u = np.full(n_steps, basal_per_step, dtype=float)
    insulin_u = basal_u + bolus_u

    bg_true = np.empty(n_steps, dtype=float)
    for k in range(n_steps):
        bg_true[k] = patient.step(
            step,
            insulin_units=float(insulin_u[k]),
            carbs_grams=float(carbs_physical[k]),
            current_time_minutes=float(k * step),
        )

    t_min = np.arange(n_steps, dtype=float) * step
    eval_offset_min = float(scenario.warmup_hours) * 60.0
    start_ts = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = start_ts + pd.to_timedelta(t_min, unit="m")

    return pd.DataFrame(
        {
            "timestamp_dt": timestamps,
            "t_min": t_min,
            "eval_hour": (t_min - eval_offset_min) / 60.0,
            "in_eval_window": t_min >= eval_offset_min,
            "minute_of_day": (timestamps.hour * 60 + timestamps.minute).to_numpy(dtype=float),
            "bg_true_mgdl": bg_true,
            "basal_u": basal_u,
            "bolus_u": bolus_u,
            "insulin_u": insulin_u,
            "carbs_physical_g": carbs_physical,
            "carbs_announced_g": carbs_announced,
        }
    )


def apply_sensor_fault_model(
    trajectory: pd.DataFrame,
    sensor_spec: SensorFaultSpec,
    seed: int,
) -> pd.DataFrame:
    """Apply interstitial lag, colored noise, drift, dropouts, and display behavior."""
    rng = np.random.default_rng(seed)
    df = trajectory.copy()
    n = len(df)
    t_min = df["t_min"].to_numpy(dtype=float)
    step = float(t_min[1] - t_min[0]) if n > 1 else 5.0
    bg = df["bg_true_mgdl"].to_numpy(dtype=float)

    # 1. Dead-time transport lag + first-order interstitial compartment
    lag_steps = int(round(sensor_spec.lag_minutes / step))
    bg_lagged = np.concatenate([np.full(min(lag_steps, n), bg[0]), bg[: max(0, n - lag_steps)]])
    alpha = 1.0 - float(np.exp(-step / sensor_spec.isf_tau_minutes))
    ig = np.empty(n, dtype=float)
    ig[0] = bg_lagged[0]
    for k in range(1, n):
        ig[k] = ig[k - 1] + alpha * (bg_lagged[k] - ig[k - 1])

    # 2. Stationary AR(1) sensor noise + systematic bias + linear/smooth drift
    noise = np.zeros(n, dtype=float)
    if sensor_spec.noise_sd_mgdl > 0.0:
        phi = float(sensor_spec.noise_ar1_phi)
        innov_sd = float(sensor_spec.noise_sd_mgdl) * float(np.sqrt(max(1e-9, 1.0 - phi * phi)))
        noise[0] = float(rng.normal(0.0, sensor_spec.noise_sd_mgdl))
        for k in range(1, n):
            noise[k] = phi * noise[k - 1] + float(rng.normal(0.0, innov_sd))

    drift = float(sensor_spec.drift_mgdl_per_day) * (t_min / 1440.0)
    cgm_clean = ig + float(sensor_spec.bias_mgdl) + drift + noise

    # 3. Optional compression-low artifact
    is_compression = np.zeros(n, dtype=bool)
    if sensor_spec.compression_low_at_hour is not None and sensor_spec.compression_low_minutes > 0:
        eval_h = df["eval_hour"].to_numpy(dtype=float)
        c_start = float(sensor_spec.compression_low_at_hour)
        c_end = c_start + float(sensor_spec.compression_low_minutes) / 60.0
        is_compression = (eval_h >= c_start - 1e-9) & (eval_h < c_end - 1e-9)
        cgm_clean = np.where(is_compression, cgm_clean - sensor_spec.compression_low_drop_mgdl, cgm_clean)

    cgm_clean = np.clip(cgm_clean, 40.0, 400.0)

    # 4. Scheduled, recurring, and random signal dropouts (applied inside evaluation window)
    is_dropout = np.zeros(n, dtype=bool)
    eval_h = df["eval_hour"].to_numpy(dtype=float)
    if sensor_spec.dropout_minutes > 0:
        dur_h = float(sensor_spec.dropout_minutes) / 60.0
        if sensor_spec.recurring_dropout_every_hours is not None:
            period = float(sensor_spec.recurring_dropout_every_hours)
            anchor = float(sensor_spec.dropout_at_hour)
            while anchor < float(np.max(eval_h)):
                is_dropout |= (eval_h >= anchor - 1e-9) & (eval_h < anchor + dur_h - 1e-9)
                anchor += period
        else:
            d_start = float(sensor_spec.dropout_at_hour)
            is_dropout |= (eval_h >= d_start - 1e-9) & (eval_h < d_start + dur_h - 1e-9)

    if sensor_spec.random_dropout_prob > 0.0:
        for k in range(n):
            if eval_h[k] >= 0.0 and float(rng.random()) < sensor_spec.random_dropout_prob:
                span = max(1, int(round(max(sensor_spec.dropout_minutes, 5) / step)))
                is_dropout[k : min(n, k + span)] = True

    # 5. Communication delay & display mode (nan_blank vs stale_hold)
    cgm_source = cgm_clean.copy()
    if sensor_spec.comm_delay_steps > 0:
        d = int(sensor_spec.comm_delay_steps)
        cgm_source = np.concatenate([np.full(min(d, n), cgm_clean[0]), cgm_clean[: max(0, n - d)]])

    cgm_observed = np.where(is_dropout, np.nan, cgm_source)
    cgm_displayed = np.empty(n, dtype=float)
    is_stale_displayed = np.zeros(n, dtype=bool)
    steps_into_dropout = np.zeros(n, dtype=int)
    steps_since_recovery = np.full(n, -1, dtype=int)

    last_valid = float(cgm_source[0])
    drop_counter = 0
    rec_counter = -1
    had_dropout = False
    for k in range(n):
        if is_dropout[k]:
            drop_counter += 1
            had_dropout = True
            rec_counter = -1
            steps_into_dropout[k] = drop_counter
            if sensor_spec.display_mode == "stale_hold":
                cgm_displayed[k] = last_valid
                is_stale_displayed[k] = True
            else:
                cgm_displayed[k] = np.nan
        else:
            if drop_counter > 0 or (had_dropout and rec_counter >= 0):
                rec_counter = 0 if drop_counter > 0 else rec_counter + 1
                steps_since_recovery[k] = rec_counter
            drop_counter = 0
            last_valid = float(cgm_source[k])
            cgm_displayed[k] = last_valid

    df["ig_mgdl"] = ig
    df["cgm_clean_mgdl"] = cgm_clean
    df["cgm_observed_mgdl"] = cgm_observed
    df["cgm_displayed_mgdl"] = cgm_displayed
    df["is_dropout"] = is_dropout
    df["is_compression_low"] = is_compression
    df["is_stale_displayed"] = is_stale_displayed
    df["steps_into_dropout"] = steps_into_dropout
    df["steps_since_recovery"] = steps_since_recovery
    return df
