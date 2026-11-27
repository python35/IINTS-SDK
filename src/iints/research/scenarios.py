"""B. Virtual Cohort & Systematic Boundary-Case Scenario Generator.

Generates:
1. Stratified virtual patient cohorts cleanly partitioned into ``dev`` and
   strictly held-out ``unseen`` profiles (preventing hyperparameter leakage).
2. Both standard in-distribution 24-hour scenarios and adversarial boundary
   ("corner-case") scenarios where CGM dropouts coincide with rapid post-meal
   spikes or insulin-induced descents toward hypoglycaemia.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from typing import List, Literal

import numpy as np

from iints.core.patient import equilibrium as eq
from iints.sdk2.contracts import MealEventSpec, PatientSpec, ScenarioSpec, SensorFaultSpec

__all__ = [
    "generate_virtual_cohort",
    "generate_boundary_scenarios",
    "generate_dropout_conditions",
]


def generate_virtual_cohort(
    n_profiles: int = 100,
    *,
    seed: int = 42,
    unseen_fraction: float = 0.30,
    model: Literal["hovorka_equilibrium", "iints_ode", "bergman", "simglucose"] = "hovorka_equilibrium",
) -> List[PatientSpec]:
    """Generate a reproducible stratified virtual cohort with held-out ``unseen`` split."""
    if n_profiles < 2:
        raise ValueError("n_profiles must be at least 2")
    rng = np.random.default_rng(seed)
    n_unseen = max(1, int(round(n_profiles * unseen_fraction)))
    n_dev = n_profiles - n_unseen

    # Stratified Latin-Hypercube-like quantiles shuffled deterministically
    u_basal = (np.arange(n_profiles) + rng.uniform(0.1, 0.9, size=n_profiles)) / n_profiles
    u_gluc = (np.arange(n_profiles) + rng.uniform(0.1, 0.9, size=n_profiles)) / n_profiles
    u_tau = (np.arange(n_profiles) + rng.uniform(0.1, 0.9, size=n_profiles)) / n_profiles
    rng.shuffle(u_basal)
    rng.shuffle(u_gluc)
    rng.shuffle(u_tau)

    cohort: List[PatientSpec] = []
    for idx in range(n_profiles):
        split: Literal["dev", "unseen"] = "dev" if idx < n_dev else "unseen"
        basal = float(np.round(0.85 + 1.65 * u_basal[idx], 3))             # 0.85 .. 2.50 U/h
        op_glucose = float(np.round(112.0 + 58.0 * u_gluc[idx], 1))        # 112 .. 170 mg/dL
        meal_tau = float(np.round(35.0 + 48.0 * u_tau[idx], 1))            # 35 .. 83 min
        bioavail = float(np.round(rng.uniform(0.70, 0.92), 3))
        isf_clin = float(np.round(1800.0 / max(18.0, basal * 24.0 * 1.9), 1))
        carb_factor = float(np.round(500.0 / max(18.0, basal * 24.0 * 1.9), 1))
        sigma = float(eq.basal_scale(basal, op_glucose)) if model == "hovorka_equilibrium" else None

        cohort.append(
            PatientSpec(
                model=model,
                profile=f"vp_{idx + 1:03d}_{split}",
                split=split,
                initial_glucose_mgdl=op_glucose,
                basal_u_per_h=basal,
                sigma=sigma,
                insulin_sensitivity_mgdl_per_u=isf_clin,
                carb_factor_g_per_u=max(4.0, carb_factor),
                meal_tau_min=meal_tau,
                bioavailability=bioavail,
                max_glucose_rate_mgdl_per_min=9.0,
            )
        )
    return cohort


def generate_boundary_scenarios(duration_hours: float = 24.0) -> List[ScenarioSpec]:
    """Return 1 in-distribution reference + 2 systematic boundary stress scenarios."""
    warmup_h = 6.0
    w_min = warmup_h * 60.0
    return [
        ScenarioSpec(
            id="standard_day",
            duration_hours=duration_hours,
            warmup_hours=warmup_h,
            meal_schedule="standard_day",
            meal_time_jitter_min=15.0,
            carb_announcement_error_sd=0.08,
            bolus_relative_error_sd=0.08,
            regime="in_distribution",
        ),
        ScenarioSpec(
            id="ood_postprandial_peak_dropout",
            duration_hours=duration_hours,
            warmup_hours=warmup_h,
            meal_schedule="custom",
            meals=[
                # Warmup breakfast
                MealEventSpec(minute=120.0, carbs_g=45.0, announced_carbs_g=45.0),
                # Large rapid-rise meal at eval_hour=7.75h so an 8.0h/8.5h dropout lands right on steepest ascent
                MealEventSpec(minute=w_min + 7.75 * 60.0, carbs_g=95.0, announced_carbs_g=80.0, bolus_delay_min=15.0),
                # Unannounced afternoon snack + heavy evening meal
                MealEventSpec(minute=w_min + 13.5 * 60.0, carbs_g=35.0, announced_carbs_g=0.0),
                MealEventSpec(minute=w_min + 19.0 * 60.0, carbs_g=85.0, announced_carbs_g=75.0),
            ],
            meal_time_jitter_min=0.0,
            carb_announcement_error_sd=0.0,
            bolus_relative_error_sd=0.10,
            regime="ood_stress",
        ),
        ScenarioSpec(
            id="ood_nocturnal_descent_dropout",
            duration_hours=duration_hours,
            warmup_hours=warmup_h,
            meal_schedule="custom",
            meals=[
                MealEventSpec(minute=120.0, carbs_g=40.0, announced_carbs_g=40.0),
                # Over-estimated carbs at eval_hour=7.5h triggers aggressive bolus & sharp descent during 8.5h dropout
                MealEventSpec(minute=w_min + 7.5 * 60.0, carbs_g=40.0, announced_carbs_g=75.0, bolus_delay_min=-10.0),
                MealEventSpec(minute=w_min + 18.5 * 60.0, carbs_g=60.0, announced_carbs_g=60.0),
            ],
            meal_time_jitter_min=0.0,
            carb_announcement_error_sd=0.0,
            bolus_relative_error_sd=0.05,
            regime="ood_stress",
        ),
    ]


def generate_dropout_conditions(
    dropout_minutes_list: List[int] | None = None,
    *,
    dropout_at_hour: float = 8.5,
    recurring_every_hours: float | None = 6.0,
    display_mode: Literal["nan_blank", "stale_hold"] = "nan_blank",
) -> List[SensorFaultSpec]:
    """Build matched sensor conditions differing ONLY in CGM dropout duration."""
    durations = dropout_minutes_list if dropout_minutes_list is not None else [0, 5, 15, 30]
    conditions: List[SensorFaultSpec] = []
    for mins in durations:
        cid = f"dropout_{int(mins)}m" if display_mode == "nan_blank" else f"dropout_{int(mins)}m_{display_mode}"
        conditions.append(
            SensorFaultSpec(
                model="cgm_v1",
                condition_id=cid,
                isf_tau_minutes=6.0,
                noise_sd_mgdl=6.0,
                noise_ar1_phi=0.75,
                dropout_minutes=int(mins),
                dropout_at_hour=dropout_at_hour,
                recurring_dropout_every_hours=recurring_every_hours if mins > 0 else None,
                display_mode=display_mode,
            )
        )
    return conditions
