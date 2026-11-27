"""Gray-Box MiniMed 780G predictor v2 — integrated SDK facade.

Unifies:
- Section-aware CareLink parsing (``carelink_sections.py``),
- Equilibrium-consistent Hovorka parameterisation (``equilibrium.py``),
- Vectorised nonlinear RK4 forward propagation (``nonlinear_propagator.py``),
- Rolling-origin walk-forward backtesting & split-conformal calibration (``minimed_backtest.py``),
- AR(2)-nested Gray-Box blending & testable meal-absorption models.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast
import warnings

import numpy as np
import pandas as pd

from iints.analysis import carelink_sections as cp
from iints.analysis import minimed_backtest as bt
from iints.analysis import nonlinear_propagator as nlp

__all__ = [
    "IDENTIFIED",
    "IDENTIFIED_LEGACY_RESIDUAL",
    "RECOMMENDED_TRAIN_DAYS",
    "load_timeline",
    "insulin_audit",
    "evaluate_carelink_csv",
    "compare_variants",
    "what_if",
]

#: Identified on 90-day MiniMed 780G + Simplera export by Nelder-Mead on blind
#: white-box error, upgraded with AR(2)-nested Gray-Box blending.
IDENTIFIED = bt.PhysicsConfig(
    propagation="nonlinear",
    sigma=0.11507,
    meal_tau_min=63.97,
    bioavailability=0.8535,
    q_scale=0.5,
    feedback_gain=0.07886,
    ar2_nested_graybox=True,
    meal_model="hovorka_2comp",
)

#: Exact configuration from the initial validation report (fixed-weight residual).
IDENTIFIED_LEGACY_RESIDUAL = replace(IDENTIFIED, ar2_nested_graybox=False)

RECOMMENDED_TRAIN_DAYS = 21.0


def load_timeline(
    csv_path: Union[str, Path],
    *,
    auto_basal_mode: str = "profile",
) -> pd.DataFrame:
    """Parse a CareLink export into the 5-minute timeline required by the engine."""
    return cp.build_timeline(cp.read_sections(csv_path), auto_basal_mode=auto_basal_mode)


def insulin_audit(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Per-day insulin balance: pump-reported vs booked doses."""
    sections = cp.read_sections(csv_path)
    return cp.reconcile_report(sections, cp.build_timeline(sections))


def evaluate_carelink_csv(
    csv_path: Union[str, Path],
    *,
    config: Optional[bt.PhysicsConfig] = None,
    train_days: Optional[float] = RECOMMENDED_TRAIN_DAYS,
    test_days: float = 7.0,
    initial_train_days: float = 30.0,
    bad_sensor_days: Sequence = (),
    horizons: Sequence[int] = bt.HORIZONS,
) -> Dict[str, Any]:
    """Rolling-origin walk-forward evaluation with baselines and split-conformal coverage."""
    timeline = load_timeline(csv_path)
    cfg = config or IDENTIFIED
    table = bt.walk_forward(
        timeline,
        cfg,
        train_days=train_days,
        test_days=test_days,
        initial_train_days=initial_train_days,
        bad_days=bad_sensor_days,
        horizons=horizons,
    )
    scores = bt.score(table)

    horizon_reports = []
    for horizon in horizons:
        rows = table[table["horizon"] == horizon]
        if rows.empty:
            continue
        sd = rows["sd_whitebox"].to_numpy() * rows["conformal_scale_band"].to_numpy()
        inside = (
            (rows["y_true"] >= rows["y_graybox"] - bt.Z_90 * sd)
            & (rows["y_true"] <= rows["y_graybox"] + bt.Z_90 * sd)
        )
        post_meal = rows[rows["post_meal"]]
        pm_rmse = (
            float(np.sqrt(np.mean((post_meal["y_graybox"] - post_meal["y_true"]) ** 2)))
            if not post_meal.empty
            else float("nan")
        )
        pm_bias = (
            float(np.mean(post_meal["y_graybox"] - post_meal["y_true"]))
            if not post_meal.empty
            else float("nan")
        )
        subset = scores[scores["horizon"] == horizon].set_index("model")
        horizon_reports.append(
            {
                "horizon_minutes": int(horizon),
                "n_origins": int(len(rows)),
                "rmse": {
                    model: round(float(cast(Any, subset.loc[model, "rmse"])), 2)
                    for model in subset.index
                },
                "bias_graybox_mgdl": round(float(cast(Any, subset.loc["graybox", "bias"])), 2),
                "post_meal_rmse_mgdl": round(pm_rmse, 2),
                "post_meal_bias_mgdl": round(pm_bias, 2),
                "coverage_90_pct": round(100.0 * float(inside.mean()), 1),
                "conformal_scale": round(float(rows["conformal_scale"].mean()), 2),
            }
        )

    return {
        "research_disclaimer": (
            "In-silico research evaluation only; not a medical device and must "
            "never be used for clinical insulin dosing."
        ),
        "source_csv": str(csv_path),
        "timeline_rows": int(len(timeline)),
        "cgm_samples": int(timeline["glucose"].notna().sum()),
        "config": cfg.label(),
        "train_days": train_days,
        "folds": int(table["fold"].nunique()),
        "horizons": horizon_reports,
        "forecasts": table,
    }


def compare_variants(
    csv_path: Union[str, Path],
    *,
    train_days: Optional[float] = RECOMMENDED_TRAIN_DAYS,
) -> pd.DataFrame:
    """Head-to-head A/B ablation of AR(2)-nesting and physiological meal models."""
    timeline = load_timeline(csv_path)
    variants: List[tuple[str, bt.PhysicsConfig]] = [
        ("v2_legacy_resid (hovorka_2comp)", IDENTIFIED_LEGACY_RESIDUAL),
        ("v2_ar2nested (hovorka_2comp)", IDENTIFIED),
        (
            "v2_ar2nested (biphasic 35% fast 25m)",
            replace(IDENTIFIED, meal_model="biphasic", fast_carb_fraction=0.35, fast_meal_tau_min=25.0),
        ),
        (
            "v2_ar2nested (glucose_gated gut)",
            replace(IDENTIFIED, meal_model="glucose_gated", gastric_glucose_gain=0.35),
        ),
    ]
    rows = []
    for name, cfg in variants:
        table = bt.walk_forward(timeline, cfg, train_days=train_days)
        scores = bt.score(table)
        entry: Dict[str, Any] = {"variant": name}
        for h in bt.HORIZONS:
            sub = scores[scores["horizon"] == h].set_index("model")
            h_rows = table[table["horizon"] == h]
            pm = h_rows[h_rows["post_meal"]]
            entry[f"ar2_{h}m"] = round(float(cast(Any, sub.loc["ar2", "rmse"])), 2)
            entry[f"wb_{h}m"] = round(float(cast(Any, sub.loc["whitebox", "rmse"])), 2)
            entry[f"gb_{h}m"] = round(float(cast(Any, sub.loc["graybox", "rmse"])), 2)
            entry[f"pm_gb_{h}m"] = round(
                float(np.sqrt(np.mean((pm["y_graybox"] - pm["y_true"]) ** 2))), 2
            )
            sd = h_rows["sd_whitebox"].to_numpy() * h_rows["conformal_scale_band"].to_numpy()
            inside = (
                (h_rows["y_true"] >= h_rows["y_graybox"] - bt.Z_90 * sd)
                & (h_rows["y_true"] <= h_rows["y_graybox"] + bt.Z_90 * sd)
            )
            entry[f"cov90_{h}m"] = round(100.0 * float(inside.mean()), 1)
        rows.append(entry)
    return pd.DataFrame(rows)


def what_if(
    timeline: pd.DataFrame,
    anchor_index: int,
    *,
    config: Optional[bt.PhysicsConfig] = None,
    horizon_minutes: int = 180,
    carbs_g: float = 0.0,
    bolus_u: float = 0.0,
    basal_pct: float = 100.0,
    sensitivity_multiplier: float = 1.0,
    absorption_multiplier: float = 1.0,
    conformal_scale: float = 3.7,
) -> Dict[str, Any]:
    """Forward scenario from the reconstructed Kalman state, integrated nonlinearly."""
    cfg = config or IDENTIFIED
    if carbs_g < 0.0 or bolus_u < 0.0 or basal_pct < 0.0:
        raise ValueError("carbs_g, bolus_u and basal_pct must be non-negative")
    if sensitivity_multiplier <= 0.0 or absorption_multiplier <= 0.0:
        raise ValueError("the scenario multipliers must be positive")

    engine = bt.GrayBoxEngine(cfg, timeline)
    history = timeline.iloc[: anchor_index + 1].reset_index(drop=True)
    filtered = engine.filter_timeline(history)
    origins = np.array([len(history) - 1])
    state = engine.physical_states(filtered, origins)

    if horizon_minutes > 120:
        warnings.warn(
            "the 90 % interval is extrapolated beyond 120 min and was not validated there",
            RuntimeWarning,
            stacklevel=2,
        )
    steps = max(1, int(round(horizon_minutes / bt.STEP_MINUTES)))
    params = replace(
        engine.plant_params,
        meal_tau_min=cfg.meal_tau_min * absorption_multiplier,
        k_b=engine.plant_params.k_b * sensitivity_multiplier,
    )

    basal = float(np.nanmean(history["auto_basal_u"].tail(12)) * 60.0 / bt.STEP_MINUTES)
    rate = np.full((1, steps), basal * basal_pct / 100.0)
    rate[0, 0] += bolus_u * 60.0 / bt.STEP_MINUTES
    carbs = np.zeros((1, steps))
    carbs[0, 0] = carbs_g

    median = nlp.propagate(
        state,
        rate,
        carbs,
        params,
        step_minutes=bt.STEP_MINUTES,
        substeps=cfg.substeps,
    )[0]
    sd = np.array([engine.forecast_sd(0)[min(k, engine.max_steps - 1)] for k in range(steps)])
    if steps > engine.max_steps:
        sd[engine.max_steps :] = sd[engine.max_steps - 1] * np.sqrt(
            np.arange(1, steps - engine.max_steps + 1) + 1.0
        )
    sd = sd * conformal_scale
    start = float(state[0, 0] / params.volume_dl)
    times = np.arange(0, steps + 1) * bt.STEP_MINUTES
    median = np.r_[start, median]
    sd = np.r_[0.0, sd]
    lower = np.clip(median - bt.Z_90 * sd, 20.0, 500.0)
    upper = np.clip(median + bt.Z_90 * sd, 20.0, 500.0)

    return {
        "times_min": times.tolist(),
        "median_mgdl": median.round(1).tolist(),
        "p05_mgdl": lower.round(1).tolist(),
        "p95_mgdl": upper.round(1).tolist(),
        "start_mgdl": round(start, 1),
        "peak_mgdl": round(float(np.max(median)), 1),
        "nadir_mgdl": round(float(np.min(median)), 1),
        "min_p05_mgdl": round(float(np.min(lower)), 1),
        "interval_validated_to_min": 120,
        "hypo_risk_p05": bool(np.min(lower) < 70.0),
        "scenario": {
            "carbs_g": carbs_g,
            "bolus_u": bolus_u,
            "basal_pct": basal_pct,
            "sensitivity_multiplier": sensitivity_multiplier,
            "absorption_multiplier": absorption_multiplier,
        },
        "research_disclaimer": "In-silico scenario only; never a dosing recommendation.",
    }
