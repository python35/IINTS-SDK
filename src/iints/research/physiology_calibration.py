"""Physiology calibration helpers for real CGM datasets.

The routines in this module do not claim to identify a clinical digital twin.
They compute reproducible, conservative parameter hints from real or prepared
research datasets so simulator presets can be tuned against empirical glucose
shape features before being used in experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


GLUCOSE_COLUMN_CANDIDATES = (
    "glucose_actual_mgdl",
    "glucose",
    "cgm_mgdl",
    "sensor_glucose_mgdl",
)
TIME_COLUMN_CANDIDATES = ("time_minutes", "timestamp", "timestamp_minutes", "datetime")
CARB_COLUMN_CANDIDATES = ("carb_grams", "carbs", "carb_intake_grams", "meal_carbs", "meal_grams")
INSULIN_COLUMN_CANDIDATES = (
    "insulin_units",
    "delivered_insulin_units",
    "bolus_units",
    "insulin",
)
EXERCISE_COLUMN_CANDIDATES = ("exercise_flag", "exercise", "activity_flag", "activity")
SUBJECT_COLUMN_CANDIDATES = ("subject_id", "patient_id", "person_id")


@dataclass(frozen=True)
class CalibrationColumns:
    """Resolved column names used by the physiology calibration audit."""

    time: str
    glucose: str
    carbs: str | None = None
    insulin: str | None = None
    exercise: str | None = None
    subject: str | None = None


def _first_existing(columns: Mapping[str, Any] | pd.Index, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def resolve_calibration_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str | None = None,
    glucose_column: str | None = None,
    carb_column: str | None = None,
    insulin_column: str | None = None,
    exercise_column: str | None = None,
    subject_column: str | None = None,
) -> CalibrationColumns:
    """Resolve common diabetes dataset column names to a normalized schema."""

    time = time_column or _first_existing(dataframe.columns, TIME_COLUMN_CANDIDATES)
    glucose = glucose_column or _first_existing(dataframe.columns, GLUCOSE_COLUMN_CANDIDATES)
    if time is None:
        raise ValueError(f"Could not resolve time column; tried {TIME_COLUMN_CANDIDATES}.")
    if glucose is None:
        raise ValueError(f"Could not resolve glucose column; tried {GLUCOSE_COLUMN_CANDIDATES}.")

    carbs = carb_column or _first_existing(dataframe.columns, CARB_COLUMN_CANDIDATES)
    insulin = insulin_column or _first_existing(dataframe.columns, INSULIN_COLUMN_CANDIDATES)
    exercise = exercise_column or _first_existing(dataframe.columns, EXERCISE_COLUMN_CANDIDATES)
    subject = subject_column or _first_existing(dataframe.columns, SUBJECT_COLUMN_CANDIDATES)
    return CalibrationColumns(
        time=str(time),
        glucose=str(glucose),
        carbs=str(carbs) if carbs is not None else None,
        insulin=str(insulin) if insulin is not None else None,
        exercise=str(exercise) if exercise is not None else None,
        subject=str(subject) if subject is not None else None,
    )


def load_calibration_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV or parquet dataset for calibration analysis."""

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported calibration dataset format: {path.suffix}")


def _normalize_time_minutes(series: pd.Series) -> pd.Series:
    """Return elapsed minutes from either numeric minutes or datetimes."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(len(series) * 0.7)):
        return numeric.astype(float)
    timestamps = pd.to_datetime(series, errors="coerce")
    if timestamps.notna().sum() < 3:
        return numeric.astype(float)
    origin = timestamps.dropna().min()
    return (timestamps - origin).dt.total_seconds() / 60.0


def standardize_calibration_dataframe(
    dataframe: pd.DataFrame,
    columns: CalibrationColumns | None = None,
) -> pd.DataFrame:
    """Standardize real/simulated glucose data for calibration metrics."""

    cols = columns or resolve_calibration_columns(dataframe)
    out = pd.DataFrame(
        {
            "time_minutes": _normalize_time_minutes(dataframe[cols.time]),
            "glucose_mgdl": pd.to_numeric(dataframe[cols.glucose], errors="coerce"),
        }
    )
    out["carbs_g"] = (
        pd.to_numeric(dataframe[cols.carbs], errors="coerce").fillna(0.0)
        if cols.carbs is not None
        else 0.0
    )
    out["insulin_u"] = (
        pd.to_numeric(dataframe[cols.insulin], errors="coerce").fillna(0.0)
        if cols.insulin is not None
        else 0.0
    )
    out["exercise_flag"] = (
        pd.to_numeric(dataframe[cols.exercise], errors="coerce").fillna(0.0)
        if cols.exercise is not None
        else 0.0
    )
    if cols.subject is not None:
        out["subject_id"] = dataframe[cols.subject].astype(str)
    else:
        out["subject_id"] = "unknown"
    out = out.dropna(subset=["time_minutes", "glucose_mgdl"]).copy()
    out = out[np.isfinite(out["glucose_mgdl"])].copy()
    out = out[(out["glucose_mgdl"] >= 20.0) & (out["glucose_mgdl"] <= 600.0)].copy()
    out["time_minutes"] = out["time_minutes"].astype(float)
    out["hour_of_day"] = (out["time_minutes"] % 1440.0) / 60.0
    return out.sort_values(["subject_id", "time_minutes"]).reset_index(drop=True)


def _safe_float(value: float | np.floating[Any]) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _percentile(values: pd.Series | np.ndarray, q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return _safe_float(np.percentile(array, q))


def glucose_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute dataset-level glucose metrics relevant to simulator calibration."""

    glucose = frame["glucose_mgdl"].astype(float)
    if len(glucose) == 0:
        raise ValueError("No valid glucose rows available for calibration.")
    ordered = frame.sort_values(["subject_id", "time_minutes"]).copy()
    ordered["dt_min"] = ordered.groupby("subject_id")["time_minutes"].diff()
    ordered["dg_mgdl"] = ordered.groupby("subject_id")["glucose_mgdl"].diff()
    valid_rate = ordered[(ordered["dt_min"] > 0) & (ordered["dt_min"] <= 30)].copy()
    roc = valid_rate["dg_mgdl"] / valid_rate["dt_min"]

    mean = float(glucose.mean())
    sd = float(glucose.std(ddof=0)) if len(glucose) > 1 else 0.0
    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "duration_hours": _safe_float((frame["time_minutes"].max() - frame["time_minutes"].min()) / 60.0),
        "mean_glucose_mgdl": round(mean, 3),
        "median_glucose_mgdl": round(float(glucose.median()), 3),
        "sd_mgdl": round(sd, 3),
        "cv_pct": round((sd / mean * 100.0) if mean else 0.0, 3),
        "tir_70_180_pct": round(float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0), 3),
        "tir_below_70_pct": round(float((glucose < 70).mean() * 100.0), 3),
        "tir_above_180_pct": round(float((glucose > 180).mean() * 100.0), 3),
        "p05_mgdl": _percentile(glucose, 5),
        "p95_mgdl": _percentile(glucose, 95),
        "median_step_minutes": _percentile(valid_rate["dt_min"], 50) if len(valid_rate) else None,
        "roc_abs_p95_mgdl_min": _percentile(np.abs(roc), 95) if len(roc) else None,
        "roc_abs_p99_mgdl_min": _percentile(np.abs(roc), 99) if len(roc) else None,
        "roc_max_abs_mgdl_min": _safe_float(np.nanmax(np.abs(roc))) if len(roc) else None,
    }


def meal_response_summary(
    frame: pd.DataFrame,
    *,
    min_carbs_g: float = 8.0,
    pre_window_min: float = 30.0,
    post_window_min: float = 240.0,
) -> dict[str, Any]:
    """Summarize post-meal peak timing and amplitude."""

    meals = frame[frame["carbs_g"] >= float(min_carbs_g)].copy()
    rows: list[dict[str, float]] = []
    for _, meal in meals.iterrows():
        subject = meal["subject_id"]
        t0 = float(meal["time_minutes"])
        subject_frame = frame[frame["subject_id"] == subject]
        pre = subject_frame[
            (subject_frame["time_minutes"] >= t0 - pre_window_min)
            & (subject_frame["time_minutes"] <= t0)
        ]
        post = subject_frame[
            (subject_frame["time_minutes"] >= t0)
            & (subject_frame["time_minutes"] <= t0 + post_window_min)
        ]
        if len(pre) < 2 or len(post) < 4:
            continue
        baseline = float(pre["glucose_mgdl"].median())
        peak_idx = post["glucose_mgdl"].idxmax()
        peak = float(post.loc[peak_idx, "glucose_mgdl"])
        peak_time = float(post.loc[peak_idx, "time_minutes"] - t0)
        rows.append(
            {
                "carbs_g": float(meal["carbs_g"]),
                "baseline_mgdl": baseline,
                "peak_delta_mgdl": peak - baseline,
                "time_to_peak_min": peak_time,
            }
        )

    if not rows:
        return {
            "meal_count": int(len(meals)),
            "eligible_meal_count": 0,
            "median_peak_delta_mgdl": None,
            "median_time_to_peak_min": None,
            "p75_time_to_peak_min": None,
            "median_peak_delta_per_10g_mgdl": None,
        }
    meal_df = pd.DataFrame(rows)
    return {
        "meal_count": int(len(meals)),
        "eligible_meal_count": int(len(meal_df)),
        "median_peak_delta_mgdl": round(float(meal_df["peak_delta_mgdl"].median()), 3),
        "median_time_to_peak_min": round(float(meal_df["time_to_peak_min"].median()), 3),
        "p75_time_to_peak_min": _percentile(meal_df["time_to_peak_min"], 75),
        "median_peak_delta_per_10g_mgdl": round(
            float((meal_df["peak_delta_mgdl"] / meal_df["carbs_g"].clip(lower=1.0) * 10.0).median()),
            3,
        ),
    }


def exercise_response_summary(
    frame: pd.DataFrame,
    *,
    pre_window_min: float = 30.0,
    post_window_min: float = 90.0,
) -> dict[str, Any]:
    """Estimate glucose movement after starts of exercise-flagged intervals."""

    exercise = frame["exercise_flag"].astype(float) > 0
    starts = frame[exercise & ~exercise.groupby(frame["subject_id"]).shift(fill_value=False)].copy()
    rows: list[dict[str, float]] = []
    for _, event in starts.iterrows():
        subject = event["subject_id"]
        t0 = float(event["time_minutes"])
        subject_frame = frame[frame["subject_id"] == subject]
        pre = subject_frame[
            (subject_frame["time_minutes"] >= t0 - pre_window_min)
            & (subject_frame["time_minutes"] <= t0)
        ]
        post = subject_frame[
            (subject_frame["time_minutes"] >= t0)
            & (subject_frame["time_minutes"] <= t0 + post_window_min)
        ]
        if len(pre) < 2 or len(post) < 3:
            continue
        rows.append(
            {
                "delta_90min_mgdl": float(post["glucose_mgdl"].iloc[-1] - pre["glucose_mgdl"].median()),
                "min_delta_mgdl": float(post["glucose_mgdl"].min() - pre["glucose_mgdl"].median()),
            }
        )
    if not rows:
        return {"exercise_event_count": int(len(starts)), "eligible_exercise_event_count": 0}
    ex_df = pd.DataFrame(rows)
    return {
        "exercise_event_count": int(len(starts)),
        "eligible_exercise_event_count": int(len(ex_df)),
        "median_90min_delta_mgdl": round(float(ex_df["delta_90min_mgdl"].median()), 3),
        "median_min_delta_mgdl": round(float(ex_df["min_delta_mgdl"].median()), 3),
    }


def dawn_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Estimate dawn rise from overnight and early-morning glucose medians."""

    overnight = frame[(frame["hour_of_day"] >= 0.0) & (frame["hour_of_day"] < 4.0)]
    dawn = frame[(frame["hour_of_day"] >= 4.0) & (frame["hour_of_day"] < 8.0)]
    if len(overnight) < 8 or len(dawn) < 8:
        return {"overnight_median_mgdl": None, "dawn_median_mgdl": None, "dawn_rise_mgdl": None}
    overnight_median = float(overnight["glucose_mgdl"].median())
    dawn_median = float(dawn["glucose_mgdl"].median())
    return {
        "overnight_median_mgdl": round(overnight_median, 3),
        "dawn_median_mgdl": round(dawn_median, 3),
        "dawn_rise_mgdl": round(dawn_median - overnight_median, 3),
    }


def build_parameter_hints(
    glucose: Mapping[str, Any],
    meal: Mapping[str, Any],
    dawn: Mapping[str, Any],
) -> dict[str, Any]:
    """Build conservative simulator parameter hints from empirical summaries."""

    median_glucose = float(glucose.get("median_glucose_mgdl") or 120.0)
    mean_glucose = float(glucose.get("mean_glucose_mgdl") or median_glucose)
    overnight = dawn.get("overnight_median_mgdl")
    dawn_rise = float(dawn.get("dawn_rise_mgdl") or 0.0)
    peak_time = meal.get("median_time_to_peak_min")
    p75_peak = meal.get("p75_time_to_peak_min")
    roc_p99 = glucose.get("roc_abs_p99_mgdl_min")
    carb_delta = meal.get("median_peak_delta_per_10g_mgdl")

    carb_duration = 240.0
    if peak_time is not None:
        carb_duration = float(np.clip(max(float(p75_peak or peak_time) * 1.7, float(peak_time) * 2.0), 120.0, 420.0))

    max_rate = 3.0
    if roc_p99 is not None:
        # Keep simulator guards slightly above empirical p99, while avoiding impossible CGM cliffs.
        max_rate = float(np.clip(float(roc_p99) * 1.35, 1.0, 4.0))

    absorption_rate = 0.03
    if carb_delta is not None:
        # CustomPatientModel absorption rates are unitless educational knobs; keep range conservative.
        absorption_rate = float(np.clip(float(carb_delta) / 350.0, 0.012, 0.06))

    hints = {
        "initial_glucose": round(float(np.clip(median_glucose, 80.0, 220.0)), 3),
        "basal_glucose_target": round(float(np.clip(float(overnight or mean_glucose), 80.0, 200.0)), 3),
        "carb_absorption_duration_minutes": round(carb_duration, 3),
        "glucose_absorption_rate": round(absorption_rate, 5),
        "max_glucose_rate_mgdl_per_min": round(max_rate, 3),
        "dawn_phenomenon_strength": round(float(np.clip(dawn_rise / 4.0, 0.0, 25.0)), 3),
        "insulin_action_duration": 300.0,
        "insulin_peak_time": 75.0,
    }
    return {
        "patient_profile_hints": hints,
        "calibration_limits": [
            "Hints are empirical starting points, not identified clinical parameters.",
            "Meal absorption is estimated from observed CGM peaks and can be confounded by insulin timing and missed carbs.",
            "Dawn strength is based on median overnight-vs-morning difference and should be validated per subject.",
            "Insulin action defaults should be replaced by known analog/pump settings when available.",
        ],
    }


def compare_simulation_to_real(real: pd.DataFrame, simulation: pd.DataFrame | None) -> dict[str, Any] | None:
    """Compare high-level physiology features between real and simulated traces."""

    if simulation is None:
        return None
    real_summary = glucose_summary(real)
    sim_summary = glucose_summary(simulation)
    keys = [
        "mean_glucose_mgdl",
        "sd_mgdl",
        "cv_pct",
        "tir_70_180_pct",
        "tir_below_70_pct",
        "tir_above_180_pct",
        "roc_abs_p95_mgdl_min",
    ]
    deltas = {}
    for key in keys:
        real_value = real_summary.get(key)
        sim_value = sim_summary.get(key)
        if real_value is None or sim_value is None:
            continue
        deltas[key] = round(float(sim_value) - float(real_value), 3)
    return {"real": real_summary, "simulation": sim_summary, "simulation_minus_real": deltas}


def physiology_calibration_report(
    real_dataframe: pd.DataFrame,
    *,
    simulation_dataframe: pd.DataFrame | None = None,
    columns: CalibrationColumns | None = None,
    simulation_columns: CalibrationColumns | None = None,
) -> dict[str, Any]:
    """Build a calibration report for a real diabetes dataset."""

    real = standardize_calibration_dataframe(real_dataframe, columns)
    simulation = (
        standardize_calibration_dataframe(simulation_dataframe, simulation_columns)
        if simulation_dataframe is not None
        else None
    )
    glucose = glucose_summary(real)
    meal = meal_response_summary(real)
    exercise = exercise_response_summary(real)
    dawn = dawn_summary(real)
    hints = build_parameter_hints(glucose, meal, dawn)
    return {
        "schema_version": 1,
        "purpose": "research physiology calibration audit; not clinical validation",
        "real_dataset": {
            "glucose_summary": glucose,
            "meal_response_summary": meal,
            "exercise_response_summary": exercise,
            "dawn_summary": dawn,
        },
        "calibration": hints,
        "simulation_comparison": compare_simulation_to_real(real, simulation),
    }
