from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from iints.research.dataset import save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the HUPA-UCM dataset for LSTM predictor training."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_packs/public/hupa_ucm"),
        help="Root directory containing HUPA-UCM CSV files (per-patient).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv"),
        help="Output dataset path (CSV or Parquet)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data_packs/public/hupa_ucm/quality_report.json"),
        help="Quality report output path",
    )
    parser.add_argument("--time-step", type=int, default=5, help="Expected CGM time step (minutes)")
    parser.add_argument("--max-gap-multiplier", type=float, default=2.5, help="Segment break multiplier")
    parser.add_argument("--dia-minutes", type=float, default=240.0, help="Insulin action duration (minutes)")
    parser.add_argument("--peak-minutes", type=float, default=75.0, help="IOB activity peak time (minutes)")
    parser.add_argument("--carb-absorb-minutes", type=float, default=120.0, help="Carb absorption duration (minutes)")
    parser.add_argument("--max-insulin", type=float, default=30.0, help="Clip insulin units above this")
    parser.add_argument("--max-carbs", type=float, default=200.0, help="Clip carb grams above this")
    parser.add_argument("--carb-serving-grams", type=float, default=10.0, help="Carb serving size (g) for HUPA carb_input")
    parser.add_argument("--basal-is-rate", action=argparse.BooleanOptionalAction, default=False,
                        help="If True, treat basal_rate as U/hr and convert to U/step. "
                             "Default assumes basal_rate already in U per interval.")
    parser.add_argument("--icr-default", type=float, default=10.0, help="Fallback ICR (g/U)")
    parser.add_argument("--isf-default", type=float, default=50.0, help="Fallback ISF (mg/dL per U)")
    parser.add_argument("--basal-default", type=float, default=0.0, help="Fallback basal rate (U/hr)")
    parser.add_argument("--meal-window-min", type=float, default=30.0, help="Window for meal→insulin matching (minutes)")
    parser.add_argument("--isf-window-min", type=float, default=60.0, help="Window for insulin sensitivity estimate (minutes)")
    parser.add_argument("--min-meal-carbs", type=float, default=5.0, help="Minimum carbs to consider a meal (g)")
    parser.add_argument("--min-bolus", type=float, default=0.1, help="Minimum insulin to consider a bolus (U)")
    return parser.parse_args()


def _read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _col_pick(df: pd.DataFrame, names: List[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _clean_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


# ---------------------------------------------------------------------------
# OpenAPS-style bilinear IOB / exponential COB
# ---------------------------------------------------------------------------

def _derive_iob_cob_openaps(
    insulin: pd.Series,
    carbs: pd.Series,
    delta_min: pd.Series,
    dia_minutes: float,
    peak_minutes: float,
    carb_absorb_minutes: float,
) -> Tuple[pd.Series, pd.Series]:
    iob_vals: List[float] = []
    cob_vals: List[float] = []
    prev_iob = 0.0
    prev_cob = 0.0

    for ins, carb, dt in zip(insulin, carbs, delta_min):
        if not np.isfinite(dt) or dt <= 0:
            prev_iob = 0.0
            prev_cob = 0.0
            iob_vals.append(0.0)
            cob_vals.append(0.0)
            continue

        iob_decay = np.exp(-dt * np.log(2) / (dia_minutes * 0.5))
        prev_iob = prev_iob * iob_decay + ins

        cob_decay = np.exp(-dt / carb_absorb_minutes)
        prev_cob = prev_cob * cob_decay + carb

        iob_vals.append(prev_iob)
        cob_vals.append(prev_cob)

    return pd.Series(iob_vals, index=insulin.index), pd.Series(cob_vals, index=carbs.index)


def _estimate_basal_rate(
    insulin_units: pd.Series,
    carbs_grams: pd.Series,
    time_step: float,
    default_basal: float,
) -> Tuple[float, int]:
    nonzero = (insulin_units > 0) & (carbs_grams <= 0)
    if nonzero.mean() < 0.5:
        return default_basal, int(nonzero.sum())
    median_units = float(insulin_units.loc[nonzero].median())
    basal_rate = median_units * (60.0 / time_step)
    return basal_rate, int(nonzero.sum())


def _estimate_icr(
    df: pd.DataFrame,
    meal_window_min: float,
    min_meal_carbs: float,
    min_bolus: float,
    default_icr: float,
) -> Tuple[float, int]:
    meals = df[df["carb_grams"] >= min_meal_carbs]
    if meals.empty:
        return default_icr, 0

    ratios: List[float] = []
    for _, meal in meals.iterrows():
        t0 = meal["timestamp"]
        window_start = t0 - pd.Timedelta(minutes=meal_window_min)
        window_end = t0 + pd.Timedelta(minutes=meal_window_min)
        insulin_window = df.loc[
            (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end),
            "insulin_units",
        ].sum()
        if insulin_window >= min_bolus:
            ratios.append(float(meal["carb_grams"] / insulin_window))

    if not ratios:
        return default_icr, 0

    icr = float(np.median(ratios))
    icr = float(np.clip(icr, 2.0, 30.0))
    return icr, len(ratios)


def _estimate_isf(
    df: pd.DataFrame,
    time_step: float,
    isf_window_min: float,
    min_bolus: float,
    min_meal_carbs: float,
    default_isf: float,
) -> Tuple[float, int]:
    step = max(1, int(round(isf_window_min / time_step)))
    ratios: List[float] = []

    for idx in range(len(df) - step):
        insulin = float(df.at[idx, "insulin_units"])
        carbs_window = float(df.loc[idx: idx + step, "carb_grams"].sum())
        if insulin < min_bolus or carbs_window > min_meal_carbs:
            continue
        if df.at[idx, "segment"] != df.at[idx + step, "segment"]:
            continue
        g_now = float(df.at[idx, "glucose_actual_mgdl"])
        g_future = float(df.at[idx + step, "glucose_actual_mgdl"])
        delta = g_now - g_future
        if delta <= 5.0:
            continue
        ratios.append(delta / insulin)

    if not ratios:
        return default_isf, 0

    isf = float(np.median(ratios))
    isf = float(np.clip(isf, 10.0, 200.0))
    return isf, len(ratios)


def _normalize_sleep_minutes(series: pd.Series) -> pd.Series:
    values = _clean_numeric(series, default=0.0)
    # If values look like hours (< 24), convert to minutes.
    if values.max() <= 24.0 and values.max() > 0:
        return values * 60.0
    return values


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    files = []
    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(args.input.rglob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {args.input}")

    frames: List[pd.DataFrame] = []
    subject_params: Dict[str, Dict[str, float]] = {}
    estimation_counts: Dict[str, Dict[str, int]] = {}

    for csv_path in files:
        df = _read_csv_auto(csv_path)

        ts_col = _col_pick(df, ["time", "timestamp", "datetime", "date_time", "DateTime", "Date"])
        if ts_col is None:
            raise SystemExit(f"No timestamp column found in {csv_path.name}")
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df[df["timestamp"].notna()].copy()

        subject_id = None
        subj_col = _col_pick(df, ["patient_id", "subject_id", "pid", "id"])
        if subj_col and df[subj_col].notna().any():
            subject_id = str(df[subj_col].dropna().iloc[0])
        else:
            subject_id = csv_path.stem
        df["subject_id"] = subject_id

        glucose_col = _col_pick(df, ["glucose", "glucose_mg_dl", "glucose_mgdl", "cgm", "sgv"])
        insulin_basal_col = _col_pick(df, ["basal_rate", "basal", "basal_units"])
        insulin_bolus_col = _col_pick(df, ["bolus_volume_delivered", "bolus", "bolus_units"])
        carb_col = _col_pick(df, ["carb_input", "carbs", "carb_grams", "carbohydrates"])
        steps_col = _col_pick(df, ["steps", "step_count"])
        calories_col = _col_pick(df, ["calories", "kcal", "energy"])
        hr_col = _col_pick(df, ["heart_rate", "hr", "bpm"])
        sleep_col = _col_pick(df, ["sleep_minutes", "sleep_duration", "sleep_hours", "sleep_quality", "sleep_score"])

        if glucose_col is None:
            raise SystemExit(f"No glucose column found in {csv_path.name}")

        df["glucose_actual_mgdl"] = _clean_numeric(df[glucose_col])

        basal_raw = _clean_numeric(df[insulin_basal_col]) if insulin_basal_col else pd.Series(0.0, index=df.index)
        if args.basal_is_rate:
            basal_units = basal_raw / 60.0 * args.time_step
        else:
            basal_units = basal_raw

        bolus_units = _clean_numeric(df[insulin_bolus_col]) if insulin_bolus_col else pd.Series(0.0, index=df.index)
        df["insulin_units"] = (basal_units + bolus_units).clip(lower=0.0, upper=args.max_insulin)

        carb_raw = _clean_numeric(df[carb_col]) if carb_col else pd.Series(0.0, index=df.index)
        if carb_col == "carb_input":
            carb_raw = carb_raw * args.carb_serving_grams
        df["carb_grams"] = carb_raw.clip(lower=0.0, upper=args.max_carbs)

        df["steps"] = _clean_numeric(df[steps_col]) if steps_col else 0.0
        df["calories"] = _clean_numeric(df[calories_col]) if calories_col else 0.0
        df["heart_rate"] = _clean_numeric(df[hr_col]) if hr_col else 0.0
        if sleep_col:
            df["sleep_minutes"] = _normalize_sleep_minutes(df[sleep_col])
        else:
            df["sleep_minutes"] = 0.0

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["delta_min"] = df["timestamp"].diff().dt.total_seconds() / 60.0
        max_gap = args.time_step * args.max_gap_multiplier
        segment_break = df["delta_min"].isna() | (df["delta_min"] <= 0) | (df["delta_min"] > max_gap)
        df.loc[segment_break, "delta_min"] = np.nan
        df["segment"] = segment_break.cumsum()

        df["glucose_trend_mgdl_min"] = (
            df.groupby("segment")["glucose_actual_mgdl"].diff() / df["delta_min"]
        )

        iob, cob = _derive_iob_cob_openaps(
            df["insulin_units"].fillna(0.0),
            df["carb_grams"].fillna(0.0),
            df["delta_min"],
            args.dia_minutes,
            args.peak_minutes,
            args.carb_absorb_minutes,
        )
        df["patient_iob_units"] = iob
        df["patient_cob_grams"] = cob

        minutes = (
            df["timestamp"].dt.hour * 60
            + df["timestamp"].dt.minute
            + df["timestamp"].dt.second / 60.0
        )
        df["time_of_day_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

        basal_rate, basal_n = _estimate_basal_rate(
            df["insulin_units"], df["carb_grams"], args.time_step, args.basal_default
        )
        icr, icr_n = _estimate_icr(
            df,
            meal_window_min=args.meal_window_min,
            min_meal_carbs=args.min_meal_carbs,
            min_bolus=args.min_bolus,
            default_icr=args.icr_default,
        )
        isf, isf_n = _estimate_isf(
            df,
            time_step=args.time_step,
            isf_window_min=args.isf_window_min,
            min_bolus=args.min_bolus,
            min_meal_carbs=args.min_meal_carbs,
            default_isf=args.isf_default,
        )

        df["effective_basal_rate_u_per_hr"] = basal_rate
        df["effective_icr"] = icr
        df["effective_isf"] = isf

        subject_params[subject_id] = {
            "effective_basal_rate_u_per_hr": float(basal_rate),
            "effective_icr": float(icr),
            "effective_isf": float(isf),
        }
        estimation_counts[subject_id] = {
            "basal_samples": int(basal_n),
            "icr_samples": int(icr_n),
            "isf_samples": int(isf_n),
        }

        df = df[df["glucose_actual_mgdl"] > 0].copy()
        df = df[df["glucose_trend_mgdl_min"].notna()].copy()
        df = df[np.isfinite(df["glucose_trend_mgdl_min"])].copy()

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.input),
        "records_total": int(len(combined)),
        "subjects": int(combined["subject_id"].nunique()),
        "subject_ids": sorted(combined["subject_id"].unique().tolist()),
        "start_time": combined["timestamp"].min().isoformat(),
        "end_time": combined["timestamp"].max().isoformat(),
        "median_interval_min": float(combined["delta_min"].median()),
        "percent_large_gaps": float((combined["delta_min"] > (args.time_step * args.max_gap_multiplier)).mean() * 100.0),
        "glucose_mean": float(combined["glucose_actual_mgdl"].mean()),
        "glucose_std": float(combined["glucose_actual_mgdl"].std()),
        "glucose_min": float(combined["glucose_actual_mgdl"].min()),
        "glucose_max": float(combined["glucose_actual_mgdl"].max()),
        "insulin_mean": float(combined["insulin_units"].mean()),
        "carb_mean": float(combined["carb_grams"].mean()),
        "time_step_minutes": args.time_step,
        "dia_minutes": args.dia_minutes,
        "peak_minutes": args.peak_minutes,
        "carb_absorb_minutes": args.carb_absorb_minutes,
        "iob_model": "openaps_bilinear",
        "fallbacks": {
            "effective_isf_default": args.isf_default,
            "effective_icr_default": args.icr_default,
            "effective_basal_default": args.basal_default,
        },
        "basal_is_rate": args.basal_is_rate,
        "carb_serving_grams": args.carb_serving_grams,
        "subject_parameters": subject_params,
        "estimation_counts": estimation_counts,
    }
    args.report.write_text(json.dumps(report, indent=2))

    output_cols = [
        "subject_id",
        "timestamp",
        "glucose_actual_mgdl",
        "glucose_trend_mgdl_min",
        "insulin_units",
        "carb_grams",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "effective_basal_rate_u_per_hr",
        "steps",
        "calories",
        "heart_rate",
        "sleep_minutes",
        "time_of_day_sin",
        "time_of_day_cos",
        "segment",
        "delta_min",
    ]
    save_dataset(combined[output_cols], args.output)
    print(f"Saved quality report: {args.report}")
    print(f"Saved training data: {args.output}")


if __name__ == "__main__":
    main()
