#!/usr/bin/env python3
"""
Prepare the OhioT1DM dataset (2018/2020) for IINTS-AF LSTM training.

Outputs a merged CSV/Parquet with the standard research feature schema:
  - glucose, insulin (basal+bolus), carbs, IOB/COB, time features, etc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from iints.research.dataset import save_dataset


def _derive_iob_cob_openaps(
    insulin: pd.Series,
    carbs: pd.Series,
    delta_min: pd.Series,
    dia_minutes: float,
    peak_minutes: float,
    carb_absorb_minutes: float,
) -> tuple[pd.Series, pd.Series]:
    """
    Derive IOB and COB using the OpenAPS-style bilinear IOB approximation and
    exponential COB decay. Mirrors the logic in prepare_azt1d.py.
    """
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
        prev_iob = prev_iob * iob_decay + float(ins)

        cob_decay = np.exp(-dt / carb_absorb_minutes)
        prev_cob = prev_cob * cob_decay + float(carb)

        iob_vals.append(prev_iob)
        cob_vals.append(prev_cob)

    return pd.Series(iob_vals, index=insulin.index), pd.Series(cob_vals, index=carbs.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the OhioT1DM dataset for LSTM training.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_packs/public/OhioT1DM"),
        help="Root directory containing OhioT1DM {2018,2020}/{train,test} XML files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_packs/public/OhioT1DM/processed/ohio_merged.csv"),
        help="Output dataset path (CSV or Parquet)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data_packs/public/OhioT1DM/processed/quality_report.json"),
        help="Quality report output path",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2018,2020",
        help="Comma-separated list of years to include (e.g., 2018,2020)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated list of splits to include (train,test)",
    )
    parser.add_argument("--time-step", type=int, default=5, help="Expected CGM time step (minutes)")
    parser.add_argument("--max-gap-multiplier", type=float, default=2.5, help="Segment break multiplier")
    parser.add_argument("--dia-minutes", type=float, default=240.0, help="Insulin action duration (minutes)")
    parser.add_argument("--peak-minutes", type=float, default=75.0, help="IOB peak time (minutes)")
    parser.add_argument("--carb-absorb-minutes", type=float, default=120.0, help="Carb absorption duration (minutes)")
    parser.add_argument("--max-basal", type=float, default=20.0, help="Clip basal values above this (U/hr)")
    parser.add_argument("--max-bolus", type=float, default=30.0, help="Clip bolus values above this (U)")
    parser.add_argument("--max-carbs", type=float, default=200.0, help="Clip carb grams above this")
    parser.add_argument("--isf-default", type=float, default=50.0, help="Fallback ISF (mg/dL per U)")
    parser.add_argument("--icr-default", type=float, default=10.0, help="Fallback ICR (g/U)")
    parser.add_argument(
        "--no-filter-meals-without-rise",
        action="store_true",
        help="Disable filtering meal events that do not produce a glucose rise.",
    )
    parser.add_argument("--meal-rise-threshold", type=float, default=10.0, help="Min glucose rise to keep meal (mg/dL)")
    parser.add_argument("--meal-pre-window", type=float, default=10.0, help="Minutes before meal for baseline")
    parser.add_argument("--meal-post-window", type=float, default=90.0, help="Minutes after meal for rise detection")
    return parser.parse_args()


def _parse_ts(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%d-%m-%Y %H:%M:%S", errors="coerce")


def _events(root: ET.Element, tag: str) -> List[ET.Element]:
    node = root.find(tag)
    if node is None:
        return []
    return list(node.findall("event"))


def _build_grid(start: pd.Timestamp, end: pd.Timestamp, step_min: int) -> pd.DatetimeIndex:
    start = start.floor(f"{step_min}min")
    end = end.ceil(f"{step_min}min")
    return pd.date_range(start, end, freq=f"{step_min}min")


def _assign_bolus(df: pd.DataFrame, bolus_events: Iterable[ET.Element], step_min: int) -> pd.Series:
    bolus = pd.Series(0.0, index=df.index)
    for ev in bolus_events:
        ts_begin = _parse_ts(ev.attrib.get("ts_begin", ""))
        ts_end = _parse_ts(ev.attrib.get("ts_end", ""))
        dose = float(ev.attrib.get("dose", "0") or 0.0)
        if not pd.notna(ts_begin) or dose <= 0:
            continue
        if not pd.notna(ts_end):
            ts_end = ts_begin
        if ts_end < ts_begin:
            ts_end = ts_begin
        duration_min = (ts_end - ts_begin).total_seconds() / 60.0
        if duration_min <= step_min:
            ts = ts_begin.floor(f"{step_min}min")
            if ts in bolus.index:
                bolus.loc[ts] += dose
            continue
        steps = int(np.ceil(duration_min / step_min))
        if steps <= 0:
            continue
        dose_per_step = dose / steps
        times = pd.date_range(ts_begin.floor(f"{step_min}min"), periods=steps, freq=f"{step_min}min")
        for ts in times:
            if ts in bolus.index:
                bolus.loc[ts] += dose_per_step
    return bolus


def _assign_meals(df: pd.DataFrame, meal_events: Iterable[ET.Element], step_min: int) -> pd.Series:
    carbs = pd.Series(0.0, index=df.index)
    for ev in meal_events:
        ts = _parse_ts(ev.attrib.get("ts", ""))
        grams = float(ev.attrib.get("carbs", "0") or 0.0)
        if not pd.notna(ts) or grams <= 0:
            continue
        ts = ts.floor(f"{step_min}min")
        if ts in carbs.index:
            carbs.loc[ts] += grams
    return carbs


def _assign_sleep(df: pd.DataFrame, sleep_events: Iterable[ET.Element], step_min: int) -> pd.Series:
    sleep = pd.Series(0.0, index=df.index)
    for ev in sleep_events:
        ts_begin = _parse_ts(ev.attrib.get("ts_begin", ""))
        ts_end = _parse_ts(ev.attrib.get("ts_end", ""))
        if not pd.notna(ts_begin) or not pd.notna(ts_end):
            continue
        # Some files store end before begin (sleep from evening to morning)
        if ts_end < ts_begin:
            start = ts_end
            end = ts_begin
        else:
            start = ts_begin
            end = ts_end
        times = pd.date_range(start.floor(f"{step_min}min"), end.ceil(f"{step_min}min"), freq=f"{step_min}min")
        for ts in times:
            if ts in sleep.index:
                sleep.loc[ts] = float(step_min)
    return sleep


def _assign_exercise_flag(df: pd.DataFrame, exercise_events: Iterable[ET.Element], step_min: int) -> pd.Series:
    active = pd.Series(0.0, index=df.index)
    for ev in exercise_events:
        ts = _parse_ts(ev.attrib.get("ts", ""))
        duration = float(ev.attrib.get("duration", "0") or 0.0)
        if not pd.notna(ts) or duration <= 0:
            continue
        steps = int(np.ceil(duration / step_min))
        times = pd.date_range(ts.floor(f"{step_min}min"), periods=steps, freq=f"{step_min}min")
        for t in times:
            if t in active.index:
                active.loc[t] = 1.0
    return active


def _time_features(timestamps: pd.Series) -> Tuple[pd.Series, pd.Series]:
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    radians = 2.0 * np.pi * (minutes / 1440.0)
    return np.sin(radians), np.cos(radians)


def _recompute_subject_segments(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Recompute time_minutes/segments/IOB-COB per subject after merge."""
    df = df.sort_values("timestamp").copy()
    df["time_minutes"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    df["delta_min"] = df["timestamp"].diff().dt.total_seconds() / 60.0
    max_gap = args.time_step * args.max_gap_multiplier
    segment_break = df["delta_min"].isna() | (df["delta_min"] <= 0) | (df["delta_min"] > max_gap)
    df["segment"] = segment_break.cumsum()

    trend = df["glucose_actual_mgdl"].diff() / df["delta_min"]
    trend = trend.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["glucose_trend_mgdl_min"] = trend

    # Optional: filter meal events without observed rise
    total_meals = 0
    filtered_meals = 0
    if not args.no_filter_meals_without_rise and "carb_intake_grams" in df.columns:
        for _, seg in df.groupby("segment", sort=False):
            meal_mask = pd.to_numeric(seg["carb_intake_grams"], errors="coerce") > 0
            meal_idx = seg.index[meal_mask]
            for idx in meal_idx:
                total_meals += 1
                t0 = df.loc[idx, "time_minutes"]
                pre = seg[(seg["time_minutes"] >= t0 - args.meal_pre_window) & (seg["time_minutes"] <= t0 + args.meal_pre_window)]
                post = seg[(seg["time_minutes"] >= t0 + args.meal_pre_window) & (seg["time_minutes"] <= t0 + args.meal_post_window)]
                if pre.empty or post.empty:
                    continue
                pre_med = pd.to_numeric(pre["glucose_actual_mgdl"], errors="coerce").median()
                post_max = pd.to_numeric(post["glucose_actual_mgdl"], errors="coerce").max()
                if not np.isfinite(pre_med) or not np.isfinite(post_max):
                    continue
                if (post_max - pre_med) < args.meal_rise_threshold:
                    df.loc[idx, "carb_intake_grams"] = 0.0
                    if "carb_grams" in df.columns:
                        df.loc[idx, "carb_grams"] = 0.0
                    filtered_meals += 1

    iob_parts = []
    cob_parts = []
    for _, seg in df.groupby("segment", sort=False):
        iob, cob = _derive_iob_cob_openaps(
            seg["insulin_units"].fillna(0.0),
            seg["carb_grams"].fillna(0.0),
            seg["delta_min"].fillna(args.time_step),
            args.dia_minutes,
            args.peak_minutes,
            args.carb_absorb_minutes,
        )
        iob_parts.append(iob)
        cob_parts.append(cob)
    df["derived_iob_units"] = pd.concat(iob_parts).sort_index()
    df["derived_cob_grams"] = pd.concat(cob_parts).sort_index()
    df["derived_iob_units"] = df["derived_iob_units"].clip(lower=0.0)
    df["derived_cob_grams"] = df["derived_cob_grams"].clip(lower=0.0)
    df["patient_iob_units"] = df["derived_iob_units"]
    df["patient_cob_grams"] = df["derived_cob_grams"]
    df.attrs["total_meals"] = total_meals
    df.attrs["filtered_meals"] = filtered_meals
    return df


def _process_xml(path: Path, time_step: int, args: argparse.Namespace) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    subject_id = root.attrib.get("id", path.stem.split("-")[0])

    glucose_events = _events(root, "glucose_level")
    if not glucose_events:
        return pd.DataFrame()

    glucose_rows = []
    for ev in glucose_events:
        ts = _parse_ts(ev.attrib.get("ts", ""))
        value = pd.to_numeric(ev.attrib.get("value", ""), errors="coerce")
        if not pd.notna(ts) or not pd.notna(value):
            continue
        glucose_rows.append((ts, float(value)))
    if not glucose_rows:
        return pd.DataFrame()

    glucose_df = pd.DataFrame(glucose_rows, columns=["timestamp", "glucose"]).sort_values("timestamp")
    grid = _build_grid(glucose_df["timestamp"].min(), glucose_df["timestamp"].max(), time_step)

    df = pd.DataFrame(index=grid)
    df["timestamp"] = df.index
    df["subject_id"] = str(subject_id)

    # Glucose: interpolate on grid
    glucose_series = glucose_df.set_index("timestamp")["glucose"].sort_index()
    # Interpolate on union of original timestamps + target grid to avoid losing all values
    glucose_series = (
        glucose_series.reindex(glucose_series.index.union(grid))
        .sort_index()
        .interpolate(method="time")
        .reindex(grid)
        .ffill()
        .bfill()
    )
    df["glucose_actual_mgdl"] = glucose_series
    df["glucose_to_algo_mgdl"] = df["glucose_actual_mgdl"]

    # Basal rate (U/hr), forward fill
    basal_events = _events(root, "basal")
    basal_rows = []
    for ev in basal_events:
        ts = _parse_ts(ev.attrib.get("ts", ""))
        value = pd.to_numeric(ev.attrib.get("value", ""), errors="coerce")
        if not pd.notna(ts) or not pd.notna(value):
            continue
        basal_rows.append((ts, float(value)))
    if basal_rows:
        basal_df = pd.DataFrame(basal_rows, columns=["timestamp", "basal"]).sort_values("timestamp")
        basal_series = basal_df.set_index("timestamp")["basal"].reindex(grid).ffill().bfill()
    else:
        basal_series = pd.Series(0.0, index=grid)
    basal_series = basal_series.clip(upper=args.max_basal)
    df["effective_basal_rate_u_per_hr"] = basal_series.to_numpy()
    df["basal_units"] = basal_series.to_numpy() * (time_step / 60.0)

    # Bolus insulin
    bolus_units = _assign_bolus(df, _events(root, "bolus"), time_step)
    bolus_units = bolus_units.clip(upper=args.max_bolus)
    df["bolus_units"] = bolus_units.to_numpy()

    # Meal carbs
    carb_grams = _assign_meals(df, _events(root, "meal"), time_step)
    carb_grams = carb_grams.clip(upper=args.max_carbs)
    df["carb_grams"] = carb_grams.to_numpy()
    df["carb_intake_grams"] = df["carb_grams"]

    # Sleep + exercise flags
    df["sleep_minutes"] = _assign_sleep(df, _events(root, "sleep"), time_step).to_numpy()
    exercise_flag = _assign_exercise_flag(df, _events(root, "exercise"), time_step)
    df["device_mode_code"] = np.where(
        df["sleep_minutes"] > 0, 1.0, np.where(exercise_flag > 0, 2.0, 0.0)
    )

    # Placeholder features not present in OhioT1DM
    df["steps"] = 0.0
    df["calories"] = 0.0
    df["heart_rate"] = 0.0

    # Derived time columns
    df["time_minutes"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60.0
    df["delta_min"] = df["timestamp"].diff().dt.total_seconds() / 60.0

    max_gap = time_step * args.max_gap_multiplier
    segment_break = df["delta_min"].isna() | (df["delta_min"] <= 0) | (df["delta_min"] > max_gap)
    df["segment"] = segment_break.cumsum()

    df["glucose_trend_mgdl_min"] = df["glucose_actual_mgdl"].diff() / df["delta_min"]
    df["glucose_trend_mgdl_min"] = df["glucose_trend_mgdl_min"].fillna(0.0)

    df["insulin_units"] = df["basal_units"] + df["bolus_units"]

    # IOB/COB
    iob, cob = _derive_iob_cob_openaps(
        df["insulin_units"].fillna(0.0),
        df["carb_grams"].fillna(0.0),
        df["delta_min"],
        args.dia_minutes,
        args.peak_minutes,
        args.carb_absorb_minutes,
    )
    df["derived_iob_units"] = iob
    df["derived_cob_grams"] = cob
    df["patient_iob_units"] = df["derived_iob_units"]
    df["patient_cob_grams"] = df["derived_cob_grams"]

    df["effective_isf"] = float(args.isf_default)
    df["effective_icr"] = float(args.icr_default)

    sin_t, cos_t = _time_features(df["timestamp"])
    df["time_of_day_sin"] = sin_t
    df["time_of_day_cos"] = cos_t

    # Final columns in standard order
    columns = [
        "subject_id",
        "timestamp",
        "time_minutes",
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
        "glucose_trend_mgdl_min",
        "insulin_units",
        "carb_grams",
        "carb_intake_grams",
        "derived_iob_units",
        "derived_cob_grams",
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
        "device_mode_code",
        "segment",
    ]
    return df[columns]


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    frames: List[pd.DataFrame] = []
    for year in years:
        year_dir = args.input / year
        for split in splits:
            split_dir = year_dir / split
            if not split_dir.exists():
                continue
            for xml_path in sorted(split_dir.glob("*.xml")):
                df = _process_xml(xml_path, args.time_step, args)
                if df.empty:
                    continue
                df["dataset_year"] = year
                df["dataset_split"] = split
                frames.append(df)

    if not frames:
        raise SystemExit("No OhioT1DM XML files were processed.")

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged = merged.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)

    # Recompute segments and derived signals after merging all files per subject.
    rebuilt = []
    total_meals = 0
    filtered_meals = 0
    for subject_id, sdf in merged.groupby("subject_id", sort=False):
        sdf = _recompute_subject_segments(sdf, args)
        sdf["subject_id"] = subject_id
        total_meals += int(sdf.attrs.get("total_meals", 0))
        filtered_meals += int(sdf.attrs.get("filtered_meals", 0))
        rebuilt.append(sdf)
    merged = pd.concat(rebuilt, ignore_index=True)
    save_dataset(merged, args.output)

    report = {
        "rows": int(len(merged)),
        "subjects": int(merged["subject_id"].nunique()),
        "years": years,
        "splits": splits,
        "time_step_minutes": args.time_step,
        "iob_model": "openaps_bilinear",
        "segment_recomputed": True,
        "meal_filter_applied": not args.no_filter_meals_without_rise,
        "meal_filter_threshold_mgdl": args.meal_rise_threshold,
        "meal_filter_pre_window_min": args.meal_pre_window,
        "meal_filter_post_window_min": args.meal_post_window,
        "meal_events_total": int(total_meals),
        "meal_events_filtered": int(filtered_meals),
        "features": [
            "glucose_actual_mgdl",
            "glucose_trend_mgdl_min",
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
            "meal_announcement_grams (optional)",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Saved dataset: {args.output}")
    print(f"Saved quality report: {args.report}")


if __name__ == "__main__":
    main()
