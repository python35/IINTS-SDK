#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument("--min-meals", type=int, default=1)
    p.add_argument("--min-meal-gap-min", type=float, default=60.0)
    p.add_argument("--min-std", type=float, default=15.0)
    p.add_argument("--max-std", type=float, default=45.0)
    p.add_argument("--min-tir", type=float, default=70.0)
    p.add_argument("--max-tir", type=float, default=95.0)
    p.add_argument("--meal-rise-min", type=float, default=10.0)
    p.add_argument("--meal-rise-window-min", type=float, default=30.0)
    p.add_argument("--premeal-trend-max", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    if "time_minutes" not in df.columns:
        raise SystemExit("time_minutes missing")

    best = None
    best_key = None

    for subj in sorted(df["subject_id"].astype(str).unique()):
        subj_df = df[df["subject_id"].astype(str) == str(subj)].copy()
        max_time = subj_df["time_minutes"].max()
        if not pd.notna(max_time):
            continue
        day_count = int((max_time - 1e-9) // 1440) + 1
        for day in range(day_count):
            start = day * 1440
            end = start + 1440
            day_df = subj_df[(subj_df["time_minutes"] >= start) & (subj_df["time_minutes"] < end)].copy()
            if len(day_df) < args.min_rows:
                continue
            glucose = day_df["glucose_actual_mgdl"].astype(float)
            if glucose.isna().all():
                continue
            tir = ((glucose >= 70) & (glucose <= 180)).mean() * 100
            below54 = (glucose < 54).mean() * 100
            below70 = (glucose < 70).mean() * 100
            above250 = (glucose > 250).mean() * 100
            std = glucose.std()

            # Meal events
            meal_col = "carb_intake_grams" if "carb_intake_grams" in day_df.columns else None
            meal_times = np.array([])
            if meal_col:
                meal_times = day_df.loc[day_df[meal_col] > 0, "time_minutes"].to_numpy()
            meal_times = np.sort(meal_times)
            deduped = []
            last = None
            for t in meal_times:
                if last is None or (t - last) >= args.min_meal_gap_min:
                    deduped.append(t)
                    last = t
            meal_count = len(deduped)

            if below54 > 2:
                continue
            if std < args.min_std or std > args.max_std:
                continue
            if tir < args.min_tir or tir > args.max_tir:
                continue
            if meal_count < args.min_meals:
                continue

            # Meal alignment: glucose should not be rising steeply before the meal,
            # and should rise modestly after meal within the window.
            meal_ok = True
            if meal_col:
                # build a simple time->glucose series for this day
                time_vals = day_df["time_minutes"].to_numpy()
                glucose_vals = day_df["glucose_actual_mgdl"].to_numpy()
                # normalize day time to 0..1440 for checks
                time_vals = time_vals - start
                for mt in deduped:
                    mt = mt - start
                    if mt < 0 or mt > 1440:
                        continue
                    # pre-meal window
                    pre_mask = (time_vals >= mt - args.meal_rise_window_min) & (time_vals < mt)
                    post_mask = (time_vals >= mt) & (time_vals <= mt + args.meal_rise_window_min)
                    if pre_mask.sum() < 3 or post_mask.sum() < 3:
                        continue
                    pre_slope = np.polyfit(time_vals[pre_mask], glucose_vals[pre_mask], 1)[0]
                    post_delta = glucose_vals[post_mask].max() - glucose_vals[post_mask].min()
                    if pre_slope > args.premeal_trend_max:
                        meal_ok = False
                        break
                    if post_delta < args.meal_rise_min:
                        meal_ok = False
                        break
            if not meal_ok:
                continue

            score = tir - 0.5 * below70 - 0.2 * above250 - 0.05 * std
            key = (score, tir, -below70)
            if best is None or key > best_key:
                best = (subj, day, start, end, tir, below70, below54, above250, std, len(day_df), meal_count)
                best_key = key

    if not best:
        raise SystemExit("No suitable day found")

    subj, day, start, end, tir, below70, below54, above250, std, n, meals = best
    print("best", best)

    best_df = df[(df["subject_id"].astype(str) == str(subj)) & (df["time_minutes"] >= start) & (df["time_minutes"] < end)].copy()
    best_df = best_df.sort_values("time_minutes")
    best_df["time_minutes"] = best_df["time_minutes"] - start

    cols = [
        "time_minutes",
        "glucose_actual_mgdl",
        "glucose_to_algo_mgdl",
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
        "carb_intake_grams",
    ]
    for c in cols:
        if c not in best_df.columns:
            best_df[c] = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_df[cols].to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
