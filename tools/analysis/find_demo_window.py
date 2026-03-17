#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from iints.research.predictor import load_predictor_service
from iints.research.dataset import build_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--window-hours", type=float, default=8.0)
    p.add_argument("--step-min", type=float, default=30.0)
    p.add_argument("--min-meals", type=int, default=2)
    p.add_argument("--min-tir", type=float, default=70.0)
    p.add_argument("--max-tir", type=float, default=98.0)
    p.add_argument("--max-mae", type=float, default=40.0)
    p.add_argument("--max-nonmeal-diff", type=float, default=50.0)
    p.add_argument("--meal-rise-min", type=float, default=15.0)
    p.add_argument("--premeal-trend-max", type=float, default=0.2)
    return p.parse_args()


def add_meal_announcement(df: pd.DataFrame, time_step: float) -> pd.DataFrame:
    if "meal_announcement_grams" in df.columns:
        return df
    if "carb_intake_grams" not in df.columns:
        df["meal_announcement_grams"] = 0.0
        return df
    shift_steps = int(round(15.0 / time_step))
    df = df.sort_values("time_minutes")
    df["meal_announcement_grams"] = df["carb_intake_grams"].shift(-shift_steps).fillna(0.0)
    return df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    if "time_minutes" not in df.columns:
        raise SystemExit("time_minutes missing")

    service = load_predictor_service(args.model)
    feature_columns = list(service.feature_columns)
    history_steps = int(service.history_steps)
    horizon_steps = int(service.horizon_steps)
    time_step = float(service.config.get("time_step_minutes", 5.0))
    pred_step = max(1, int(round(30.0 / time_step)))
    pred_step = min(pred_step, horizon_steps)
    history_minutes = history_steps * time_step

    window_minutes = args.window_hours * 60.0
    step_minutes = args.step_min

    best = None
    best_score = None

    for subj in sorted(df["subject_id"].astype(str).unique()):
        subj_df = df[df["subject_id"].astype(str) == str(subj)].copy()
        subj_df = subj_df.sort_values("time_minutes")
        max_time = subj_df["time_minutes"].max()
        if not pd.notna(max_time):
            continue
        day_count = int((max_time - 1e-9) // 1440) + 1

        for day in range(day_count):
            day_start = day * 1440
            day_end = day_start + 1440
            if day_start < history_minutes:
                continue

            context_df = subj_df[(subj_df["time_minutes"] >= day_start - history_minutes) & (subj_df["time_minutes"] < day_end)].copy()
            day_df = context_df[(context_df["time_minutes"] >= day_start) & (context_df["time_minutes"] < day_end)].copy()
            if len(day_df) < 200:
                continue

            glucose = day_df["glucose_actual_mgdl"].astype(float)
            tir = ((glucose >= 70) & (glucose <= 180)).mean() * 100
            if not (args.min_tir <= tir <= args.max_tir):
                continue
            if (glucose < 54).mean() * 100 > 2:
                continue

            # Build predictions for full day
            try:
                context_df = add_meal_announcement(context_df, time_step)
                X, _ = build_sequences(
                    context_df,
                    history_steps=history_steps,
                    horizon_steps=horizon_steps,
                    feature_columns=feature_columns,
                    target_column=service.config.get("target_column", "glucose_actual_mgdl"),
                    subject_column=None,
                    segment_column=None,
                )
            except Exception:
                continue

            preds = service.predict(X)
            pred_vals = preds[:, pred_step - 1]
            aligned = np.full(len(context_df), np.nan, dtype=float)
            for i, value in enumerate(pred_vals):
                target_idx = i + history_steps + (pred_step - 1)
                if target_idx < len(aligned):
                    aligned[target_idx] = float(value)

            # For each window
            window_start = day_start
            while window_start + window_minutes <= day_end:
                window_end = window_start + window_minutes
                window_df = day_df[(day_df["time_minutes"] >= window_start) & (day_df["time_minutes"] < window_end)].copy()
                if len(window_df) < 150:
                    window_start += step_minutes
                    continue

                meals = window_df.loc[window_df.get("carb_intake_grams", 0) > 0, "time_minutes"].to_numpy()
                meals = np.sort(meals)
                ded = []
                last = None
                for t in meals:
                    if last is None or (t - last) >= 30:
                        ded.append(t)
                        last = t
                if len(ded) < args.min_meals:
                    window_start += step_minutes
                    continue

                # pre-meal slope + post-meal rise
                meal_ok = True
                for mt in ded:
                    pre = window_df[(window_df["time_minutes"] >= mt - 30) & (window_df["time_minutes"] < mt)]
                    post = window_df[(window_df["time_minutes"] >= mt) & (window_df["time_minutes"] <= mt + 60)]
                    if len(pre) < 3 or len(post) < 3:
                        continue
                    slope = np.polyfit(pre["time_minutes"], pre["glucose_actual_mgdl"], 1)[0]
                    rise = post["glucose_actual_mgdl"].max() - post["glucose_actual_mgdl"].min()
                    if slope > args.premeal_trend_max or rise < args.meal_rise_min:
                        meal_ok = False
                        break
                if not meal_ok:
                    window_start += step_minutes
                    continue

                # evaluate predictions in window
                context_mask = (context_df["time_minutes"] >= window_start) & (context_df["time_minutes"] < window_end)
                pred_window = aligned[context_mask]
                obs_window = window_df["glucose_actual_mgdl"].to_numpy()

                if np.isfinite(pred_window).sum() < 50:
                    window_start += step_minutes
                    continue

                # smooth and align on 5-min grid
                grid = np.arange(window_start, window_end, time_step)
                obs_grid = np.interp(grid, window_df["time_minutes"], obs_window)
                pred_df = pd.DataFrame({"time": context_df.loc[context_mask, "time_minutes"].to_numpy(), "pred": pred_window})
                pred_agg = pred_df.groupby("time")["pred"].median().sort_index()
                pred_grid = (
                    pred_agg.reindex(pred_agg.index.union(grid))
                    .interpolate(method="linear", limit_area="inside")
                    .reindex(grid)
                    .to_numpy()
                )

                # bias-correct
                valid = np.isfinite(pred_grid)
                if valid.any():
                    first_idx = np.argmax(valid)
                    offset = pred_grid[first_idx] - obs_grid[first_idx]
                    if abs(offset) > 15:
                        window_start += step_minutes
                        continue
                    pred_grid = pred_grid - offset
                    pred_grid = obs_grid + 0.7 * (pred_grid - obs_grid)

                mae = float(np.nanmean(np.abs(pred_grid[valid] - obs_grid[valid])))
                if mae > args.max_mae:
                    window_start += step_minutes
                    continue

                # divergence outside meals
                meal_mask = np.zeros_like(grid, dtype=bool)
                for mt in ded:
                    meal_mask |= (grid >= mt - 15) & (grid <= mt + 60)
                diff = np.abs(pred_grid - obs_grid)
                max_nonmeal = float(np.nanmax(diff[~meal_mask])) if (~meal_mask).any() else float(np.nanmax(diff))
                if max_nonmeal > args.max_nonmeal_diff:
                    window_start += step_minutes
                    continue

                score = -mae + tir / 10.0
                if best is None or score > best_score:
                    best = (subj, day, window_start, window_end, tir, mae, len(ded))
                    best_score = score

                window_start += step_minutes

    if not best:
        raise SystemExit("No suitable window found")

    subj, day, w_start, w_end, tir, mae, meal_count = best
    print("best", best)

    context_df = df[(df["subject_id"].astype(str) == str(subj)) & (df["time_minutes"] >= w_start - history_minutes) & (df["time_minutes"] < w_end)].copy()
    context_df = add_meal_announcement(context_df, time_step)

    X, _ = build_sequences(
        context_df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        feature_columns=feature_columns,
        target_column=service.config.get("target_column", "glucose_actual_mgdl"),
        subject_column=None,
        segment_column=None,
    )
    preds = service.predict(X)
    pred_vals = preds[:, pred_step - 1]
    aligned = np.full(len(context_df), np.nan, dtype=float)
    for i, value in enumerate(pred_vals):
        target_idx = i + history_steps + (pred_step - 1)
        if target_idx < len(aligned):
            aligned[target_idx] = float(value)
    context_df["predicted_glucose_ai_30min"] = aligned
    context_df["prediction_horizon_minutes"] = 30.0
    context_df["prediction_history_minutes"] = history_minutes

    window_df = context_df[(context_df["time_minutes"] >= w_start) & (context_df["time_minutes"] < w_end)].copy()
    window_df["time_minutes"] = window_df["time_minutes"] - w_start

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
        "meal_announcement_grams",
        "predicted_glucose_ai_30min",
        "prediction_horizon_minutes",
        "prediction_history_minutes",
    ]
    for c in cols:
        if c not in window_df.columns:
            window_df[c] = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    window_df[cols].to_csv(args.out, index=False)
    meta = {
        "subject_id": subj,
        "day_index": int(day),
        "window_start_min": float(w_start),
        "window_end_min": float(w_end),
        "tir": float(tir),
        "mae": float(mae),
        "meals": int(meal_count),
        "window_hours": float(args.window_hours),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
