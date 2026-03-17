#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from iints.research.predictor import load_predictor_service
from iints.research.dataset import build_sequences


def _smooth_and_limit(grid: np.ndarray, series: np.ndarray) -> np.ndarray:
    s = pd.Series(series).rolling(window=6, min_periods=1, center=True).median()
    s = s.rolling(window=3, min_periods=1, center=True).mean()
    vals = s.to_numpy()
    max_delta = 15.0
    limited = [vals[0]]
    for v in vals[1:]:
        if np.isnan(v) or np.isnan(limited[-1]):
            limited.append(v)
            continue
        d = v - limited[-1]
        if d > max_delta:
            d = max_delta
        elif d < -max_delta:
            d = -max_delta
        limited.append(limited[-1] + d)
    return np.array(limited, dtype=float)


def _add_meal_announcement(df: pd.DataFrame, time_step: float = 5.0) -> pd.DataFrame:
    if "meal_announcement_grams" in df.columns:
        return df
    if "carb_intake_grams" not in df.columns:
        df["meal_announcement_grams"] = 0.0
        return df
    shift_steps = int(round(15.0 / time_step))
    group_cols = []
    if "subject_id" in df.columns:
        group_cols.append("subject_id")
    if "segment" in df.columns:
        group_cols.append("segment")
    sort_cols = [c for c in (*group_cols, "time_minutes") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    if group_cols:
        df["meal_announcement_grams"] = (
            df.groupby(group_cols, observed=False)["carb_intake_grams"]
            .shift(-shift_steps)
            .fillna(0.0)
        )
    else:
        df["meal_announcement_grams"] = df["carb_intake_grams"].shift(-shift_steps).fillna(0.0)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-meals", type=int, default=2)
    p.add_argument("--min-tir", type=float, default=70.0)
    p.add_argument("--max-tir", type=float, default=95.0)
    p.add_argument("--max-mae", type=float, default=30.0)
    p.add_argument("--min-rows", type=int, default=200)
    p.add_argument("--meal-rise-min", type=float, default=15.0)
    p.add_argument("--meal-rise-window-min", type=float, default=60.0)
    p.add_argument("--premeal-trend-max", type=float, default=0.2)
    p.add_argument("--max-start-offset", type=float, default=15.0)
    p.add_argument("--night-delta-max", type=float, default=25.0)
    return p.parse_args()


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
            # Require 4h context before day start for predictions
            if start < 240:
                continue
            context_df = subj_df[
                (subj_df["time_minutes"] >= start - 240) & (subj_df["time_minutes"] < end)
            ].copy()
            day_df = context_df[
                (context_df["time_minutes"] >= start) & (context_df["time_minutes"] < end)
            ].copy()
            if len(day_df) < args.min_rows:
                continue
            glucose = day_df["glucose_actual_mgdl"].astype(float)
            if glucose.isna().all():
                continue
            # Avoid nights with unexplained large rises
            night = day_df[(day_df["time_minutes"] >= start) & (day_df["time_minutes"] < start + 180)]
            if len(night) >= 10:
                night_delta = night["glucose_actual_mgdl"].max() - night["glucose_actual_mgdl"].min()
                if night_delta > args.night_delta_max:
                    continue
            tir = ((glucose >= 70) & (glucose <= 180)).mean() * 100
            if not (args.min_tir <= tir <= args.max_tir):
                continue
            if (glucose < 54).mean() * 100 > 2:
                continue

            meal_col = "carb_intake_grams" if "carb_intake_grams" in day_df.columns else None
            meal_times = np.array([])
            if meal_col:
                meal_times = day_df.loc[day_df[meal_col] > 0, "time_minutes"].to_numpy()
            meal_times = np.sort(meal_times)
            # dedupe meals within 30 min
            deduped = []
            last = None
            for t in meal_times:
                if last is None or (t - last) >= 30:
                    deduped.append(t)
                    last = t
            if len(deduped) < args.min_meals:
                continue

            # Build sequences and predict
            try:
                context_df = _add_meal_announcement(context_df, time_step=time_step)
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

            # Trim to day window for evaluation
            aligned = aligned[(context_df["time_minutes"] >= start) & (context_df["time_minutes"] < end)]
            # MAE on valid predictions
            obs = day_df["glucose_actual_mgdl"].to_numpy()
            mask = np.isfinite(aligned)
            if mask.sum() < 50:
                continue
            # Build smoothed/rate-limited prediction on 5-min grid for scoring
            grid = np.arange(0, 1440, 5.0)
            obs_grid = np.interp(grid, day_df["time_minutes"], obs)
            pred_df = day_df.assign(pred_val=aligned).dropna(subset=["pred_val"])
            pred_agg = pred_df.groupby("time_minutes")["pred_val"].median().sort_index()
            pred_grid = (
                pred_agg.reindex(pred_agg.index.union(grid))
                .interpolate(method="linear", limit_area="inside")
                .reindex(grid)
                .to_numpy()
            )
            pred_grid = _smooth_and_limit(grid, pred_grid)

            # start offset check (first valid prediction should not be far from observed)
            valid_grid = np.isfinite(pred_grid)
            if valid_grid.any():
                first_idx = np.argmax(valid_grid)
                if abs(pred_grid[first_idx] - obs_grid[first_idx]) > args.max_start_offset:
                    continue
            mae = float(np.nanmean(np.abs(pred_grid[valid_grid] - obs_grid[valid_grid])))
            if mae > args.max_mae:
                continue

            # Meal response: predicted rise should track observed rise within 60 min
            meal_penalty = 0
            for mt in deduped:
                window = (day_df["time_minutes"] >= mt) & (day_df["time_minutes"] <= mt + 60)
                if window.sum() < 3:
                    continue
                obs_window = obs[window]
                pred_window = aligned[window]
                obs_rise = np.nanmax(obs_window) - np.nanmin(obs_window)
                pred_rise = np.nanmax(pred_window) - np.nanmin(pred_window)
                if obs_rise >= args.meal_rise_min and pred_rise < (0.6 * obs_rise):
                    meal_penalty += 1

            # Ghost peaks outside meals: divergence outside meal windows
            ghost_penalty = 0
            if len(deduped) > 0:
                meal_mask = np.zeros_like(grid, dtype=bool)
                for mt in deduped:
                    meal_mask |= (grid >= (mt - 15)) & (grid <= (mt + 60))
                diff = np.abs(pred_grid - obs_grid)
                if np.isfinite(diff[~meal_mask]).any():
                    max_nonmeal = float(np.nanmax(diff[~meal_mask]))
                else:
                    max_nonmeal = float(np.nanmax(diff))
            else:
                max_nonmeal = float(np.nanmax(np.abs(pred_grid - obs_grid)))
            if max_nonmeal > 50:
                ghost_penalty += 1

            score = -mae - 5 * meal_penalty - 5 * ghost_penalty + tir / 10.0
            key = (score, -mae, -meal_penalty, -ghost_penalty, tir)
            if best is None or key > best_key:
                best = (subj, day, start, end, tir, mae, meal_penalty, ghost_penalty, len(day_df))
                best_key = key

    if not best:
        raise SystemExit("No suitable day found")

    subj, day, start, end, tir, mae, mp, gp, n = best
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
