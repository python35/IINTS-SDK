#!/usr/bin/env python3
"""
Attach model-based glucose predictions to a results CSV.

Usage:
  PYTHONPATH=src python3 tools/attach_ai_predictions.py \
    --results results/realistic_run/results.csv \
    --model models/hupa_finetuned_v2/predictor.pt \
    --out results/realistic_run/results_with_ai.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from iints.research.predictor import load_predictor_service
from iints.research.dataset import build_sequences


def _ensure_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time_minutes" in df.columns:
        minutes = df["time_minutes"].to_numpy()
    else:
        minutes = np.arange(len(df)) * 5.0
        df["time_minutes"] = minutes

    day_minutes = np.mod(minutes, 1440.0)
    radians = 2.0 * np.pi * (day_minutes / 1440.0)
    if "time_of_day_sin" not in df.columns:
        df["time_of_day_sin"] = np.sin(radians)
    if "time_of_day_cos" not in df.columns:
        df["time_of_day_cos"] = np.cos(radians)
    return df


def _ensure_optional_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    defaults = {
        "steps": 0.0,
        "calories": 0.0,
        "heart_rate": 0.0,
        "sleep_minutes": 0.0,
    }
    if any(col in feature_columns for col in ("time_of_day_sin", "time_of_day_cos")):
        df = _ensure_time_features(df)
    for col, value in defaults.items():
        if col in feature_columns and col not in df.columns:
            df[col] = float(value)
    return df


def _infer_time_step_minutes(df: pd.DataFrame) -> float:
    if "time_minutes" in df.columns:
        vals = pd.to_numeric(df["time_minutes"], errors="coerce").dropna().to_numpy()
        if vals.size > 1:
            diffs = np.diff(np.unique(vals))
            diffs = diffs[diffs > 0]
            if diffs.size:
                return float(np.median(diffs))
    return 5.0


def _apply_meal_announcement(
    df: pd.DataFrame,
    feature_columns: List[str],
    source_column: str,
    feature_name: str,
    announce_minutes: float | None,
) -> pd.DataFrame:
    if announce_minutes is None or feature_name not in feature_columns:
        return df
    if source_column not in df.columns:
        df[feature_name] = 0.0
        return df

    shift_steps = int(round(announce_minutes / _infer_time_step_minutes(df)))
    if shift_steps <= 0:
        df[feature_name] = df[source_column]
        return df

    group_cols = []
    if "subject_id" in df.columns:
        group_cols.append("subject_id")
    if "segment" in df.columns:
        group_cols.append("segment")

    sort_cols = [c for c in (*group_cols, "time_minutes") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if group_cols:
        df[feature_name] = (
            df.groupby(group_cols, observed=False)[source_column]
            .shift(-shift_steps)
            .fillna(0.0)
        )
    else:
        df[feature_name] = df[source_column].shift(-shift_steps).fillna(0.0)
    return df


def _assert_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--model", required=True, help="Path to predictor.pt")
    parser.add_argument("--out", required=True, help="Output CSV with predictions")
    parser.add_argument(
        "--meal-announce-minutes",
        type=float,
        default=None,
        help="Optional pre-announced meal lead time (minutes).",
    )
    parser.add_argument(
        "--meal-announce-column",
        default="carb_intake_grams",
        help="Source column for meal announcements.",
    )
    parser.add_argument(
        "--meal-announce-feature",
        default="meal_announcement_grams",
        help="Feature name used by the model for meal announcements.",
    )
    parser.add_argument(
        "--prediction-minutes",
        type=float,
        default=30.0,
        help="Prediction horizon (minutes) to extract from the model output.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    model_path = Path(args.model)
    out_path = Path(args.out)

    df = pd.read_csv(results_path)
    if "time_minutes" in df.columns:
        df = df.sort_values("time_minutes").reset_index(drop=True)

    service = load_predictor_service(model_path)
    feature_columns = list(service.feature_columns)
    history_steps = int(service.history_steps)
    horizon_steps = int(service.horizon_steps)
    target_column = service.config.get("target_column", "glucose_actual_mgdl")

    if target_column not in df.columns and "glucose_to_algo_mgdl" in df.columns:
        df[target_column] = df["glucose_to_algo_mgdl"]

    df = _ensure_optional_features(df, feature_columns)
    df = _apply_meal_announcement(
        df,
        feature_columns,
        source_column=args.meal_announce_column,
        feature_name=args.meal_announce_feature,
        announce_minutes=args.meal_announce_minutes,
    )

    _assert_columns(df, feature_columns + [target_column])

    X, _ = build_sequences(
        df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        feature_columns=feature_columns,
        target_column=target_column,
        subject_column=None,
        segment_column=None,
    )

    preds = service.predict(X)  # [N, horizon_steps]
    time_step = float(service.config.get("time_step_minutes", 5.0))
    pred_minutes = float(args.prediction_minutes)
    pred_step = max(1, int(round(pred_minutes / time_step)))
    pred_step = min(pred_step, preds.shape[1])  # 1..horizon_steps
    pred_values = preds[:, pred_step - 1]

    aligned = np.full(len(df), np.nan, dtype=float)
    for i, value in enumerate(pred_values):
        target_idx = i + history_steps + (pred_step - 1)
        if target_idx < len(aligned):
            aligned[target_idx] = float(value)

    df["prediction_horizon_minutes"] = pred_minutes
    df["prediction_history_minutes"] = float(service.config.get("history_steps", 0)) * float(
        service.config.get("time_step_minutes", 5.0)
    )
    if int(pred_minutes) == 30:
        df["predicted_glucose_ai_30min"] = aligned
    else:
        df[f"predicted_glucose_ai_{int(pred_minutes)}min"] = aligned
    # Backfill default column for plotting compatibility
    if "predicted_glucose_30min" not in df.columns or df["predicted_glucose_30min"].isna().all():
        if int(pred_minutes) == 30:
            df["predicted_glucose_30min"] = aligned

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
