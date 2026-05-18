from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


CONTROL_FEATURE_COLUMNS = [
    "glucose_actual_mgdl",
    "glucose_trend_mgdl_min",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "carb_intake_grams",
]

CONTROL_TARGET_COLUMN = "teacher_insulin_units"


def _resolve_steps_path(run_dir: Path) -> Path:
    candidates = [run_dir / "raw" / "steps.csv", run_dir / "results.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No raw/steps.csv or results.csv found in {run_dir}")


def _prepare_control_frame(run_dir: Path, source_label: str) -> pd.DataFrame:
    steps_path = _resolve_steps_path(run_dir)
    df = pd.read_csv(steps_path).copy()
    required = [*CONTROL_FEATURE_COLUMNS, "delivered_insulin_units"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{steps_path}: missing columns {missing}")

    frame = df[[*CONTROL_FEATURE_COLUMNS, "delivered_insulin_units"]].copy()
    frame[CONTROL_TARGET_COLUMN] = pd.to_numeric(frame["delivered_insulin_units"], errors="coerce").fillna(0.0)
    frame["source_run"] = source_label
    time_minutes = df["time_minutes"] if "time_minutes" in df.columns else pd.Series(0.0, index=df.index)
    frame["time_minutes"] = pd.to_numeric(time_minutes, errors="coerce").fillna(0.0)
    frame["safety_triggered"] = (
        df.get("safety_triggered", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    )
    frame["algo_recommended_insulin_units"] = pd.to_numeric(
        df.get("algo_recommended_insulin_units", frame[CONTROL_TARGET_COLUMN]),
        errors="coerce",
    ).fillna(frame[CONTROL_TARGET_COLUMN])
    return frame.drop(columns=["delivered_insulin_units"])


def build_control_dataset_from_runs(
    run_dirs: Iterable[Tuple[str, Path]],
    *,
    output_path: Path,
    manifest_path: Path | None = None,
) -> Dict[str, Any]:
    frames = [_prepare_control_frame(path, label) for label, path in run_dirs]
    if not frames:
        raise ValueError("At least one run directory is required.")
    dataset = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    manifest = summarize_control_dataset(dataset)
    manifest["output_path"] = str(output_path)
    manifest["sources"] = sorted(dataset["source_run"].unique().tolist())
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def summarize_control_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    target = pd.to_numeric(df[CONTROL_TARGET_COLUMN], errors="coerce").fillna(0.0)
    glucose = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce")
    safety = df.get("safety_triggered", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    return {
        "rows": int(len(df)),
        "feature_columns": CONTROL_FEATURE_COLUMNS,
        "target_column": CONTROL_TARGET_COLUMN,
        "mean_teacher_insulin_units": round(float(target.mean()), 6) if len(target) else 0.0,
        "max_teacher_insulin_units": round(float(target.max()), 6) if len(target) else 0.0,
        "hypo_rows_below_70": int((glucose < 70.0).sum()),
        "safety_intervention_rows": int(safety.sum()),
    }


def train_linear_imitation_controller(
    df: pd.DataFrame,
    *,
    ridge_lambda: float = 1e-3,
) -> Dict[str, Any]:
    X = df[CONTROL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df[CONTROL_TARGET_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(X) < 2:
        raise ValueError("Need at least two rows to train a controller.")
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    X_scaled = (X - feature_mean) / feature_std
    design = np.c_[np.ones(len(X_scaled)), X_scaled]
    regularizer = ridge_lambda * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    predictions = np.clip(design @ weights, 0.0, None)
    return {
        "model_type": "linear_imitation_controller",
        "feature_columns": CONTROL_FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "ridge_lambda": ridge_lambda,
        "train_metrics": evaluate_controller_predictions(df, predictions),
    }


def predict_linear_controller(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    X = df[model["feature_columns"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    std = np.asarray(model["feature_std"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    scaled = (X - mean) / std
    return np.clip(float(model["intercept"]) + scaled @ weights, 0.0, None)


def evaluate_controller_predictions(df: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
    target = pd.to_numeric(df[CONTROL_TARGET_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    glucose = pd.to_numeric(df["glucose_actual_mgdl"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    error = predictions - target
    return {
        "rows": int(len(df)),
        "mae_units": round(float(np.mean(np.abs(error))), 6),
        "rmse_units": round(float(np.sqrt(np.mean(error**2))), 6),
        "max_prediction_units": round(float(np.max(predictions)), 6) if len(predictions) else 0.0,
        "unsafe_hypo_proposal_rows": int(((glucose < 70.0) & (predictions > 0.0)).sum()),
        "over_5u_proposal_rows": int((predictions > 5.0).sum()),
    }


def save_linear_controller(model: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_linear_controller(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
