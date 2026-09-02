from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iints.research.dual_stream import decompose_dual_stream, extract_dual_stream_pre_meal_features


@dataclass(frozen=True)
class PPGRTrajectoryMetrics:
    """Evaluation metrics for a 2-hour postprandial glucose trajectory."""

    mae_mgdl: float
    rmse_mgdl: float
    pearson_r: float
    peak_glucose_mae_mgdl: float
    time_to_peak_mae_minutes: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PPGRBenchmarkResult:
    """Summary of model comparison across multiple PPGR architectures."""

    sensor: str
    sample_count: int
    train_count: int
    test_count: int
    train_subject_count: int
    test_subject_count: int
    split_strategy: str
    group_disjoint: bool
    glucofm_checkpoint_sha256: str | None
    models: Mapping[str, PPGRTrajectoryMetrics]
    winning_model: str
    relative_mae_gain_pct: float
    report_json: Path
    report_md: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "sample_count": self.sample_count,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "train_subject_count": self.train_subject_count,
            "test_subject_count": self.test_subject_count,
            "split_strategy": self.split_strategy,
            "group_disjoint": self.group_disjoint,
            "glucofm_checkpoint_sha256": self.glucofm_checkpoint_sha256,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "winning_model": self.winning_model,
            "relative_mae_gain_pct": self.relative_mae_gain_pct,
            "report_json": str(self.report_json),
            "report_md": str(self.report_md),
        }


def compute_trajectory_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_minutes: int = 5,
) -> PPGRTrajectoryMetrics:
    """
    Compute postprandial trajectory metrics between ground truth and predicted curves.
    y_true, y_pred: shape (N, T) where T is the number of forecast steps (e.g. 24 for 2h at 5min).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return PPGRTrajectoryMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    diffs = np.abs(y_true[mask] - y_pred[mask])
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

    # Correlation across valid pairs
    if len(y_true[mask]) > 1:
        corr_mat = np.corrcoef(y_true[mask], y_pred[mask])
        r = float(corr_mat[0, 1]) if np.isfinite(corr_mat[0, 1]) else 0.0
    else:
        r = 0.0

    # Peak metrics per sample
    true_peaks = np.nanmax(y_true, axis=1)
    pred_peaks = np.nanmax(y_pred, axis=1)
    peak_mask = np.isfinite(true_peaks) & np.isfinite(pred_peaks)
    peak_mae = float(np.mean(np.abs(true_peaks[peak_mask] - pred_peaks[peak_mask]))) if np.any(peak_mask) else 0.0

    true_t_peak = np.nanargmax(y_true, axis=1) * step_minutes
    pred_t_peak = np.nanargmax(y_pred, axis=1) * step_minutes
    ttp_mae = float(np.mean(np.abs(true_t_peak[peak_mask] - pred_t_peak[peak_mask]))) if np.any(peak_mask) else 0.0

    return PPGRTrajectoryMetrics(
        mae_mgdl=round(mae, 2),
        rmse_mgdl=round(rmse, 2),
        pearson_r=round(r, 3),
        peak_glucose_mae_mgdl=round(peak_mae, 2),
        time_to_peak_mae_minutes=round(ttp_mae, 1),
    )


class BasePPGRModel:
    """Base interface for postprandial glucose trajectory predictors."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> BasePPGRModel:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CarbOnlyLinearPPGR(BasePPGRModel):
    """
    Standard carbohydrate-only linear regression model (1 feature: carb_grams).
    Predicts Delta G(t) for each step t in [1..24].
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None  # (1, T)
        self.intercept: np.ndarray | None = None  # (T,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CarbOnlyLinearPPGR:
        # X[:, 0] is assumed to be carbs_g
        carbs = X[:, 0:1]  # (N, 1)
        valid = np.isfinite(carbs.squeeze()) & np.all(np.isfinite(y), axis=1)
        if not np.any(valid):
            self.weights = np.zeros((1, y.shape[1]))
            self.intercept = np.zeros(y.shape[1])
            return self

        xc = carbs[valid]
        yc = y[valid]
        X_design = np.hstack([xc, np.ones((len(xc), 1))])
        # Solve least squares for all time steps
        sol, _, _, _ = np.linalg.lstsq(X_design, yc, rcond=None)
        self.weights = sol[:-1]  # (1, T)
        self.intercept = sol[-1]  # (T,)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        carbs = X[:, 0:1]
        if self.weights is None or self.intercept is None:
            return np.zeros((len(X), 24))
        return carbs @ self.weights + self.intercept


class MultiMacroLinearPPGR(BasePPGRModel):
    """
    Biphasic macronutrient model: Carbs, Protein, Fat, Fiber, Calories.
    Captures delayed gastric emptying and late glucose elevation.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.intercept: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultiMacroLinearPPGR:
        # X features: carbs, protein, fat, fiber, calories (first 5 columns)
        feats = X[:, :5]
        valid = np.all(np.isfinite(feats), axis=1) & np.all(np.isfinite(y), axis=1)
        if not np.any(valid):
            self.weights = np.zeros((feats.shape[1], y.shape[1]))
            self.intercept = np.zeros(y.shape[1])
            return self

        xf = feats[valid]
        yf = y[valid]
        # Ridge regression solution: (X'X + alpha*I)^-1 X'Y
        mean_x = np.mean(xf, axis=0)
        mean_y = np.mean(yf, axis=0)
        xc = xf - mean_x
        yc = yf - mean_y

        reg = self.alpha * np.eye(xc.shape[1])
        w = np.linalg.solve(xc.T @ xc + reg, xc.T @ yc)
        self.weights = w
        self.intercept = mean_y - mean_x @ w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = X[:, :5]
        if self.weights is None or self.intercept is None:
            return np.zeros((len(X), 24))
        return feats @ self.weights + self.intercept


class ContextFeatureRidgePPGR(BasePPGRModel):
    """
    Ridge model over measured meal, subject, and optional pre-meal features.
    Combines:
    - Slower circadian baseline context (x_base_last, x_base_slope, diurnal phase)
    - Acute event context (x_event_last, x_event_auc, event_velocity)
    - Full meal macronutrients (carbs, protein, fat, fiber, calories)
    - Subject metabolic covariates (BMI, HbA1c, fasting glucose)
    """

    def __init__(self, alpha: float = 2.0) -> None:
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.intercept: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ContextFeatureRidgePPGR:
        valid = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
        if not np.any(valid):
            self.weights = np.zeros((X.shape[1], y.shape[1]))
            self.intercept = np.zeros(y.shape[1])
            return self

        xf = X[valid]
        yf = y[valid]

        mean_x = np.mean(xf, axis=0)
        mean_y = np.mean(yf, axis=0)
        xc = xf - mean_x
        yc = yf - mean_y

        reg = self.alpha * np.eye(xc.shape[1])
        w = np.linalg.solve(xc.T @ xc + reg, xc.T @ yc)
        self.weights = w
        self.intercept = mean_y - mean_x @ w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.intercept is None:
            return np.zeros((len(X), 24))
        return X @ self.weights + self.intercept


# Compatibility alias for callers written before the scientific naming fix.
# This estimator does not implement GlucoFM and is never labelled GlucoFM in
# generated reports.
DualStreamGlucoFMPPGR = ContextFeatureRidgePPGR


def _parse_pre_meal_history(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        values = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    finite = values[np.isfinite(values)]
    return finite.tolist() if finite.size >= 3 else None


def _build_ppgr_dataset(
    meals_df: pd.DataFrame,
    sensor: str = "dexcom",
    subjects_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Extract feature matrix X and target trajectory y (24 steps at 5-min intervals) from meals table.
    """
    prefix = "dexcom" if sensor.lower() == "dexcom" else "libre"
    target_cols = [f"{prefix}_t{t}" for t in range(5, 125, 5)]

    subj_map: dict[str, dict[str, float]] = {}
    if subjects_df is not None and not subjects_df.empty:
        for _, s_row in subjects_df.iterrows():
            sid = str(s_row.get("subject_id", ""))
            subj_map[sid] = {
                "bmi": float(s_row.get("bmi", 25.0)) if pd.notna(s_row.get("bmi")) else 25.0,
                "hba1c": float(s_row.get("hba1c_pct", 5.6)) if pd.notna(s_row.get("hba1c_pct")) else 5.6,
                "fast_glu": float(s_row.get("fasting_glucose_mgdl", 95.0)) if pd.notna(s_row.get("fasting_glucose_mgdl")) else 95.0,
            }

    history_column = next(
        (
            column
            for column in (
                f"pre_meal_cgm_{prefix}_json",
                "pre_meal_cgm_json",
                "pre_meal_glucose_history_json",
            )
            if column in meals_df.columns
        ),
        None,
    )
    feature_names = [
        "carbs_g",
        "protein_g",
        "fat_g",
        "fiber_g",
        "calories_kcal",
        "pre_meal_glucose",
        "tod_sin",
        "tod_cos",
        "bmi",
        "hba1c",
        "fasting_glucose",
    ]
    if history_column is not None:
        feature_names.extend(
            [
                "pre_meal_baseline_last",
                "pre_meal_baseline_slope_60min",
                "pre_meal_event_last",
                "pre_meal_event_auc_60min",
            ]
        )

    x_list: list[list[float]] = []
    y_list: list[list[float]] = []
    subject_ids: list[str] = []

    for _, row in meals_df.iterrows():
        sid = str(row.get("subject_id", ""))
        g0 = pd.to_numeric(row.get(f"pre_meal_glucose_{prefix}", 100.0), errors="coerce")
        if pd.isna(g0) or g0 <= 0:
            g0 = 100.0

        # Extract target trajectory relative to G0: Delta G(t)
        traj: list[float] = []
        for col in target_cols:
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            traj.append(float(val) if pd.notna(val) else g0)
        delta_traj = [v - g0 for v in traj]

        carbs = float(pd.to_numeric(row.get("carbs_g", 0.0), errors="coerce") or 0.0)
        protein = float(pd.to_numeric(row.get("protein_g", 0.0), errors="coerce") or 0.0)
        fat = float(pd.to_numeric(row.get("fat_g", 0.0), errors="coerce") or 0.0)
        fiber = float(pd.to_numeric(row.get("fiber_g", 0.0), errors="coerce") or 0.0)
        cal = float(pd.to_numeric(row.get("calories_kcal", 0.0), errors="coerce") or 0.0)

        t_min = float(pd.to_numeric(row.get("time_minutes", 0.0), errors="coerce") or 0.0)
        tod_sin = float(np.sin(2 * np.pi * t_min / 1440.0))
        tod_cos = float(np.cos(2 * np.pi * t_min / 1440.0))

        s_meta = subj_map.get(sid, {"bmi": 25.0, "hba1c": 5.6, "fast_glu": 95.0})

        feat_vector = [
            carbs,
            protein,
            fat,
            fiber,
            cal,
            g0,
            tod_sin,
            tod_cos,
            s_meta["bmi"],
            s_meta["hba1c"],
            s_meta["fast_glu"],
        ]

        if history_column is not None:
            history = _parse_pre_meal_history(row.get(history_column))
            if history is None:
                # Missing context remains missing. It is not silently replaced
                # by a zero-valued "dual stream" feature.
                feat_vector.extend([np.nan, np.nan, np.nan, np.nan])
            else:
                stream = extract_dual_stream_pre_meal_features(
                    history,
                    sampling_interval_minutes=5.0,
                    time_of_day_minutes=t_min,
                )
                feat_vector.extend(
                    [
                        float(stream["baseline_last"]),
                        float(stream["baseline_slope_60min"]),
                        float(stream["event_last"]),
                        float(stream["event_auc_60min"]),
                    ]
                )

        x_list.append(feat_vector)
        y_list.append(delta_traj)
        subject_ids.append(sid)

    return (
        np.array(x_list, dtype=float),
        np.array(y_list, dtype=float),
        feature_names,
        np.asarray(subject_ids, dtype=str),
    )


def build_ppgr_dataset(
    meals_df: pd.DataFrame,
    sensor: str = "dexcom",
    subjects_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build measured PPGR features without inventing missing context values."""

    X, y, feature_names, _ = _build_ppgr_dataset(meals_df, sensor, subjects_df)
    return X, y, feature_names


def _json_array(value: Any, field: str) -> list[Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must contain a JSON array")
    return parsed


def _build_glucofm_embeddings(
    meals_df: pd.DataFrame,
    *,
    sensor: str,
    checkpoint: Path | str,
) -> tuple[np.ndarray, str]:
    """Embed measured 24-hour pre-meal histories with trained weights only."""

    from iints.research.glucofm import (
        embed_cgm_with_glucofm,
        load_glucofm_checkpoint,
        sha256_file,
    )

    prefix = "dexcom" if sensor.lower() == "dexcom" else "libre"
    history_column = next(
        (
            column
            for column in (
                f"pre_meal_cgm_{prefix}_json",
                "pre_meal_cgm_json",
                "pre_meal_glucose_history_json",
            )
            if column in meals_df.columns
        ),
        None,
    )
    if history_column is None:
        raise ValueError(
            "GlucoFM PPGR evaluation requires a pre-meal 24-hour history column "
            "(pre_meal_cgm_json or sensor-specific equivalent)"
        )
    timestamp_column = next(
        (
            column
            for column in (
                f"pre_meal_timestamps_{prefix}_json",
                "pre_meal_timestamps_json",
            )
            if column in meals_df.columns
        ),
        None,
    )
    encoder, _ = load_glucofm_checkpoint(checkpoint)
    embeddings: list[np.ndarray] = []
    for row_index, row in meals_df.iterrows():
        values = _json_array(row.get(history_column), history_column)
        timestamps = (
            _json_array(row.get(timestamp_column), timestamp_column)
            if timestamp_column is not None
            else None
        )
        try:
            embedding = embed_cgm_with_glucofm(
                values,
                encoder=encoder,
                timestamps=timestamps,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not embed PPGR row {row_index}: {exc}"
            ) from exc
        embeddings.append(np.asarray(embedding, dtype=float))
    return np.vstack(embeddings), sha256_file(checkpoint)


def run_ppgr_benchmark(
    meals_path: Path | str,
    output_dir: Path | str,
    *,
    sensor: str = "dexcom",
    subjects_path: Path | str | None = None,
    glucofm_checkpoint: Path | str | None = None,
    test_split: float = 0.25,
    seed: int = 42,
) -> PPGRBenchmarkResult:
    """
    Train and benchmark multiple PPGR forecasting models against 2-hour glucose trajectories.
    """
    m_path = Path(meals_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not m_path.is_file():
        raise FileNotFoundError(f"meals file not found: {m_path}")

    meals_df = pd.read_csv(m_path)
    subjects_df = pd.read_csv(subjects_path) if subjects_path and Path(subjects_path).is_file() else None

    X, y, feat_names, subject_ids = _build_ppgr_dataset(
        meals_df, sensor=sensor, subjects_df=subjects_df
    )
    n_samples = len(X)
    if n_samples < 4:
        raise ValueError(f"insufficient meal samples for PPGR benchmarking (found {n_samples})")
    if "subject_id" not in meals_df.columns:
        raise ValueError(
            "PPGR benchmarking requires a subject_id column for group-disjoint evaluation"
        )
    unique_subjects = np.unique(subject_ids[subject_ids != ""])
    if unique_subjects.size < 2:
        raise ValueError(
            "PPGR benchmarking requires at least two subjects; row-level random "
            "splitting is not permitted"
        )

    rng = np.random.default_rng(seed)
    shuffled_subjects = rng.permutation(unique_subjects)
    n_test_subjects = max(1, int(round(len(shuffled_subjects) * test_split)))
    n_test_subjects = min(n_test_subjects, len(shuffled_subjects) - 1)
    test_subjects = set(shuffled_subjects[:n_test_subjects].tolist())
    train_subjects = set(shuffled_subjects[n_test_subjects:].tolist())
    train_idx = np.flatnonzero(np.isin(subject_ids, list(train_subjects)))
    test_idx = np.flatnonzero(np.isin(subject_ids, list(test_subjects)))
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("subject-grouped PPGR split produced an empty partition")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model_inputs: dict[str, tuple[BasePPGRModel, np.ndarray, np.ndarray]] = {
        "Carb-Only Baseline": (CarbOnlyLinearPPGR(), X_train, X_test),
        "Multi-Macronutrient": (
            MultiMacroLinearPPGR(alpha=1.0),
            X_train,
            X_test,
        ),
    }
    if np.all(np.isfinite(X_train)) and np.all(np.isfinite(X_test)):
        context_name = (
            "Measured Pre-Meal Context Ridge"
            if "pre_meal_event_auc_60min" in feat_names
            else "Meal + Subject Covariate Ridge"
        )
        model_inputs[context_name] = (
            ContextFeatureRidgePPGR(alpha=2.0),
            X_train,
            X_test,
        )

    checkpoint_sha: str | None = None
    if glucofm_checkpoint is not None:
        embeddings, checkpoint_sha = _build_glucofm_embeddings(
            meals_df,
            sensor=sensor,
            checkpoint=glucofm_checkpoint,
        )
        glucofm_X = np.hstack([X, embeddings])
        glucofm_train = glucofm_X[train_idx]
        glucofm_test = glucofm_X[test_idx]
        if not np.all(np.isfinite(glucofm_train)) or not np.all(np.isfinite(glucofm_test)):
            raise ValueError("GlucoFM PPGR features contain non-finite values")
        model_inputs["IINTS GlucoFM v2 reproduction (frozen encoder + ridge)"] = (
            ContextFeatureRidgePPGR(alpha=2.0),
            glucofm_train,
            glucofm_test,
        )

    metrics: dict[str, PPGRTrajectoryMetrics] = {}
    for name, (model, model_X_train, model_X_test) in model_inputs.items():
        model.fit(model_X_train, y_train)
        pred_y = model.predict(model_X_test)
        metrics[name] = compute_trajectory_metrics(y_test, pred_y)

    carb_mae = metrics["Carb-Only Baseline"].mae_mgdl
    winning = min(metrics.keys(), key=lambda k: metrics[k].mae_mgdl)
    winning_mae = metrics[winning].mae_mgdl
    gain_pct = round(((carb_mae - winning_mae) / max(0.1, carb_mae)) * 100.0, 1)

    report_json_path = out_dir / "ppgr_benchmark_report.json"
    report_md_path = out_dir / "ppgr_benchmark_report.md"

    res = PPGRBenchmarkResult(
        sensor=sensor.lower(),
        sample_count=n_samples,
        train_count=len(train_idx),
        test_count=len(test_idx),
        train_subject_count=len(train_subjects),
        test_subject_count=len(test_subjects),
        split_strategy="subject-grouped",
        group_disjoint=True,
        glucofm_checkpoint_sha256=checkpoint_sha,
        models=metrics,
        winning_model=winning,
        relative_mae_gain_pct=gain_pct,
        report_json=report_json_path,
        report_md=report_md_path,
    )

    report_json_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")

    metric_rows = "\n".join(
        "| **{name}** | `{mae}` | `{rmse}` | `{corr}` | `{peak}` | `{ttp}` |".format(
            name=name,
            mae=metric.mae_mgdl,
            rmse=metric.rmse_mgdl,
            corr=metric.pearson_r,
            peak=metric.peak_glucose_mae_mgdl,
            ttp=metric.time_to_peak_mae_minutes,
        )
        for name, metric in metrics.items()
    )
    md_content = f"""# Postprandial Glycemic Response (PPGR) Benchmark Report

- **Sensor Type:** `{sensor.upper()}`
- **Total Meals Evaluated:** `{n_samples}` (Train: `{len(train_idx)}`, Test: `{len(test_idx)}`)
- **Subject Split:** `{len(train_subjects)}` train / `{len(test_subjects)}` test subjects (group-disjoint)
- **Forecast Horizon:** 120 minutes (24 steps at 5-minute sampling)
- **GlucoFM Checkpoint:** `{checkpoint_sha or 'not used'}`
- **Winning Architecture:** `{winning}`
- **Relative MAE Improvement over Carb-Only Baseline:** `{gain_pct}%`

## Model Performance Table

| Architecture | 2h Trajectory MAE (mg/dL) | RMSE (mg/dL) | Pearson r | Peak Glucose MAE (mg/dL) | Time-to-Peak Error (min) |
| :--- | :--- | :--- | :--- | :--- | :--- |
{metric_rows}

## Interpretation Guardrails
1. Results are computed on held-out subjects; no meal from a test subject appears in training.
2. A model is labelled GlucoFM only when a trained IINTS GlucoFM v2 reproduction checkpoint and measured 24-hour pre-meal history were supplied.
3. A lower error on this split is descriptive evidence, not a treatment or clinical-validity claim.
"""
    report_md_path.write_text(md_content, encoding="utf-8")

    return res


__all__ = [
    "PPGRTrajectoryMetrics",
    "PPGRBenchmarkResult",
    "BasePPGRModel",
    "CarbOnlyLinearPPGR",
    "MultiMacroLinearPPGR",
    "ContextFeatureRidgePPGR",
    "DualStreamGlucoFMPPGR",
    "compute_trajectory_metrics",
    "build_ppgr_dataset",
    "run_ppgr_benchmark",
]
