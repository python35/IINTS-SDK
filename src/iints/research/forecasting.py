from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from iints.research.evaluation import (
    forecast_error_report,
    hypoglycemia_detection_report,
    uncertainty_reliability_report,
)


DEFAULT_FORECAST_FEATURE_COLUMNS = [
    "glucose_actual_mgdl",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "glucose_trend_mgdl_min",
    "effective_dia_minutes",
    "steps",
    "heart_rate",
    "exercise_intensity",
    "stress_intensity",
    "insulin_antibody_binding_fraction",
    "insulin_antibody_release_fraction",
    "antibody_bound_insulin_units",
]


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for research-only glucose forecasting helpers."""

    history_minutes: int = 240
    horizon_minutes: int = 30
    time_step_minutes: int = 5
    max_rate_mgdl_min: float = 4.0
    trend_decay_minutes: float = 35.0
    insulin_action_minutes: float = 240.0
    carb_absorption_minutes: float = 120.0
    low_threshold_mgdl: float = 70.0
    urgent_low_threshold_mgdl: float = 54.0
    high_threshold_mgdl: float = 180.0
    very_high_threshold_mgdl: float = 250.0
    critical_high_threshold_mgdl: float = 300.0
    uncertainty_std_threshold_mgdl: float = 35.0

    def __post_init__(self) -> None:
        positive = {
            "history_minutes": self.history_minutes,
            "horizon_minutes": self.horizon_minutes,
            "time_step_minutes": self.time_step_minutes,
            "max_rate_mgdl_min": self.max_rate_mgdl_min,
            "trend_decay_minutes": self.trend_decay_minutes,
            "insulin_action_minutes": self.insulin_action_minutes,
            "carb_absorption_minutes": self.carb_absorption_minutes,
            "uncertainty_std_threshold_mgdl": self.uncertainty_std_threshold_mgdl,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        thresholds = (
            float(self.urgent_low_threshold_mgdl),
            float(self.low_threshold_mgdl),
            float(self.high_threshold_mgdl),
            float(self.very_high_threshold_mgdl),
            float(self.critical_high_threshold_mgdl),
        )
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError("forecast thresholds must all be finite")
        if list(thresholds) != sorted(thresholds) or len(set(thresholds)) != 5:
            raise ValueError("forecast thresholds must be strictly increasing")

    @property
    def history_steps(self) -> int:
        return max(1, int(round(self.history_minutes / self.time_step_minutes)))

    @property
    def horizon_steps(self) -> int:
        return max(1, int(round(self.horizon_minutes / self.time_step_minutes)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _safe_clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _bounded_feature(value: float, *, name: str, low: float, high: float) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or not low <= numeric <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return numeric


def _feature_index(feature_columns: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    for candidate in candidates:
        if candidate in feature_columns:
            return feature_columns.index(candidate)
    return None


class PhysiologyAwareBaseline:
    """
    Transparent glucose forecast baseline using trend, IOB, COB, stress and activity.

    This is intentionally not a medical model. It is a research baseline that gives
    the neural predictor a stronger, interpretable comparator than "last value".
    """

    def __init__(
        self,
        horizon_steps: int,
        *,
        time_step_minutes: float = 5.0,
        feature_columns: Optional[Sequence[str]] = None,
        max_rate_mgdl_min: float = 4.0,
        trend_decay_minutes: float = 35.0,
        insulin_action_minutes: float = 240.0,
        carb_absorption_minutes: float = 120.0,
    ) -> None:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0")
        numeric_positive = {
            "time_step_minutes": time_step_minutes,
            "max_rate_mgdl_min": max_rate_mgdl_min,
            "trend_decay_minutes": trend_decay_minutes,
            "insulin_action_minutes": insulin_action_minutes,
            "carb_absorption_minutes": carb_absorption_minutes,
        }
        for name, value in numeric_positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.horizon_steps = int(horizon_steps)
        self.time_step_minutes = float(time_step_minutes)
        self.feature_columns = list(feature_columns or DEFAULT_FORECAST_FEATURE_COLUMNS)
        self.max_rate_mgdl_min = float(max_rate_mgdl_min)
        self.trend_decay_minutes = float(trend_decay_minutes)
        self.insulin_action_minutes = float(insulin_action_minutes)
        self.carb_absorption_minutes = float(carb_absorption_minutes)

    def name(self) -> str:
        return "PhysiologyAware"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("X must have shape [N, T, F]")
        if X.shape[1] == 0 or X.shape[2] == 0:
            raise ValueError("X must contain at least one time step and feature")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite feature values")
        predictions = np.empty((X.shape[0], self.horizon_steps), dtype=np.float32)
        for row_index in range(X.shape[0]):
            predictions[row_index] = self._predict_one(X[row_index])
        return predictions

    def _read_last(self, window: np.ndarray, candidates: Sequence[str], default: float) -> float:
        index = _feature_index(self.feature_columns, candidates)
        if index is None or index >= window.shape[1]:
            return default
        value = float(window[-1, index])
        if not np.isfinite(value):
            raise ValueError(
                f"Forecast feature {self.feature_columns[index]!r} must be finite"
            )
        return value

    def _predict_one(self, window: np.ndarray) -> np.ndarray:
        glucose_index = _feature_index(
            self.feature_columns,
            ("glucose_actual_mgdl", "glucose_to_algo_mgdl", "glucose"),
        )
        if glucose_index is None or glucose_index >= window.shape[1]:
            raise ValueError("A glucose feature column is required for physiology-aware forecasting.")

        glucose_history = window[:, glucose_index].astype(float)
        current_glucose = _bounded_feature(
            float(glucose_history[-1]),
            name="current_glucose",
            low=0.0,
            high=1_000.0,
        )
        trend = self._read_last(window, ("glucose_trend_mgdl_min", "glucose_rate_mgdl_min"), float("nan"))
        if not np.isfinite(trend):
            if len(glucose_history) >= 2:
                trend = (glucose_history[-1] - glucose_history[-2]) / max(self.time_step_minutes, 1e-6)
            else:
                trend = 0.0
        trend = _safe_clip(trend, -self.max_rate_mgdl_min, self.max_rate_mgdl_min)

        iob = _bounded_feature(
            self._read_last(window, ("patient_iob_units", "iob", "insulin_on_board"), 0.0),
            name="IOB",
            low=0.0,
            high=100.0,
        )
        cob = _bounded_feature(
            self._read_last(window, ("patient_cob_grams", "cob", "carbs_on_board"), 0.0),
            name="COB",
            low=0.0,
            high=1_000.0,
        )
        isf = _bounded_feature(
            self._read_last(window, ("effective_isf", "isf"), 50.0),
            name="ISF",
            low=5.0,
            high=250.0,
        )
        icr = _bounded_feature(
            self._read_last(window, ("effective_icr", "icr"), 10.0),
            name="ICR",
            low=1.0,
            high=80.0,
        )
        dia = _bounded_feature(
            self._read_last(window, ("effective_dia_minutes", "dia_minutes"), self.insulin_action_minutes),
            name="DIA",
            low=30.0,
            high=480.0,
        )
        heart_rate = _bounded_feature(
            self._read_last(window, ("heart_rate",), 0.0),
            name="heart_rate",
            low=0.0,
            high=300.0,
        )
        steps = _bounded_feature(
            self._read_last(window, ("steps",), 0.0),
            name="steps",
            low=0.0,
            high=1_000_000.0,
        )
        exercise_intensity = _bounded_feature(
            self._read_last(window, ("exercise_intensity",), 0.0),
            name="exercise_intensity",
            low=0.0,
            high=1.0,
        )
        stress_intensity = _bounded_feature(
            self._read_last(window, ("stress_intensity", "illness_intensity"), 0.0),
            name="stress_intensity",
            low=0.0,
            high=1.0,
        )
        antibody_binding = _bounded_feature(
            self._read_last(window, ("insulin_antibody_binding_fraction",), 0.0),
            name="insulin_antibody_binding_fraction",
            low=0.0,
            high=0.95,
        )
        antibody_release = _bounded_feature(
            self._read_last(window, ("insulin_antibody_release_fraction",), 0.0),
            name="insulin_antibody_release_fraction",
            low=0.0,
            high=0.75,
        )
        antibody_bound_pool = _bounded_feature(
            self._read_last(window, ("antibody_bound_insulin_units",), 0.0),
            name="antibody_bound_insulin_units",
            low=0.0,
            high=100.0,
        )

        exercise_signal = max(0.0, (heart_rate - 105.0) / 75.0) + max(0.0, steps / 2500.0)
        exercise_signal = _safe_clip(max(exercise_signal, exercise_intensity), 0.0, 1.5)
        stress_signal = _safe_clip(stress_intensity, 0.0, 1.0)

        glucose = current_glucose
        remaining_iob = iob * (1.0 - antibody_binding)
        antibody_bound_pool += iob * antibody_binding
        remaining_cob = cob
        values = []
        for step_index in range(self.horizon_steps):
            future_minutes = (step_index + 1) * self.time_step_minutes
            trend_decay = float(np.exp(-future_minutes / max(self.trend_decay_minutes, 1.0)))
            trend_delta = trend * self.time_step_minutes * trend_decay

            insulin_step_fraction = min(self.time_step_minutes / max(dia, 1.0), 1.0)
            release_fraction = antibody_release
            if antibody_binding > 0.0 and release_fraction <= 0.0:
                release_fraction = min(self.time_step_minutes / 360.0, 0.08)
            released_insulin = antibody_bound_pool * release_fraction
            antibody_bound_pool = max(0.0, antibody_bound_pool - released_insulin)
            exercise_multiplier = 1.0 + 0.35 * exercise_signal
            insulin_delta = -(remaining_iob + released_insulin) * isf * insulin_step_fraction * exercise_multiplier
            remaining_iob = max(0.0, remaining_iob * (1.0 - insulin_step_fraction))

            carb_step_fraction = min(
                self.time_step_minutes / max(self.carb_absorption_minutes, 1.0),
                1.0,
            )
            carb_delta = remaining_cob * (isf / icr) * carb_step_fraction
            remaining_cob = max(0.0, remaining_cob * (1.0 - carb_step_fraction))

            exercise_delta = -exercise_signal * self.time_step_minutes * 0.35
            stress_delta = stress_signal * self.time_step_minutes * 0.45
            raw_delta = trend_delta + insulin_delta + carb_delta + exercise_delta + stress_delta
            max_step_delta = self.max_rate_mgdl_min * self.time_step_minutes
            # The rate envelope is part of this comparator's declared model.
            # Absolute glucose is deliberately not clipped: evaluation must be
            # able to observe and count impossible forecasts.
            glucose = glucose + _safe_clip(
                raw_delta, -max_step_delta, max_step_delta
            )
            values.append(glucose)
        return np.asarray(values, dtype=np.float32)


def assess_forecast_risk(
    predicted_glucose: Sequence[float] | np.ndarray,
    *,
    current_glucose: Optional[float] = None,
    predicted_std: Optional[Sequence[float] | np.ndarray | float] = None,
    config: ForecastConfig = ForecastConfig(),
) -> Dict[str, Any]:
    """Classify a forecast into research guardrail levels."""
    predictions = np.asarray(predicted_glucose, dtype=float).reshape(-1)
    if len(predictions) == 0:
        raise ValueError("predicted_glucose must not be empty")

    finite_predictions = predictions[np.isfinite(predictions)]
    if len(finite_predictions) == 0:
        return {
            "risk_level": "invalid",
            "guardrail_action": "fallback_required",
            "reason": "forecast contains no finite values",
            "min_predicted_mgdl": None,
            "max_predicted_mgdl": None,
            "end_predicted_mgdl": None,
        }

    min_pred = float(np.min(finite_predictions))
    max_pred = float(np.max(finite_predictions))
    end_pred = float(finite_predictions[-1])
    reasons: list[str] = []
    risk_level = "in_range"
    guardrail_action = "monitor"

    if min_pred < config.urgent_low_threshold_mgdl:
        risk_level = "critical_low"
        guardrail_action = "block_or_reduce_insulin"
        reasons.append(f"predicted glucose below {config.urgent_low_threshold_mgdl:g} mg/dL")
    elif min_pred < config.low_threshold_mgdl:
        risk_level = "hypo_risk"
        guardrail_action = "block_extra_insulin"
        reasons.append(f"predicted glucose below {config.low_threshold_mgdl:g} mg/dL")
    elif max_pred > config.critical_high_threshold_mgdl:
        risk_level = "critical_high"
        guardrail_action = "clinical_review_required"
        reasons.append(f"predicted glucose above {config.critical_high_threshold_mgdl:g} mg/dL")
    elif max_pred > config.very_high_threshold_mgdl:
        risk_level = "severe_hyper_risk"
        guardrail_action = "review_correction_context"
        reasons.append(f"predicted glucose above {config.very_high_threshold_mgdl:g} mg/dL")
    elif max_pred > config.high_threshold_mgdl:
        risk_level = "hyper_risk"
        guardrail_action = "monitor_and_explain"
        reasons.append(f"predicted glucose above {config.high_threshold_mgdl:g} mg/dL")

    if current_glucose is not None and np.isfinite(current_glucose):
        slope = (end_pred - float(current_glucose)) / max(config.horizon_minutes, 1)
        if slope <= -2.0:
            reasons.append("forecast is falling fast")
            if risk_level == "in_range":
                risk_level = "falling_fast"
                guardrail_action = "monitor_or_reduce_aggression"
        elif slope >= 2.0:
            reasons.append("forecast is rising fast")
            if risk_level == "in_range":
                risk_level = "rising_fast"
                guardrail_action = "monitor_and_explain"

    std_value: Optional[float] = None
    if predicted_std is not None:
        std_arr = np.asarray(predicted_std, dtype=float).reshape(-1)
        if len(std_arr) > 0:
            finite_std = std_arr[np.isfinite(std_arr)]
            if len(finite_std) > 0:
                std_value = float(np.max(finite_std))
                if std_value > config.uncertainty_std_threshold_mgdl:
                    reasons.append(
                        f"uncertainty above {config.uncertainty_std_threshold_mgdl:g} mg/dL"
                    )
                    if risk_level == "in_range":
                        risk_level = "uncertain"
                    guardrail_action = "fallback_or_human_review"

    return {
        "risk_level": risk_level,
        "guardrail_action": guardrail_action,
        "reason": "; ".join(reasons) if reasons else "forecast inside configured research bounds",
        "min_predicted_mgdl": min_pred,
        "max_predicted_mgdl": max_pred,
        "end_predicted_mgdl": end_pred,
        "max_predicted_std_mgdl": std_value,
    }


def _coerce_forecast_input(df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    frame = df.copy()
    glucose_column = _first_existing(frame, ("glucose_actual_mgdl", "glucose_to_algo_mgdl", "glucose"))
    if glucose_column is None:
        raise ValueError("Input data must include glucose_actual_mgdl, glucose_to_algo_mgdl, or glucose")
    glucose_source = (
        frame[glucose_column]
        if "glucose_actual_mgdl" not in frame.columns
        else frame["glucose_actual_mgdl"]
    )
    glucose = pd.to_numeric(glucose_source, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    malformed = glucose.isna() & glucose_source.notna()
    if malformed.any():
        raise ValueError(
            f"Glucose input contains {int(malformed.sum())} malformed values"
        )
    glucose = glucose.interpolate(method="linear", limit=1, limit_area="inside")
    if glucose.isna().any():
        raise ValueError(
            "Glucose input contains unresolved missing values; boundary values "
            "are not replaced with a synthetic 120 mg/dL default"
        )
    frame["glucose_actual_mgdl"] = glucose.astype(float)

    if "time_minutes" not in frame.columns:
        frame["time_minutes"] = np.arange(len(frame)) * config.time_step_minutes
    else:
        time_values = pd.to_numeric(
            frame["time_minutes"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        if time_values.isna().any():
            raise ValueError("time_minutes must contain only finite numeric values")
        if (time_values.diff().dropna() <= 0.0).any():
            raise ValueError("time_minutes must be strictly increasing")
        frame["time_minutes"] = time_values.astype(float)
    if "glucose_trend_mgdl_min" not in frame.columns:
        frame["glucose_trend_mgdl_min"] = (
            frame["glucose_actual_mgdl"].diff().fillna(0.0) / max(config.time_step_minutes, 1)
        ).clip(-config.max_rate_mgdl_min, config.max_rate_mgdl_min)

    defaults = {
        "patient_iob_units": 0.0,
        "patient_cob_grams": 0.0,
        "effective_isf": 50.0,
        "effective_icr": 10.0,
        "effective_basal_rate_u_per_hr": 0.8,
        "effective_dia_minutes": config.insulin_action_minutes,
        "steps": 0.0,
        "heart_rate": 0.0,
        "exercise_intensity": 0.0,
        "stress_intensity": 0.0,
        "insulin_antibody_binding_fraction": 0.0,
        "insulin_antibody_release_fraction": 0.0,
        "antibody_bound_insulin_units": 0.0,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
            continue
        source = frame[column]
        parsed = pd.to_numeric(source, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        malformed = parsed.isna() & source.notna()
        if malformed.any():
            raise ValueError(
                f"Forecast feature {column!r} contains malformed values"
            )
        frame[column] = parsed.fillna(default)
    return frame


def _prediction_endpoint(output: Any, horizon_steps: int) -> tuple[float, np.ndarray]:
    arr = np.asarray(output, dtype=float)
    if arr.ndim == 2:
        curve = arr[0]
    elif arr.ndim == 1:
        curve = arr
    else:
        curve = np.asarray([float("nan")])
    if len(curve) == 0:
        return float("nan"), curve
    index = min(max(horizon_steps - 1, 0), len(curve) - 1)
    return float(curve[index]), curve.astype(float)


def attach_forecasts_to_frame(
    df: pd.DataFrame,
    *,
    predictor_service: Optional[Any] = None,
    config: ForecastConfig = ForecastConfig(),
    feature_columns: Optional[Sequence[str]] = None,
    feature_overrides: Optional[Mapping[str, float]] = None,
    mc_samples: int = 30,
) -> pd.DataFrame:
    """Attach transparent and optional neural forecasts to a CGM/run dataframe."""
    frame = _coerce_forecast_input(df, config)
    service_feature_columns = getattr(predictor_service, "feature_columns", None)
    selected_features = list(feature_columns or service_feature_columns or DEFAULT_FORECAST_FEATURE_COLUMNS)
    for column in selected_features:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if feature_overrides:
        for column, value in feature_overrides.items():
            if column not in frame.columns:
                frame[column] = float(value)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(float(value))
                frame[column] = float(value)
            if column not in selected_features:
                selected_features.append(column)

    service_history_steps = getattr(predictor_service, "history_steps", None)
    history_steps = int(service_history_steps) if isinstance(service_history_steps, int) else config.history_steps
    horizon_steps = config.horizon_steps
    horizon_label = f"{config.horizon_minutes}min"

    physiology = PhysiologyAwareBaseline(
        horizon_steps,
        time_step_minutes=config.time_step_minutes,
        feature_columns=selected_features,
        max_rate_mgdl_min=config.max_rate_mgdl_min,
        trend_decay_minutes=config.trend_decay_minutes,
        insulin_action_minutes=config.insulin_action_minutes,
        carb_absorption_minutes=config.carb_absorption_minutes,
    )

    physiology_col = f"predicted_glucose_physiology_{horizon_label}"
    last_value_col = f"predicted_glucose_last_value_{horizon_label}"
    ai_col = f"predicted_glucose_ai_{horizon_label}"
    std_col = f"predicted_glucose_ai_std_{horizon_label}"
    observed_col = f"observed_glucose_{horizon_label}"

    for column in (
        physiology_col,
        last_value_col,
        ai_col,
        std_col,
        observed_col,
        "forecast_risk_level",
        "forecast_guardrail_action",
        "forecast_reason",
        "forecast_source",
    ):
        frame[column] = np.nan if column not in {"forecast_risk_level", "forecast_guardrail_action", "forecast_reason", "forecast_source"} else ""

    feature_matrix = frame[selected_features].to_numpy(dtype=np.float32)
    glucose_values = frame["glucose_actual_mgdl"].to_numpy(dtype=float)
    for idx in range(history_steps - 1, len(frame)):
        window = feature_matrix[idx - history_steps + 1 : idx + 1][None, :, :]
        physiology_curve = physiology.predict(window)[0]
        physiology_endpoint = float(physiology_curve[-1])
        frame.at[idx, physiology_col] = physiology_endpoint
        frame.at[idx, last_value_col] = float(glucose_values[idx])

        predicted_curve = physiology_curve.astype(float)
        predicted_endpoint = physiology_endpoint
        predicted_std: Optional[np.ndarray] = None
        source = "physiology_baseline"

        if predictor_service is not None:
            predict_with_uncertainty = getattr(predictor_service, "predict_with_uncertainty", None)
            try:
                if callable(predict_with_uncertainty):
                    mean, std = predict_with_uncertainty(window, n_samples=mc_samples)
                    predicted_endpoint, predicted_curve = _prediction_endpoint(mean, horizon_steps)
                    _, std_curve = _prediction_endpoint(std, horizon_steps)
                    predicted_std = std_curve
                    frame.at[idx, std_col] = float(std_curve[min(horizon_steps - 1, len(std_curve) - 1)])
                    source = "neural_predictor_mc_dropout"
                else:
                    predict = getattr(predictor_service, "predict", None)
                    if callable(predict):
                        output = predict(window)
                        predicted_endpoint, predicted_curve = _prediction_endpoint(output, horizon_steps)
                        source = "neural_predictor"
            except Exception:
                source = "physiology_baseline_after_predictor_error"
                predicted_endpoint = physiology_endpoint
                predicted_curve = physiology_curve.astype(float)

        frame.at[idx, ai_col] = predicted_endpoint
        frame.at[idx, "forecast_source"] = source
        future_idx = idx + horizon_steps
        if future_idx < len(frame):
            frame.at[idx, observed_col] = float(glucose_values[future_idx])
        risk = assess_forecast_risk(
            predicted_curve,
            current_glucose=float(glucose_values[idx]),
            predicted_std=predicted_std,
            config=config,
        )
        frame.at[idx, "forecast_risk_level"] = risk["risk_level"]
        frame.at[idx, "forecast_guardrail_action"] = risk["guardrail_action"]
        frame.at[idx, "forecast_reason"] = risk["reason"]

    return frame


def summarize_forecast_frame(
    frame: pd.DataFrame,
    *,
    config: ForecastConfig = ForecastConfig(),
) -> Dict[str, Any]:
    horizon_label = f"{config.horizon_minutes}min"
    observed_col = f"observed_glucose_{horizon_label}"
    predictions = {
        "physiology": f"predicted_glucose_physiology_{horizon_label}",
        "last_value": f"predicted_glucose_last_value_{horizon_label}",
        "ai": f"predicted_glucose_ai_{horizon_label}",
    }
    report: Dict[str, Any] = {
        "horizon_minutes": config.horizon_minutes,
        "history_minutes": config.history_minutes,
        "rows": int(len(frame)),
        "risk_counts": frame["forecast_risk_level"].value_counts(dropna=False).to_dict()
        if "forecast_risk_level" in frame.columns
        else {},
        "guardrail_counts": frame["forecast_guardrail_action"].value_counts(dropna=False).to_dict()
        if "forecast_guardrail_action" in frame.columns
        else {},
        "models": {},
    }
    if observed_col not in frame.columns:
        return report

    observed = pd.to_numeric(frame[observed_col], errors="coerce")
    for label, column in predictions.items():
        if column not in frame.columns:
            continue
        predicted = pd.to_numeric(frame[column], errors="coerce")
        mask = observed.notna() & predicted.notna()
        if not bool(mask.any()):
            continue
        std = None
        std_col = f"predicted_glucose_{label}_std_{horizon_label}"
        if std_col in frame.columns:
            std_values = pd.to_numeric(frame[std_col], errors="coerce")
            if bool((mask & std_values.notna()).any()):
                std = std_values[mask].to_numpy(dtype=float)
        metrics = forecast_error_report(
            observed[mask].to_numpy(dtype=float),
            predicted[mask].to_numpy(dtype=float),
            std,
        )
        metrics["hypoglycemia_detection"] = hypoglycemia_detection_report(
            observed[mask].to_numpy(dtype=float),
            predicted[mask].to_numpy(dtype=float),
        )
        if std is not None:
            metrics["uncertainty_reliability"] = uncertainty_reliability_report(
                observed[mask].to_numpy(dtype=float),
                predicted[mask].to_numpy(dtype=float),
                std,
                bins=5,
            )
        report["models"][label] = metrics
    return report


def _render_forecast_markdown(report: Mapping[str, Any], artifacts: Mapping[str, str]) -> str:
    lines = [
        "# Glucose Forecast Research Report",
        "",
        "IINTS glucose prediction evidence bundle. Research and education only; not for treatment decisions.",
        "",
        "## Scope",
        "",
        f"- Horizon: `{report.get('horizon_minutes')}` minutes",
        f"- History: `{report.get('history_minutes')}` minutes",
        f"- Rows: `{report.get('rows')}`",
        "",
        "## Risk Summary",
        "",
    ]
    risk_counts = report.get("risk_counts", {})
    if risk_counts:
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(risk_counts.items()))
    else:
        lines.append("- No risk rows available.")

    lines.extend(["", "## Forecast Metrics", ""])
    models = report.get("models", {})
    if not models:
        lines.append("- No observed future glucose values were available for scoring yet.")
    for label, metrics in models.items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- MAE: `{metrics['mae']:.2f}` mg/dL",
                f"- RMSE: `{metrics['rmse']:.2f}` mg/dL",
                f"- Bias: `{metrics['bias']:.2f}` mg/dL",
                f"- Within +/-20 mg/dL: `{metrics['within_20_mgdl_pct']:.2f}%`",
                f"- Missed hypo rate: `{metrics['missed_hypo_rate_pct']:.2f}%`",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Predictions CSV: `{artifacts['predictions_csv']}`",
            f"- Metrics JSON: `{artifacts['report_json']}`",
            "",
            "## Guardrails",
            "",
            "- A predictor must beat transparent baselines before it is useful for research claims.",
            "- Low-glucose misses are treated as safety-critical evidence, not just ordinary regression error.",
            "- High uncertainty or out-of-distribution inputs should trigger fallback/review logic.",
        ]
    )
    return "\n".join(lines)


def resolve_forecast_input(path: Path) -> Path:
    """Accept either a CSV file or an IINTS run directory."""
    if path.is_file():
        return path
    candidates = [
        path / "raw" / "steps.csv",
        path / "results.csv",
        path / "research" / "predictor_training.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No forecast input found in {path}. Expected a CSV, raw/steps.csv, results.csv, or research/predictor_training.csv."
    )


def write_forecast_bundle(
    input_path: Path,
    output_dir: Path,
    *,
    predictor_service: Optional[Any] = None,
    config: ForecastConfig = ForecastConfig(),
    feature_columns: Optional[Sequence[str]] = None,
    feature_overrides: Optional[Mapping[str, float]] = None,
    mc_samples: int = 30,
) -> Dict[str, Any]:
    """Build a forecast CSV, JSON metrics report, and Markdown explanation."""
    source_path = resolve_forecast_input(input_path)
    df = pd.read_csv(source_path)
    forecast_frame = attach_forecasts_to_frame(
        df,
        predictor_service=predictor_service,
        config=config,
        feature_columns=feature_columns,
        feature_overrides=feature_overrides,
        mc_samples=mc_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output_dir / "forecast_predictions.csv"
    report_json = output_dir / "forecast_report.json"
    report_md = output_dir / "forecast_report.md"
    manifest_json = output_dir / "forecast_manifest.json"

    forecast_frame.to_csv(predictions_csv, index=False)
    report = summarize_forecast_frame(forecast_frame, config=config)
    report["source_path"] = str(source_path)
    report["source_sha256"] = _sha256_file(source_path)
    report["prediction_sha256"] = _sha256_file(predictions_csv)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    artifacts = {
        "predictions_csv": str(predictions_csv),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "manifest_json": str(manifest_json),
    }
    report_md.write_text(_render_forecast_markdown(report, artifacts), encoding="utf-8")
    manifest = {
        "schema_version": "iints_forecast_bundle_v1",
        "source_path": str(source_path),
        "source_sha256": report["source_sha256"],
        "config": config.__dict__,
        "feature_overrides": dict(feature_overrides or {}),
        "artifacts": artifacts,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "report": report,
        "artifacts": artifacts,
    }
