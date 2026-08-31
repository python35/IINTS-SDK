from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DualStreamDecomposition:
    """Decomposition of continuous glucose data into multiscale baseline and event streams."""

    raw_glucose: np.ndarray
    baseline_stream: np.ndarray
    event_stream: np.ndarray
    sampling_interval_minutes: float
    filter_window_minutes: float

    @property
    def length(self) -> int:
        return len(self.raw_glucose)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "glucose_raw": self.raw_glucose,
            "glucose_baseline": self.baseline_stream,
            "glucose_event": self.event_stream,
        })


def decompose_dual_stream(
    glucose_values: Sequence[float] | np.ndarray,
    sampling_interval_minutes: float = 5.0,
    filter_window_minutes: float = 120.0,
    filter_type: str = "gaussian",
) -> DualStreamDecomposition:
    """
    Decompose a CGM time-series into a slow circadian/basal baseline stream
    and a fast acute event stream (GlucoFM dual-stream architecture).

    Parameters:
    - glucose_values: 1D array of continuous glucose measurements (mg/dL)
    - sampling_interval_minutes: sampling cadence (e.g. 5.0 min for Dexcom, 1.0 min for interpolated)
    - filter_window_minutes: temporal window for baseline extraction (default 120 min)
    - filter_type: 'gaussian', 'rolling_mean', or 'exponential'
    """
    raw = np.asarray(glucose_values, dtype=float).copy()
    if len(raw) == 0:
        return DualStreamDecomposition(
            raw_glucose=np.array([]),
            baseline_stream=np.array([]),
            event_stream=np.array([]),
            sampling_interval_minutes=sampling_interval_minutes,
            filter_window_minutes=filter_window_minutes,
        )

    # Replace NaNs with linear interpolation for filtering
    valid_mask = np.isfinite(raw)
    if not np.any(valid_mask):
        return DualStreamDecomposition(
            raw_glucose=raw,
            baseline_stream=np.full_like(raw, np.nan),
            event_stream=np.full_like(raw, np.nan),
            sampling_interval_minutes=sampling_interval_minutes,
            filter_window_minutes=filter_window_minutes,
        )

    indices = np.arange(len(raw))
    clean_signal = np.interp(indices, indices[valid_mask], raw[valid_mask])

    window_steps = max(3, int(round(filter_window_minutes / sampling_interval_minutes)))
    if window_steps % 2 == 0:
        window_steps += 1

    if filter_type == "gaussian":
        sigma = window_steps / 4.0
        x = np.arange(-window_steps // 2 + 1, window_steps // 2 + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(clean_signal, (len(kernel) // 2, len(kernel) // 2), mode="edge")
        baseline = np.convolve(padded, kernel, mode="valid")[:len(raw)]
    elif filter_type == "exponential":
        alpha = 2.0 / (window_steps + 1.0)
        s = pd.Series(clean_signal)
        baseline = s.ewm(alpha=alpha, adjust=False).mean().values
    else:  # rolling_mean
        s = pd.Series(clean_signal)
        baseline = s.rolling(window=window_steps, min_periods=1, center=True).mean().values

    # Restore NaNs where original was NaN
    baseline[~valid_mask] = np.nan
    event = raw - baseline

    return DualStreamDecomposition(
        raw_glucose=raw,
        baseline_stream=baseline,
        event_stream=event,
        sampling_interval_minutes=sampling_interval_minutes,
        filter_window_minutes=filter_window_minutes,
    )


def extract_dual_stream_pre_meal_features(
    glucose_pre_meal: Sequence[float] | np.ndarray,
    sampling_interval_minutes: float = 5.0,
    filter_window_minutes: float = 120.0,
    time_of_day_minutes: float = 0.0,
) -> dict[str, float]:
    """
    Extract multiscale GlucoFM features from 60 minutes of pre-meal CGM context.
    """
    decomp = decompose_dual_stream(
        glucose_pre_meal,
        sampling_interval_minutes=sampling_interval_minutes,
        filter_window_minutes=filter_window_minutes,
    )

    base = decomp.baseline_stream[np.isfinite(decomp.baseline_stream)]
    event = decomp.event_stream[np.isfinite(decomp.event_stream)]

    if len(base) == 0:
        return {
            "baseline_last": 120.0,
            "baseline_slope_60min": 0.0,
            "event_last": 0.0,
            "event_auc_60min": 0.0,
            "event_velocity": 0.0,
            "tod_sin": np.sin(2 * np.pi * time_of_day_minutes / 1440.0),
            "tod_cos": np.cos(2 * np.pi * time_of_day_minutes / 1440.0),
        }

    base_last = float(base[-1])
    base_slope = float((base[-1] - base[0]) / max(1, len(base) - 1)) if len(base) > 1 else 0.0

    event_last = float(event[-1]) if len(event) > 0 else 0.0
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    event_auc = float(trapz_fn(np.abs(event))) if trapz_fn and len(event) > 1 else float(np.sum(np.abs(event)))
    event_velocity = float((event[-1] - event[0]) / max(1, len(event) - 1)) if len(event) > 1 else 0.0

    tod_sin = float(np.sin(2 * np.pi * time_of_day_minutes / 1440.0))
    tod_cos = float(np.cos(2 * np.pi * time_of_day_minutes / 1440.0))

    return {
        "baseline_last": base_last,
        "baseline_slope_60min": base_slope,
        "event_last": event_last,
        "event_auc_60min": event_auc,
        "event_velocity": event_velocity,
        "tod_sin": tod_sin,
        "tod_cos": tod_cos,
    }


__all__ = [
    "DualStreamDecomposition",
    "decompose_dual_stream",
    "extract_dual_stream_pre_meal_features",
]
