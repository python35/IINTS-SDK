from __future__ import annotations

from typing import Any, Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE/RMSE/Bias for regression arrays of equal shape."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }


def band_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0,
) -> Dict[str, Dict[str, float]]:
    """Compute MAE/RMSE/Bias separately in hypo/in-range/hyper observed bands."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    err = yp - yt

    def _stats(mask: np.ndarray) -> Dict[str, float]:
        count = int(mask.sum())
        if count == 0:
            return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
        e = err[mask]
        return {
            "count": count,
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "bias": float(np.mean(e)),
        }

    return {
        "hypo_lt_70": _stats(yt < low_threshold),
        "in_range_70_180": _stats((yt >= low_threshold) & (yt <= high_threshold)),
        "hyper_gt_180": _stats(yt > high_threshold),
    }


def interval_coverage_metrics(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    confidence: float = 0.95,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0,
) -> Dict[str, Any]:
    """
    Compute interval coverage + sharpness using normal approximation.

    confidence currently supports 0.95 only (z=1.96), which is the default
    interval commonly used for uncertainty reporting.
    """
    if y_true.shape != mean_pred.shape or y_true.shape != std_pred.shape:
        raise ValueError(
            "Shape mismatch: "
            f"y_true={y_true.shape}, mean_pred={mean_pred.shape}, std_pred={std_pred.shape}"
        )
    if confidence != 0.95:
        raise ValueError("Only confidence=0.95 is currently supported.")

    z = 1.96
    yt = y_true.reshape(-1)
    mean = mean_pred.reshape(-1)
    std = np.maximum(std_pred.reshape(-1), 1e-6)
    lower = mean - z * std
    upper = mean + z * std
    covered = (yt >= lower) & (yt <= upper)
    target = confidence

    def _coverage(mask: np.ndarray) -> Dict[str, float]:
        count = int(mask.sum())
        if count == 0:
            return {"count": 0, "coverage": float("nan"), "calibration_abs_error": float("nan")}
        cov = float(np.mean(covered[mask]))
        return {
            "count": count,
            "coverage": cov,
            "calibration_abs_error": abs(cov - target),
        }

    return {
        "confidence": confidence,
        "target_coverage": target,
        "overall_coverage": float(np.mean(covered)),
        "overall_calibration_abs_error": abs(float(np.mean(covered)) - target),
        "mean_interval_width": float(np.mean(upper - lower)),
        "bands": {
            "hypo_lt_70": _coverage(yt < low_threshold),
            "in_range_70_180": _coverage((yt >= low_threshold) & (yt <= high_threshold)),
            "hyper_gt_180": _coverage(yt > high_threshold),
        },
    }
