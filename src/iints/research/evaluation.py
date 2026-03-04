from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _band_mask(observed: np.ndarray, band: str) -> np.ndarray:
    if band == "hypo":
        return observed < 70.0
    if band == "target":
        return (observed >= 70.0) & (observed <= 180.0)
    if band == "hyper":
        return observed > 180.0
    raise ValueError(f"Unknown glycemic band: {band}")


def forecast_error_report(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted must have the same shape")
    if len(obs) == 0:
        raise ValueError("empty inputs")

    error = pred - obs
    abs_error = np.abs(error)
    sq_error = error ** 2

    report: Dict[str, Any] = {
        "n": int(len(obs)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(sq_error))),
        "bias": float(np.mean(error)),
        "within_10_mgdl_pct": float(np.mean(abs_error <= 10.0) * 100.0),
        "within_20_mgdl_pct": float(np.mean(abs_error <= 20.0) * 100.0),
        "false_hypo_alarm_rate_pct": float(np.mean((pred < 70.0) & (obs >= 70.0)) * 100.0),
        "missed_hypo_rate_pct": float(np.mean((pred >= 70.0) & (obs < 70.0)) * 100.0),
    }

    band_metrics: Dict[str, Dict[str, float]] = {}
    for band in ("hypo", "target", "hyper"):
        mask = _band_mask(obs, band)
        if np.any(mask):
            band_metrics[band] = {
                "count": float(np.sum(mask)),
                "mae": float(np.mean(abs_error[mask])),
                "rmse": float(np.sqrt(np.mean(sq_error[mask]))),
                "bias": float(np.mean(error[mask])),
            }
        else:
            band_metrics[band] = {"count": 0.0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    report["band_metrics"] = band_metrics

    if predicted_std is not None:
        std = np.asarray(predicted_std, dtype=float).reshape(-1)
        if std.shape != obs.shape:
            raise ValueError("predicted_std must have same shape as observed")
        lower = pred - (1.96 * std)
        upper = pred + (1.96 * std)
        coverage = (obs >= lower) & (obs <= upper)
        report["interval_95_coverage_pct"] = float(np.mean(coverage) * 100.0)
        report["mean_predicted_std_mgdl"] = float(np.mean(std))

    return report

