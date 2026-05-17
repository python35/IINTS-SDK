from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

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


def hypoglycemia_detection_report(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold_mgdl: float = 70.0,
) -> Dict[str, Any]:
    """Report binary hypo-detection performance at a clinically explicit threshold."""
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    if obs.shape != pred.shape:
        raise ValueError("observed and predicted must have the same shape")
    if len(obs) == 0:
        raise ValueError("empty inputs")

    observed_hypo = obs < threshold_mgdl
    predicted_hypo = pred < threshold_mgdl
    tp = int(np.sum(observed_hypo & predicted_hypo))
    fn = int(np.sum(observed_hypo & ~predicted_hypo))
    fp = int(np.sum(~observed_hypo & predicted_hypo))
    tn = int(np.sum(~observed_hypo & ~predicted_hypo))

    def _ratio(numerator: int, denominator: int) -> Optional[float]:
        return None if denominator == 0 else float(numerator / denominator)

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    false_positive_rate = _ratio(fp, fp + tn)
    false_negative_rate = _ratio(fn, fn + tp)
    return {
        "threshold_mgdl": float(threshold_mgdl),
        "counts": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
        },
        "sensitivity": sensitivity,
        "sensitivity_pct": None if sensitivity is None else float(sensitivity * 100.0),
        "specificity": specificity,
        "specificity_pct": None if specificity is None else float(specificity * 100.0),
        "precision": precision,
        "precision_pct": None if precision is None else float(precision * 100.0),
        "negative_predictive_value": npv,
        "negative_predictive_value_pct": None if npv is None else float(npv * 100.0),
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_pct": (
            None if false_positive_rate is None else float(false_positive_rate * 100.0)
        ),
        "missed_hypo_rate": false_negative_rate,
        "missed_hypo_rate_pct": (
            None if false_negative_rate is None else float(false_negative_rate * 100.0)
        ),
    }


def uncertainty_reliability_report(
    observed: np.ndarray,
    predicted: np.ndarray,
    predicted_std: np.ndarray,
    *,
    bins: int = 5,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Bin predictions by uncertainty and compare nominal vs empirical coverage."""
    if confidence != 0.95:
        raise ValueError("Only confidence=0.95 is currently supported.")
    if bins <= 0:
        raise ValueError("bins must be > 0")

    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    std = np.asarray(predicted_std, dtype=float).reshape(-1)
    if obs.shape != pred.shape or obs.shape != std.shape:
        raise ValueError("observed, predicted, and predicted_std must have the same shape")
    if len(obs) == 0:
        raise ValueError("empty inputs")

    std = np.maximum(std, 1e-6)
    abs_error = np.abs(pred - obs)
    lower = pred - (1.96 * std)
    upper = pred + (1.96 * std)
    covered = (obs >= lower) & (obs <= upper)
    quantiles = np.quantile(std, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(quantiles)
    if len(edges) == 1:
        edges = np.array([edges[0], edges[0]])

    bucket_rows: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (std >= low) & (std <= high)
        else:
            mask = (std >= low) & (std < high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        coverage = float(np.mean(covered[mask]))
        bucket_rows.append(
            {
                "bin": index + 1,
                "count": count,
                "std_min_mgdl": low,
                "std_max_mgdl": high,
                "mean_predicted_std_mgdl": float(np.mean(std[mask])),
                "mean_absolute_error_mgdl": float(np.mean(abs_error[mask])),
                "interval_95_coverage_pct": float(coverage * 100.0),
                "calibration_abs_error_pct": float(abs(coverage - confidence) * 100.0),
            }
        )
    weighted_error = sum(row["calibration_abs_error_pct"] * row["count"] for row in bucket_rows)
    weighted_error /= max(1, len(obs))
    return {
        "confidence": confidence,
        "target_coverage_pct": float(confidence * 100.0),
        "overall_coverage_pct": float(np.mean(covered) * 100.0),
        "weighted_calibration_abs_error_pct": float(weighted_error),
        "bins": bucket_rows,
    }


def subgroup_error_report(
    observed: np.ndarray,
    predicted: np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    predicted_std: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate forecast quality separately for each provided subgroup label."""
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    group_values = np.asarray(groups).reshape(-1)
    if obs.shape != pred.shape or obs.shape != group_values.shape:
        raise ValueError("observed, predicted, and groups must have the same flattened shape")
    std = None if predicted_std is None else np.asarray(predicted_std, dtype=float).reshape(-1)
    if std is not None and std.shape != obs.shape:
        raise ValueError("predicted_std must have the same flattened shape as observed")

    report: Dict[str, Dict[str, Any]] = {}
    for group in sorted({str(value) for value in group_values}):
        mask = np.asarray([str(value) == group for value in group_values], dtype=bool)
        report[group] = forecast_error_report(
            obs[mask],
            pred[mask],
            None if std is None else std[mask],
        )
        report[group]["hypoglycemia_detection"] = hypoglycemia_detection_report(
            obs[mask],
            pred[mask],
        )
    return report


def feature_drift_report(
    reference_features: np.ndarray,
    candidate_features: np.ndarray,
    *,
    feature_names: Iterable[str],
) -> Dict[str, Any]:
    """Compare raw feature distributions using robust location and scale deltas."""
    reference = np.asarray(reference_features, dtype=float)
    candidate = np.asarray(candidate_features, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference_features and candidate_features must be 2D")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("reference_features and candidate_features must have equal feature counts")
    names = list(feature_names)
    if len(names) != reference.shape[1]:
        raise ValueError("feature_names length must match feature count")

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        ref = reference[:, index]
        cand = candidate[:, index]
        ref_clean = ref[np.isfinite(ref)]
        cand_clean = cand[np.isfinite(cand)]
        if len(ref_clean) == 0 or len(cand_clean) == 0:
            rows.append(
                {
                    "feature": name,
                    "status": "insufficient_data",
                    "robust_shift_score": None,
                    "reference_missing_pct": float(np.mean(~np.isfinite(ref)) * 100.0),
                    "candidate_missing_pct": float(np.mean(~np.isfinite(cand)) * 100.0),
                }
            )
            continue
        ref_median = float(np.median(ref_clean))
        cand_median = float(np.median(cand_clean))
        ref_iqr = float(np.percentile(ref_clean, 75) - np.percentile(ref_clean, 25))
        robust_scale = max(ref_iqr, 1e-6)
        shift_score = abs(cand_median - ref_median) / robust_scale
        rows.append(
            {
                "feature": name,
                "status": "ok",
                "reference_median": ref_median,
                "candidate_median": cand_median,
                "reference_iqr": ref_iqr,
                "robust_shift_score": float(shift_score),
                "reference_missing_pct": float(np.mean(~np.isfinite(ref)) * 100.0),
                "candidate_missing_pct": float(np.mean(~np.isfinite(cand)) * 100.0),
            }
        )
    ok_scores = [row["robust_shift_score"] for row in rows if row["robust_shift_score"] is not None]
    return {
        "feature_count": len(rows),
        "max_robust_shift_score": None if not ok_scores else float(max(ok_scores)),
        "mean_robust_shift_score": None if not ok_scores else float(np.mean(ok_scores)),
        "features": rows,
    }
