from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from mdmp_core.contracts import DataContract
from mdmp_core.runner import dataframe_fingerprint


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(str(col))
    return columns


def _summary_for_columns(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "mean": _safe_float(series.mean()),
            "std": _safe_float(series.std(ddof=0)),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "p05": _safe_float(series.quantile(0.05)),
            "p95": _safe_float(series.quantile(0.95)),
            "missing_ratio": _safe_float(series.isna().mean()),
        }
    return out


def _bounds_violations(df: pd.DataFrame, contract: DataContract | None) -> Dict[str, Dict[str, int]]:
    if contract is None:
        return {}
    violations: Dict[str, Dict[str, int]] = {}
    for col in contract.schema.columns:
        if col.bounds is None or col.name not in df.columns:
            continue
        lo, hi = col.bounds
        values = pd.to_numeric(df[col.name], errors="coerce")
        violations[col.name] = {
            "below": int((values < lo).fillna(False).sum()),
            "above": int((values > hi).fillna(False).sum()),
        }
    return violations


def compare_datasets(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    contract: DataContract | None = None,
) -> Dict[str, Any]:
    baseline_fingerprint = f"sha256:{dataframe_fingerprint(baseline_df)}"
    candidate_fingerprint = f"sha256:{dataframe_fingerprint(candidate_df)}"

    baseline_cols = {str(col) for col in baseline_df.columns}
    candidate_cols = {str(col) for col in candidate_df.columns}
    shared_cols = sorted(baseline_cols & candidate_cols)

    numeric_shared = sorted(set(_numeric_columns(baseline_df)) & set(_numeric_columns(candidate_df)))
    baseline_stats = _summary_for_columns(baseline_df, numeric_shared)
    candidate_stats = _summary_for_columns(candidate_df, numeric_shared)

    stats_delta: Dict[str, Dict[str, float | None]] = {}
    for col in numeric_shared:
        before = baseline_stats.get(col, {})
        after = candidate_stats.get(col, {})
        stats_delta[col] = {
            "mean_delta": _safe_float((after.get("mean") or 0.0) - (before.get("mean") or 0.0)),
            "std_delta": _safe_float((after.get("std") or 0.0) - (before.get("std") or 0.0)),
            "missing_ratio_delta": _safe_float((after.get("missing_ratio") or 0.0) - (before.get("missing_ratio") or 0.0)),
        }

    baseline_bounds = _bounds_violations(baseline_df, contract)
    candidate_bounds = _bounds_violations(candidate_df, contract)
    bounds_delta: Dict[str, Dict[str, int]] = {}
    for col in sorted(set(baseline_bounds) | set(candidate_bounds)):
        before = baseline_bounds.get(col, {"below": 0, "above": 0})
        after = candidate_bounds.get(col, {"below": 0, "above": 0})
        bounds_delta[col] = {
            "below_delta": int(after["below"]) - int(before["below"]),
            "above_delta": int(after["above"]) - int(before["above"]),
        }

    return {
        "row_count": {
            "baseline": int(len(baseline_df)),
            "candidate": int(len(candidate_df)),
            "delta": int(len(candidate_df) - len(baseline_df)),
        },
        "columns": {
            "added": sorted(candidate_cols - baseline_cols),
            "removed": sorted(baseline_cols - candidate_cols),
            "shared": shared_cols,
        },
        "fingerprints": {
            "baseline": baseline_fingerprint,
            "candidate": candidate_fingerprint,
        },
        "has_changes": bool(
            len(baseline_df) != len(candidate_df)
            or baseline_cols != candidate_cols
            or baseline_fingerprint != candidate_fingerprint
        ),
        "numeric_summary": {
            "baseline": baseline_stats,
            "candidate": candidate_stats,
            "delta": stats_delta,
        },
        "bounds_violations": {
            "baseline": baseline_bounds,
            "candidate": candidate_bounds,
            "delta": bounds_delta,
        },
    }
