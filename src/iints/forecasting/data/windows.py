"""Step 2b — Leakage-Resistant Contiguous CGM Window Builder with Embargo & Cross-Split Guard.

Sampling convention (``WindowSpec`` default):
- ``L = 12`` valid CGM samples at 5-minute intervals: ``[G_{t-55}, G_{t-50}, ..., G_t]``.
- Elapsed time between first input sample ``t-55`` and origin ``t`` is ``55.0 minutes``
  (representing 12 five-minute bins = 60 minutes of nominal sampling history).
- Target ``y = G_{t+30}`` lies strictly ``30.0 ± 2.5 minutes`` after ``t`` in the SAME split.

Anti-leakage guarantees:
1. NEVER interpolates across missing/invalid CGM rows; any invalid row between ``t-55``
   and ``t`` increases the valid-to-valid ``dt_min`` above ``5 + tol`` (e.g. ``>= 10 min``),
   rejecting any window that spans a dropped or out-of-bounds reading.
2. NEVER crosses ``train`` / ``test`` file boundaries, and raises ``DataLeakageError`` if
   a subject's ``test`` stream contains timestamps earlier than or equal to the end of
   their ``train`` stream.
3. Enforces a chronological **embargo purge zone** between ``train_fit`` and ``val``:
   ``min(val.window_start_ts) > max(train_fit.target_ts)`` holds strictly per subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from iints.forecasting.schemas import WindowSpec

__all__ = [
    "DataLeakageError",
    "SplitWindows",
    "PartitionedWindowDataset",
    "build_partitioned_windows",
]


class DataLeakageError(ValueError):
    """Raised when input streams violate strict chronological split isolation."""


@dataclass(frozen=True)
class SplitWindows:
    X: np.ndarray                     # shape (N, L) in mg/dL
    y: np.ndarray                     # shape (N,) in mg/dL
    subject_ids: np.ndarray           # shape (N,) str
    window_start_ts: np.ndarray       # shape (N,) datetime64[ns]
    origin_ts: np.ndarray             # shape (N,) datetime64[ns]
    target_ts: np.ndarray             # shape (N,) datetime64[ns]

    def __len__(self) -> int:
        return int(len(self.y))


@dataclass(frozen=True)
class PartitionedWindowDataset:
    spec: WindowSpec
    train_fit: SplitWindows
    val: SplitWindows
    test: SplitWindows
    embargo_purged_count: int

    def split_counts(self) -> Dict[str, int]:
        return {
            "train_fit": len(self.train_fit),
            "val": len(self.val),
            "test": len(self.test),
            "embargo_purged": int(self.embargo_purged_count),
        }


def _empty_split(input_len: int) -> SplitWindows:
    return SplitWindows(
        X=np.empty((0, input_len), dtype=float),
        y=np.empty((0,), dtype=float),
        subject_ids=np.empty((0,), dtype=object),
        window_start_ts=np.empty((0,), dtype="datetime64[ns]"),
        origin_ts=np.empty((0,), dtype="datetime64[ns]"),
        target_ts=np.empty((0,), dtype="datetime64[ns]"),
    )


def _extract_stream_windows(
    stream_df: pd.DataFrame,
    subject_id: str,
    spec: WindowSpec,
) -> SplitWindows:
    """Extract contiguous windows from ``stream_df``, rejecting any window containing an invalid row."""
    L = spec.input_length
    # Keep rows with valid timestamps in chronological order (dropping exact duplicate timestamps)
    ordered = (
        stream_df[stream_df["timestamp"].notna()]
        .loc[~stream_df["validity_flags"].astype(str).str.contains("duplicate_timestamp")]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(ordered) <= L:
        return _empty_split(L)

    ts = ordered["timestamp"].to_numpy(dtype="datetime64[ns]")
    g = ordered["glucose_mgdl"].to_numpy(dtype=float)
    row_valid = ordered["is_valid"].to_numpy(dtype=bool)
    t_min = (ts - ts[0]) / np.timedelta64(1, "m")

    min_step = spec.sample_interval_minutes - spec.step_tolerance_minutes
    max_step = spec.sample_interval_minutes + spec.step_tolerance_minutes
    expected_span = spec.elapsed_span_minutes
    span_tol = max(2.5, spec.step_tolerance_minutes * 2.5)

    step_diffs = np.diff(t_min)
    step_ok = (step_diffs >= min_step) & (step_diffs <= max_step)
    bad_steps_prefix = np.concatenate([[0], np.cumsum(~step_ok)])
    bad_rows_prefix = np.concatenate([[0], np.cumsum(~row_valid)])

    h_min = float(spec.prediction_horizon_minutes)
    t_tol = float(spec.target_tolerance_minutes)

    xs: List[np.ndarray] = []
    ys: List[float] = []
    w_starts: List[np.datetime64] = []
    origins: List[np.datetime64] = []
    targets: List[np.datetime64] = []

    n = len(g)
    for i in range(L - 1, n - 1):
        start_i = i - L + 1
        # Every single row in [start_i .. i] must be valid AND all L-1 intervals contiguous
        if bad_rows_prefix[i + 1] - bad_rows_prefix[start_i] > 0:
            continue
        if bad_steps_prefix[i] - bad_steps_prefix[start_i] > 0:
            continue
        span = t_min[i] - t_min[start_i]
        if abs(span - expected_span) > span_tol:
            continue

        desired_target_min = t_min[i] + h_min
        idx = int(np.searchsorted(t_min, desired_target_min))
        best_r = -1
        best_err = t_tol + 1.0
        for cand in range(max(i + 1, idx - 3), min(n, idx + 4)):
            if not row_valid[cand]:
                continue
            err = abs(t_min[cand] - desired_target_min)
            if err <= t_tol and err < best_err:
                best_err = err
                best_r = cand
        if best_r < 0:
            continue

        xs.append(g[start_i : i + 1].copy())
        ys.append(float(g[best_r]))
        w_starts.append(ts[start_i])
        origins.append(ts[i])
        targets.append(ts[best_r])

    if not ys:
        return _empty_split(L)

    return SplitWindows(
        X=np.vstack(xs),
        y=np.asarray(ys, dtype=float),
        subject_ids=np.full(len(ys), str(subject_id), dtype=object),
        window_start_ts=np.asarray(w_starts, dtype="datetime64[ns]"),
        origin_ts=np.asarray(origins, dtype="datetime64[ns]"),
        target_ts=np.asarray(targets, dtype="datetime64[ns]"),
    )


def _concat_splits(parts: List[SplitWindows], input_len: int) -> SplitWindows:
    non_empty = [p for p in parts if len(p) > 0]
    if not non_empty:
        return _empty_split(input_len)
    return SplitWindows(
        X=np.vstack([p.X for p in non_empty]),
        y=np.concatenate([p.y for p in non_empty]),
        subject_ids=np.concatenate([p.subject_ids for p in non_empty]),
        window_start_ts=np.concatenate([p.window_start_ts for p in non_empty]),
        origin_ts=np.concatenate([p.origin_ts for p in non_empty]),
        target_ts=np.concatenate([p.target_ts for p in non_empty]),
    )


def _slice_split(sw: SplitWindows, mask: np.ndarray) -> SplitWindows:
    return SplitWindows(
        X=sw.X[mask],
        y=sw.y[mask],
        subject_ids=sw.subject_ids[mask],
        window_start_ts=sw.window_start_ts[mask],
        origin_ts=sw.origin_ts[mask],
        target_ts=sw.target_ts[mask],
    )


def build_partitioned_windows(
    canonical: pd.DataFrame,
    spec: WindowSpec | None = None,
) -> PartitionedWindowDataset:
    """Build leakage-free `train_fit`, `val` (chronological + embargoed), and official `test` windows."""
    wspec = spec or WindowSpec()
    L = wspec.input_length

    # Cross-split chronological guard per subject: official test must strictly follow official train!
    valid_ts_df = canonical[canonical["timestamp"].notna()]
    for subj, s_grp in valid_ts_df.groupby("subject_id", sort=True):
        tr_sub = s_grp[s_grp["source_split"] == "train"]["timestamp"]
        te_sub = s_grp[s_grp["source_split"] == "test"]["timestamp"]
        if not tr_sub.empty and not te_sub.empty:
            max_tr = tr_sub.max()
            min_te = te_sub.min()
            if min_te <= max_tr:
                raise DataLeakageError(
                    f"Cross-split temporal overlap detected for subject '{subj}': "
                    f"earliest test timestamp ({min_te}) <= latest train timestamp ({max_tr})."
                )

    train_fit_parts: List[SplitWindows] = []
    val_parts: List[SplitWindows] = []
    test_parts: List[SplitWindows] = []
    purged_total = 0

    for (subj, split), grp in canonical.groupby(["subject_id", "source_split"], sort=True):
        sw = _extract_stream_windows(grp, str(subj), wspec)
        if len(sw) == 0:
            continue
        if split == "test":
            test_parts.append(sw)
        elif split == "train":
            n_w = len(sw)
            cut_idx = max(1, int(np.floor(n_w * (1.0 - wspec.validation_fraction))))
            if cut_idx >= n_w:
                train_fit_parts.append(sw)
                continue
            cutoff_target_ts = sw.target_ts[cut_idx - 1]
            is_train_fit = sw.target_ts <= cutoff_target_ts
            # Strict embargo: val window's EARLIEST input sample must be strictly AFTER max(train_fit.target_ts)!
            max_train_target = sw.target_ts[is_train_fit].max()
            is_val = sw.window_start_ts > max_train_target
            purged_total += int(np.sum(~is_train_fit & ~is_val))
            if np.any(is_train_fit):
                train_fit_parts.append(_slice_split(sw, is_train_fit))
            if np.any(is_val):
                val_parts.append(_slice_split(sw, is_val))

    return PartitionedWindowDataset(
        spec=wspec,
        train_fit=_concat_splits(train_fit_parts, L),
        val=_concat_splits(val_parts, L),
        test=_concat_splits(test_parts, L),
        embargo_purged_count=purged_total,
    )
