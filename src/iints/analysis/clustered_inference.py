#!/usr/bin/env python3
"""Uncertainty intervals that respect the study design.

The EUCYS study matrix is a fully crossed design: every algorithm is run on
every (virtual patient profile, scenario, seed) block. Runs are therefore not
independent draws - each profile contributes many runs, and runs that share a
profile share its physiology.

Treating all N runs as independent (SE = SD / sqrt(N)) understates the
uncertainty, sometimes by a large factor: it counts repeated measurements of
six virtual patients as if they were six hundred patients. That is
pseudo-replication, and it is the single most common statistical error in
simulation benchmarks.

This module provides three intervals for the same contrast so the difference
is explicit and auditable:

``naive_ci``
    The pseudo-replicated interval. Kept only for comparison; do not report it.

``cluster_t_ci``
    Recommended default. Reduces each cluster (profile) to one number, then
    applies a t interval with G-1 degrees of freedom. With few clusters this
    is the conservative and defensible choice.

``hierarchical_bootstrap_ci``
    Percentile bootstrap that resamples clusters first, then blocks within the
    resampled clusters. Makes no normality assumption, but with a small number
    of clusters it can be anti-conservative - report it alongside, not instead
    of, the cluster-t interval.

Because the design is crossed, algorithm contrasts should be computed as
within-block paired differences (:func:`paired_block_differences`) before any
interval is taken: pairing removes between-profile and between-scenario
variance, which is what makes the comparison sensitive despite only six
profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "Interval",
    "naive_ci",
    "cluster_t_ci",
    "hierarchical_bootstrap_ci",
    "paired_block_differences",
    "compare_algorithms",
    "DEFAULT_BLOCK_KEYS",
    "DEFAULT_CLUSTER_KEY",
    "MIN_CLUSTERS_FOR_INTERVAL",
]

#: Unit of independence in the study matrix: the virtual patient.
DEFAULT_CLUSTER_KEY = "profile_id"

#: Fewest clusters for which a reported interval carries information.
#:
#: With two clusters the cluster-t interval has one degree of freedom, so its
#: width is driven by the t multiplier (12.7 at 95%) rather than by the data.
#: Such an interval is not wrong, but printing it invites the reader to treat
#: the design as better powered than it is, so callers omit it and say why.
#: ``research/export_arm_comparison.py`` applies the same threshold to
#: subjects; keep the two in step.
MIN_CLUSTERS_FOR_INTERVAL = 3

#: A block is one fully crossed cell in which every algorithm is observed.
DEFAULT_BLOCK_KEYS = ("profile_id", "scenario_slug", "seed")


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval and its provenance."""

    estimate: float
    ci_low: float
    ci_high: float
    method: str
    n_observations: int
    n_clusters: Optional[int] = None

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["half_width"] = self.half_width
        out["excludes_zero"] = self.excludes_zero
        return out


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def naive_ci(values: Sequence[float], confidence: float = 0.95) -> Interval:
    """Interval that assumes every observation is independent.

    This reproduces the historical ``1.96 * SD / sqrt(n)`` behaviour so a
    report can show what the correction changed. It is not a valid interval
    for clustered data.
    """
    arr = _as_array(values)
    if arr.size < 2:
        return Interval(float(arr.mean()) if arr.size else float("nan"),
                        float("nan"), float("nan"), "naive", int(arr.size))
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    mean = float(arr.mean())
    half = z * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return Interval(mean, mean - half, mean + half, "naive", int(arr.size))


def cluster_t_ci(values: Sequence[float],
                 clusters: Sequence,
                 confidence: float = 0.95) -> Interval:
    """Cluster-level t interval: the recommended default.

    Each cluster is collapsed to its mean, so the effective sample size is the
    number of clusters, not the number of runs.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float),
                          "cluster": list(clusters)}).dropna()
    means = frame.groupby("cluster")["value"].mean().to_numpy()
    n_clusters = means.size
    mean = float(means.mean()) if n_clusters else float("nan")
    if n_clusters < 2:
        return Interval(mean, float("nan"), float("nan"),
                        "cluster_t", len(frame), n_clusters)
    t = stats.t.ppf(0.5 + confidence / 2.0, df=n_clusters - 1)
    half = t * float(means.std(ddof=1)) / np.sqrt(n_clusters)
    return Interval(mean, mean - half, mean + half,
                    "cluster_t", len(frame), n_clusters)


def hierarchical_bootstrap_ci(values: Sequence[float],
                              clusters: Sequence,
                              confidence: float = 0.95,
                              n_boot: int = 10_000,
                              seed: int = 0) -> Interval:
    """Percentile bootstrap resampling clusters, then observations within them."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float),
                          "cluster": list(clusters)}).dropna()
    groups = [g["value"].to_numpy() for _, g in frame.groupby("cluster")]
    n_clusters = len(groups)
    mean = float(frame["value"].mean()) if len(frame) else float("nan")
    if n_clusters < 2:
        return Interval(mean, float("nan"), float("nan"),
                        "hierarchical_bootstrap", len(frame), n_clusters)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, n_clusters, n_clusters)
        # Resample within each drawn cluster, then average cluster means so
        # every cluster carries equal weight regardless of its run count.
        draws[i] = np.mean([
            groups[c][rng.integers(0, groups[c].size, groups[c].size)].mean()
            for c in picked
        ])
    lo, hi = np.percentile(draws, [(1 - confidence) / 2 * 100,
                                   (1 + confidence) / 2 * 100])
    return Interval(mean, float(lo), float(hi),
                    "hierarchical_bootstrap", len(frame), n_clusters)


def paired_block_differences(df: pd.DataFrame,
                             value: str,
                             group_column: str,
                             treatment: str,
                             reference: str,
                             block_keys: Sequence[str] = DEFAULT_BLOCK_KEYS) -> pd.DataFrame:
    """Within-block ``treatment - reference`` differences.

    Blocks in which either arm is missing are dropped rather than imputed, so
    the contrast is always like-for-like.
    """
    keys = list(block_keys)
    subset = df[df[group_column].isin([treatment, reference])]
    wide = subset.pivot_table(index=keys, columns=group_column,
                              values=value, aggfunc="mean")
    if treatment not in wide.columns or reference not in wide.columns:
        return pd.DataFrame(columns=keys + ["difference"])
    wide = wide.dropna(subset=[treatment, reference])
    out = wide.reset_index()[keys].copy()
    out["difference"] = (wide[treatment] - wide[reference]).to_numpy()
    return out


def compare_algorithms(df: pd.DataFrame,
                       value: str = "tir_70_180",
                       group_column: str = "algorithm_id",
                       reference: str = "standard_pump",
                       treatments: Optional[Sequence[str]] = None,
                       block_keys: Sequence[str] = DEFAULT_BLOCK_KEYS,
                       cluster_key: str = DEFAULT_CLUSTER_KEY,
                       confidence: float = 0.95,
                       n_boot: int = 10_000,
                       seed: int = 0) -> pd.DataFrame:
    """Paired contrasts of every algorithm against a reference.

    Returns one row per contrast with the naive, cluster-t and hierarchical
    bootstrap intervals side by side.
    """
    if treatments is None:
        treatments = [a for a in sorted(df[group_column].unique()) if a != reference]

    rows: List[Dict[str, object]] = []
    for treatment in treatments:
        diffs = paired_block_differences(df, value, group_column,
                                         treatment, reference, block_keys)
        if diffs.empty:
            continue
        d = diffs["difference"].to_numpy().tolist()
        clusters = diffs[cluster_key].tolist()
        naive = naive_ci(d, confidence)
        clustered = cluster_t_ci(d, clusters, confidence)
        boot = hierarchical_bootstrap_ci(d, clusters, confidence, n_boot, seed)
        rows.append({
            "treatment": treatment,
            "reference": reference,
            "metric": value,
            "n_blocks": len(d),
            "n_clusters": clustered.n_clusters,
            "mean_difference": clustered.estimate,
            "naive_ci_low": naive.ci_low,
            "naive_ci_high": naive.ci_high,
            "naive_half_width": naive.half_width,
            "cluster_t_ci_low": clustered.ci_low,
            "cluster_t_ci_high": clustered.ci_high,
            "cluster_t_half_width": clustered.half_width,
            "bootstrap_ci_low": boot.ci_low,
            "bootstrap_ci_high": boot.ci_high,
            "significant_naive": naive.excludes_zero,
            "significant_clustered": clustered.excludes_zero,
            "inflation_factor": (clustered.half_width / naive.half_width
                                 if naive.half_width else float("nan")),
        })
    return pd.DataFrame(rows)
