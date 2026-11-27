"""D. Within-Patient Paired Statistical Analysis (Zero Pseudoreplication).

Fundamental statistical principle:
Thousands of 5-minute CGM samples from one virtual patient are NOT independent
observations. Every analysis here first aggregates per virtual patient ``i``,
computes within-patient paired differences against the pre-registered control
condition ``Delta_i = Metric_i(cond) - Metric_i(control)``, and then computes
patient-level bootstrap confidence intervals, Wilcoxon signed-rank tests, and
Holm-Bonferroni multiplicity-adjusted p-values across the virtual cohort.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from iints.research.designer import ResearchPlan

__all__ = [
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "analyze_research_results",
]


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni family-wise error rate adjustment."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: (np.isnan(x[1]), x[1]))
    adjusted = [float("nan")] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        if np.isnan(p_val):
            adjusted[orig_idx] = float("nan")
            continue
        scaled = min(1.0, float(p_val) * (m - rank))
        running_max = max(running_max, scaled)
        adjusted[orig_idx] = running_max
    return adjusted


def paired_bootstrap_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric patient-level bootstrap CI for the mean."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, clean.size, size=(n_boot, clean.size))
    means = np.mean(clean[idx], axis=1)
    alpha = 0.5 * (1.0 - confidence)
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _wilcoxon_p(deltas: np.ndarray) -> float:
    clean = deltas[np.isfinite(deltas)]
    nonzero = clean[np.abs(clean) > 1e-12]
    if nonzero.size < 5:
        return 1.0 if clean.size > 0 and np.all(np.abs(clean) <= 1e-12) else float("nan")
    try:
        res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def analyze_research_results(
    cell_metrics: pd.DataFrame,
    plan: ResearchPlan,
) -> Dict[str, Any]:
    """Compute patient-level condition summaries, paired deltas, generalization & failure audits."""
    proto = plan.statistical_protocol
    horizon = int(proto.primary_horizon_minutes)
    metric = proto.primary_metric
    control_id = plan.control_condition_id

    h_df = cell_metrics[cell_metrics["horizon_min"] == horizon].copy()
    if h_df.empty:
        raise ValueError(f"No rows found for primary_horizon_minutes={horizon}")

    # 1. Aggregate across in-distribution scenarios per (profile_id, split, simulator_model, condition_id, algorithm_id)
    in_dist = h_df[h_df["regime"] == "in_distribution"]
    if in_dist.empty:
        in_dist = h_df

    patient_level = (
        in_dist.groupby(
            ["profile_id", "split", "simulator_model", "condition_id", "dropout_minutes", "algorithm_id"],
            as_index=False,
        )[
            ["mae", "rmse", "dropout_mae", "dropout_recovery_mae", "clarke_ab_pct", "coverage_90_pct"]
        ]
        .mean()
    )

    # 2. Condition summary per (condition_id, algorithm_id) across virtual patients
    cond_rows: List[Dict[str, Any]] = []
    for (cond_id, drop_min, algo_id), grp in patient_level.groupby(
        ["condition_id", "dropout_minutes", "algorithm_id"]
    ):
        vals = grp[metric].to_numpy(dtype=float)
        ci_lo, ci_hi = paired_bootstrap_ci(
            vals, confidence=proto.confidence_level, n_boot=proto.bootstrap_resamples, seed=plan.master_seed
        )
        cond_rows.append(
            {
                "condition_id": str(cond_id),
                "dropout_minutes": int(drop_min),
                "algorithm_id": str(algo_id),
                "n_patients": int(len(grp)),
                "mean_mae": float(np.mean( grp["mae"])),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "median_mae": float(np.median(vals)),
                "iqr_mae": float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25)),
                "p95_patient_mae": float(np.quantile(vals, 0.95)),
                "mean_rmse": float(np.mean(grp["rmse"])),
                "mean_dropout_mae": float(np.nanmean(grp["dropout_mae"])) if grp["dropout_mae"].notna().any() else float("nan"),
                "mean_recovery_mae": (
                    float(np.nanmean(grp["dropout_recovery_mae"]))
                    if grp["dropout_recovery_mae"].notna().any()
                    else float("nan")
                ),
                "mean_clarke_ab_pct": float(np.mean(grp["clarke_ab_pct"])),
                "mean_coverage_90_pct": float(np.mean(grp["coverage_90_pct"])),
            }
        )
    condition_summary = pd.DataFrame(cond_rows).sort_values(["dropout_minutes", "mean_mae"]).reset_index(drop=True)

    # 3. Within-patient paired differences vs control condition
    ctrl = patient_level[patient_level["condition_id"] == control_id][
        ["profile_id", "algorithm_id", metric, "rmse"]
    ].rename(columns={metric: "ctrl_metric", "rmse": "ctrl_rmse"})

    paired_rows: List[Dict[str, Any]] = []
    merged = patient_level[patient_level["condition_id"] != control_id].merge(
        ctrl, on=["profile_id", "algorithm_id"], how="inner"
    )
    for (cond_id, drop_min, algo_id), grp in merged.groupby(["condition_id", "dropout_minutes", "algorithm_id"]):
        deltas = (grp[metric] - grp["ctrl_metric"]).to_numpy(dtype=float)
        ci_lo, ci_hi = paired_bootstrap_ci(
            deltas, confidence=proto.confidence_level, n_boot=proto.bootstrap_resamples, seed=plan.master_seed
        )
        p_raw = _wilcoxon_p(deltas)
        paired_rows.append(
            {
                "condition_id": str(cond_id),
                "dropout_minutes": int(drop_min),
                "algorithm_id": str(algo_id),
                "n_paired_patients": int(len(deltas)),
                "mean_delta_mae": float(np.mean(deltas)),
                "ci95_delta_low": ci_lo,
                "ci95_delta_high": ci_hi,
                "median_delta_mae": float(np.median(deltas)),
                "pct_patients_degraded": 100.0 * float(np.mean(deltas > 0.0)),
                "worst_patient_delta_mae": float(np.max(deltas)),
                "wilcoxon_p_raw": p_raw,
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    if not paired_summary.empty:
        paired_summary["p_holm_bonferroni"] = holm_bonferroni(paired_summary["wilcoxon_p_raw"].tolist())
        paired_summary = paired_summary.sort_values(["dropout_minutes", "mean_delta_mae"]).reset_index(drop=True)

    # 4. Generalization on strictly held-out UNSEEN profiles vs DEV profiles
    split_summary = (
        patient_level.groupby(["split", "condition_id", "dropout_minutes", "algorithm_id"], as_index=False)[
            ["mae", "rmse", "dropout_mae"]
        ]
        .mean()
        .sort_values(["split", "dropout_minutes", "mae"])
        .reset_index(drop=True)
    )

    # 5. Generalization on OOD Adversarial Boundary Scenarios
    ood_df = h_df[h_df["regime"] == "ood_stress"]
    if not ood_df.empty:
        ood_summary = (
            ood_df.groupby(["scenario_id", "condition_id", "dropout_minutes", "algorithm_id"], as_index=False)[
                ["mae", "rmse", "dropout_mae", "dropout_recovery_mae"]
            ]
            .mean()
            .sort_values(["scenario_id", "dropout_minutes", "mae"])
            .reset_index(drop=True)
        )
    else:
        ood_summary = pd.DataFrame()

    # 6. Cross-simulator rank concordance (if >1 simulator backend present)
    simulators = sorted(patient_level["simulator_model"].unique())
    cross_sim_summary = pd.DataFrame()
    if len(simulators) > 1:
        cross_sim_summary = (
            patient_level.groupby(["simulator_model", "condition_id", "algorithm_id"], as_index=False)[
                ["mae", "rmse", "dropout_mae"]
            ]
            .mean()
            .sort_values(["simulator_model", "condition_id", "mae"])
            .reset_index(drop=True)
        )

    # 7. Worst-case failure audit (top 10 worst individual cells)
    worst_cases = (
        h_df.sort_values("mae", ascending=False)
        .head(10)[
            [
                "profile_id",
                "split",
                "scenario_id",
                "regime",
                "condition_id",
                "algorithm_id",
                "mae",
                "rmse",
                "dropout_mae",
            ]
        ]
        .reset_index(drop=True)
    )

    return {
        "primary_horizon_minutes": horizon,
        "primary_metric": metric,
        "control_condition_id": control_id,
        "patient_level": patient_level,
        "condition_summary": condition_summary,
        "paired_vs_control": paired_summary,
        "split_generalization": split_summary,
        "ood_boundary_summary": ood_summary,
        "cross_simulator_summary": cross_sim_summary,
        "worst_cases": worst_cases,
    }
