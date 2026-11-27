"""Step 4 — Per-Subject, Glycemic-Range & Small-N Rigorous Forecasting Evaluation.

Evaluates models on identical test windows, reporting:
- Pooled MAE, RMSE, Median Absolute Error (`medae`), and Clarke A+B %,
- Per-subject error distributions (`per_subject`: MAE, RMSE, MedAE, P95, Clarke A+B),
- Glycemic range breakdown (`hypo < 70`, `target 70-180`, `hyper > 180 mg/dL`),
- Input rate-of-change breakdown (`falling`, `steady`, `rising`),
- Rigorous small-N subject-level paired comparison (`compare_models_by_subject`):
  exact two-sided Wilcoxon signed-rank test (`method='exact'` for 5 <= N <= 25),
  subject-level paired bootstrap 95% CI, IQR, minimum attainable exact p-value
  `2^(1-N)` qualification, and anti-pseudoreplication enforcement (multi-seed runs
  must average per subject before cross-subject testing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from iints.analysis.error_grid import clarke_zones
from iints.analysis.prediction_accuracy import glycemic_range
from iints.forecasting.data.ohio_adapter import (
    load_carelink_benchmark_dataset,
    load_ohiot1dm_dataset,
    load_virtual_benchmark_dataset,
)
from iints.forecasting.data.windows import SplitWindows, build_partitioned_windows
from iints.forecasting.models.base import WindowForecaster
from iints.forecasting.models.persistence import PersistencePredictor
from iints.forecasting.models.ridge import RidgeCGMPredictor
from iints.forecasting.schemas import (
    FORECASTING_DISCLAIMER,
    DataQualityReport,
    ForecastingExperimentConfig,
    WindowSpec,
)
from iints.research.statistics import paired_bootstrap_ci
from iints.sdk2.reproducibility import capture_provenance, sha256_json

__all__ = [
    "evaluate_predictions",
    "compare_models_by_subject",
    "run_phase2_baseline_benchmark",
]


def _scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        raise ValueError("evaluate_predictions requires strictly finite y_true and y_pred arrays.")
    err = np.abs(y_true - y_pred)
    zones = clarke_zones(y_true, y_pred)
    return {
        "n_samples": int(len(y_true)),
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_abs_error": float(np.median(err)),
        "p95_abs_error": float(np.quantile(err, 0.95)),
        "bias": float(np.mean(y_pred - y_true)),
        "clarke_ab_pct": 100.0 * float(np.mean(np.isin(zones, ["A", "B"]))),
    }


def evaluate_predictions(
    split: SplitWindows,
    y_pred: np.ndarray,
    *,
    model_id: str,
    sample_interval_minutes: float = 5.0,
) -> Dict[str, Any]:
    """Compute pooled, per-subject, glycemic-range, and rate-of-change evaluations."""
    y_true = np.asarray(split.y, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != pred.shape or len(y_true) == 0:
        raise ValueError("Non-empty matching split.y and y_pred required.")

    pooled = _scalar_metrics(y_true, pred)

    subj_rows: List[Dict[str, Any]] = []
    for subj in sorted(set(split.subject_ids.tolist())):
        mask = split.subject_ids == subj
        m = _scalar_metrics(y_true[mask], pred[mask])
        subj_rows.append({"model_id": model_id, "subject_id": str(subj), **m})
    per_subject_df = pd.DataFrame(subj_rows)

    subj_maes = per_subject_df["mae"].to_numpy(dtype=float)
    subject_distribution = {
        "n_subjects": int(len(per_subject_df)),
        "mean_subject_mae": float(np.mean(subj_maes)),
        "sd_subject_mae": float(np.std(subj_maes, ddof=1)) if len(subj_maes) > 1 else 0.0,
        "median_subject_mae": float(np.median(subj_maes)),
        "iqr_subject_mae": float(np.quantile(subj_maes, 0.75) - np.quantile(subj_maes, 0.25)),
        "min_subject_mae": float(np.min(subj_maes)),
        "max_subject_mae": float(np.max(subj_maes)),
    }

    ranges = glycemic_range(y_true)
    range_rows: List[Dict[str, Any]] = []
    for r_label in ("hypo", "target", "hyper"):
        mask = ranges == r_label
        if np.any(mask):
            range_rows.append(
                {"model_id": model_id, "glycemic_range": r_label, **_scalar_metrics(y_true[mask], pred[mask])}
            )
    by_range_df = pd.DataFrame(range_rows)

    lag_steps = min(3, split.X.shape[1] - 1)
    if lag_steps >= 1:
        roc = (split.X[:, -1] - split.X[:, -1 - lag_steps]) / (lag_steps * sample_interval_minutes)
    else:
        roc = np.zeros(len(y_true))
    roc_labels = np.where(roc < -1.0, "falling_fast_or_mod", np.where(roc > 1.0, "rising_fast_or_mod", "steady"))
    roc_rows: List[Dict[str, Any]] = []
    for r_lbl in ("falling_fast_or_mod", "steady", "rising_fast_or_mod"):
        mask = roc_labels == r_lbl
        if np.any(mask):
            roc_rows.append(
                {"model_id": model_id, "rate_regime": r_lbl, **_scalar_metrics(y_true[mask], pred[mask])}
            )
    by_rate_df = pd.DataFrame(roc_rows)

    return {
        "model_id": model_id,
        "pooled": pooled,
        "subject_distribution": subject_distribution,
        "per_subject": per_subject_df,
        "by_glycemic_range": by_range_df,
        "by_rate_regime": by_rate_df,
    }


def compare_models_by_subject(
    reference_eval: Dict[str, Any],
    candidate_eval: Dict[str, Any],
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired per-subject comparison (`candidate - reference`) with small-N exact statistics."""
    ref_df = reference_eval["per_subject"][["subject_id", "mae", "rmse"]].rename(
        columns={"mae": "ref_mae", "rmse": "ref_rmse"}
    )
    cand_df = candidate_eval["per_subject"][["subject_id", "mae", "rmse"]].rename(
        columns={"mae": "cand_mae", "rmse": "cand_rmse"}
    )
    merged = ref_df.merge(cand_df, on="subject_id", how="inner")
    deltas = (merged["cand_mae"] - merged["ref_mae"]).to_numpy(dtype=float)
    n_subj = int(len(deltas))

    ci_lo, ci_hi = paired_bootstrap_ci(deltas, confidence=0.95, n_boot=2000, seed=seed)
    iqr_d = float(np.quantile(deltas, 0.75) - np.quantile(deltas, 0.25)) if n_subj > 0 else float("nan")

    p_val = float("nan")
    nonzero = deltas[np.abs(deltas) > 1e-12]
    n_nz = int(nonzero.size)
    if n_nz >= 6:
        method = "exact" if n_nz <= 25 else "approx"
        p_val = float(stats.wilcoxon(nonzero, alternative="two-sided", method=method).pvalue)
        min_p = 2.0 ** (1 - n_nz) if n_nz <= 25 else 0.0
        stat_note = (
            f"Two-sided Wilcoxon ({method}) on N={n_subj} independent subjects "
            f"(nonzero pairs={n_nz}, minimum attainable exact p={min_p:.6f}). "
            "For multi-seed neural models, predictions/errors must be averaged per subject first."
        )
    elif n_nz > 0:
        min_p = 2.0 ** (1 - n_nz)
        stat_note = (
            f"Underpowered cohort (nonzero N={n_nz} < 6; minimum two-sided exact p={min_p:.4f} > 0.05). "
            "Wilcoxon p-value omitted to prevent over-interpretation; inspect per-subject table."
        )
    else:
        stat_note = "Zero non-tied subject differences."

    return {
        "reference_model": reference_eval["model_id"],
        "candidate_model": candidate_eval["model_id"],
        "n_subjects": n_subj,
        "mean_delta_mae": float(np.mean(deltas)) if n_subj else float("nan"),
        "ci95_delta_low": ci_lo,
        "ci95_delta_high": ci_hi,
        "median_delta_mae": float(np.median(deltas)) if n_subj else float("nan"),
        "iqr_delta_mae": iqr_d,
        "min_delta_mae": float(np.min(deltas)) if n_subj else float("nan"),
        "max_delta_mae": float(np.max(deltas)) if n_subj else float("nan"),
        "subjects_improved": int(np.sum(deltas < 0.0)),
        "wilcoxon_p_value": p_val,
        "statistical_qualification": stat_note,
        "paired_table": merged.assign(delta_mae=deltas),
    }


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No observations in this stratum._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [f"{row[c]:.2f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run_phase2_baseline_benchmark(
    *,
    dataset: str = "ohiot1dm",
    data_path: Union[str, Path, None] = None,
    output_dir: Union[str, Path] = "results/neural_forecasting_phase2",
    config: ForecastingExperimentConfig | None = None,
    virtual_subjects: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the complete Phase 1–2 pipeline: Import -> Audit -> Embargo Windows -> Baselines -> Register -> Report."""
    cfg = config or ForecastingExperimentConfig(dataset=dataset)  # type: ignore[arg-type]
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    canonical: pd.DataFrame
    quality: DataQualityReport
    if dataset == "ohiot1dm":
        if data_path is None:
            raise ValueError("data_path is required when dataset='ohiot1dm'.")
        canonical, quality = load_ohiot1dm_dataset(data_path)
    elif dataset == "carelink":
        if data_path is None:
            raise ValueError("data_path is required when dataset='carelink'.")
        canonical, quality = load_carelink_benchmark_dataset(data_path)
    elif dataset == "iints_virtual":
        canonical, quality = load_virtual_benchmark_dataset(n_subjects=virtual_subjects, seed=seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'ohiot1dm', 'iints_virtual', or 'carelink'.")

    wspec = WindowSpec(
        n_input_samples=cfg.n_input_samples,
        history_minutes=cfg.history_minutes,
        prediction_horizon_minutes=cfg.prediction_horizon_minutes,
        sample_interval_minutes=cfg.sample_interval_minutes,
        validation_fraction=cfg.validation_fraction,
    )
    partitioned = build_partitioned_windows(canonical, wspec)
    if len(partitioned.train_fit) == 0 or len(partitioned.test) == 0:
        raise RuntimeError(f"Insufficient contiguous windows extracted: {partitioned.split_counts()}")

    persistence = PersistencePredictor().fit(partitioned.train_fit.X, partitioned.train_fit.y)
    ridge = RidgeCGMPredictor(alpha_grid=cfg.ridge_alphas).fit(
        partitioned.train_fit.X,
        partitioned.train_fit.y,
        X_val=partitioned.val.X if len(partitioned.val) > 0 else None,
        y_val=partitioned.val.y if len(partitioned.val) > 0 else None,
    )

    models: Sequence[WindowForecaster] = (persistence, ridge)
    val_evals: Dict[str, Dict[str, Any]] = {}
    test_evals: Dict[str, Dict[str, Any]] = {}
    from iints.forecasting.training.registry import ForecastingModelRegistry

    registry = ForecastingModelRegistry(out_root / "model_registry")
    ds_hash = sha256_json(quality.model_dump(mode="json"))

    for m in models:
        if len(partitioned.val) > 0:
            v_ev = evaluate_predictions(
                partitioned.val,
                m.predict(partitioned.val.X),
                model_id=m.model_id,
                sample_interval_minutes=cfg.sample_interval_minutes,
            )
            val_metrics = v_ev["pooled"]
            val_evals[m.model_id] = v_ev
        else:
            val_metrics = {}
        t_ev = evaluate_predictions(
            partitioned.test,
            m.predict(partitioned.test.X),
            model_id=m.model_id,
            sample_interval_minutes=cfg.sample_interval_minutes,
        )
        test_evals[m.model_id] = t_ev
        registry.register(
            m,
            dataset_id=quality.dataset_id,
            is_synthetic=quality.is_synthetic,
            benchmark_tier=quality.benchmark_tier,
            dataset_sha256=ds_hash,
            input_length=wspec.input_length,
            elapsed_input_span_minutes=wspec.elapsed_span_minutes,
            horizon_minutes=wspec.prediction_horizon_minutes,
            training_config={**cfg.model_dump(mode="json"), "seed": int(seed)},
            split_counts=partitioned.split_counts(),
            validation_metrics=val_metrics,
            test_metrics_available=True,
        )

    comparison = compare_models_by_subject(
        test_evals[persistence.model_id],
        test_evals[ridge.model_id],
        seed=seed,
    )

    (out_root / "experiment_config.json").write_text(
        json.dumps({**cfg.model_dump(mode="json"), "seed": int(seed), "is_synthetic": quality.is_synthetic}, indent=2),
        encoding="utf-8",
    )
    (out_root / "data_quality_report.json").write_text(
        json.dumps(quality.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    pooled_df = pd.DataFrame(
        [
            {
                "dataset_id": quality.dataset_id,
                "is_synthetic": quality.is_synthetic,
                "benchmark_tier": quality.benchmark_tier,
                "model_id": mid,
                **ev["pooled"],
                **ev["subject_distribution"],
            }
            for mid, ev in test_evals.items()
        ]
    )
    per_subj_all = pd.concat([ev["per_subject"] for ev in test_evals.values()], ignore_index=True)
    per_subj_all.insert(0, "is_synthetic", quality.is_synthetic)
    per_subj_all.insert(0, "dataset_id", quality.dataset_id)
    by_range_all = pd.concat([ev["by_glycemic_range"] for ev in test_evals.values()], ignore_index=True)
    by_rate_all = pd.concat([ev["by_rate_regime"] for ev in test_evals.values()], ignore_index=True)

    pooled_csv = out_root / "test_pooled_summary.csv"
    subj_csv = out_root / "test_per_subject.csv"
    range_csv = out_root / "test_by_glycemic_range.csv"
    pooled_df.to_csv(pooled_csv, index=False)
    per_subj_all.to_csv(subj_csv, index=False)
    by_range_all.to_csv(range_csv, index=False)

    prov = capture_provenance(
        {**cfg.model_dump(mode="json"), "seed": int(seed), "is_synthetic": quality.is_synthetic},
        [pooled_csv, subj_csv, range_csv],
    )
    (out_root / "provenance_manifest.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    synth_banner = (
        "\n> [!WARNING]\n> **SYNTHETIC SOFTWARE VALIDATION ONLY (`is_synthetic=True`)**: "
        "This run executed on simulated IINTS Virtual Patient data to validate pipeline integrity. "
        "It is NOT human clinical data and NOT evidence of real-world OhioT1DM forecasting accuracy.\n"
        if quality.is_synthetic
        else ""
    )

    report_lines = [
        f"# IINTS Neural Forecasting v1 — Phase 2 Baseline Report (`{quality.dataset_id}`)",
        f"> **Disclaimer:** {FORECASTING_DISCLAIMER}",
        synth_banner,
        "## 1. Dataset Integrity, Sampling Convention & Partition Summary",
        f"- **Dataset ID:** `{quality.dataset_id}` (`benchmark_tier={quality.benchmark_tier}`, `is_synthetic={quality.is_synthetic}`)",
        f"- **Source Provenance:** `{quality.source_provenance}`",
        f"- **Sampling Convention:** `{wspec.input_length}` valid samples at `{wspec.sample_interval_minutes:.1f} min` intervals "
        f"(`elapsed_span_minutes={wspec.elapsed_span_minutes:.1f} min` from `t-55m` to `t`, forecasting `t+{wspec.prediction_horizon_minutes}m`)",
        f"- **Subjects:** `{quality.n_subjects}` | Raw rows: `{quality.total_raw_rows}` | Valid rows: `{quality.total_valid_rows}`",
        f"- **Flagged Anomalies:** duplicates=`{quality.total_duplicates_flagged}`, out-of-bounds=`{quality.total_out_of_bounds_flagged}`, "
        f"missing=`{quality.total_missing_values_flagged}`, gaps > 6.5m=`{quality.total_gaps_over_threshold}`",
        f"- **Contiguous Windows:** `train_fit={len(partitioned.train_fit)}`, `val={len(partitioned.val)}`, "
        f"`embargo_purged={partitioned.embargo_purged_count}`, `official_test={len(partitioned.test)}`",
        f"- **Selected Ridge Penalty (on `val` only):** `alpha = {ridge.selected_alpha}` "
        f"(`scaler mean={ridge.scaler.mean_mgdl:.2f}`, `std={ridge.scaler.std_mgdl:.2f} mg/dL`)",
        "",
        "## 2. Independent Test Set Summary (Pooled & Subject-Level Distribution)",
        _md_table(
            pooled_df[
                [
                    "model_id",
                    "is_synthetic",
                    "n_samples",
                    "mae",
                    "rmse",
                    "median_abs_error",
                    "clarke_ab_pct",
                    "mean_subject_mae",
                    "sd_subject_mae",
                    "median_subject_mae",
                ]
            ]
        ),
        "",
        "## 3. Paired Per-Subject Comparison (`Ridge - Persistence`)",
        f"- **Paired Subjects:** `{comparison['n_subjects']}` | Improved: `{comparison['subjects_improved']}/{comparison['n_subjects']}`",
        f"- **Mean Paired ΔMAE:** `{comparison['mean_delta_mae']:.3f} mg/dL` "
        f"(95% Subject Bootstrap CI: `[{comparison['ci95_delta_low']:.3f}, {comparison['ci95_delta_high']:.3f}]`, "
        f"Median ΔMAE: `{comparison['median_delta_mae']:.3f}`, IQR: `{comparison['iqr_delta_mae']:.3f}`)",
        f"- **Wilcoxon Signed-Rank p-value:** `{comparison['wilcoxon_p_value']}`",
        f"- **Statistical Qualification:** {comparison['statistical_qualification']}",
        "",
        _md_table(comparison["paired_table"]),
        "",
        "## 4. Stratified Error by True Target Glycemic Range",
        _md_table(by_range_all[["model_id", "glycemic_range", "n_samples", "mae", "rmse", "median_abs_error"]]),
        "",
        "## 5. Stratified Error by Recent CGM Rate-of-Change Regime",
        _md_table(by_rate_all[["model_id", "rate_regime", "n_samples", "mae", "rmse", "median_abs_error"]]),
        "",
        "## 6. Limitations",
        "- Version 1 uses univariate CGM history (12 samples, 55-minute elapsed span) without insulin or meal features.",
        "- All numbers are deterministic computations on stored evaluation windows; clinical validation is `none`.",
        "",
    ]
    report_path = out_root / "BASELINE_EVALUATION_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "quality": quality,
        "windows": partitioned,
        "val_evaluations": val_evals,
        "test_evaluations": test_evals,
        "subject_comparison": comparison,
        "registry_entries": registry.list_entries(),
        "report_path": report_path,
        "pooled_summary": pooled_df,
    }
