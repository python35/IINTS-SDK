"""End-to-end runner for IINTS Neural Forecasting v1 (Phases 2 -> 3 -> 4 -> 5).

Executes:
1. Dataset import & quality audit (`ohiot1dm` if `--ohio-dir` exists, else explicitly
   labeled `iints_virtual` software-validation cohort — never mixed!).
2. Leakage-free embargoed window extraction (`12` samples = `55.0 min` span -> `+30 min`).
3. Training & subject-level comparison of `Persistence`, `Ridge`, 5-seed `GRU-32`,
   and 5-seed `Causal TCN-16`.
4. 10-condition CGM Robustness Benchmark (Clean, 5/15/30m Dropout+LOCF, 5/10/15 mg/dL
   Colored Noise, 5/10/15m Sensor Delay).
5. Edge export (`.iints_edge.json` + `.onnx` if PyTorch/ONNX installed), numerical
   parity verification (`max_abs_diff`), and batch-size-1 latency/RAM/telemetry benchmark.
6. Writes deterministic `NEURAL_FORECASTING_FULL_REPORT.md` + 4-panel figure.

Usage:
    python tools/forecasting/run_full_neural_jetson_pipeline.py
    # With official OhioT1DM XMLs:
    python tools/forecasting/run_full_neural_jetson_pipeline.py --dataset ohiot1dm --data-path "OhioT1DM-volledig"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from iints.forecasting import (
    FORECASTING_DISCLAIMER,
    NeuralTrainingConfig,
    WindowSpec,
    benchmark_edge_inference,
    build_partitioned_windows,
    export_model_for_edge,
    load_ohiot1dm_dataset,
    load_virtual_benchmark_dataset,
    run_cgm_robustness_benchmark,
    train_and_compare_all_models,
)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = [f"{r[c]:.3f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _plot_dashboard(
    summary_df: pd.DataFrame,
    rob_df: pd.DataFrame,
    edge_rows: List[Dict],
    out_png: Path,
    is_synthetic: bool,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    fig.patch.set_facecolor("white")
    colors = ["#1b4965", "#2a9d8f", "#ca6702", "#ae2012", "#6c757d", "#3a86ff"]

    # Panel A: Clean Test MAE across models
    ax = axes[0, 0]
    s_ord = summary_df.sort_values("mae")
    ax.barh(s_ord["model_id"], s_ord["mae"], color=colors[: len(s_ord)])
    ax.invert_yaxis()
    ax.set_xlabel("30-min Test MAE (mg/dL) — lower is better")
    ax.set_title("A. Independent Test Accuracy (Univariate 12-Sample CGM)", fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="x", alpha=0.25)

    # Panel B: Dropout Robustness (0 / 5 / 15 / 30 min LOCF)
    ax = axes[0, 1]
    drop_sub = rob_df[rob_df["family"].isin(["clean", "dropout_locf"])]
    for i, mid in enumerate(dict.fromkeys(drop_sub["model_id"])):
        m_df = drop_sub[drop_sub["model_id"] == mid].sort_values("severity_value")
        ax.plot(
            m_df["severity_value"],
            m_df["mean_subject_mae"],
            marker="o",
            linewidth=2.0,
            color=colors[i % len(colors)],
            label=mid,
        )
    ax.set_xlabel("CGM Dropout Duration (min, LOCF)")
    ax.set_ylabel("Mean Subject MAE (mg/dL)")
    ax.set_title("B. Robustness to Simulated CGM Signal Dropout", fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # Panel C: Sensor Noise & Delay Degradation
    ax = axes[1, 0]
    nd_sub = rob_df[rob_df["condition_id"].isin(["A_clean", "C_noise_10mgdl", "D_delay_10m"])]
    conds = ["A_clean", "C_noise_10mgdl", "D_delay_10m"]
    mids = list(dict.fromkeys(nd_sub["model_id"]))
    x = np.arange(len(mids))
    w = 0.25
    for j, cid in enumerate(conds):
        vals = [
            float(nd_sub[(nd_sub["model_id"] == m) & (nd_sub["condition_id"] == cid)]["mean_subject_mae"].mean())
            for m in mids
        ]
        ax.bar(x + (j - 1) * w, vals, width=w, label=cid)
    ax.set_xticks(x)
    ax.set_xticklabels(mids, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Mean Subject MAE (mg/dL)")
    ax.set_title("C. Clean vs +10 mg/dL Noise vs 10-min Delay", fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel D: Batch-Size-1 Edge Inference Latency (Median vs P99 ms)
    ax = axes[1, 1]
    e_df = pd.DataFrame(edge_rows)
    xe = np.arange(len(e_df))
    ax.bar(xe - 0.18, e_df["median_latency_ms"], width=0.35, color="#2a9d8f", label="Median (P50) ms")
    ax.bar(xe + 0.18, e_df["p99_latency_ms"], width=0.35, color="#ae2012", label="Tail (P99) ms")
    ax.set_xticks(xe)
    ax.set_xticklabels(e_df["model_id"], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Batch-Size-1 Latency (ms)")
    ax.set_title("D. Local Edge Inference Latency (Batch Size = 1)", fontweight="bold", fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    tag = "SYNTHETIC VALIDATION COHORT" if is_synthetic else "OHIOT1DM COHORT"
    fig.suptitle(f"IINTS Neural Forecasting v1 — {tag} [Research Only, clinical_use=False]", fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="IINTS Neural Forecasting v1 — Complete Phase 2-5 Study Runner.")
    parser.add_argument("--dataset", choices=["iints_virtual", "ohiot1dm"], default="iints_virtual")
    parser.add_argument("--data-path", type=str, default=None, help="Path to OhioT1DM XML directory when --dataset ohiot1dm.")
    parser.add_argument("--virtual-subjects", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seeds", type=str, default="11,23,42,67,89")
    parser.add_argument("--out-dir", type=str, default="results/neural_forecasting_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seed_tuple = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    print(f"[1/5] Loading dataset '{args.dataset}' and running quality audit...")
    if args.dataset == "ohiot1dm":
        if not args.data_path:
            raise SystemExit("--data-path pointing to official OhioT1DM XMLs is required when --dataset ohiot1dm.")
        canonical, quality = load_ohiot1dm_dataset(args.data_path)
    else:
        canonical, quality = load_virtual_benchmark_dataset(n_subjects=args.virtual_subjects, seed=42)

    part = build_partitioned_windows(canonical, WindowSpec())
    print(f"      Subjects={quality.n_subjects} | Windows={part.split_counts()}\n")

    print(f"[2/5] Training Persistence, Ridge, 5-seed GRU-32 & 5-seed Causal TCN-16 (epochs={args.epochs})...")
    tr_out = train_and_compare_all_models(
        part,
        quality,
        out_dir,
        train_cfg=NeuralTrainingConfig(seeds=seed_tuple, max_epochs=args.epochs),
    )
    print(tr_out["summary_table"][["model_id", "mae", "rmse", "median_abs_error", "mean_subject_mae"]].to_string(index=False))
    print()

    print("[3/5] Running 10-condition CGM Robustness Benchmark (Clean, Dropout, Noise, Delay)...")
    rob_out = run_cgm_robustness_benchmark(
        part.test,
        tr_out["models"],
        is_synthetic=quality.is_synthetic,
        dataset_id=quality.dataset_id,
        output_dir=out_dir / "robustness",
    )

    print("[4/5] Exporting models for Edge/Jetson, verifying numerical parity & benchmarking batch-size-1 latency...")
    edge_dir = out_dir / "edge_exports"
    edge_rows: List[Dict] = []
    for mid, model in tr_out["models"].items():
        exp = export_model_for_edge(model, edge_dir, reference_windows=part.test.X[:64])
        bench = benchmark_edge_inference(
            exp["edge_bundle_path"],
            sample_windows=part.test.X[:128],
            n_warmup=30,
            n_runs=300,
            output_json_path=edge_dir / f"{mid}_benchmark.json",
        )
        edge_rows.append(
            {
                "model_id": mid,
                "runtime": bench["runtime"],
                "max_parity_diff_mgdl": exp["parity"]["max_abs_diff_mgdl"],
                "parity_passed": exp["parity"]["passed"],
                "median_latency_ms": bench["median_latency_ms"],
                "p95_latency_ms": bench["p95_latency_ms"],
                "p99_latency_ms": bench["p99_latency_ms"],
                "throughput_preds_per_sec": bench["throughput_preds_per_sec"],
                "peak_ram_mb": bench["peak_ram_mb"],
            }
        )
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(edge_dir / "edge_benchmark_summary.csv", index=False)
    print(edge_df.to_string(index=False))

    print("\n[5/5] Writing deterministic NEURAL_FORECASTING_FULL_REPORT.md & 4-panel figure...")
    fig_path = out_dir / "neural_forecasting_dashboard.png"
    has_fig = _plot_dashboard(tr_out["summary_table"], rob_out["condition_summary"], edge_rows, fig_path, quality.is_synthetic)

    banner = (
        "\n> [!WARNING]\n> **SYNTHETIC SOFTWARE VALIDATION (`is_synthetic=True`)**: "
        "Executed on IINTS Virtual Patient trajectories, NOT human clinical OhioT1DM data.\n"
        if quality.is_synthetic
        else ""
    )
    report_md = "\n".join(
        [
            f"# IINTS Neural Forecasting v1 — Full Study Report (`{quality.dataset_id}`)",
            f"> **Disclaimer:** {FORECASTING_DISCLAIMER}",
            banner,
            f"![Dashboard]({fig_path.name})" if has_fig else "",
            "## 1. Clean Test Set Comparison (Persistence vs Ridge vs 5-Seed GRU vs 5-Seed TCN)",
            _md_table(
                tr_out["summary_table"][
                    ["model_id", "is_synthetic", "n_samples", "mae", "rmse", "median_abs_error", "clarke_ab_pct", "mean_subject_mae", "sd_subject_mae"]
                ]
            ),
            "",
            "## 2. Seed Initialization Sensitivity (`seeds = " + str(list(seed_tuple)) + "`)",
            _md_table(tr_out["seed_sensitivity"]),
            "",
            "## 3. CGM Fault Robustness Paired Degradation vs Clean (`Delta MAE` per Subject)",
            _md_table(
                rob_out["paired_degradation"][
                    ["condition_id", "model_id", "n_subjects", "clean_mean_subject_mae", "perturbed_mean_subject_mae", "mean_delta_mae", "ci95_delta_low", "ci95_delta_high", "p_holm_bonferroni"]
                ]
            ),
            "",
            "## 4. Edge Export Numerical Parity & Batch-Size-1 Latency Benchmark",
            _md_table(edge_df),
            "",
        ]
    )
    rep_path = out_dir / "NEURAL_FORECASTING_FULL_REPORT.md"
    rep_path.write_text(report_md, encoding="utf-8")
    print(f"      Report : {rep_path.resolve()}")
    if has_fig:
        print(f"      Figure : {fig_path.resolve()}")


if __name__ == "__main__":
    main()
