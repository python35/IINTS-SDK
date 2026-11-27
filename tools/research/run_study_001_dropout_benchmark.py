"""Execute Study 001 — Robustness of Glucose Forecasters Under Simulated CGM Data Loss.

Implements the full pre-registered Research Lab workflow:
1. Proposes Study 001 across N virtual patient profiles (70% dev / 30% held-out unseen),
   standard + OOD adversarial boundary scenarios, and 4 matched CGM dropout arms
   (0, 5, 15, 30 min).
2. Locks the pre-registered SHA-256 plan digest (`approve_research_plan`).
3. Runs the simulation matrix with checkpointing, hierarchical seeding, and SQLite indexing.
4. Computes within-patient paired differences (Delta MAE vs 0-min control), 2,000-sample
   patient-level bootstrap 95% CIs, Wilcoxon signed-rank p-values, and Holm-Bonferroni FWER.
5. Emits `SCIENTIFIC_REPORT.md` + 4-panel `study_dashboard.png` strictly from stored data.

Usage:
    # Fast 20-profile benchmark (~20-30s):
    python tools/research/run_study_001_dropout_benchmark.py --profiles 20 --workers 4

    # Full 100-profile publication benchmark:
    python tools/research/run_study_001_dropout_benchmark.py --profiles 100 --workers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iints.research import (
    analyze_research_results,
    approve_research_plan,
    execute_research_plan,
    generate_verifiable_report_bundle,
    propose_research_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IINTS Research Lab — Study 001 CGM Dropout Robustness Benchmark."
    )
    parser.add_argument(
        "--template",
        type=str,
        default="study_001_cgm_dropout",
        choices=["study_001_cgm_dropout", "study_002_inclusive_stale_ui", "study_003_cross_simulator"],
        help="Pre-built scientific benchmark template.",
    )
    parser.add_argument("--profiles", type=int, default=20, help="Number of virtual patient profiles (use 100 for full study).")
    parser.add_argument("--seed", type=int, default=42, help="Master deterministic seed.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads.")
    parser.add_argument("--reviewer", type=str, default="Rune Bobbaers", help="Researcher approving the pre-registered plan.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/study_001_cgm_dropout",
        help="Directory for checkpoints, Parquet/CSV, SQLite, dashboard PNG, and SCIENTIFIC_REPORT.md.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[1/4] Designing pre-registered plan '{args.template}' (N={args.profiles} virtual profiles)...")
    proposed = propose_research_plan(
        args.template,
        n_profiles=args.profiles,
        seed=args.seed,
    )
    approved = approve_research_plan(proposed, reviewer=args.reviewer)
    print(f"      Status: {approved.status.upper()} | Reviewer: {approved.approved_by}")
    print(f"      Pre-registration lock SHA-256: {approved.plan_lock_sha256}")
    print(f"      Total simulation cells: {approved.total_simulation_cells()}\n")

    print(f"[2/4] Executing simulation campaign (workers={args.workers}, checkpoint resume enabled)...")
    run_out = execute_research_plan(
        approved,
        out_dir,
        max_workers=args.workers,
        resume=True,
        progress_callback=lambda done, tot: print(f"      Progress: {done}/{tot} cells", end="\r", flush=True),
    )
    print(
        f"\n      Executed {run_out['executed_cells']} new cells "
        f"(resumed {run_out['resumed_cells']} from checkpoint).\n"
    )

    print("[3/4] Running within-patient paired statistical analysis (zero pseudoreplication)...")
    analysis = analyze_research_results(run_out["cell_metrics"], approved)
    print("\n--- Condition Summary (30-min Horizon, Patient-Level Mean & 95% CI) ---")
    print(
        analysis["condition_summary"][
            [
                "condition_id",
                "algorithm_id",
                "n_patients",
                "mean_mae",
                "ci95_low",
                "ci95_high",
                "mean_dropout_mae",
                "mean_recovery_mae",
            ]
        ].to_string(index=False)
    )

    if not analysis["paired_vs_control"].empty:
        print("\n--- Within-Patient Paired Degradation vs Control (Holm-Bonferroni Adjusted) ---")
        print(
            analysis["paired_vs_control"][
                [
                    "condition_id",
                    "algorithm_id",
                    "mean_delta_mae",
                    "ci95_delta_low",
                    "ci95_delta_high",
                    "pct_patients_degraded",
                    "p_holm_bonferroni",
                ]
            ].to_string(index=False)
        )

    print("\n[4/4] Generating deterministic SCIENTIFIC_REPORT.md and 4-panel dashboard...")
    bundle = generate_verifiable_report_bundle(run_out, analysis)
    for label, path in bundle.items():
        print(f"      {label:15s}: {path.resolve()}")


if __name__ == "__main__":
    main()
