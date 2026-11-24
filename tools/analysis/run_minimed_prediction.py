#!/usr/bin/env python3
"""Run the Hybrid Gray-Box MiniMed Glucose Predictor & In-Silico BoxQP Benchmark.

Usage:
    $env:PYTHONPATH = "src"
    python tools/analysis/run_minimed_prediction.py "c:\\Users\\runeb\\Downloads\\Rune Bobbaers 24-11-2026.csv"

Research & educational evaluation only — never use for clinical insulin dosing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iints.analysis.minimed_predictor import PredictorConfig, evaluate_carelink_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run out-of-sample MiniMed 780G Gray-Box glucose prediction benchmark."
    )
    parser.add_argument("csv_path", type=Path, help="Path to Medtronic CareLink CSV export")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output path to save the full JSON benchmark report",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological fraction of timeline used for fitting (default: 0.70)",
    )
    args = parser.parse_args()

    cfg = PredictorConfig(train_fraction=args.train_fraction)
    report = evaluate_carelink_csv(args.csv_path, config=cfg)

    print("\n=== IINTS-SDK MiniMed Hybrid Gray-Box Prediction Benchmark ===")
    print(f"Source CSV : {report['source_csv']}")
    print(
        f"Timeline   : {report['total_timeline_rows']} CGM rows "
        f"(train={report['train_rows']}, blind test={report['test_rows']})"
    )
    fit = report["fit_summary"]
    print(
        f"Physics OP : G0={fit['operating_glucose_mgdl']:.1f} mg/dL | "
        f"Basal={fit['basal_rate_u_per_h']:.2f} U/h | "
        f"Est TDD={fit.get('estimated_tdd_u_per_day', 0.0):.1f} U/day | "
        f"Est ISF={fit.get('estimated_isf_mgdl_per_u', 50.0):.1f} mg/dL/U"
    )
    print(
        "\nHorizon |  N   | Naive RMSE | WhiteBox RMSE | GrayBox RMSE | MAE  | Clarke A+B | 90% Cov | Dir Acc | Dir Err"
    )
    print("-" * 106)
    for row in report["horizon_evaluations"]:
        print(
            f"{row['horizon_minutes']:4d} min | {row['n_samples']:4d} | "
            f"{row['naive_rmse_mgdl']:10.2f} | {row['whitebox_rmse_mgdl']:13.2f} | "
            f"{row['graybox_rmse_mgdl']:12.2f} | {row['graybox_mae_mgdl']:4.1f} | "
            f"{row['graybox_clarke_ab_pct']:9.2f}% | {row['graybox_coverage_90_pct']:6.2f}% | "
            f"{row['graybox_directional_accurate_pct']:6.2f}% | {row['graybox_directional_erroneous_pct']:6.2f}%"
        )

    qp = report["latest_in_silico_boxqp"]
    print("\n=== Latest-State In-Silico Chance-Constrained BoxQP (Research Only) ===")
    print(
        f"Rate: {qp['recommended_rate_u_per_h']:.3f} U/h "
        f"({qp['recommended_step_units']:.3f} U / 5 min) | "
        f"Pred 60m: {qp['predicted_60m_mgdl']:.1f} mg/dL | "
        f"Min p05: {qp['min_predicted_p05_mgdl']:.1f} mg/dL | "
        f"Safety Suspend: {qp['safety_floor_suspended']} | "
        f"KKT res: {qp['kkt_residual']:.2e}"
    )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {args.out_json}")


if __name__ == "__main__":
    main()
