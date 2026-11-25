"""Run head-to-head A/B benchmark of MiniMed v2 (AR(2)-nested + meal models).

Usage:
    python tools/analysis/run_minimed_v2_ablation.py "c:/Users/runeb/Downloads/Rune Bobbaers 24-11-2026.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iints.analysis import minimed_predictor_v2 as v2


def main() -> None:
    parser = argparse.ArgumentParser(description="IINTS-SDK MiniMed v2 ablation & AR(2)-nesting benchmark.")
    parser.add_argument("csv_path", type=str, help="Path to Medtronic CareLink CSV export.")
    parser.add_argument("--train-days", type=float, default=21.0, help="Rolling training window in days.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="docs/presentation_visuals/minimed_v2_ablation.csv",
        help="Output CSV path for the ablation table.",
    )
    args = parser.parse_args()

    print(f"Running 9-fold walk-forward A/B ablation on: {args.csv_path}")
    print(f"Rolling train window: {args.train_days:.0f} days (in-silico research only)\n")

    audit = v2.insulin_audit(args.csv_path)
    max_err = float(audit["balance_error_u"].abs().max())
    print(f"Insulin bookkeeping audit: max |balance_error_u| = {max_err:.4f} U across {len(audit)} days\n")

    table = v2.compare_variants(args.csv_path, train_days=args.train_days)
    print(table.to_string(index=False))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nSaved ablation table to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
