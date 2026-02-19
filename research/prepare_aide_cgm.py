from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iints.research.dataset import save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_packs/public/aide_t1d/Data Tables/AIDEDeviceCGM.txt"),
        help="Path to AIDEDeviceCGM.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_packs/public/aide_t1d/processed/aide_cgm.csv"),
        help="Output Parquet path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data_packs/public/aide_t1d/quality_report.json"),
        help="Quality report output path",
    )
    parser.add_argument("--time-step", type=int, default=5, help="Expected CGM time step (minutes)")
    parser.add_argument("--max-gap-multiplier", type=float, default=2.5, help="Segment break multiplier")
    parser.add_argument("--min-glucose", type=float, default=40.0, help="Minimum glucose (mg/dL)")
    parser.add_argument("--max-glucose", type=float, default=400.0, help="Maximum glucose (mg/dL)")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap after sorting")
    return parser.parse_args()


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input, sep="|", low_memory=False)
    df["timestamp"] = pd.to_datetime(df.get("DataDtTm"), errors="coerce")
    df = df[df["timestamp"].notna()].copy()

    if "RecordType" in df.columns:
        df = df[df["RecordType"].astype(str).str.upper() == "CGM"]

    if "Unusable" in df.columns:
        df = df[~df["Unusable"].apply(_to_bool)]

    df["glucose_actual_mgdl"] = pd.to_numeric(df.get("GlucValue"), errors="coerce")
    if "Units" in df.columns:
        mmol_mask = df["Units"].astype(str).str.contains("mmol", case=False, na=False)
        df.loc[mmol_mask, "glucose_actual_mgdl"] = df.loc[mmol_mask, "glucose_actual_mgdl"] * 18.0

    df = df[df["glucose_actual_mgdl"].notna()].copy()
    df = df[
        (df["glucose_actual_mgdl"] >= args.min_glucose)
        & (df["glucose_actual_mgdl"] <= args.max_glucose)
    ].copy()

    df = df.sort_values(["PtID", "timestamp"]).reset_index(drop=True)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df["delta_min"] = df.groupby("PtID")["timestamp"].diff().dt.total_seconds() / 60.0
    max_gap = args.time_step * args.max_gap_multiplier
    segment_break = df["delta_min"].isna() | (df["delta_min"] <= 0) | (df["delta_min"] > max_gap)
    df["segment"] = segment_break.groupby(df["PtID"]).cumsum()

    df["glucose_trend_mgdl_min"] = (
        df.groupby(["PtID", "segment"])["glucose_actual_mgdl"].diff() / df["delta_min"]
    )

    minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
        + df["timestamp"].dt.second / 60.0
    )
    df["time_of_day_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    df["time_of_day_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

    # Drop rows where trend is undefined (segment starts or invalid step)
    df = df[df["glucose_trend_mgdl_min"].notna()].copy()

    report = {
        "source": str(args.input),
        "records_total": int(len(df)),
        "patients": int(df["PtID"].nunique()),
        "start_time": df["timestamp"].min().isoformat(),
        "end_time": df["timestamp"].max().isoformat(),
        "median_interval_min": float(df["delta_min"].median()),
        "percent_large_gaps": float((df["delta_min"] > max_gap).mean() * 100.0),
        "glucose_mean": float(df["glucose_actual_mgdl"].mean()),
        "glucose_std": float(df["glucose_actual_mgdl"].std()),
        "glucose_min": float(df["glucose_actual_mgdl"].min()),
        "glucose_max": float(df["glucose_actual_mgdl"].max()),
        "max_rows_applied": int(args.max_rows) if args.max_rows else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    output_cols = [
        "glucose_actual_mgdl",
        "glucose_trend_mgdl_min",
        "time_of_day_sin",
        "time_of_day_cos",
    ]
    save_dataset(df[output_cols], args.output)
    print(f"Saved quality report: {args.report}")
    print(f"Saved training data: {args.output}")


if __name__ == "__main__":
    main()
