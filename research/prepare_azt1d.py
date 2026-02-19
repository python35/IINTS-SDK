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
        default=Path("data_packs/public/azt1d/AZT1D 2025/CGM Records"),
        help="Root directory containing Subject folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_packs/public/azt1d/processed/azt1d_merged.csv"),
        help="Output dataset path (CSV or Parquet)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data_packs/public/azt1d/quality_report.json"),
        help="Quality report output path",
    )
    parser.add_argument("--time-step", type=int, default=5, help="Expected CGM time step (minutes)")
    parser.add_argument("--max-gap-multiplier", type=float, default=2.5, help="Segment break multiplier")
    parser.add_argument("--dia-minutes", type=float, default=240.0, help="Insulin action duration (minutes)")
    parser.add_argument("--carb-absorb-minutes", type=float, default=120.0, help="Carb absorption duration (minutes)")
    parser.add_argument("--max-basal", type=float, default=20.0, help="Clip basal values above this")
    parser.add_argument("--max-bolus", type=float, default=30.0, help="Clip bolus values above this")
    parser.add_argument("--max-carbs", type=float, default=200.0, help="Clip carb grams above this")
    return parser.parse_args()


def _clean_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def _device_mode_code(series: pd.Series) -> pd.Series:
    mapping = {"sleep": 1.0, "sleepsleep": 1.0, "exercise": 2.0, "0": 0.0}
    return series.fillna("0").astype(str).str.lower().map(mapping).fillna(0.0)


def _derive_iob_cob(
    insulin: pd.Series,
    carbs: pd.Series,
    delta_min: pd.Series,
    dia_minutes: float,
    carb_absorb_minutes: float,
) -> tuple[pd.Series, pd.Series]:
    iob = []
    cob = []
    prev_iob = 0.0
    prev_cob = 0.0
    for ins, carb, dt in zip(insulin, carbs, delta_min):
        if not np.isfinite(dt) or dt <= 0:
            prev_iob = 0.0
            prev_cob = 0.0
            iob.append(0.0)
            cob.append(0.0)
            continue
        iob_decay = np.exp(-dt / dia_minutes)
        cob_decay = np.exp(-dt / carb_absorb_minutes)
        prev_iob = prev_iob * iob_decay + ins
        prev_cob = prev_cob * cob_decay + carb
        iob.append(prev_iob)
        cob.append(prev_cob)
    return pd.Series(iob, index=insulin.index), pd.Series(cob, index=carbs.index)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    frames = []
    for csv_path in sorted(args.input.glob("Subject */Subject *.csv")):
        subject_id = csv_path.parent.name.replace("Subject", "").strip()
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df.get("EventDateTime"), errors="coerce")
        df = df[df["timestamp"].notna()].copy()
        df["subject_id"] = subject_id

        df["glucose_actual_mgdl"] = _clean_column(df, "CGM")
        df["basal_units"] = _clean_column(df, "Basal").clip(upper=args.max_basal)
        df["bolus_units"] = _clean_column(df, "TotalBolusInsulinDelivered").clip(upper=args.max_bolus)
        df["correction_units"] = _clean_column(df, "CorrectionDelivered").clip(upper=args.max_bolus)
        df["insulin_units"] = df["basal_units"] + df["bolus_units"] + df["correction_units"]

        carb_size = _clean_column(df, "CarbSize")
        food_delivered = _clean_column(df, "FoodDelivered")
        df["carb_grams"] = carb_size.where(carb_size > 0, food_delivered).clip(upper=args.max_carbs)

        df["device_mode_code"] = _device_mode_code(df.get("DeviceMode", pd.Series(index=df.index)))

        df = df.sort_values("timestamp").reset_index(drop=True)
        df["delta_min"] = df["timestamp"].diff().dt.total_seconds() / 60.0
        max_gap = args.time_step * args.max_gap_multiplier
        segment_break = df["delta_min"].isna() | (df["delta_min"] <= 0) | (df["delta_min"] > max_gap)
        df["segment"] = segment_break.cumsum()

        df["glucose_trend_mgdl_min"] = df["glucose_actual_mgdl"].diff() / df["delta_min"]

        iob, cob = _derive_iob_cob(
            df["insulin_units"].fillna(0.0),
            df["carb_grams"].fillna(0.0),
            df["delta_min"],
            args.dia_minutes,
            args.carb_absorb_minutes,
        )
        df["derived_iob_units"] = iob
        df["derived_cob_grams"] = cob

        minutes = (
            df["timestamp"].dt.hour * 60
            + df["timestamp"].dt.minute
            + df["timestamp"].dt.second / 60.0
        )
        df["time_of_day_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * minutes / 1440.0)

        df = df[df["glucose_actual_mgdl"] > 0].copy()
        df = df[df["glucose_trend_mgdl_min"].notna()].copy()
        df = df[np.isfinite(df["glucose_trend_mgdl_min"])].copy()

        frames.append(
            df[
                [
                    "subject_id",
                    "timestamp",
                    "glucose_actual_mgdl",
                    "glucose_trend_mgdl_min",
                    "insulin_units",
                    "carb_grams",
                    "derived_iob_units",
                    "derived_cob_grams",
                    "time_of_day_sin",
                    "time_of_day_cos",
                    "device_mode_code",
                ]
            ]
        )

    combined = pd.concat(frames, ignore_index=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.input),
        "records_total": int(len(combined)),
        "subjects": int(combined["subject_id"].nunique()),
        "start_time": combined["timestamp"].min().isoformat(),
        "end_time": combined["timestamp"].max().isoformat(),
        "glucose_mean": float(combined["glucose_actual_mgdl"].mean()),
        "glucose_std": float(combined["glucose_actual_mgdl"].std()),
        "glucose_min": float(combined["glucose_actual_mgdl"].min()),
        "glucose_max": float(combined["glucose_actual_mgdl"].max()),
        "insulin_mean": float(combined["insulin_units"].mean()),
        "carb_mean": float(combined["carb_grams"].mean()),
    }
    args.report.write_text(json.dumps(report, indent=2))

    save_dataset(combined.drop(columns=["timestamp"]), args.output)
    print(f"Saved quality report: {args.report}")
    print(f"Saved training data: {args.output}")


if __name__ == "__main__":
    main()
