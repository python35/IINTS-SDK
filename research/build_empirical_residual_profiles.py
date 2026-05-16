from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_INPUTS = {
    "azt1d_daily": Path("data_packs/public/azt1d/processed/azt1d_merged.csv"),
    "hupa_ucm_daily": Path("data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build subtle empirical physiology residual templates from real CGM days."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/iints/data/physiology_residual_profiles.json"),
        help="JSON output path for packaged residual profiles.",
    )
    parser.add_argument(
        "--templates-per-dataset",
        type=int,
        default=12,
        help="Maximum number of daily residual templates kept per dataset.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=25,
        help="Centered rolling-median window used to remove meal-scale trends.",
    )
    parser.add_argument(
        "--clip-mgdl",
        type=float,
        default=18.0,
        help="Absolute residual clipping limit in mg/dL.",
    )
    return parser.parse_args()


def _load_daily_templates(
    path: Path,
    *,
    rolling_window: int,
    clip_mgdl: float,
) -> list[list[float]]:
    dataframe = pd.read_csv(path)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe["glucose_actual_mgdl"] = pd.to_numeric(
        dataframe["glucose_actual_mgdl"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["subject_id", "timestamp", "glucose_actual_mgdl"]).copy()
    dataframe["day"] = dataframe["timestamp"].dt.floor("D")

    templates: list[tuple[float, list[float]]] = []
    expected_index = pd.timedelta_range(start="0min", periods=288, freq="5min")
    for (_, day), subset in dataframe.groupby(["subject_id", "day"], sort=True):
        day_start = pd.Timestamp(day)
        subset = subset.sort_values("timestamp")
        subset["minute_of_day"] = subset["timestamp"] - day_start
        series = subset.set_index("minute_of_day")["glucose_actual_mgdl"]
        series = series.groupby(level=0).mean().reindex(expected_index)
        coverage = float(series.notna().mean())
        if coverage < 0.90:
            continue
        series = series.interpolate(limit=6, limit_direction="both")
        if series.isna().any():
            continue
        trend = series.rolling(
            window=rolling_window,
            center=True,
            min_periods=max(3, rolling_window // 3),
        ).median()
        trend = trend.interpolate(limit_direction="both").fillna(series.median())
        residual = (series - trend).clip(lower=-clip_mgdl, upper=clip_mgdl)
        residual = (residual - residual.median()).clip(lower=-clip_mgdl, upper=clip_mgdl)
        templates.append(
            (
                float(residual.abs().mean()),
                [round(float(value), 3) for value in residual.tolist()],
            )
        )
    # Keep days with visible but not pathological residual structure.
    templates.sort(key=lambda item: item[0], reverse=True)
    return [template for _, template in templates]


def _take_evenly(values: list[list[float]], count: int) -> list[list[float]]:
    if len(values) <= count:
        return values
    if count <= 1:
        return [values[0]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[position] for position in positions]


def build_profiles(
    *,
    templates_per_dataset: int,
    rolling_window: int,
    clip_mgdl: float,
) -> list[dict[str, object]]:
    per_dataset = {
        dataset_id: _take_evenly(
            _load_daily_templates(
                path,
                rolling_window=rolling_window,
                clip_mgdl=clip_mgdl,
            ),
            templates_per_dataset,
        )
        for dataset_id, path in DATASET_INPUTS.items()
    }
    pooled = _take_evenly(
        [template for templates in per_dataset.values() for template in templates],
        templates_per_dataset * len(per_dataset),
    )
    return [
        {
            "id": "free_living_t1d",
            "label": "Pooled free-living T1D empirical residuals",
            "source_dataset_ids": ["azt1d", "hupa_ucm"],
            "sample_interval_minutes": 5,
            "templates": pooled,
        },
        {
            "id": "azt1d_daily",
            "label": "AZT1D empirical residuals",
            "source_dataset_ids": ["azt1d"],
            "sample_interval_minutes": 5,
            "templates": per_dataset["azt1d_daily"],
        },
        {
            "id": "hupa_ucm_daily",
            "label": "HUPA-UCM empirical residuals",
            "source_dataset_ids": ["hupa_ucm"],
            "sample_interval_minutes": 5,
            "templates": per_dataset["hupa_ucm_daily"],
        },
    ]


def main() -> None:
    args = parse_args()
    profiles = build_profiles(
        templates_per_dataset=args.templates_per_dataset,
        rolling_window=args.rolling_window,
        clip_mgdl=args.clip_mgdl,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profiles, indent=2))
    counts = {profile["id"]: len(profile["templates"]) for profile in profiles}
    print(json.dumps({"out": str(args.out), "template_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
