from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.data.realism_reference import RealismReferenceProfile, get_realism_reference
from iints.data.realism_validator import validate_realism_dataset
from iints.highlevel import run_simulation
from iints.presets import get_preset


DATASET_CONFIG = {
    "azt1d_daily": {
        "path": Path("data_packs/public/azt1d/processed/azt1d_merged.csv"),
        "preset": "realistic_azt1d_day",
        "label": "AZT1D",
    },
    "hupa_ucm_daily": {
        "path": Path("data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv"),
        "preset": "realistic_hupa_ucm_day",
        "label": "HUPA-UCM",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render simulator-versus-real daily CGM comparison figures."
    )
    parser.add_argument(
        "--references",
        default="azt1d_daily,hupa_ucm_daily",
        help="Comma-separated reference ids to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/realism_gallery"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _standardize_simulation(results: pd.DataFrame) -> pd.DataFrame:
    return results.rename(
        columns={
            "time_minutes": "timestamp",
            "glucose_actual_mgdl": "glucose",
            "carb_intake_grams": "carbs",
            "delivered_insulin_units": "insulin",
        }
    )[["timestamp", "glucose", "carbs", "insulin"]]


def _daily_windows(path: Path) -> list[pd.DataFrame]:
    dataframe = pd.read_csv(path)
    dataframe["timestamp_dt"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe["glucose_actual_mgdl"] = pd.to_numeric(
        dataframe["glucose_actual_mgdl"],
        errors="coerce",
    )
    dataframe = dataframe.dropna(subset=["subject_id", "timestamp_dt", "glucose_actual_mgdl"]).copy()
    dataframe["day"] = dataframe["timestamp_dt"].dt.floor("D")
    windows: list[pd.DataFrame] = []
    for (_, day), subset in dataframe.groupby(["subject_id", "day"], sort=True):
        subset = subset.sort_values("timestamp_dt").copy()
        if len(subset) < 250:
            continue
        subset["timestamp"] = (
            subset["timestamp_dt"] - pd.Timestamp(day)
        ).dt.total_seconds() / 60.0
        windows.append(
            pd.DataFrame(
                {
                    "timestamp": subset["timestamp"],
                    "glucose": subset["glucose_actual_mgdl"],
                    "carbs": pd.to_numeric(subset.get("carb_grams", 0.0), errors="coerce").fillna(0.0),
                    "insulin": pd.to_numeric(subset.get("insulin_units", 0.0), errors="coerce").fillna(0.0),
                    "subject_id": subset["subject_id"],
                    "day": day,
                }
            )
        )
    return windows


def normalized_glucose_distance(
    dataframe: pd.DataFrame,
    reference_profile: RealismReferenceProfile,
) -> float:
    report = validate_realism_dataset(dataframe, reference=reference_profile)
    wanted = {
        "mean_glucose_mgdl",
        "sd_mgdl",
        "cv_pct",
        "tir_70_180_pct",
        "tir_above_180_pct",
        "tir_below_70_pct",
        "glucose_range_mgdl",
    }
    distances: list[float] = []
    comparisons = {
        comparison.metric_key: comparison
        for comparison in report.reference_comparisons
        if comparison.metric_key in wanted and comparison.observed_value is not None
    }
    for metric_key, comparison in comparisons.items():
        band = reference_profile.metric_bands[metric_key]
        half_span = max((band.target_high - band.target_low) / 2.0, 1e-9)
        distances.append(abs(float(comparison.observed_value) - band.median) / half_span)
    return sum(distances) / max(len(distances), 1)


def select_representative_real_day(
    path: Path,
    reference_profile: RealismReferenceProfile,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    verdict_rank = {
        "likely_realistic": 2,
        "needs_review": 1,
        "likely_unrealistic": 0,
    }
    ranked: list[tuple[int, float, float, pd.DataFrame]] = []
    for window in _daily_windows(path):
        report = validate_realism_dataset(window, reference=reference_profile)
        ranked.append(
            (
                verdict_rank[report.verdict],
                float(report.realism_score),
                -normalized_glucose_distance(window, reference_profile),
                window,
            )
        )
    _, _, negative_distance, best = max(ranked, key=lambda row: row[:3])
    distance = -negative_distance
    first = best.iloc[0]
    metadata = {
        "subject_id": str(first["subject_id"]),
        "day": str(pd.Timestamp(first["day"]).date()),
        "normalized_glucose_distance": round(float(distance), 4),
    }
    return best[["timestamp", "glucose", "carbs", "insulin"]].copy(), metadata


def simulate_preset(preset_name: str, *, seed: int, output_dir: Path) -> pd.DataFrame:
    preset = get_preset(preset_name)
    outputs = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=preset["scenario"],
        patient_config=preset["patient_config"],
        duration_minutes=preset["duration_minutes"],
        time_step=preset["time_step_minutes"],
        physiology_variation_profile=preset.get("physiology_variation_profile"),
        physiology_variation_scale=float(preset.get("physiology_variation_scale", 1.0)),
        seed=seed,
        output_dir=output_dir,
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )
    return _standardize_simulation(outputs["results"])


def build_gallery(
    references: list[str],
    *,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(len(references), 1, figsize=(13, 4.2 * len(references)), sharex=True)
    if len(references) == 1:
        axes = [axes]

    for axis, reference_id in zip(axes, references):
        config = DATASET_CONFIG[reference_id]
        profile = get_realism_reference(reference_id)
        real_day, real_meta = select_representative_real_day(config["path"], profile)
        sim_day = simulate_preset(
            str(config["preset"]),
            seed=seed,
            output_dir=output_dir / "simulations" / reference_id,
        )
        real_report = validate_realism_dataset(real_day, reference=profile)
        sim_report = validate_realism_dataset(sim_day, reference=profile)
        hours_real = real_day["timestamp"] / 60.0
        hours_sim = sim_day["timestamp"] / 60.0

        axis.axhspan(70, 180, color="#DCFCE7", alpha=0.55, label="70-180 mg/dL")
        axis.plot(hours_real, real_day["glucose"], color="#0F172A", linewidth=1.4, label="Real day")
        axis.plot(hours_sim, sim_day["glucose"], color="#0F766E", linewidth=1.6, label="Simulator")
        axis.set_title(
            f"{config['label']}: real representative day vs calibrated simulator",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_ylabel("Glucose (mg/dL)")
        axis.set_ylim(40, max(260, float(real_day["glucose"].max()) + 10))
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
        rows.append(
            {
                "reference": reference_id,
                "real_day": real_meta,
                "real_verdict": real_report.verdict,
                "realism_score_real_day": round(real_report.realism_score, 4),
                "sim_preset": config["preset"],
                "sim_verdict": sim_report.verdict,
                "realism_score_simulation": round(sim_report.realism_score, 4),
            }
        )

    axes[-1].set_xlabel("Hour of day")
    fig.tight_layout()
    figure_path = output_dir / "simulator_vs_real_daily_traces.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {"figure": str(figure_path), "comparisons": rows}
    metadata_path = output_dir / "simulator_vs_real_daily_traces.json"
    metadata_path.write_text(json.dumps(payload, indent=2))
    payload["metadata"] = str(metadata_path)
    return payload


def main() -> None:
    args = parse_args()
    payload = build_gallery(
        _parse_csv(args.references),
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
