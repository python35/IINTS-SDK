from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.highlevel import run_simulation
from iints.presets import get_preset


REAL_INPUTS = [
    Path("data_packs/public/azt1d/processed/azt1d_merged.csv"),
    Path("data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv"),
]

SHARED_COLUMNS = [
    "subject_id",
    "segment",
    "time_minutes",
    "glucose_actual_mgdl",
    "glucose_trend_mgdl_min",
    "patient_iob_units",
    "patient_cob_grams",
    "effective_isf",
    "effective_icr",
    "effective_basal_rate_u_per_hr",
    "steps",
    "calories",
    "heart_rate",
    "sleep_minutes",
    "time_of_day_sin",
    "time_of_day_cos",
    "carb_intake_grams",
    "insulin_units",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a real+simulator multimodal training set for predictor retraining."
    )
    parser.add_argument(
        "--presets",
        default="free_living_t1d,realistic_reference_day,realistic_azt1d_day,realistic_hupa_ucm_day",
    )
    parser.add_argument("--synthetic-runs-per-preset", type=int, default=6)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_packs/generated/realism_augmented_multimodal.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_packs/generated/realism_augmented_multimodal_manifest.json"),
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _standardize_real(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    source_prefix = "AZT1D" if "azt1d" in str(path).lower() else "HUPA"
    dataframe["subject_id"] = source_prefix + "_" + dataframe["subject_id"].astype(str)
    if "segment" not in dataframe.columns:
        dataframe["segment"] = dataframe.groupby("subject_id").ngroup()
    if "time_minutes" not in dataframe.columns:
        timestamp = pd.to_datetime(dataframe["timestamp"], errors="coerce")
        dataframe["time_minutes"] = (
            timestamp - timestamp.dt.floor("D")
        ).dt.total_seconds() / 60.0
    if "carb_intake_grams" not in dataframe.columns:
        dataframe["carb_intake_grams"] = pd.to_numeric(
            dataframe.get("carb_grams", 0.0),
            errors="coerce",
        ).fillna(0.0)
    for column in SHARED_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = 0.0
    return dataframe[SHARED_COLUMNS].copy()


def _standardize_simulation(
    results: pd.DataFrame,
    *,
    subject_id: str,
    segment: int,
) -> pd.DataFrame:
    dataframe = results.copy()
    dataframe["subject_id"] = subject_id
    dataframe["segment"] = segment
    dataframe["insulin_units"] = dataframe["delivered_insulin_units"]
    dataframe["time_of_day_sin"] = np.sin(2 * np.pi * dataframe["time_minutes"] / 1440.0)
    dataframe["time_of_day_cos"] = np.cos(2 * np.pi * dataframe["time_minutes"] / 1440.0)
    for column in ("steps", "calories", "heart_rate", "sleep_minutes"):
        dataframe[column] = 0.0
    for column in SHARED_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = 0.0
    return dataframe[SHARED_COLUMNS].copy()


def build_dataset(
    *,
    presets: list[str],
    synthetic_runs_per_preset: int,
    seed: int,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = [_standardize_real(path) for path in REAL_INPUTS]
    synthetic_rows = 0
    synthetic_subjects: list[str] = []
    segment = 10_000
    for preset_name in presets:
        preset = get_preset(preset_name)
        for index in range(synthetic_runs_per_preset):
            run_seed = seed + len(synthetic_subjects)
            outputs = run_simulation(
                algorithm=ClinicalBaselineAlgorithm(),
                scenario=preset["scenario"],
                patient_config=preset["patient_config"],
                duration_minutes=preset["duration_minutes"],
                time_step=preset["time_step_minutes"],
                seed=run_seed,
                output_dir=output_root / preset_name / f"seed_{run_seed}",
                compare_baselines=False,
                export_audit=False,
                generate_report=False,
                physiology_variation_profile=preset.get("physiology_variation_profile"),
                physiology_variation_scale=float(preset.get("physiology_variation_scale", 1.0)),
            )
            subject_id = f"SIM_{preset_name}_{index:02d}"
            synthetic_subjects.append(subject_id)
            synthetic_frame = _standardize_simulation(
                outputs["results"],
                subject_id=subject_id,
                segment=segment,
            )
            synthetic_rows += len(synthetic_frame)
            frames.append(synthetic_frame)
            segment += 1
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["subject_id", "segment", "time_minutes"]).reset_index(drop=True)
    manifest = {
        "real_inputs": [str(path) for path in REAL_INPUTS],
        "presets": presets,
        "synthetic_runs_per_preset": synthetic_runs_per_preset,
        "synthetic_subject_count": len(synthetic_subjects),
        "synthetic_rows": synthetic_rows,
        "total_rows": len(dataset),
        "subject_count": int(dataset["subject_id"].nunique()),
        "columns": SHARED_COLUMNS,
    }
    return dataset, manifest


def main() -> None:
    args = parse_args()
    dataset, manifest = build_dataset(
        presets=_parse_csv(args.presets),
        synthetic_runs_per_preset=args.synthetic_runs_per_preset,
        seed=args.seed,
        output_root=args.output.parent / "synthetic_runs",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output), "manifest": str(args.manifest), **manifest}, indent=2))


if __name__ == "__main__":
    main()
