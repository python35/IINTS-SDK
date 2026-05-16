from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.data.realism_reference import RealismReferenceProfile, get_realism_reference
from iints.data.realism_validator import RealismReport, validate_realism_dataset
from iints.highlevel import run_simulation
from iints.presets import get_preset
from iints.validation import load_patient_config_by_name


VERDICT_RANK = {
    "likely_unrealistic": 0,
    "needs_review": 1,
    "likely_realistic": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a simulator patient profile against an empirical real-data realism envelope."
    )
    parser.add_argument(
        "--preset",
        default="realistic_reference_day",
        help="Preset whose scenario and base patient profile should be calibrated.",
    )
    parser.add_argument(
        "--reference",
        default="free_living_t1d",
        help="Realism reference profile id.",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,42,99",
        help="Comma-separated deterministic seeds used for robust calibration.",
    )
    parser.add_argument(
        "--initial-glucose",
        default="135,140,145,150,155",
        help="Candidate initial glucose values in mg/dL.",
    )
    parser.add_argument(
        "--dawn-strength",
        default="0,4,8,10,12",
        help="Candidate dawn-phenomenon strengths in mg/dL per hour.",
    )
    parser.add_argument(
        "--meal-mismatch",
        default="0.95,1.0,1.05,1.1",
        help="Candidate meal mismatch multipliers.",
    )
    parser.add_argument(
        "--glucose-decay",
        default="0.02,0.03,0.04",
        help="Candidate homeostatic glucose-decay rates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked candidates to retain in the JSON report.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/simulator_calibration.json"),
        help="JSON output path for the calibration report.",
    )
    parser.add_argument(
        "--best-profile-out",
        type=Path,
        default=None,
        help="Optional YAML path for the best calibrated patient profile.",
    )
    return parser.parse_args()


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _standardize(results):
    return results.rename(
        columns={
            "time_minutes": "timestamp",
            "glucose_actual_mgdl": "glucose",
            "carb_intake_grams": "carbs",
            "delivered_insulin_units": "insulin",
        }
    )[["timestamp", "glucose", "carbs", "insulin"]]


def build_candidate_grid(
    base_profile: dict[str, Any],
    *,
    initial_glucose_values: list[float],
    dawn_strength_values: list[float],
    meal_mismatch_values: list[float],
    glucose_decay_values: list[float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for initial_glucose, dawn_strength, meal_mismatch, glucose_decay in itertools.product(
        initial_glucose_values,
        dawn_strength_values,
        meal_mismatch_values,
        glucose_decay_values,
    ):
        profile = dict(base_profile)
        profile.update(
            {
                "initial_glucose": float(initial_glucose),
                "dawn_phenomenon_strength": float(dawn_strength),
                "meal_mismatch_epsilon": float(meal_mismatch),
                "glucose_decay_rate": float(glucose_decay),
            }
        )
        candidates.append(profile)
    return candidates


def normalized_reference_distance(
    report: RealismReport,
    reference_profile: RealismReferenceProfile,
) -> float:
    """Measure how far a report sits from empirical reference medians."""
    distances: list[float] = []
    comparisons = {comparison.metric_key: comparison for comparison in report.reference_comparisons}
    for metric_key, band in reference_profile.metric_bands.items():
        comparison = comparisons.get(metric_key)
        if comparison is None or comparison.observed_value is None:
            continue
        target_span = max(float(band.target_high - band.target_low), 1e-9)
        half_span = max(target_span / 2.0, 1e-9)
        distances.append(abs(float(comparison.observed_value) - float(band.median)) / half_span)
    if not distances:
        return float("inf")
    return mean(distances)


def _row_for_candidate(
    *,
    candidate_id: int,
    patient_profile: dict[str, Any],
    preset: dict[str, Any],
    seed: int,
    reference_profile: RealismReferenceProfile,
    output_dir: Path,
) -> dict[str, Any]:
    outputs = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=preset["scenario"],
        patient_config=patient_profile,
        duration_minutes=preset["duration_minutes"],
        time_step=preset["time_step_minutes"],
        seed=seed,
        output_dir=output_dir,
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )
    dataframe = _standardize(outputs["results"])
    report = validate_realism_dataset(dataframe, reference=reference_profile)
    failed_checks = sum(1 for check in report.checks if check.status == "failed")
    warning_checks = sum(1 for check in report.checks if check.status == "warning")
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "verdict": report.verdict,
        "realism_score": round(report.realism_score, 4),
        "normalized_reference_distance": round(
            normalized_reference_distance(report, reference_profile),
            4,
        ),
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "mean_glucose_mgdl": report.metrics.get("mean_glucose_mgdl"),
        "sd_mgdl": report.metrics.get("sd_mgdl"),
        "cv_pct": report.metrics.get("cv_pct"),
        "tir_70_180_pct": report.metrics.get("tir_70_180_pct"),
        "glucose_range_mgdl": report.metrics.get("glucose_range_mgdl"),
    }


def summarize_candidate(
    *,
    candidate_id: int,
    patient_profile: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    verdict_counts = Counter(str(row["verdict"]) for row in rows)
    return {
        "candidate_id": candidate_id,
        "patient_profile": patient_profile,
        "runs": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "likely_realistic_runs": verdict_counts.get("likely_realistic", 0),
        "mean_verdict_rank": round(
            mean(VERDICT_RANK[str(row["verdict"])] for row in rows),
            4,
        ),
        "mean_realism_score": round(mean(float(row["realism_score"]) for row in rows), 4),
        "mean_normalized_reference_distance": round(
            mean(float(row["normalized_reference_distance"]) for row in rows),
            4,
        ),
        "failed_checks": sum(int(row["failed_checks"]) for row in rows),
        "warning_checks": sum(int(row["warning_checks"]) for row in rows),
    }


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Prefer robust realism first, then closeness to the empirical center."""
    return (
        float(candidate["likely_realistic_runs"]),
        float(candidate["mean_verdict_rank"]),
        float(candidate["mean_realism_score"]),
        -float(candidate["failed_checks"]),
        -float(candidate["warning_checks"]),
        -float(candidate["mean_normalized_reference_distance"]),
    )


def build_report(
    *,
    preset_name: str,
    reference: str,
    seeds: list[int],
    initial_glucose_values: list[float],
    dawn_strength_values: list[float],
    meal_mismatch_values: list[float],
    glucose_decay_values: list[float],
    top_k: int,
) -> dict[str, Any]:
    preset = get_preset(preset_name)
    reference_profile = get_realism_reference(reference)
    base_profile = load_patient_config_by_name(str(preset["patient_config"])).model_dump()
    candidates = build_candidate_grid(
        base_profile,
        initial_glucose_values=initial_glucose_values,
        dawn_strength_values=dawn_strength_values,
        meal_mismatch_values=meal_mismatch_values,
        glucose_decay_values=glucose_decay_values,
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="iints_calibration_") as tmp:
        root = Path(tmp)
        for candidate_id, patient_profile in enumerate(candidates, start=1):
            for seed in seeds:
                rows.append(
                    _row_for_candidate(
                        candidate_id=candidate_id,
                        patient_profile=patient_profile,
                        preset=preset,
                        seed=seed,
                        reference_profile=reference_profile,
                        output_dir=root / f"candidate_{candidate_id}" / f"seed_{seed}",
                    )
                )

    summaries = [
        summarize_candidate(
            candidate_id=candidate_id,
            patient_profile=patient_profile,
            rows=[row for row in rows if row["candidate_id"] == candidate_id],
        )
        for candidate_id, patient_profile in enumerate(candidates, start=1)
    ]
    ranked = sorted(summaries, key=candidate_rank_key, reverse=True)
    best = ranked[0]

    return {
        "preset": preset_name,
        "reference": reference,
        "seeds": seeds,
        "search_space": {
            "initial_glucose": initial_glucose_values,
            "dawn_phenomenon_strength": dawn_strength_values,
            "meal_mismatch_epsilon": meal_mismatch_values,
            "glucose_decay_rate": glucose_decay_values,
        },
        "candidate_count": len(candidates),
        "best_candidate": best,
        "top_candidates": ranked[: max(top_k, 1)],
        "runs": rows,
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        preset_name=args.preset,
        reference=args.reference,
        seeds=_parse_csv_ints(args.seeds),
        initial_glucose_values=_parse_csv_floats(args.initial_glucose),
        dawn_strength_values=_parse_csv_floats(args.dawn_strength),
        meal_mismatch_values=_parse_csv_floats(args.meal_mismatch),
        glucose_decay_values=_parse_csv_floats(args.glucose_decay),
        top_k=args.top_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Saved simulator calibration: {args.out}")
    print(json.dumps(report["best_candidate"], indent=2))

    if args.best_profile_out is not None:
        args.best_profile_out.parent.mkdir(parents=True, exist_ok=True)
        args.best_profile_out.write_text(
            yaml.safe_dump(report["best_candidate"]["patient_profile"], sort_keys=False)
        )
        print(f"Saved best patient profile: {args.best_profile_out}")


if __name__ == "__main__":
    main()
