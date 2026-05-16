from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.data.realism_validator import validate_realism_dataset
from iints.highlevel import run_simulation
from iints.presets import get_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score simulator presets against an empirical real-data realism envelope."
    )
    parser.add_argument(
        "--presets",
        default="realistic_reference_day,baseline_t1d,free_living_t1d",
        help="Comma-separated preset names to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3,42,99",
        help="Comma-separated deterministic seeds.",
    )
    parser.add_argument(
        "--reference",
        default="free_living_t1d",
        help="Realism reference profile id.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path for the benchmark report.",
    )
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_text(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _standardize(results):
    return results.rename(
        columns={
            "time_minutes": "timestamp",
            "glucose_actual_mgdl": "glucose",
            "carb_intake_grams": "carbs",
            "delivered_insulin_units": "insulin",
        }
    )[["timestamp", "glucose", "carbs", "insulin"]]


def _row_for_run(
    *,
    preset_name: str,
    seed: int,
    reference: str,
    output_dir: Path,
) -> dict[str, Any]:
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
    dataframe = _standardize(outputs["results"])
    report = validate_realism_dataset(dataframe, reference=reference)
    statuses = {check.code: check.status for check in report.checks}
    return {
        "preset": preset_name,
        "seed": seed,
        "verdict": report.verdict,
        "realism_score": round(report.realism_score, 4),
        "mean_glucose_mgdl": report.metrics.get("mean_glucose_mgdl"),
        "sd_mgdl": report.metrics.get("sd_mgdl"),
        "cv_pct": report.metrics.get("cv_pct"),
        "glucose_range_mgdl": report.metrics.get("glucose_range_mgdl"),
        "quality_basics": statuses.get("quality_basics"),
        "meal_response": statuses.get("meal_response"),
        "causal_alignment": statuses.get("causal_alignment"),
        "reference_envelope": statuses.get("reference_envelope"),
    }


def build_report(
    *,
    presets: list[str],
    seeds: list[int],
    reference: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="iints_realism_") as tmp:
        root = Path(tmp)
        for preset_name in presets:
            for seed in seeds:
                rows.append(
                    _row_for_run(
                        preset_name=preset_name,
                        seed=seed,
                        reference=reference,
                        output_dir=root / preset_name / f"seed_{seed}",
                    )
                )

    by_preset: dict[str, dict[str, Any]] = {}
    for preset_name in presets:
        subset = [row for row in rows if row["preset"] == preset_name]
        verdict_counts = Counter(row["verdict"] for row in subset)
        by_preset[preset_name] = {
            "runs": len(subset),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "mean_realism_score": round(
                sum(float(row["realism_score"]) for row in subset) / max(len(subset), 1),
                4,
            ),
        }

    return {
        "reference": reference,
        "presets": presets,
        "seeds": seeds,
        "summary": by_preset,
        "runs": rows,
    }


def main() -> None:
    args = parse_args()
    presets = _parse_csv_text(args.presets)
    seeds = _parse_csv_ints(args.seeds)
    report = build_report(presets=presets, seeds=seeds, reference=args.reference)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Saved realism benchmark: {args.out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
