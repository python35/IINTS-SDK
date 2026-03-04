#!/usr/bin/env python3
"""
Demo 04 — Scenario Bank Scorecard

What this demonstrates:
1) Running one algorithm across multiple bundled presets.
2) Evaluating each run with the same validation profile.
3) Generating a compact CSV/JSON scorecard for comparison.

Run:
    PYTHONPATH=src python3 examples/demos/04_scenario_bank_scorecard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import iints
from iints.validation import evaluate_run, load_validation_profiles


def load_presets() -> List[Dict[str, Any]]:
    # Read bundled presets from the installed package data.
    import sys

    if sys.version_info >= (3, 9):
        from importlib.resources import files

        content = files("iints.presets").joinpath("presets.json").read_text()
    else:
        from importlib import resources

        content = resources.read_text("iints.presets", "presets.json")
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("presets.json must contain a list of presets")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small scenario-bank scorecard demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo_04_scorecard"),
        help="Directory for scorecard outputs.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="research_default",
        help="Validation profile id.",
    )
    parser.add_argument(
        "--max-presets",
        type=int,
        default=3,
        help="Maximum number of presets to run (for quick demos).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_validation_profiles()
    if args.profile not in profiles:
        raise ValueError(f"Unknown validation profile: {args.profile}")
    profile = profiles[args.profile]

    presets = load_presets()[: max(1, args.max_presets)]
    rows: List[Dict[str, Any]] = []

    for index, preset in enumerate(presets):
        preset_name = str(preset.get("name", f"preset_{index}"))
        scenario = preset.get("scenario")
        if not isinstance(scenario, dict):
            continue

        run_output = args.output_dir / preset_name
        outputs = iints.run_simulation(
            algorithm=iints.StandardPumpAlgorithm(),
            scenario=scenario,
            patient_config=str(preset.get("patient_config", "default_patient")),
            duration_minutes=int(preset.get("duration_minutes", 720)),
            time_step=int(preset.get("time_step_minutes", 5)),
            seed=args.seed + index,
            output_dir=run_output,
            compare_baselines=False,
            export_audit=True,
            generate_report=False,
        )
        report = evaluate_run(
            outputs["results"],
            profile=profile,
            safety_report=outputs.get("safety_report", {}),
            duration_minutes=int(preset.get("duration_minutes", 720)),
        )
        (run_output / "validation_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        rows.append(
            {
                "preset": preset_name,
                "profile": profile.profile_id,
                "passed": report.passed,
                "required_checks_passed": report.required_checks_passed,
                "required_checks_total": report.required_checks_total,
                "validation_score": report.score,
                "tir_70_180": report.metrics.get("tir_70_180"),
                "tir_below_70": report.metrics.get("tir_below_70"),
                "cv": report.metrics.get("cv"),
                "safety_index": report.metrics.get("safety_index"),
                "output_dir": str(run_output),
            }
        )

    if not rows:
        raise RuntimeError("No scorecard rows produced.")

    scorecard_df = pd.DataFrame(rows).sort_values(by=["passed", "validation_score"], ascending=[False, False])
    scorecard_csv = args.output_dir / "scorecard.csv"
    scorecard_json = args.output_dir / "scorecard.json"
    scorecard_df.to_csv(scorecard_csv, index=False)
    scorecard_json.write_text(json.dumps(rows, indent=2))

    print(json.dumps(
        {
            "rows": len(rows),
            "profile": profile.profile_id,
            "scorecard_csv": str(scorecard_csv),
            "scorecard_json": str(scorecard_json),
        },
        indent=2,
    ))
    print("\nDemo 04 complete.")


if __name__ == "__main__":
    main()
