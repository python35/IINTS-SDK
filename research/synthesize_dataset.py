from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

import iints
from iints.core.algorithms.pid_controller import PIDController
from iints.validation import load_scenario
from iints.scenarios import ScenarioGeneratorConfig, generate_random_scenario
from iints.research.dataset import save_parquet, concat_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of synthetic runs")
    parser.add_argument("--duration", type=int, default=720, help="Duration minutes")
    parser.add_argument("--time-step", type=int, default=5, help="Time step minutes")
    parser.add_argument("--preset", type=str, default="baseline_t1d", help="Preset name")
    parser.add_argument("--scenario", type=Path, default=None, help="Scenario JSON path (optional)")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--seed", type=int, default=123, help="Base seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []

    if args.scenario:
        scenario_payload = load_scenario(args.scenario).model_dump()
    else:
        config = ScenarioGeneratorConfig(
            name="Synthetic Scenario",
            duration_minutes=args.duration,
            meal_count=3,
            exercise_count=1,
            sensor_error_count=0,
            seed=args.seed,
        )
        scenario_payload = generate_random_scenario(config)

    for idx in range(args.runs):
        seed = args.seed + idx
        outputs = iints.run_simulation(
            algorithm=PIDController(),
            scenario=scenario_payload,
            patient_config="default_patient",
            duration_minutes=args.duration,
            time_step=args.time_step,
            seed=seed,
            compare_baselines=False,
            export_audit=False,
            generate_report=False,
            output_dir=None,
        )
        df = outputs["results"].copy()
        df["run_id"] = outputs.get("run_id", f"run_{idx}")
        frames.append(df)

    combined = concat_runs(frames)
    save_parquet(combined, args.output)
    print(f"Saved synthetic dataset to {args.output}")


if __name__ == "__main__":
    main()
