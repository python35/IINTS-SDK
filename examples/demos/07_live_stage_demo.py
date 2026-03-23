#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if SDK_SRC.exists():
    sys.path.insert(0, str(SDK_SRC))

from iints.analysis.booth_demo import build_booth_demo

# FAIR DEMO SCRIPT
# ----------------
# This is the file to show first on a booth stand.
# Point at the values below and explain:
# - PATIENT_CONFIG: swap the patient, rerun, and the whole bundle updates
# - OUTPUT_DIR: every artifact lands in one clean folder
# - DURATION_MINUTES / TIME_STEP_MINUTES: simulation horizon and resolution
# - SEED: same seed = same result, which keeps the demo reproducible
#
# The script then creates three stories:
# 1. Normal Run
# 2. Meal Stress Test
# 3. Supervisor Override
# After the run, open the poster + per-scenario folders.
PATIENT_CONFIG = "default_patient"
OUTPUT_DIR = "results/booth_demo_live"
DURATION_MINUTES = 360
TIME_STEP_MINUTES = 5
SEED = 42
PREPARE_AI = True

# Other good packaged patient configs to mention live:
# - patient_559_config
# - clinic_safe_baseline
# - clinic_safe_stress_meal
# - clinic_safe_hypo_prone
# - clinic_safe_hyper_challenge
# - clinic_safe_midnight
# - clinic_safe_pizza


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair-friendly live demo runner for the IINTS-AF SDK.",
    )
    parser.add_argument("--patient-config", default=PATIENT_CONFIG, help="Packaged patient profile name or path to a YAML config.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory where the demo bundle should be written.")
    parser.add_argument("--duration", type=int, default=DURATION_MINUTES, help="Simulation duration in minutes.")
    parser.add_argument("--time-step", type=int, default=TIME_STEP_MINUTES, help="Simulation time step in minutes.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic random seed.")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI-ready artifact generation.")
    args = parser.parse_args()

    outputs = build_booth_demo(
        output_dir=args.output_dir,
        patient_config=args.patient_config,
        duration_minutes=args.duration,
        time_step=args.time_step,
        seed=args.seed,
        prepare_ai=not args.skip_ai,
    )

    print("IINTS Live Stage Demo complete.")
    print(f"Patient config: {args.patient_config}")
    print("")
    print("What to show next:")
    print(f"1. Poster: {outputs['poster_png']}")
    print(f"2. Jury guide: {outputs['jury_talk_track']}")
    print(f"3. Live demo script: {outputs['live_demo_script']}")
    print(f"4. Commands: {outputs['run_commands']}")
    print("")
    print("Three scenario folders:")
    print(f"- Normal Run: {outputs['01_normal_run_dir']}")
    print(f"- Meal Stress Test: {outputs['02_meal_stress_test_dir']}")
    print(f"- Supervisor Override: {outputs['03_supervisor_override_dir']}")
    print("")
    print("Suggested booth flow:")
    print("- Show the constants at the top of this file.")
    print("- Run this script once.")
    print("- Open the poster and explain the three panels from left to right.")
    print("- If people want proof, open a scenario folder and show the CSV, PDF, and manifest.")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
