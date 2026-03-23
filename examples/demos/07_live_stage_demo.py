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

# These are the three knobs to point at on a fair stand.
# Swap the patient profile, rerun, and the poster + reports update automatically.
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
    print("Key outputs:")
    print(f"- Poster: {outputs['poster_png']}")
    print(f"- Jury guide: {outputs['jury_talk_track']}")
    print(f"- Live demo script: {outputs['live_demo_script']}")
    print(f"- Commands: {outputs['run_commands']}")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
