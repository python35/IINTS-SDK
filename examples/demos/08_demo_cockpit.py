#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if SDK_SRC.exists():
    sys.path.insert(0, str(SDK_SRC))

from iints.analysis.demo_cockpit import build_demo_cockpit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the visual IINTS demo cockpit HTML app.")
    parser.add_argument("--output-dir", default="results/demo_cockpit", help="Directory where the demo cockpit bundle should be written.")
    parser.add_argument("--patient-config", default="default_patient", help="Packaged patient profile name or path to a YAML config.")
    parser.add_argument("--duration", type=int, default=360, help="Simulation duration in minutes for each scenario.")
    parser.add_argument("--time-step", type=int, default=5, help="Simulation step size in minutes.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI-ready artifact generation.")
    args = parser.parse_args()

    outputs = build_demo_cockpit(
        output_dir=args.output_dir,
        patient_config=args.patient_config,
        duration_minutes=args.duration,
        time_step=args.time_step,
        seed=args.seed,
        prepare_ai=not args.skip_ai,
    )

    print("IINTS Demo Cockpit complete.")
    print(f"Open in browser: {outputs['html_path']}")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
