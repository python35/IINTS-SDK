#!/usr/bin/env python3
"""
Demo 02 — Dual-Guard Forecasting (Predictor + Deterministic Supervisor)

What this demonstrates:
1) Loading a trained predictor checkpoint as an advisory layer.
2) Enabling uncertainty and out-of-distribution (OOD) gates.
3) Inspecting predictor gate telemetry in simulation outputs.

Run:
    PYTHONPATH=src python3 examples/demos/02_dual_guard_predictor.py \
      --predictor models/hupa_finetuned_v2/predictor.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import iints
from iints import SafetyConfig, StandardPumpAlgorithm


def build_demo_scenario() -> Dict[str, Any]:
    """
    Mixed scenario with meals + temporary sensor error.

    The sensor_error event helps show why deterministic gates are useful:
    predictive signals are not trusted blindly.
    """
    return {
        "scenario_name": "Dual Guard Demo",
        "schema_version": "1.1",
        "scenario_version": "1.0",
        "description": "Dual-guard demo with meals and one sensor disturbance.",
        "stress_events": [
            {"start_time": 80, "event_type": "meal", "value": 55, "duration": 60, "absorption_delay_minutes": 10},
            {"start_time": 360, "event_type": "meal", "value": 70, "duration": 90, "absorption_delay_minutes": 20},
            {"start_time": 430, "event_type": "sensor_error", "value": -45, "duration": 25},
        ],
    }


def resolve_predictor(path_value: Optional[str]) -> Optional[object]:
    if not path_value:
        return None

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Predictor checkpoint not found: {path}")

    # Imported lazily so this demo can still run in environments without research extras.
    from iints.research.predictor import load_predictor_service

    return load_predictor_service(path)


def summarize_predictor_telemetry(results_df: Any) -> Dict[str, Any]:
    gate_counts = {}
    if "predictor_gate_reason" in results_df.columns:
        gate_counts = results_df["predictor_gate_reason"].fillna("none").value_counts().to_dict()

    ood_fraction = None
    if "predictor_in_distribution" in results_df.columns:
        mask = results_df["predictor_in_distribution"].fillna(True).astype(bool)
        ood_fraction = float((~mask).mean())

    mean_std = None
    if "predictor_uncertainty_std_mgdl" in results_df.columns:
        series = results_df["predictor_uncertainty_std_mgdl"].dropna()
        if len(series) > 0:
            mean_std = float(series.mean())

    return {
        "gate_reason_counts": gate_counts,
        "ood_fraction": ood_fraction,
        "mean_uncertainty_std_mgdl": mean_std,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dual-guard predictor demo.")
    parser.add_argument(
        "--predictor",
        type=str,
        default=None,
        help="Path to predictor checkpoint (.pt). If omitted, deterministic-only run is executed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/demo_02_dual_guard"),
        help="Directory for demo outputs.",
    )
    parser.add_argument("--duration-minutes", type=int, default=720, help="Simulation duration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = resolve_predictor(args.predictor)

    safety = SafetyConfig(
        predictor_uncertainty_gate_enabled=True,
        predictor_uncertainty_max_std_mgdl=35.0,
        predictor_ood_gate_enabled=True,
        predictor_ood_zscore_threshold=4.0,
        predictor_ood_max_feature_fraction=0.25,
    )

    outputs = iints.run_simulation(
        algorithm=StandardPumpAlgorithm(),
        scenario=build_demo_scenario(),
        patient_config="default_patient",
        duration_minutes=args.duration_minutes,
        time_step=5,
        seed=args.seed,
        output_dir=args.output_dir,
        compare_baselines=False,
        export_audit=True,
        generate_report=True,
        safety_config=safety,
        predictor=predictor,
    )

    summary = {
        "output_dir": outputs.get("output_dir"),
        "results_csv": outputs.get("results_csv"),
        "report_pdf": outputs.get("report_pdf"),
        "audit_summary": outputs.get("audit", {}).get("summary") if isinstance(outputs.get("audit"), dict) else None,
        "predictor_loaded": predictor is not None,
        "predictor_telemetry": summarize_predictor_telemetry(outputs["results"]),
    }
    print(json.dumps(summary, indent=2, default=str))
    print("\nDemo 02 complete.")


if __name__ == "__main__":
    main()
