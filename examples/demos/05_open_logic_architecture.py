#!/usr/bin/env python3
"""
Demo 05 — Open Logic Dual-Guard Architecture

This script is presentation-oriented: it produces concrete evidence for the
three architecture layers in one run.

Layer 1: InputValidator (plausibility + fail-soft)
Layer 2: Intelligence (optional predictor advisory signal)
Layer 3: Independent Supervisor (deterministic safety override)

Run:
    PYTHONPATH=src python3 examples/demos/05_open_logic_architecture.py

Optional predictor:
    PYTHONPATH=src python3 examples/demos/05_open_logic_architecture.py \
      --predictor models/hupa_finetuned_v2/predictor.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import iints
from iints import RunawayAIAlgorithm, SafetyConfig


def build_architecture_scenario() -> Dict[str, Any]:
    # Includes two sensor artifacts + meal stress so all layers are exercised.
    return {
        "scenario_name": "Open Logic Architecture Demo",
        "schema_version": "1.1",
        "scenario_version": "1.0",
        "description": "Deliberately stressful run to demonstrate validator + supervisor + predictor telemetry.",
        "stress_events": [
            {"start_time": 50, "event_type": "meal", "value": 45, "duration": 50, "absorption_delay_minutes": 10},
            {"start_time": 180, "event_type": "sensor_error", "value": -90, "duration": 15},
            {"start_time": 240, "event_type": "meal", "value": 65, "duration": 70, "absorption_delay_minutes": 15},
            {"start_time": 320, "event_type": "sensor_error", "value": 120, "duration": 10},
        ],
    }


def resolve_predictor(path_value: Optional[str]) -> Optional[object]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Predictor checkpoint not found: {path}")
    from iints.research.predictor import load_predictor_service

    return load_predictor_service(path)


def summarize_layers(results_df: pd.DataFrame, outputs: Dict[str, Any], predictor_loaded: bool) -> Dict[str, Any]:
    layer1_count = int(
        results_df.get("input_validator_fail_soft", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    layer3_overrides = int(results_df.get("safety_triggered", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    layer3_reasons = (
        results_df.loc[results_df.get("safety_triggered", False) == True, "safety_reason"].value_counts().head(5).to_dict()
        if "safety_triggered" in results_df.columns and "safety_reason" in results_df.columns
        else {}
    )
    gate_counts = (
        results_df.get("predictor_gate_reason", pd.Series(dtype=str)).fillna("none").value_counts().to_dict()
        if "predictor_gate_reason" in results_df.columns
        else {}
    )
    return {
        "layer_1_input_validator": {
            "fail_soft_corrections": layer1_count,
            "evidence": "results.csv -> input_validator_fail_soft",
        },
        "layer_2_intelligence": {
            "predictor_loaded": predictor_loaded,
            "predictor_gate_reason_counts": gate_counts,
            "evidence": "results.csv -> predictor_* columns",
        },
        "layer_3_independent_supervisor": {
            "safety_overrides": layer3_overrides,
            "top_safety_reasons": layer3_reasons,
            "evidence": "results.csv -> safety_triggered/safety_reason",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an architecture evidence demo.")
    parser.add_argument("--predictor", type=str, default=None, help="Optional predictor checkpoint (.pt).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo_05_open_logic"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = resolve_predictor(args.predictor)

    # Strict enough to make supervisor interventions visible during stress.
    safety = SafetyConfig(
        max_insulin_per_bolus=2.0,
        max_iob=4.5,
        predictor_uncertainty_gate_enabled=True,
        predictor_uncertainty_max_std_mgdl=35.0,
        predictor_ood_gate_enabled=True,
        predictor_ood_zscore_threshold=4.0,
        predictor_ood_max_feature_fraction=0.25,
    )

    outputs = iints.run_simulation(
        algorithm=RunawayAIAlgorithm(),
        scenario=build_architecture_scenario(),
        patient_config="default_patient",
        duration_minutes=420,
        time_step=5,
        seed=args.seed,
        output_dir=args.output_dir,
        compare_baselines=False,
        export_audit=True,
        generate_report=True,
        safety_config=safety,
        predictor=predictor,
    )

    results_df: pd.DataFrame = outputs["results"]
    layer_summary = summarize_layers(results_df, outputs, predictor_loaded=predictor is not None)
    evidence = {
        "output_dir": outputs.get("output_dir"),
        "results_csv": outputs.get("results_csv"),
        "report_pdf": outputs.get("report_pdf"),
        "audit_jsonl": (outputs.get("audit") or {}).get("jsonl"),
        "audit_summary": (outputs.get("audit") or {}).get("summary"),
        "layers": layer_summary,
    }

    artifact_path = Path(outputs["output_dir"]) / "open_logic_summary.json"
    artifact_path.write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2, default=str))
    print("\nDemo 05 complete.")


if __name__ == "__main__":
    main()
