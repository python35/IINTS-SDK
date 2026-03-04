#!/usr/bin/env python3
"""
Demo 01 — Basic End-to-End Simulation

What this demonstrates:
1) How to implement a custom algorithm class.
2) How to run a deterministic simulation with stress events.
3) How to generate report artifacts (CSV + PDF + audit trail).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import iints
from iints import AlgorithmInput, AlgorithmMetadata, InsulinAlgorithm


class DemoRuleAlgorithm(InsulinAlgorithm):
    """
    A simple, transparent rule-based controller.

    This is intentionally conservative and easy to explain in a live demo.
    """

    def __init__(self) -> None:
        super().__init__()
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="DemoRuleAlgorithm",
                version="1.0.0",
                author="IINTS Demo",
                description="Simple glucose-threshold based dosing demo",
                algorithm_type="rule_based",
                requires_training=False,
            )
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        glucose = float(data.current_glucose)
        iob = float(data.insulin_on_board)

        # Small basal suggestion in nominal range.
        basal = 0.05
        correction = 0.0
        reason = "Nominal glucose range"

        # Conservative correction logic.
        if glucose > 180:
            correction = min((glucose - 120.0) / 90.0, 2.0)
            reason = "Hyperglycemia correction"
        elif glucose < 90:
            basal = 0.0
            reason = "Low glucose protection"

        # Basic anti-stacking behavior.
        if iob > 2.0:
            correction *= 0.5
            reason += " + IOB damping"

        total = max(0.0, basal + correction)
        return {
            "total_insulin_delivered": total,
            "basal_insulin": basal,
            "correction_bolus": correction,
            "primary_reason": reason,
        }


def build_demo_scenario() -> Dict[str, Any]:
    return {
        "scenario_name": "Demo Basic Scenario",
        "schema_version": "1.1",
        "scenario_version": "1.0",
        "description": "Single-day meal + exercise stress test",
        "stress_events": [
            {"start_time": 60, "event_type": "meal", "value": 45, "absorption_delay_minutes": 10, "duration": 60},
            {"start_time": 300, "event_type": "meal", "value": 70, "absorption_delay_minutes": 20, "duration": 90},
            {"start_time": 500, "event_type": "exercise", "value": 0.4, "duration": 45},
        ],
    }


def main() -> None:
    output_dir = Path("results/demo_01_basic")
    outputs = iints.run_simulation(
        algorithm=DemoRuleAlgorithm(),
        scenario=build_demo_scenario(),
        patient_config="default_patient",
        duration_minutes=720,
        time_step=5,
        seed=42,
        output_dir=output_dir,
        compare_baselines=True,
        export_audit=True,
        generate_report=True,
    )

    # Keep terminal output concise and easy to read.
    summary = {k: v for k, v in outputs.items() if k != "results"}
    print(json.dumps(summary, indent=2, default=str))
    print("\nDemo 01 complete.")
    print(f"Open: {output_dir}")


if __name__ == "__main__":
    main()
