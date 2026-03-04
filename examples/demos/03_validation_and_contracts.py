#!/usr/bin/env python3
"""
Demo 03 — Validation Profiles + Formal Safety Contract Verification

What this demonstrates:
1) Running a deterministic simulation and collecting outputs.
2) Evaluating the run against a validation profile.
3) Verifying the deterministic safety contract on an input grid.

Run:
    PYTHONPATH=src python3 examples/demos/03_validation_and_contracts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

import iints
from iints.validation import (
    evaluate_run,
    load_contract_spec,
    load_validation_profiles,
    verify_safety_contract,
)


def build_demo_scenario() -> Dict[str, Any]:
    return {
        "scenario_name": "Validation and Contract Demo",
        "schema_version": "1.1",
        "scenario_version": "1.0",
        "description": "Scenario used to demonstrate validation and contract verification.",
        "stress_events": [
            {"start_time": 90, "event_type": "meal", "value": 45, "duration": 50, "absorption_delay_minutes": 10},
            {"start_time": 300, "event_type": "meal", "value": 65, "duration": 80, "absorption_delay_minutes": 15},
            {"start_time": 540, "event_type": "exercise", "value": 0.5, "duration": 40},
        ],
    }


def main() -> None:
    output_dir = Path("results/demo_03_validation")
    outputs = iints.run_simulation(
        algorithm=iints.ConstantDoseAlgorithm(dose=0.2),
        scenario=build_demo_scenario(),
        patient_config="default_patient",
        duration_minutes=720,
        time_step=5,
        seed=42,
        output_dir=output_dir,
        compare_baselines=False,
        export_audit=True,
        generate_report=False,
    )

    profiles = load_validation_profiles()
    profile = profiles["research_default"]
    validation_report = evaluate_run(
        outputs["results"],
        profile=profile,
        safety_report=outputs.get("safety_report", {}),
        duration_minutes=720,
    )

    contract = load_contract_spec()
    contract_report = verify_safety_contract(
        contract,
        glucose_values=np.arange(50.0, 160.1, 5.0).tolist(),
        trend_values=np.arange(-5.0, 3.1, 0.5).tolist(),
        proposed_doses=[0.0, 0.5, 1.0, 2.0, 4.0],
    )

    artifacts = {
        "validation_report": validation_report.to_dict(),
        "contract_report": contract_report.to_dict(),
        "run_paths": {
            "output_dir": outputs.get("output_dir"),
            "results_csv": outputs.get("results_csv"),
            "audit_summary": outputs.get("audit", {}).get("summary") if isinstance(outputs.get("audit"), dict) else None,
        },
    }

    report_path = output_dir / "validation_and_contracts.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(artifacts, indent=2))

    print(json.dumps(
        {
            "validation_status": "PASS" if validation_report.passed else "FAIL",
            "validation_checks": f"{validation_report.required_checks_passed}/{validation_report.required_checks_total}",
            "validation_score": round(validation_report.score, 2),
            "contract_status": "PASS" if contract_report.passed else "FAIL",
            "contract_violations": len(contract_report.violations),
            "artifact_json": str(report_path),
        },
        indent=2,
    ))
    print("\nDemo 03 complete.")


if __name__ == "__main__":
    main()
