#!/usr/bin/env python3
"""
Generate audit trail exports and a clinical PDF report in one run.
"""

import inspect
import json
import sys
from pathlib import Path

# Prefer local source tree over any installed package
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from iints.core.simulator import Simulator, StressEvent
from iints.core.patient.models import PatientModel
from iints.core.algorithms.pid_controller import PIDController
import iints


def main() -> None:
    patient = PatientModel(initial_glucose=120)
    algorithm = PIDController()

    simulator_kwargs = {"patient_model": patient, "algorithm": algorithm, "time_step": 5}
    # Critical hypo safety stop (override defaults if supported)
    critical_threshold = 40.0
    critical_duration_minutes = 30
    if "critical_glucose_threshold" in inspect.signature(Simulator.__init__).parameters:
        simulator_kwargs["critical_glucose_threshold"] = critical_threshold
    if "critical_glucose_duration_minutes" in inspect.signature(Simulator.__init__).parameters:
        simulator_kwargs["critical_glucose_duration_minutes"] = critical_duration_minutes
    if "enable_profiling" in inspect.signature(Simulator.__init__).parameters:
        simulator_kwargs["enable_profiling"] = True
    simulator = Simulator(**simulator_kwargs)
    print(
        "Critical hypo stop:",
        f"{critical_threshold} mg/dL for {critical_duration_minutes} minutes",
    )

    # Example meal
    simulator.add_stress_event(StressEvent(start_time=8 * 60, event_type='meal', value=60))

    results_df, safety_report = simulator.run_batch(duration_minutes=24 * 60)

    output_dir = Path("results/audit")
    if hasattr(simulator, "export_audit_trail"):
        audit_paths = simulator.export_audit_trail(results_df, output_dir=str(output_dir))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_columns = [
            "time_minutes",
            "glucose_actual_mgdl",
            "glucose_to_algo_mgdl",
            "algo_recommended_insulin_units",
            "delivered_insulin_units",
            "safety_reason",
            "safety_triggered",
            "supervisor_latency_ms",
        ]
        available = [c for c in audit_columns if c in results_df.columns]
        audit_df = results_df[available].copy()

        jsonl_path = output_dir / "audit_trail.jsonl"
        csv_path = output_dir / "audit_trail.csv"
        summary_path = output_dir / "audit_summary.json"

        audit_df.to_json(jsonl_path, orient="records", lines=True)
        audit_df.to_csv(csv_path, index=False)

        overrides = audit_df[audit_df.get("safety_triggered", False) == True] if "safety_triggered" in audit_df.columns else audit_df.iloc[0:0]
        reasons = overrides["safety_reason"].value_counts().to_dict() if "safety_reason" in overrides.columns else {}

        summary = {
            "total_steps": int(len(audit_df)),
            "total_overrides": int(len(overrides)),
            "top_reasons": reasons,
        }
        summary_path.write_text(json.dumps(summary, indent=2))

        audit_paths = {"jsonl": str(jsonl_path), "csv": str(csv_path), "summary": str(summary_path)}
    print("Audit exports:")
    for key, path in audit_paths.items():
        print(f"  {key}: {path}")

    report_output = "results/clinical_report.pdf"
    if hasattr(iints, "generate_report"):
        report_path = iints.generate_report(results_df, report_output, safety_report)
    else:
        from iints.analysis.reporting import ClinicalReportGenerator
        generator = ClinicalReportGenerator()
        report_path = generator.generate_pdf(results_df, safety_report, report_output)
    print(f"PDF report saved to: {report_path}")


if __name__ == "__main__":
    main()
