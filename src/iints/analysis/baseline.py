from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import pandas as pd

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.standard_pump_algo import StandardPumpAlgorithm
from iints.core.patient.models import PatientModel
from iints.core.simulator import Simulator
from iints.validation import build_stress_events


def compute_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    calculator = ClinicalMetricsCalculator()
    duration_hours = results_df["time_minutes"].max() / 60.0 if len(results_df) else 0.0
    metrics = calculator.calculate(
        glucose=results_df["glucose_actual_mgdl"],
        duration_hours=duration_hours,
    )
    return metrics.to_dict()


def run_baseline_comparison(
    patient_params: Dict[str, Any],
    stress_event_payloads: List[Dict[str, Any]],
    duration: int,
    time_step: int,
    primary_label: str,
    primary_results: pd.DataFrame,
    primary_safety: Dict[str, Any],
    compare_standard_pump: bool = True,
    seed: int | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    primary_metrics = compute_metrics(primary_results)
    rows.append(
        {
            "algorithm": primary_label,
            "tir_70_180": primary_metrics.get("tir_70_180", 0.0),
            "tir_below_70": primary_metrics.get("tir_below_70", 0.0),
            "tir_above_180": primary_metrics.get("tir_above_180", 0.0),
            "bolus_interventions": primary_safety.get("bolus_interventions_count", 0),
            "total_violations": primary_safety.get("total_violations", 0),
        }
    )

    baselines: List[Tuple[str, InsulinAlgorithm]] = [("Standard PID", PIDController())]
    if compare_standard_pump:
        baselines.append(("Standard Pump", StandardPumpAlgorithm()))

    for label, algo in baselines:
        patient_model = PatientModel(**patient_params)
        simulator = Simulator(
            patient_model=patient_model,
            algorithm=algo,
            time_step=time_step,
            seed=seed,
        )
        for event in build_stress_events(stress_event_payloads):
            simulator.add_stress_event(event)
        results_df, safety_report = simulator.run_batch(duration)
        metrics = compute_metrics(results_df)
        rows.append(
            {
                "algorithm": label,
                "tir_70_180": metrics.get("tir_70_180", 0.0),
                "tir_below_70": metrics.get("tir_below_70", 0.0),
                "tir_above_180": metrics.get("tir_above_180", 0.0),
                "bolus_interventions": safety_report.get("bolus_interventions_count", 0),
                "total_violations": safety_report.get("total_violations", 0),
            }
        )

    return {
        "reference": "Standard PID",
        "rows": rows,
    }


def write_baseline_comparison(comparison: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "baseline_comparison.json"
    csv_path = output_dir / "baseline_comparison.csv"
    json_path.write_text(json.dumps(comparison, indent=2))
    pd.DataFrame(comparison.get("rows", [])).to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}
