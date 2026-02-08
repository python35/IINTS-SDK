from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.patient.models import PatientModel
from iints.core.patient.profile import PatientProfile
from iints.core.simulator import Simulator
from iints.analysis.baseline import run_baseline_comparison, write_baseline_comparison
from iints.analysis.reporting import ClinicalReportGenerator
from iints.validation import (
    build_stress_events,
    load_scenario,
    scenario_to_payloads,
    validate_patient_config_dict,
    validate_scenario_dict,
    load_patient_config_by_name,
)


def _resolve_patient_config(patient_config: Union[str, Path, Dict[str, Any], PatientProfile]) -> Dict[str, Any]:
    if isinstance(patient_config, PatientProfile):
        return validate_patient_config_dict(patient_config.to_patient_config()).model_dump()
    if isinstance(patient_config, dict):
        return validate_patient_config_dict(patient_config).model_dump()
    patient_config_path = Path(patient_config)
    if patient_config_path.is_file():
        data = yaml.safe_load(patient_config_path.read_text())
        return validate_patient_config_dict(data).model_dump()
    # Treat as a named config in the packaged directory.
    return load_patient_config_by_name(str(patient_config)).model_dump()


def _resolve_scenario_payloads(
    scenario: Optional[Union[str, Path, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if scenario is None:
        return None
    if isinstance(scenario, dict):
        scenario_model = validate_scenario_dict(scenario)
        return scenario_model.model_dump()
    scenario_model = load_scenario(scenario)
    return scenario_model.model_dump()


def run_simulation(
    algorithm: Union[InsulinAlgorithm, type],
    scenario: Optional[Union[str, Path, Dict[str, Any]]] = None,
    patient_config: Union[str, Path, Dict[str, Any]] = "default_patient",
    duration_minutes: int = 720,
    time_step: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    compare_baselines: bool = True,
    export_audit: bool = True,
    generate_report: bool = True,
) -> Dict[str, Any]:
    """
    One-line simulation runner with audit + report + baseline comparison.
    """
    algorithm_instance = algorithm() if isinstance(algorithm, type) else algorithm
    patient_params = _resolve_patient_config(patient_config)
    patient_model = PatientModel(**patient_params)

    scenario_payload = _resolve_scenario_payloads(scenario)
    stress_event_payloads = scenario_payload.get("stress_events", []) if scenario_payload else []

    simulator = Simulator(
        patient_model=patient_model,
        algorithm=algorithm_instance,
        time_step=time_step,
        seed=seed,
    )
    for event in build_stress_events(stress_event_payloads):
        simulator.add_stress_event(event)

    results_df, safety_report = simulator.run_batch(duration_minutes)

    outputs: Dict[str, Any] = {
        "results": results_df,
        "safety_report": safety_report,
    }

    if compare_baselines:
        comparison = run_baseline_comparison(
            patient_params=patient_params,
            stress_event_payloads=stress_event_payloads,
            duration=duration_minutes,
            time_step=time_step,
            primary_label=algorithm_instance.get_algorithm_metadata().name,
            primary_results=results_df,
            primary_safety=safety_report,
            seed=seed,
        )
        safety_report["baseline_comparison"] = comparison
        outputs["baseline_comparison"] = comparison

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_csv = output_path / "results.csv"
        results_df.to_csv(results_csv, index=False)
        outputs["results_csv"] = str(results_csv)

        if export_audit:
            audit_paths = simulator.export_audit_trail(results_df, output_dir=str(output_path / "audit"))
            outputs["audit"] = audit_paths

        if compare_baselines:
            outputs["baseline_files"] = write_baseline_comparison(
                safety_report.get("baseline_comparison", {}),
                output_path / "baseline",
            )

        if generate_report:
            report_path = output_path / "clinical_report.pdf"
            generator = ClinicalReportGenerator()
            generator.generate_pdf(results_df, safety_report, str(report_path))
            outputs["report_pdf"] = str(report_path)

    return outputs


def run_full(
    algorithm: Union[InsulinAlgorithm, type],
    scenario: Optional[Union[str, Path, Dict[str, Any]]] = None,
    patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = "default_patient",
    duration_minutes: int = 720,
    time_step: int = 5,
    seed: Optional[int] = None,
    output_dir: Union[str, Path] = "results/run_full",
) -> Dict[str, Any]:
    """
    One-line runner that always exports results + audit + PDF + baseline comparison.
    """
    return run_simulation(
        algorithm=algorithm,
        scenario=scenario,
        patient_config=patient_config,
        duration_minutes=duration_minutes,
        time_step=time_step,
        seed=seed,
        output_dir=output_dir,
        compare_baselines=True,
        export_audit=True,
        generate_report=True,
    )
