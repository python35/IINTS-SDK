from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

from iints.api.base_algorithm import InsulinAlgorithm
from iints.core.patient.models import PatientModel
from iints.core.patient.profile import PatientProfile
from iints.core.simulator import Simulator
from iints.core.safety import SafetyConfig
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
from iints.utils.run_io import (
    build_run_metadata,
    build_run_manifest,
    generate_run_id,
    maybe_sign_manifest,
    resolve_output_dir,
    resolve_seed,
    write_json,
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
    patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = "default_patient",
    duration_minutes: int = 720,
    time_step: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    compare_baselines: bool = True,
    export_audit: bool = True,
    generate_report: bool = True,
    safety_config: Optional[SafetyConfig] = None,
    predictor: Optional[object] = None,
) -> Dict[str, Any]:
    """
    One-line simulation runner with audit + report + baseline comparison.
    """
    algorithm_instance = algorithm() if isinstance(algorithm, type) else algorithm
    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)
    output_path = resolve_output_dir(output_dir, run_id)

    patient_params = _resolve_patient_config(patient_config)
    patient_model = PatientModel(**patient_params)

    scenario_payload = _resolve_scenario_payloads(scenario)
    stress_event_payloads = scenario_payload.get("stress_events", []) if scenario_payload else []
    effective_safety_config = safety_config or SafetyConfig()

    simulator = Simulator(
        patient_model=patient_model,
        algorithm=algorithm_instance,
        time_step=time_step,
        seed=resolved_seed,
        safety_config=effective_safety_config,
        predictor=predictor,
    )
    for event in build_stress_events(stress_event_payloads):
        simulator.add_stress_event(event)

    results_df, safety_report = simulator.run_batch(duration_minutes)

    outputs: Dict[str, Any] = {
        "results": results_df,
        "safety_report": safety_report,
        "run_id": run_id,
        "output_dir": str(output_path),
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

    config_payload: Dict[str, Any] = {
        "algorithm": {
            "class": f"{algorithm_instance.__class__.__module__}.{algorithm_instance.__class__.__name__}",
            "metadata": algorithm_instance.get_algorithm_metadata().to_dict(),
        },
        "patient_config": patient_params,
        "scenario": scenario_payload,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
        "compare_baselines": compare_baselines,
        "export_audit": export_audit,
        "generate_report": generate_report,
        "safety_config": asdict(effective_safety_config),
    }
    config_path = output_path / "config.json"
    write_json(config_path, config_payload)
    outputs["config_path"] = str(config_path)

    run_metadata = build_run_metadata(run_id, resolved_seed, config_payload, output_path)
    run_metadata_path = output_path / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)
    outputs["run_metadata_path"] = str(run_metadata_path)

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

    if "performance_report" in safety_report:
        profiling_path = output_path / "profiling.json"
        write_json(profiling_path, safety_report["performance_report"])
        outputs["profiling_path"] = str(profiling_path)

    manifest_files = {
        "config": config_path,
        "run_metadata": run_metadata_path,
        "results_csv": results_csv,
    }
    if "report_pdf" in outputs:
        manifest_files["report_pdf"] = Path(outputs["report_pdf"])
    if "audit" in outputs:
        audit_paths = outputs["audit"]
        manifest_files["audit_summary"] = Path(audit_paths.get("summary", ""))
    if "baseline_files" in outputs:
        baseline_files = outputs["baseline_files"]
        manifest_files["baseline_json"] = Path(baseline_files.get("json", ""))
        manifest_files["baseline_csv"] = Path(baseline_files.get("csv", ""))
    if "profiling_path" in outputs:
        manifest_files["profiling"] = Path(outputs["profiling_path"])

    run_manifest = build_run_manifest(output_path, manifest_files)
    run_manifest_path = output_path / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)
    outputs["run_manifest_path"] = str(run_manifest_path)
    signature_path = maybe_sign_manifest(run_manifest_path)
    if signature_path:
        outputs["run_manifest_signature"] = str(signature_path)

    return outputs


def run_full(
    algorithm: Union[InsulinAlgorithm, type],
    scenario: Optional[Union[str, Path, Dict[str, Any]]] = None,
    patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = "default_patient",
    duration_minutes: int = 720,
    time_step: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    enable_profiling: bool = True,
    safety_config: Optional[SafetyConfig] = None,
    predictor: Optional[object] = None,
) -> Dict[str, Any]:
    """
    One-line runner that always exports results + audit + PDF + baseline comparison.
    """
    algorithm_instance = algorithm() if isinstance(algorithm, type) else algorithm
    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)
    output_path = resolve_output_dir(output_dir, run_id)

    patient_params = _resolve_patient_config(patient_config)
    patient_model = PatientModel(**patient_params)

    scenario_payload = _resolve_scenario_payloads(scenario)
    stress_event_payloads = scenario_payload.get("stress_events", []) if scenario_payload else []
    effective_safety_config = safety_config or SafetyConfig()

    simulator = Simulator(
        patient_model=patient_model,
        algorithm=algorithm_instance,
        time_step=time_step,
        seed=resolved_seed,
        enable_profiling=enable_profiling,
        safety_config=effective_safety_config,
        predictor=predictor,
    )
    for event in build_stress_events(stress_event_payloads):
        simulator.add_stress_event(event)

    results_df, safety_report = simulator.run_batch(duration_minutes)

    outputs: Dict[str, Any] = {
        "results": results_df,
        "safety_report": safety_report,
        "run_id": run_id,
        "output_dir": str(output_path),
    }

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

    config_payload: Dict[str, Any] = {
        "algorithm": {
            "class": f"{algorithm_instance.__class__.__module__}.{algorithm_instance.__class__.__name__}",
            "metadata": algorithm_instance.get_algorithm_metadata().to_dict(),
        },
        "patient_config": patient_params,
        "scenario": scenario_payload,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
        "compare_baselines": True,
        "export_audit": True,
        "generate_report": True,
        "enable_profiling": enable_profiling,
        "safety_config": asdict(effective_safety_config),
    }
    config_path = output_path / "config.json"
    write_json(config_path, config_payload)
    outputs["config_path"] = str(config_path)

    run_metadata = build_run_metadata(run_id, resolved_seed, config_payload, output_path)
    run_metadata_path = output_path / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)
    outputs["run_metadata_path"] = str(run_metadata_path)

    results_csv = output_path / "results.csv"
    results_df.to_csv(results_csv, index=False)
    outputs["results_csv"] = str(results_csv)

    outputs["audit"] = simulator.export_audit_trail(results_df, output_dir=str(output_path / "audit"))
    outputs["baseline_files"] = write_baseline_comparison(comparison, output_path / "baseline")

    report_path = output_path / "clinical_report.pdf"
    generator = ClinicalReportGenerator()
    generator.generate_pdf(results_df, safety_report, str(report_path))
    outputs["report_pdf"] = str(report_path)

    if "performance_report" in safety_report:
        profiling_path = output_path / "profiling.json"
        write_json(profiling_path, safety_report["performance_report"])
        outputs["profiling_path"] = str(profiling_path)

    manifest_files = {
        "config": config_path,
        "run_metadata": run_metadata_path,
        "results_csv": results_csv,
        "report_pdf": report_path,
    }
    if "audit" in outputs:
        audit_paths = outputs["audit"]
        manifest_files["audit_summary"] = Path(audit_paths.get("summary", ""))
    if "baseline_files" in outputs:
        baseline_files = outputs["baseline_files"]
        manifest_files["baseline_json"] = Path(baseline_files.get("json", ""))
        manifest_files["baseline_csv"] = Path(baseline_files.get("csv", ""))
    if "profiling_path" in outputs:
        manifest_files["profiling"] = Path(outputs["profiling_path"])

    run_manifest = build_run_manifest(output_path, manifest_files)
    run_manifest_path = output_path / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)
    outputs["run_manifest_path"] = str(run_manifest_path)
    signature_path = maybe_sign_manifest(run_manifest_path)
    if signature_path:
        outputs["run_manifest_signature"] = str(signature_path)

    return outputs


# ---------------------------------------------------------------------------
# Population (Monte Carlo) evaluation
# ---------------------------------------------------------------------------

def run_population(
    algo_path: Optional[Union[str, Path]] = None,
    algo_class_name: Optional[str] = None,
    n_patients: int = 100,
    scenario: Optional[Union[str, Path, Dict[str, Any]]] = None,
    patient_config: Union[str, Path, Dict[str, Any], PatientProfile] = "default_patient",
    duration_minutes: int = 720,
    time_step: int = 5,
    seed: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    max_workers: Optional[int] = None,
    safety_config: Optional[SafetyConfig] = None,
    safety_weights: Optional[Dict[str, float]] = None,
    patient_model_type: str = "custom",
    population_cv: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Run a Monte Carlo population evaluation.

    Generates *n_patients* virtual patients with physiological variation
    around a base profile, runs each through the simulator in parallel,
    and returns aggregate statistics with 95 % confidence intervals for
    TIR, hypo-risk, and the IINTS Safety Index.
    """
    from iints.population.generator import PopulationConfig, PopulationGenerator
    from iints.population.runner import PopulationRunner
    from iints.analysis.population_report import PopulationReportGenerator

    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)
    output_path = resolve_output_dir(output_dir, run_id)

    # --- Resolve base patient profile ---
    patient_params = _resolve_patient_config(patient_config)
    base_profile = PatientProfile(
        isf=patient_params.get("insulin_sensitivity", 50.0),
        icr=patient_params.get("carb_factor", 10.0),
        basal_rate=patient_params.get("basal_insulin_rate", 0.8),
        initial_glucose=patient_params.get("initial_glucose", 120.0),
        insulin_action_duration=patient_params.get("insulin_action_duration", 300.0),
        dawn_phenomenon_strength=patient_params.get("dawn_phenomenon_strength", 0.0),
        dawn_start_hour=patient_params.get("dawn_start_hour", 4.0),
        dawn_end_hour=patient_params.get("dawn_end_hour", 8.0),
        glucose_decay_rate=patient_params.get("glucose_decay_rate", 0.002),
        glucose_absorption_rate=patient_params.get("glucose_absorption_rate", 0.03),
        insulin_peak_time=patient_params.get("insulin_peak_time", 75.0),
        meal_mismatch_epsilon=patient_params.get("meal_mismatch_epsilon", 1.0),
    )

    pop_config = PopulationConfig(
        n_patients=n_patients,
        base_profile=base_profile,
        seed=resolved_seed,
    )
    if population_cv:
        for param_name, cv_value in population_cv.items():
            if param_name in pop_config.parameter_distributions:
                pop_config.parameter_distributions[param_name].cv = cv_value

    generator = PopulationGenerator(pop_config)
    profiles = generator.generate()

    # --- Resolve scenario ---
    scenario_payload = _resolve_scenario_payloads(scenario)
    stress_event_payloads = (
        scenario_payload.get("stress_events", []) if scenario_payload else []
    )

    # --- Run population ---
    runner = PopulationRunner(
        algo_path=algo_path,
        algo_class_name=algo_class_name,
        scenario_payloads=stress_event_payloads,
        duration_minutes=duration_minutes,
        time_step=time_step,
        base_seed=resolved_seed,
        max_workers=max_workers,
        safety_config=safety_config,
        safety_weights=safety_weights,
        patient_model_type=patient_model_type,
    )
    result = runner.run(profiles)

    # --- Save outputs ---
    summary_csv = output_path / "population_summary.csv"
    result.summary_df.to_csv(summary_csv, index=False)

    population_report = {
        "run_id": run_id,
        "n_patients": n_patients,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
        "patient_model_type": patient_model_type,
        "aggregate_metrics": result.aggregate_metrics,
        "aggregate_safety": result.aggregate_safety,
    }
    write_json(output_path / "population_report.json", population_report)

    # --- PDF report ---
    report_generator = PopulationReportGenerator()
    report_pdf_path = output_path / "population_report.pdf"
    report_generator.generate_pdf(
        summary_df=result.summary_df,
        aggregate_metrics=result.aggregate_metrics,
        aggregate_safety=result.aggregate_safety,
        output_path=str(report_pdf_path),
    )

    return {
        "result": result,
        "run_id": run_id,
        "output_dir": str(output_path),
        "population_summary_csv": str(summary_csv),
        "population_report": population_report,
        "report_pdf": str(report_pdf_path),
    }
