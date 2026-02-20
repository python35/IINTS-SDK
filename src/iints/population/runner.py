"""
Population Runner — IINTS-AF
==============================
Runs N virtual patients through the simulator in parallel using
``concurrent.futures.ProcessPoolExecutor``.  Each worker loads
the algorithm from a file path (same pattern as ``run-parallel``
in the CLI) so that all algorithm classes are safely picklable.

After all patients complete, aggregate clinical metrics and safety
indices are computed with 95 % confidence intervals.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("iints.population")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatientResult:
    """Result for a single patient in the population."""
    patient_index: int
    profile: Dict[str, Any]
    clinical_metrics: Dict[str, float]
    safety_index: Dict[str, Any]
    safety_report: Dict[str, Any]
    terminated_early: bool = False
    error: Optional[str] = None


@dataclass
class PopulationResult:
    """Aggregate result for the entire population run."""
    n_patients: int
    patient_results: List[PatientResult]
    aggregate_metrics: Dict[str, Any]
    aggregate_safety: Dict[str, Any]
    summary_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Worker function  (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _load_algorithm_from_path(algo_path: Path):
    """Load an InsulinAlgorithm subclass from a .py file (worker-safe)."""
    import iints
    module_name = algo_path.stem
    spec = importlib.util.spec_from_file_location(module_name, algo_path)
    if spec is None:
        raise ImportError(f"Could not load module spec for {algo_path}")
    module = importlib.util.module_from_spec(spec)
    module.iints = iints  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    if spec.loader:
        spec.loader.exec_module(module)
    else:
        raise ImportError(f"Could not load module loader for {algo_path}")
    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            return obj()
    raise ImportError(f"No subclass of InsulinAlgorithm found in {algo_path}")


def _run_single_patient(job: Dict[str, Any]) -> Dict[str, Any]:
    """Run one patient simulation.  Must be picklable (top-level function)."""
    import iints
    from iints.core.patient.models import PatientModel
    from iints.core.simulator import Simulator
    from iints.core.safety import SafetyConfig
    from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
    from iints.analysis.safety_index import compute_safety_index
    from iints.validation import build_stress_events

    patient_index: int = job["patient_index"]
    patient_config: Dict[str, Any] = job["patient_config"]
    algo_path: Optional[str] = job.get("algo_path")
    algo_class_name: Optional[str] = job.get("algo_class_name")
    stress_event_payloads: List[Dict] = job.get("stress_event_payloads", [])
    duration_minutes: int = job["duration_minutes"]
    time_step: int = job["time_step"]
    seed: int = job["seed"]
    safety_config_dict: Optional[Dict] = job.get("safety_config_dict")
    safety_weights: Optional[Dict[str, float]] = job.get("safety_weights")
    patient_model_type: str = job.get("patient_model_type", "custom")

    try:
        # --- Instantiate patient model ---
        if patient_model_type == "bergman":
            from iints.core.patient.bergman_model import BergmanPatientModel
            patient_model = BergmanPatientModel(**patient_config)
        else:
            patient_model = PatientModel(**patient_config)

        # --- Load algorithm ---
        if algo_path:
            algorithm_instance = _load_algorithm_from_path(Path(algo_path))
        elif algo_class_name:
            # Built-in algorithm by qualified class name
            module_path, class_name = algo_class_name.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            algo_cls = getattr(mod, class_name)
            algorithm_instance = algo_cls()
        else:
            raise ValueError("Either algo_path or algo_class_name must be provided")

        safety_config = SafetyConfig(**safety_config_dict) if safety_config_dict else SafetyConfig()

        simulator = Simulator(
            patient_model=patient_model,
            algorithm=algorithm_instance,
            time_step=time_step,
            seed=seed,
            safety_config=safety_config,
        )
        for event in build_stress_events(stress_event_payloads):
            simulator.add_stress_event(event)

        results_df, safety_report = simulator.run_batch(duration_minutes)

        # --- Clinical metrics ---
        calculator = ClinicalMetricsCalculator()
        clinical = calculator.calculate(
            glucose=results_df["glucose_actual_mgdl"],
            duration_hours=duration_minutes / 60.0,
        )

        # --- Safety index ---
        safety_idx = compute_safety_index(
            results_df=results_df,
            safety_report=safety_report,
            duration_minutes=duration_minutes,
            weights=safety_weights,
            time_step_minutes=float(time_step),
        )

        return {
            "patient_index": patient_index,
            "profile": patient_config,
            "clinical_metrics": clinical.to_dict(),
            "safety_index": safety_idx.to_dict(),
            "safety_report_summary": {
                "total_violations": safety_report.get("total_violations", 0),
                "bolus_interventions_count": safety_report.get("bolus_interventions_count", 0),
            },
            "terminated_early": safety_report.get("terminated_early", False),
            "error": None,
        }
    except Exception as exc:
        logger.warning("Patient %d failed: %s", patient_index, exc)
        return {
            "patient_index": patient_index,
            "profile": patient_config,
            "clinical_metrics": {},
            "safety_index": {},
            "safety_report_summary": {},
            "terminated_early": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# PopulationRunner
# ---------------------------------------------------------------------------

class PopulationRunner:
    """Runs a virtual patient population through the simulator in parallel."""

    def __init__(
        self,
        algo_path: Optional[Union[str, Path]] = None,
        algo_class_name: Optional[str] = None,
        scenario_payloads: Optional[List[Dict[str, Any]]] = None,
        duration_minutes: int = 720,
        time_step: int = 5,
        base_seed: int = 42,
        max_workers: Optional[int] = None,
        safety_config: Optional[Any] = None,
        safety_weights: Optional[Dict[str, float]] = None,
        patient_model_type: str = "custom",
    ):
        if algo_path is None and algo_class_name is None:
            raise ValueError("Provide either algo_path or algo_class_name")
        self.algo_path = str(algo_path) if algo_path else None
        self.algo_class_name = algo_class_name
        self.scenario_payloads = scenario_payloads or []
        self.duration_minutes = duration_minutes
        self.time_step = time_step
        self.base_seed = base_seed
        self.max_workers = max_workers
        self.safety_config = safety_config
        self.safety_weights = safety_weights
        self.patient_model_type = patient_model_type

    def run(self, profiles) -> PopulationResult:
        """Run all patients and return aggregated results."""
        from dataclasses import asdict
        from iints.core.safety import SafetyConfig

        safety_config_dict = asdict(self.safety_config) if self.safety_config else None

        jobs = []
        for i, profile in enumerate(profiles):
            jobs.append({
                "patient_index": i,
                "patient_config": profile.to_patient_config(),
                "algo_path": self.algo_path,
                "algo_class_name": self.algo_class_name,
                "stress_event_payloads": self.scenario_payloads,
                "duration_minutes": self.duration_minutes,
                "time_step": self.time_step,
                "seed": self.base_seed + i,
                "safety_config_dict": safety_config_dict,
                "safety_weights": self.safety_weights,
                "patient_model_type": self.patient_model_type,
            })

        raw_results: List[Dict[str, Any]] = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(_run_single_patient, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_map):
                raw_results.append(future.result())

        raw_results.sort(key=lambda r: r["patient_index"])

        patient_results: List[PatientResult] = []
        summary_rows: List[Dict[str, Any]] = []

        for r in raw_results:
            patient_results.append(PatientResult(
                patient_index=r["patient_index"],
                profile=r["profile"],
                clinical_metrics=r.get("clinical_metrics", {}),
                safety_index=r.get("safety_index", {}),
                safety_report=r.get("safety_report_summary", {}),
                terminated_early=r.get("terminated_early", False),
                error=r.get("error"),
            ))

            row: Dict[str, Any] = {"patient_index": r["patient_index"]}
            row.update(r.get("profile", {}))
            row.update(r.get("clinical_metrics", {}))
            si = r.get("safety_index", {})
            if si:
                row["safety_index_score"] = si.get("safety_index")
                row["safety_grade"] = si.get("grade")
            row["terminated_early"] = r.get("terminated_early", False)
            row["error"] = r.get("error", "")
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        return PopulationResult(
            n_patients=len(profiles),
            patient_results=patient_results,
            aggregate_metrics=_compute_aggregate_metrics(summary_df),
            aggregate_safety=_compute_aggregate_safety(summary_df),
            summary_df=summary_df,
        )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_METRICS_OF_INTEREST = [
    "tir_70_180", "tir_below_70", "tir_below_54",
    "tir_above_180", "mean_glucose", "cv", "gmi",
]


def _compute_aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Population-level stats with 95 % CI (percentile method)."""
    agg: Dict[str, Any] = {}
    for metric in _METRICS_OF_INTEREST:
        if metric not in df.columns:
            continue
        values = df[metric].dropna().values
        if len(values) == 0:
            continue
        agg[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return agg


def _compute_aggregate_safety(df: pd.DataFrame) -> Dict[str, Any]:
    """Population-level safety aggregates."""
    agg: Dict[str, Any] = {}
    if "safety_index_score" in df.columns:
        scores = df["safety_index_score"].dropna().values
        if len(scores) > 0:
            agg["safety_index"] = {
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "ci_lower": float(np.percentile(scores, 2.5)),
                "ci_upper": float(np.percentile(scores, 97.5)),
            }
    if "safety_grade" in df.columns:
        grade_counts = df["safety_grade"].value_counts().to_dict()
        agg["grade_distribution"] = {str(k): int(v) for k, v in grade_counts.items()}
    if "terminated_early" in df.columns:
        agg["early_termination_rate"] = float(df["terminated_early"].mean())
    return agg
