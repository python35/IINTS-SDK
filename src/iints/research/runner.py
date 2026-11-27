"""C. Reproducible Batch Experiment Runner with Checkpointing, Resume & Parallelism.

Executes an **approved** ``ResearchPlan`` across all ``(patient, scenario,
sensor_condition, replicate)`` cells:
- Verifies ``plan.status == 'approved'`` and cryptographic ``plan_lock_sha256`` integrity.
- Simulates each ``(patient, scenario, replicate)`` ODE trajectory ONCE and shares
  that exact physiological ground truth across all matched sensor conditions (both
  4x faster and guaranteeing exact paired control!).
- Checkpoints every completed cell to ``checkpoints.jsonl`` so interrupted campaigns
  resume cleanly without recomputing finished cells.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from iints.research.designer import PlanNotApprovedError, ResearchPlan, save_research_plan
from iints.sdk2.contracts import PatientSpec, ScenarioSpec, SensorFaultSpec
from iints.sdk2.forecasters import evaluate_forecasters_on_trace
from iints.sdk2.reproducibility import (
    ExperimentStore,
    capture_provenance,
    derive_seed,
    read_dataframe_artifact,
    sha256_json,
    summarize_forecast_metrics,
    write_dataframe_artifact,
)
from iints.sdk2.sensors import apply_sensor_fault_model, simulate_patient_trajectory

__all__ = ["execute_research_plan", "load_study_cell_metrics"]


def _cell_key(profile_id: str, scenario_id: str, condition_id: str, replicate: int) -> str:
    return f"{profile_id}::{scenario_id}::{condition_id}::rep{replicate}"


def _run_patient_scenario_bundle(
    master_seed: int,
    patient: PatientSpec,
    scenario: ScenarioSpec,
    pending_conditions: List[SensorFaultSpec],
    plan: ResearchPlan,
    replicate: int,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Simulate true physiology ONCE, then evaluate all pending sensor conditions on it."""
    pat_seed = derive_seed(master_seed, "physiology", patient.profile, scenario.id, replicate)
    sen_seed = derive_seed(master_seed, "sensor_noise", patient.profile, scenario.id, replicate)

    traj = simulate_patient_trajectory(patient, scenario, seed=pat_seed)
    bundle_results: List[Tuple[str, List[Dict[str, Any]]]] = []

    for sensor in pending_conditions:
        key = _cell_key(patient.profile, scenario.id, sensor.condition_id, replicate)
        trace = apply_sensor_fault_model(traj, sensor, seed=sen_seed)
        fc = evaluate_forecasters_on_trace(
            trace,
            plan.algorithms,
            plan.evaluation,
            step_minutes=scenario.step_minutes,
        )
        metrics_df = summarize_forecast_metrics(fc)
        rows: List[Dict[str, Any]] = []
        for _, r in metrics_df.iterrows():
            rec = r.to_dict()
            rec.update(
                {
                    "cell_key": key,
                    "profile_id": patient.profile,
                    "simulator_model": patient.model,
                    "split": patient.split,
                    "scenario_id": scenario.id,
                    "regime": scenario.regime,
                    "condition_id": sensor.condition_id,
                    "dropout_minutes": int(sensor.dropout_minutes),
                    "display_mode": sensor.display_mode,
                    "replicate": int(replicate),
                }
            )
            rows.append(rec)
        bundle_results.append((key, rows))
    return bundle_results


def execute_research_plan(
    plan: ResearchPlan,
    output_dir: Union[str, Path],
    *,
    max_workers: int = 1,
    resume: bool = True,
    strict: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Execute an approved ``ResearchPlan`` with checkpointing and full provenance."""
    if plan.status != "approved":
        raise PlanNotApprovedError(
            f"ResearchPlan '{plan.plan_id}' has status='{plan.status}'. "
            "Review and call approve_research_plan(plan, reviewer=...) before executing."
        )
    expected_lock = sha256_json(plan.design_Payload_without_lock())
    if plan.plan_lock_sha256 != expected_lock:
        raise PlanNotApprovedError(
            "ResearchPlan was modified after approval (plan_lock_sha256 mismatch). "
            "Re-approve the modified plan or document a formal protocol amendment."
        )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = save_research_plan(plan, out_root / "research_plan_locked.yaml")
    ckpt_path = out_root / "checkpoints.jsonl"
    fail_path = out_root / "failures.jsonl"

    completed_rows: Dict[str, List[Dict[str, Any]]] = {}
    if resume and ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            completed_rows[item["cell_key"]] = item["rows"]

    resumed_count = len(completed_rows)
    total_planned = plan.total_simulation_cells()

    bundles: List[Tuple[PatientSpec, ScenarioSpec, List[SensorFaultSpec], int]] = []
    pending_cells = 0
    for rep in range(int(plan.replicates_per_profile)):
        for pat in plan.patients:
            for scen in plan.scenarios:
                missing_conds = [
                    cond
                    for cond in plan.sensor_conditions
                    if _cell_key(pat.profile, scen.id, cond.condition_id, rep) not in completed_rows
                ]
                if missing_conds:
                    bundles.append((pat, scen, missing_conds, rep))
                    pending_cells += len(missing_conds)

    failures: List[Dict[str, str]] = []
    executed_count = 0
    if bundles:
        with ckpt_path.open("a", encoding="utf-8") as ckpt_handle:
            if max_workers <= 1:
                for pat, scen, conds, rep in bundles:
                    try:
                        res_list = _run_patient_scenario_bundle(plan.master_seed, pat, scen, conds, plan, rep)
                        for key, rows in res_list:
                            completed_rows[key] = rows
                            ckpt_handle.write(json.dumps({"cell_key": key, "rows": rows}) + "\n")
                            executed_count += 1
                        ckpt_handle.flush()
                        if progress_callback is not None:
                            progress_callback(resumed_count + executed_count, total_planned)
                    except Exception as exc:
                        err = {"bundle": f"{pat.profile}::{scen.id}::rep{rep}", "error": f"{type(exc).__name__}: {exc}"}
                        failures.append(err)
                        with fail_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(err) + "\n")
                        if strict:
                            raise
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_map = {
                        pool.submit(_run_patient_scenario_bundle, plan.master_seed, pat, scen, conds, plan, rep): (
                            pat.profile,
                            scen.id,
                            rep,
                        )
                        for pat, scen, conds, rep in bundles
                    }
                    for fut in concurrent.futures.as_completed(future_map):
                        p_id, s_id, rep = future_map[fut]
                        try:
                            res_list = fut.result()
                            for key, rows in res_list:
                                completed_rows[key] = rows
                                ckpt_handle.write(json.dumps({"cell_key": key, "rows": rows}) + "\n")
                                executed_count += 1
                            ckpt_handle.flush()
                            if progress_callback is not None:
                                progress_callback(resumed_count + executed_count, total_planned)
                        except Exception as exc:
                            err = {"bundle": f"{p_id}::{s_id}::rep{rep}", "error": f"{type(exc).__name__}: {exc}"}
                            failures.append(err)
                            with fail_path.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(err) + "\n")
                            if strict:
                                raise

    flat_rows: List[Dict[str, Any]] = []
    for key in sorted(completed_rows.keys()):
        flat_rows.extend(completed_rows[key])
    cell_df = pd.DataFrame(flat_rows)
    cells_artifact = write_dataframe_artifact(cell_df, out_root / "cell_metrics")

    provenance = capture_provenance(
        plan.model_dump(mode="json"),
        artifact_paths=[plan_path, cells_artifact],
    )
    manifest_path = out_root / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    store = ExperimentStore(out_root.parent / "experiments.sqlite")
    store.record_experiment(
        plan.plan_id,
        provenance,
        plan.master_seed,
        out_root,
        {
            "total_cells_planned": total_planned,
            "cells_completed": len(completed_rows),
            "failures": len(failures),
            "approved_by": plan.approved_by,
            "plan_lock_sha256": plan.plan_lock_sha256,
        },
    )

    return {
        "plan": plan,
        "output_dir": out_root,
        "cell_metrics": cell_df,
        "cells_artifact": cells_artifact,
        "provenance": provenance,
        "failures": failures,
        "resumed_cells": resumed_count,
        "executed_cells": executed_count,
    }


def load_study_cell_metrics(study_dir: Union[str, Path]) -> pd.DataFrame:
    return read_dataframe_artifact(Path(study_dir) / "cell_metrics")
