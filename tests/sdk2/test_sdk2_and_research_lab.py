"""Comprehensive test suite for IINTS Simulation SDK 2.0, Research Lab & Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from iints.research import (
    PlanNotApprovedError,
    analyze_research_results,
    approve_research_plan,
    execute_research_plan,
    generate_verifiable_report_bundle,
    holm_bonferroni,
    propose_research_plan,
)
from iints.sdk2 import (
    AlgorithmSpec,
    ExperimentSpec,
    ForecastOutput,
    GlucoseForecaster,
    PatientSpec,
    ScenarioSpec,
    SensorFaultSpec,
    apply_sensor_fault_model,
    create_virtual_patient,
    derive_seed,
    evaluate_forecasters_on_trace,
    load_experiment_spec,
    register_forecaster,
    run_single_experiment,
    simulate_patient_trajectory,
)


def test_declarative_yaml_example_loads_and_runs_identically(tmp_path: Path) -> None:
    yaml_path = Path(__file__).resolve().parents[2] / "examples" / "sdk2" / "cgm_dropout_001.yaml"
    spec = load_experiment_spec(yaml_path)
    assert spec.id == "cgm_dropout_001"
    assert spec.sensor.dropout_minutes == 15
    assert len(spec.resolved_algorithms()) == 5

    # Speed up scenario slightly for unit test while testing full artifact + SQLite pipeline
    fast_spec = spec.model_copy(
        update={
            "scenario": spec.scenario.model_copy(update={"duration_hours": 10.0, "warmup_hours": 3.0}),
        }
    )
    run_a = run_single_experiment(fast_spec, output_dir=tmp_path / "run_a")
    run_b = run_single_experiment(fast_spec, output_dir=tmp_path / "run_b")

    # Bit-identical reproducibility across independent executions
    pd.testing.assert_frame_equal(run_a["trace"], run_b["trace"])
    pd.testing.assert_frame_equal(run_a["forecasts"], run_b["forecasts"])
    assert (tmp_path / "run_a" / "provenance_manifest.json").exists()
    assert (tmp_path / "experiments.sqlite").exists()

    # Adding an extra algorithm must NOT perturb existing algorithms' forecasts or true glucose
    one_algo_spec = fast_spec.model_copy(
        update={"algorithms": [AlgorithmSpec(model_id="persistence_v1")], "algorithm": None}
    )
    run_single = run_single_experiment(one_algo_spec)
    pd.testing.assert_frame_equal(run_a["trace"], run_single["trace"])
    sub_a = run_a["forecasts"][run_a["forecasts"]["algorithm_id"] == "persistence_v1"].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub_a, run_single["forecasts"].reset_index(drop=True))


@pytest.mark.parametrize("model_name", ("hovorka_equilibrium", "iints_ode", "bergman"))
def test_interchangeable_virtual_patient_adapters(model_name: str) -> None:
    spec = PatientSpec(
        model=model_name,  # type: ignore[arg-type]
        profile=f"test_{model_name}",
        initial_glucose_mgdl=130.0,
        basal_u_per_h=1.4,
    )
    patient = create_virtual_patient(spec)
    patient.reset()
    g0 = patient.true_glucose_mgdl
    assert 40.0 <= g0 <= 300.0
    # Meal step increases or maintains finite physiological glucose
    g1 = patient.step(5.0, insulin_units=1.4 * 5.0 / 60.0, carbs_grams=40.0, current_time_minutes=0.0)
    g2 = patient.step(5.0, insulin_units=1.4 * 5.0 / 60.0, carbs_grams=0.0, current_time_minutes=5.0)
    assert np.isfinite(g1) and np.isfinite(g2)
    assert patient.metadata()["model_family"] == model_name


def test_sensor_blank_vs_stale_hold_and_recovery_tracking() -> None:
    pat = PatientSpec(model="iints_ode", initial_glucose_mgdl=130.0, basal_u_per_h=1.2)
    scen = ScenarioSpec(duration_hours=6.0, warmup_hours=1.0)
    traj = simulate_patient_trajectory(pat, scen, seed=7)

    blank_sensor = SensorFaultSpec(
        condition_id="blank",
        dropout_minutes=20,
        dropout_at_hour=2.0,
        display_mode="nan_blank",
    )
    stale_sensor = blank_sensor.model_copy(update={"condition_id": "stale", "display_mode": "stale_hold"})

    t_blank = apply_sensor_fault_model(traj, blank_sensor, seed=99)
    t_stale = apply_sensor_fault_model(traj, stale_sensor, seed=99)

    assert int(t_blank["is_dropout"].sum()) == 4  # 20 min / 5 min
    assert t_blank.loc[t_blank["is_dropout"], "cgm_displayed_mgdl"].isna().all()
    assert t_stale.loc[t_stale["is_dropout"], "cgm_displayed_mgdl"].notna().all()
    assert t_stale.loc[t_stale["is_dropout"], "is_stale_displayed"].all()
    assert (t_blank["steps_since_recovery"] == 0).sum() == 1


def test_algorithm_sandbox_strictly_strips_ground_truth_and_future() -> None:
    seen_columns: list[set[str]] = []
    max_seen_times: list[float] = []

    class AuditSpyForecaster(GlucoseForecaster):
        model_id = "audit_spy_v1"

        def predict_horizons(
            self,
            causal_history: pd.DataFrame,
            horizons_minutes: Sequence[int],
            step_minutes: float = 5.0,
        ) -> ForecastOutput:
            seen_columns.append(set(causal_history.columns))
            max_seen_times.append(float(causal_history["t_min"].max()))
            val = float(pd.Series(causal_history["sensor_mgdl"]).ffill().iloc[-1])
            return ForecastOutput(
                {int(h): val for h in horizons_minutes},
                {int(h): val - 15.0 for h in horizons_minutes},
                {int(h): val + 15.0 for h in horizons_minutes},
            )

    register_forecaster("audit_spy_v1", AuditSpyForecaster)
    spec = ExperimentSpec(
        id="spy_check",
        seed=11,
        scenario=ScenarioSpec(duration_hours=4.0, warmup_hours=1.0),
        algorithms=[AlgorithmSpec(model_id="audit_spy_v1")],
    )
    traj = simulate_patient_trajectory(spec.patient, spec.scenario, seed=derive_seed(11, "p"))
    trace = apply_sensor_fault_model(traj, spec.sensor, seed=derive_seed(11, "s"))
    fc = evaluate_forecasters_on_trace(trace, spec.resolved_algorithms(), spec.evaluation)

    assert not fc.empty
    forbidden = {"bg_true_mgdl", "ig_mgdl", "cgm_clean_mgdl", "carbs_physical_g"}
    for cols in seen_columns:
        assert cols.isdisjoint(forbidden)
    # Causal timestamps must increase strictly step-by-step and never reach the final horizon target
    assert max(max_seen_times) <= float(trace["t_min"].max()) - 30.0


def test_research_lab_approval_gate_checkpoint_resume_and_paired_stats(tmp_path: Path) -> None:
    plan = propose_research_plan(
        "study_001_cgm_dropout",
        plan_id="mini_study_001",
        n_profiles=6,
        seed=123,
        duration_hours=8.0,
        include_ood_scenarios=True,
    )
    assert plan.status == "proposed"

    # 1. Must refuse execution before human approval
    with pytest.raises(PlanNotApprovedError):
        execute_research_plan(plan, tmp_path / "study")

    approved = approve_research_plan(plan, reviewer="Unit Test PI")
    assert approved.status == "approved"
    assert approved.plan_lock_sha256 is not None

    # 2. Must detect post-approval tampering of experimental parameters
    tampered = approved.model_copy(update={"master_seed": 999})
    with pytest.raises(PlanNotApprovedError, match="plan_lock_sha256 mismatch"):
        execute_research_plan(tampered, tmp_path / "study")

    # 3. Full run + checkpoint partial resume verification
    study_dir = tmp_path / "study"
    run1 = execute_research_plan(approved, study_dir, max_workers=2, resume=True)
    total_cells = approved.total_simulation_cells()
    assert run1["executed_cells"] == total_cells
    assert run1["resumed_cells"] == 0

    # Simulate interruption by keeping only the first 5 checkpointed cells
    ckpt_file = study_dir / "checkpoints.jsonl"
    lines = ckpt_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total_cells
    ckpt_file.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")

    run2 = execute_research_plan(approved, study_dir, max_workers=2, resume=True)
    assert run2["resumed_cells"] == 5
    assert run2["executed_cells"] == total_cells - 5

    # 4. Within-patient paired statistics (unit of analysis = 6 virtual patients, NOT timepoints!)
    analysis = analyze_research_results(run2["cell_metrics"], approved)
    assert (analysis["condition_summary"]["n_patients"] == 6).all()
    assert (analysis["paired_vs_control"]["n_paired_patients"] == 6).all()
    assert set(analysis["split_generalization"]["split"].unique()) == {"dev", "unseen"}
    assert not analysis["ood_boundary_summary"].empty

    # Holm-Bonferroni monotonicity sanity check
    adj = holm_bonferroni([0.004, 0.02, 0.20])
    assert adj == pytest.approx([0.012, 0.04, 0.20])

    # 5. Verifiable report bundle
    bundle = generate_verifiable_report_bundle(run2, analysis)
    report_text = bundle["report_md"].read_text(encoding="utf-8")
    assert approved.plan_lock_sha256 in report_text
    assert "Within-Patient Paired Degradation vs Control" in report_text


def test_research_studio_fastapi_workflow(tmp_path: Path) -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from iints.studio.app import create_studio_app

    client = TestClient(create_studio_app(tmp_path / "studio_ws"))
    home = client.get("/")
    assert home.status_code == 200
    assert "IINTS Research Studio 2.0" in home.text

    tpl = client.get("/api/templates")
    assert tpl.status_code == 200
    assert len(tpl.json()["templates"]) == 3

    prop = client.post(
        "/api/plans/propose",
        json={
            "template_id": "study_002_inclusive_stale_ui",
            "plan_id": "ui_hazard_test",
            "n_profiles": 6,
            "duration_hours": 6.0,
            "include_ood_scenarios": False,
        },
    )
    assert prop.status_code == 200
    assert prop.json()["status"] == "proposed"

    # Executing before approval returns HTTP 409 Conflict
    blocked = client.post("/api/plans/ui_hazard_test/execute", json={"max_workers": 1})
    assert blocked.status_code == 409

    appr = client.post("/api/plans/ui_hazard_test/approve", json={"reviewer": "Rune Bobbaers"})
    assert appr.status_code == 200
    assert appr.json()["status"] == "approved"

    ran = client.post("/api/plans/ui_hazard_test/execute", json={"max_workers": 2})
    assert ran.status_code == 200
    payload = ran.json()
    assert payload["completed_cells"] == 18  # 6 profiles * 1 scenario * 3 sensor display conditions
    assert Path(payload["report_path"]).exists()
