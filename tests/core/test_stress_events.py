from __future__ import annotations

from pathlib import Path

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.highlevel import run_simulation


def test_simulator_applies_physiological_stress_event(tmp_path: Path) -> None:
    patient_config = {
        "initial_glucose": 115.0,
        "basal_insulin_rate": 0.0,
        "insulin_sensitivity": 50.0,
        "carb_factor": 10.0,
        "glucose_decay_rate": 0.02,
        "glucose_absorption_rate": 0.03,
        "insulin_action_duration": 300.0,
        "insulin_peak_time": 75.0,
        "meal_mismatch_epsilon": 1.0,
    }
    base_scenario = {
        "scenario_name": "No Stress Baseline",
        "scenario_version": "1.0",
        "stress_events": [],
    }
    stress_scenario = {
        "scenario_name": "Illness Stress Block",
        "scenario_version": "1.0",
        "stress_events": [
            {"start_time": 30, "event_type": "stress", "value": 0.8, "duration": 60}
        ],
    }

    base = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=base_scenario,
        patient_config=patient_config,
        duration_minutes=120,
        time_step=5,
        seed=42,
        output_dir=tmp_path / "base",
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )["results"]
    stressed = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=stress_scenario,
        patient_config=patient_config,
        duration_minutes=120,
        time_step=5,
        seed=42,
        output_dir=tmp_path / "stress",
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )["results"]

    assert stressed["glucose_actual_mgdl"].max() > base["glucose_actual_mgdl"].max() + 25.0
    assert stressed["glucose_actual_mgdl"].iloc[-1] > base["glucose_actual_mgdl"].iloc[-1]
