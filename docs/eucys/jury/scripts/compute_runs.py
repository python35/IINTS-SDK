"""Produce the computed evidence for the EUCYS jury folder.

Every figure and number in 02_PUMP_MECHANICS, 03_PHYSIOLOGICAL_MODEL,
04_AI_ALGORITHMS and the single-run illustration in 08_FAILURE_CASES comes from
these runs. Nothing is hand-set: the controllers are the ones shipped in
iints.core.algorithms, the scenarios are the shipped presets, the seed is fixed
so a reviewer can reproduce the CSVs byte for byte.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path("/Users/runebobbaers/Documents/IINTS-AF-SSD/IINTS-SDK-main")
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

import iints
import iints.presets as presets
from iints.core.clinical_metrics import ClinicalMetricsCalculator

OUT = pathlib.Path("jury_build/fresh/runs")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

CONTROLLERS = {
    "clinical_baseline": ("clinical_baseline", "ClinicalBaselineAlgorithm"),
    "mpc": ("mpc_controller", "MPCController"),
    "pid": ("pid_controller", "PIDController"),
    "fixed_basal_bolus": ("fixed_basal_bolus", "FixedBasalBolus"),
}

JOBS = [
    # (run label, preset name, controller key)
    ("baseline_t1d__clinical_baseline", "baseline_t1d", "clinical_baseline"),
    ("baseline_t1d__mpc", "baseline_t1d", "mpc"),
    ("baseline_t1d__pid", "baseline_t1d", "pid"),
    ("baseline_t1d__fixed_basal_bolus", "baseline_t1d", "fixed_basal_bolus"),
    ("hypo_prone_night__clinical_baseline", "hypo_prone_night", "clinical_baseline"),
]


def instantiate(key: str):
    module_name, class_name = CONTROLLERS[key]
    module = importlib.import_module(f"iints.core.algorithms.{module_name}")
    return getattr(module, class_name)()


def main() -> None:
    calculator = ClinicalMetricsCalculator()
    index_rows = []
    for label, preset_name, controller_key in JOBS:
        preset = presets.get_preset(preset_name)
        algorithm = instantiate(controller_key)
        with tempfile.TemporaryDirectory(prefix="jury_") as tmp:
            outputs = iints.run_simulation(
                algorithm=algorithm,
                scenario=preset["scenario"],
                patient_config=preset["patient_config"],
                duration_minutes=int(preset["duration_minutes"]),
                time_step=int(preset["time_step_minutes"]),
                seed=SEED,
                output_dir=pathlib.Path(tmp),
                compare_baselines=False,
                export_audit=False,
                generate_report=False,
            )
        frame = outputs["results"]
        frame.to_csv(OUT / f"{label}.csv", index=False)
        (OUT / f"{label}__safety_report.json").write_text(
            json.dumps(outputs.get("safety_report", {}), indent=1, default=str), encoding="utf-8"
        )

        glucose = frame["glucose_actual_mgdl"]
        metrics = calculator.calculate(glucose).to_dict()
        metrics.update(
            run=label,
            preset=preset_name,
            controller=controller_key,
            patient_config=preset["patient_config"],
            duration_minutes=int(preset["duration_minutes"]),
            seed=SEED,
            steps=int(len(frame)),
            total_insulin_units=float(frame["delivered_insulin_units"].sum()),
            total_carbs_grams=float(frame["carb_intake_grams"].sum()),
            safety_interventions=int(frame["safety_triggered"].astype(bool).sum()),
            min_glucose_mgdl=float(glucose.min()),
            max_glucose_mgdl=float(glucose.max()),
        )
        index_rows.append(metrics)
        print(f"done {label}: {len(frame)} steps, min {glucose.min():.1f}, max {glucose.max():.1f}", flush=True)

    table = pd.DataFrame(index_rows)
    lead = ["run", "preset", "controller", "patient_config", "seed", "steps"]
    table = table[lead + [c for c in table.columns if c not in lead]]
    table.to_csv(OUT / "run_metrics.csv", index=False)
    print(table[["run", "steps", "total_insulin_units", "min_glucose_mgdl", "max_glucose_mgdl"]].to_string(index=False))


if __name__ == "__main__":
    main()
