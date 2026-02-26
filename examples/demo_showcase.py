#!/usr/bin/env python3
"""
IINTS-AF SDK Demo Showcase
=========================
Single script to demonstrate:
- Baseline simulation
- Chaos testing (insulin stacking + runaway AI)
- Safety supervisor interventions with Trust API callback
- Optional population study
- Edge energy estimate (device-agnostic)

Run:
  python3 examples/demo_showcase.py --output-dir results/demo_showcase
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

# Ensure local SDK source is used (avoids stale site-packages)
SDK_SRC = Path(__file__).resolve().parents[1] / "src"
if SDK_SRC.exists():
    import sys

    sys.path.insert(0, str(SDK_SRC))

from iints.core.simulator import Simulator
from iints.core.patient.models import PatientModel
from iints.core.safety import SafetyConfig
from iints.core.algorithms.pid_controller import PIDController
from iints.core.algorithms.mock_algorithms import RunawayAIAlgorithm, StackingAIAlgorithm
from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.analysis.edge_efficiency import estimate_energy_per_decision
from iints.utils.run_io import write_json
from iints.validation import load_scenario, scenario_to_payloads, build_stress_events
from iints.highlevel import run_population
from iints.research import load_predictor_service


def _print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _load_template_scenario(name: str) -> List[Dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "src" / "iints" / "templates" / "scenarios" / name
    if path.exists():
        model = load_scenario(path)
        return scenario_to_payloads(model)
    try:
        from importlib.resources import files

        data = files("iints.templates.scenarios").joinpath(name).read_text()
    except Exception:
        from importlib import resources

        data = resources.read_text("iints.templates.scenarios", name)
    payloads = json.loads(data).get("stress_events", [])
    return payloads


def _run_simulation(
    label: str,
    algorithm,
    duration_minutes: int,
    time_step: int,
    safety_config: SafetyConfig,
    output_dir: Path,
    predictor=None,
    stress_payloads: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safety_events: List[Dict[str, Any]] = []

    def on_safety_event(payload: Dict[str, Any]) -> None:
        safety_events.append(payload)
        msg = (
            f"[SafetyEvent] t={payload['time_minutes']}m | "
            f"AI {payload.get('ai_requested_units', 0):.2f}U -> "
            f"Approved {payload.get('supervisor_approved_units', 0):.2f}U | "
            f"{payload.get('safety_reason', '')}"
        )
        print(msg)

    patient = PatientModel(initial_glucose=140.0)
    sim = Simulator(
        patient_model=patient,
        algorithm=algorithm,
        time_step=time_step,
        seed=42,
        safety_config=safety_config,
        predictor=predictor,
        on_safety_event=on_safety_event,
    )

    if stress_payloads:
        for event in build_stress_events(stress_payloads):
            sim.add_stress_event(event)

    df, safety_report = sim.run_batch(duration_minutes)
    calc = ClinicalMetricsCalculator()
    metrics = calc.calculate(glucose=df["glucose_actual_mgdl"], duration_hours=duration_minutes / 60)

    # Persist outputs
    df_path = output_dir / f"{label}_results.csv"
    safety_path = output_dir / f"{label}_safety_report.json"
    metrics_path = output_dir / f"{label}_metrics.json"
    events_path = output_dir / f"{label}_safety_events.json"

    df.to_csv(df_path, index=False)
    write_json(safety_path, safety_report)
    write_json(metrics_path, metrics.to_dict())
    write_json(events_path, {"events": safety_events})

    summary = {
        "label": label,
        "results_csv": str(df_path),
        "safety_report": str(safety_path),
        "metrics": metrics.to_dict(),
        "safety_events_logged": len(safety_events),
        "early_termination": safety_report.get("early_termination", False),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IINTS-AF SDK demo showcase")
    parser.add_argument("--output-dir", default="results/demo_showcase", help="Where to store demo outputs")
    parser.add_argument("--duration", type=int, default=360, help="Simulation duration in minutes")
    parser.add_argument("--time-step", type=int, default=5, help="Simulation step in minutes")
    parser.add_argument("--population", action="store_true", help="Run population demo")
    parser.add_argument("--population-size", type=int, default=30, help="Population size")
    parser.add_argument("--device-power-watts", type=float, default=5.0, help="Power draw for energy estimate")
    parser.add_argument(
        "--predictor",
        default=None,
        help="Path to predictor.pt (enables AI predictor integration)",
    )
    parser.add_argument(
        "--export-onnx",
        default=None,
        help="If set, export predictor to ONNX at this path",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_section("IINTS-AF DEMO SHOWCASE")
    print(f"Output: {output_dir}")
    print(f"Duration: {args.duration} min | Step: {args.time_step} min")

    predictor = None
    if args.predictor:
        predictor_path = Path(args.predictor).expanduser().resolve()
        if predictor_path.exists():
            predictor = load_predictor_service(predictor_path)
            print(f"Predictor loaded: {predictor_path}")
        else:
            print(f"Predictor not found: {predictor_path}")

    if args.export_onnx:
        if predictor is None:
            print("ONNX export requested, but predictor not loaded. Provide --predictor.")
        else:
            try:
                import torch
                from iints.research.predictor import load_predictor

                model, cfg = load_predictor(Path(args.predictor))
                model.eval()
                dummy = torch.zeros(1, cfg["history_steps"], cfg["input_size"], dtype=torch.float32)
                onnx_path = Path(args.export_onnx).expanduser().resolve()
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                torch.onnx.export(
                    model,
                    dummy,
                    onnx_path,
                    input_names=["inputs"],
                    output_names=["predictions"],
                    dynamic_axes={"inputs": {0: "batch"}, "predictions": {0: "batch"}},
                    opset_version=17,
                )
                print(f"ONNX export saved: {onnx_path}")
            except Exception as exc:
                print(f"ONNX export failed: {exc}")

    # Energy estimate (device-agnostic)
    estimate = estimate_energy_per_decision(args.device_power_watts, latency_ms=0.002, decisions_per_day=288)
    energy_path = output_dir / "edge_energy_estimate.json"
    write_json(
        energy_path,
        {
            "device_power_watts": args.device_power_watts,
            "latency_ms": 0.002,
            "energy_joules": estimate.energy_joules,
            "energy_microjoules": estimate.energy_microjoules,
            "energy_joules_per_day": estimate.energy_joules_per_day,
            "energy_millijoules_per_day": estimate.energy_millijoules_per_day,
        },
    )
    print(f"Edge energy estimate saved: {energy_path}")

    safety_config = SafetyConfig()

    # Baseline run
    _print_section("1) Baseline Simulation (PID)")
    baseline_summary = _run_simulation(
        label="baseline_pid",
        algorithm=PIDController(),
        duration_minutes=args.duration,
        time_step=args.time_step,
        safety_config=safety_config,
        output_dir=output_dir,
        predictor=predictor,
        stress_payloads=_load_template_scenario("exercise_stress.json"),
    )
    print(json.dumps(baseline_summary, indent=2))

    # Chaos test: insulin stacking
    _print_section("2) Chaos Test - Insulin Stacking")
    stacking_summary = _run_simulation(
        label="chaos_stacking",
        algorithm=StackingAIAlgorithm(bolus_units=4.0, stack_steps=3),
        duration_minutes=args.duration,
        time_step=args.time_step,
        safety_config=safety_config,
        output_dir=output_dir,
        predictor=predictor,
        stress_payloads=_load_template_scenario("chaos_insulin_stacking.json"),
    )
    print(json.dumps(stacking_summary, indent=2))

    # Chaos test: runaway AI
    _print_section("3) Chaos Test - Runaway AI")
    runaway_summary = _run_simulation(
        label="chaos_runaway",
        algorithm=RunawayAIAlgorithm(max_bolus=5.0),
        duration_minutes=args.duration,
        time_step=args.time_step,
        safety_config=safety_config,
        output_dir=output_dir,
        predictor=predictor,
        stress_payloads=_load_template_scenario("chaos_runaway_ai.json"),
    )
    print(json.dumps(runaway_summary, indent=2))

    # Optional population run
    if args.population:
        _print_section("4) Population Study (Optional)")
        pop_result = run_population(
            algo_class_name="PIDController",
            algo_path=None,
            n_patients=args.population_size,
            duration_minutes=args.duration,
            time_step=args.time_step,
            output_dir=output_dir / "population",
        )
        write_json(output_dir / "population_summary.json", pop_result)
        print(f"Population outputs: {output_dir / 'population'}")

    _print_section("DONE")
    print("Demo outputs saved. Use the CSV + JSON files for quick plots or slides.")


if __name__ == "__main__":
    main()
