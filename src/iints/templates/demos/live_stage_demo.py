#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SDK_SRC = Path(__file__).resolve().parents[2] / "src"
if SDK_SRC.exists():
    sys.path.insert(0, str(SDK_SRC))

from iints.ai.prepare import prepare_ai_ready_artifacts
from iints.analysis.poster import generate_results_poster
from iints.core.algorithms.mock_algorithms import RunawayAIAlgorithm
from iints.core.algorithms.pid_controller import PIDController
from iints.highlevel import run_full

# FAIR DEMO SCRIPT
# ----------------
# This is the file to show first on a booth stand.
# It deliberately calls visible SDK features so you can explain the pipeline:
#
# 1. `run_full(...)`
#    Runs a scenario and writes a real run bundle: CSV, PDF report, audit trail,
#    baseline comparison, metadata, and a reproducibility manifest.
# 2. `generate_results_poster(...)`
#    Turns the three run bundles into one visual poster you can show to a jury.
# 3. `prepare_ai_ready_artifacts(...)`
#    Prepares the supervisor case for a local AI explanation, gated by MDMP.
#
# The knobs below are the first things to point at live.
PATIENT_CONFIG = "default_patient"
OUTPUT_DIR = "results/booth_demo_live"
DURATION_MINUTES = 360
TIME_STEP_MINUTES = 5
SEED = 42
PREPARE_AI = True

# Other good packaged patient configs to mention live:
# - patient_559_config
# - clinic_safe_baseline
# - clinic_safe_stress_meal
# - clinic_safe_hypo_prone
# - clinic_safe_hyper_challenge
# - clinic_safe_midnight
# - clinic_safe_pizza


ScenarioSpec = dict[str, Any]


SCENARIOS: list[ScenarioSpec] = [
    {
        "slug": "01_normal_run",
        "label": "Normal Run",
        "headline": "The controller keeps glucose in range during a calm day.",
        "algorithm_factory": PIDController,
        "scenario": {
            "scenario_name": "Live Demo - Normal Run",
            "schema_version": "1.1",
            "scenario_version": "1.0",
            "description": "One moderate meal to show stable closed-loop control.",
            "stress_events": [
                {
                    "start_time": 45,
                    "event_type": "meal",
                    "value": 35,
                    "reported_value": 35,
                    "absorption_delay_minutes": 10,
                    "duration": 45,
                }
            ],
        },
    },
    {
        "slug": "02_meal_stress_test",
        "label": "Meal Stress Test",
        "headline": "The same controller handles meals and exercise under stress.",
        "algorithm_factory": PIDController,
        "scenario": {
            "scenario_name": "Live Demo - Meal Stress Test",
            "schema_version": "1.1",
            "scenario_version": "1.0",
            "description": "Two bigger meals plus exercise to stress the controller.",
            "stress_events": [
                {
                    "start_time": 45,
                    "event_type": "meal",
                    "value": 70,
                    "reported_value": 70,
                    "absorption_delay_minutes": 15,
                    "duration": 75,
                },
                {
                    "start_time": 150,
                    "event_type": "meal",
                    "value": 45,
                    "reported_value": 45,
                    "absorption_delay_minutes": 10,
                    "duration": 50,
                },
                {
                    "start_time": 210,
                    "event_type": "exercise",
                    "value": 0.5,
                    "duration": 35,
                },
            ],
        },
    },
    {
        "slug": "03_supervisor_override",
        "label": "Supervisor Override",
        "headline": "A deliberately unsafe AI policy is blocked by the safety supervisor.",
        "algorithm_factory": lambda: RunawayAIAlgorithm(max_bolus=5.0),
        "scenario": {
            "scenario_name": "Live Demo - Supervisor Override",
            "schema_version": "1.1",
            "scenario_version": "1.0",
            "description": "A chaos run where a bad AI keeps asking for too much insulin.",
            "stress_events": [
                {
                    "start_time": 30,
                    "event_type": "meal",
                    "value": 60,
                    "reported_value": 60,
                },
                {
                    "start_time": 120,
                    "event_type": "exercise",
                    "value": 0.8,
                    "duration": 60,
                },
                {
                    "start_time": 200,
                    "event_type": "sensor_error",
                    "value": 180,
                },
            ],
        },
    },
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_single_scenario(
    spec: ScenarioSpec,
    *,
    output_root: Path,
    patient_config: str,
    duration_minutes: int,
    time_step_minutes: int,
    seed: int,
) -> dict[str, str]:
    run_dir = output_root / spec["slug"]
    outputs = run_full(
        algorithm=spec["algorithm_factory"](),
        scenario=spec["scenario"],
        patient_config=patient_config,
        duration_minutes=duration_minutes,
        time_step=time_step_minutes,
        seed=seed,
        output_dir=run_dir,
        enable_profiling=False,
    )
    return {
        "slug": spec["slug"],
        "label": spec["label"],
        "headline": spec["headline"],
        "output_dir": str(run_dir),
        "results_csv": str(outputs["results_csv"]),
        "report_pdf": str(outputs["report_pdf"]),
        "run_manifest_path": str(outputs["run_manifest_path"]),
        "run_metadata_path": str(outputs["run_metadata_path"]),
    }


def _build_demo_summary(
    *,
    output_dir: Path,
    patient_config: str,
    duration_minutes: int,
    time_step_minutes: int,
    seed: int,
    scenario_outputs: list[dict[str, str]],
    poster_outputs: dict[str, str],
    ai_outputs: dict[str, str],
    ai_status: str,
) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "patient_config": patient_config,
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step_minutes,
        "seed": seed,
        "poster_png": poster_outputs["poster_png"],
        "poster_summary_json": poster_outputs["summary_json"],
        "ai_status": ai_status,
        "scenarios": scenario_outputs,
        "ai_outputs": ai_outputs,
    }


def _build_jury_talk_track(summary: dict[str, Any]) -> str:
    lines = [
        "# IINTS-AF Jury Talk Track",
        "",
        "## What this script shows",
        "",
        "- `run_full(...)` for real reproducible run bundles",
        "- `generate_results_poster(...)` for one visual story",
        "- `prepare_ai_ready_artifacts(...)` for optional local AI explanation",
        "",
        "## Walkthrough",
        "",
        "1. Show the constants at the top of `examples/demos/07_live_stage_demo.py`.",
        "2. Say that swapping `PATIENT_CONFIG` reruns the same pipeline for another patient.",
        "3. Run `./scripts/run_live_stage_demo.sh`.",
        "4. Open the poster and explain the three panels from left to right.",
        "5. If someone wants proof, open a scenario folder and show `results.csv`, `clinical_report.pdf`, and `run_manifest.json`.",
        "",
        "## Scenario story",
        "",
    ]
    for scenario in summary["scenarios"]:
        lines.extend(
            [
                f"### {scenario['label']}",
                "",
                f"- {scenario['headline']}",
                f"- Run folder: `{scenario['output_dir']}`",
                f"- CSV: `{scenario['results_csv']}`",
                f"- PDF: `{scenario['report_pdf']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Optional AI step",
            "",
            summary["ai_status"],
            "",
            "```bash",
            "iints ai local-check --model ministral-3:3b",
            f"iints ai report {summary['scenarios'][2]['output_dir']} --model ministral-3:3b",
            f"iints ai review {summary['scenarios'][2]['output_dir']} --model ministral-3:3b",
            f"iints ai explain {summary['scenarios'][2]['output_dir']} --model ministral-3:3b",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_live_demo_notes(summary: dict[str, Any]) -> str:
    return (
        "IINTS-AF LIVE DEMO NOTES\n"
        "========================\n\n"
        "1. Show `examples/demos/07_live_stage_demo.py`.\n"
        "2. Point to PATIENT_CONFIG, OUTPUT_DIR, DURATION_MINUTES, TIME_STEP_MINUTES, and SEED.\n"
        "3. Explain that the script visibly uses these SDK features:\n"
        "   - run_full(...)\n"
        "   - generate_results_poster(...)\n"
        "   - prepare_ai_ready_artifacts(...)\n\n"
        "4. Run: ./scripts/run_live_stage_demo.sh\n"
        f"5. Open poster: {summary['poster_png']}\n"
        f"6. Normal run folder: {summary['scenarios'][0]['output_dir']}\n"
        f"7. Stress run folder: {summary['scenarios'][1]['output_dir']}\n"
        f"8. Supervisor run folder: {summary['scenarios'][2]['output_dir']}\n\n"
        "9. Explain the three panels:\n"
        "   - Normal Run = control case\n"
        "   - Meal Stress Test = harder physiology\n"
        "   - Supervisor Override = bad AI gets blocked\n"
    )


def _build_run_commands(summary: dict[str, Any]) -> str:
    supervisor_dir = summary["scenarios"][2]["output_dir"]
    return (
        "# Live Demo Commands\n\n"
        "## Showable script\n\n"
        "```bash\n"
        "python3 examples/demos/07_live_stage_demo.py\n"
        "```\n\n"
        "## Recommended live command\n\n"
        "```bash\n"
        "./scripts/run_live_stage_demo.sh\n"
        "```\n\n"
        "## Installed CLI alternative\n\n"
        "```bash\n"
        "iints demo-booth --output-dir results/booth_demo\n"
        "```\n\n"
        "## Optional AI step\n\n"
        "```bash\n"
        "iints ai local-check --model ministral-3:3b\n"
        f"iints ai report {supervisor_dir} --model ministral-3:3b\n"
        f"iints ai review {supervisor_dir} --model ministral-3:3b\n"
        f"iints ai explain {supervisor_dir} --model ministral-3:3b\n"
        "```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-friendly live demo runner for the IINTS-AF SDK.")
    parser.add_argument("--patient-config", default=PATIENT_CONFIG, help="Packaged patient profile name or path to a YAML config.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory where the demo bundle should be written.")
    parser.add_argument("--duration", type=int, default=DURATION_MINUTES, help="Simulation duration in minutes.")
    parser.add_argument("--time-step", type=int, default=TIME_STEP_MINUTES, help="Simulation time step in minutes.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic random seed.")
    parser.add_argument(
        "--prepare-ai",
        dest="prepare_ai",
        action="store_true",
        default=PREPARE_AI,
        help="Prepare AI-ready artifacts for the Supervisor Override scenario.",
    )
    parser.add_argument(
        "--skip-ai",
        dest="prepare_ai",
        action="store_false",
        help="Skip AI-ready artifact generation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_outputs = [
        _run_single_scenario(
            spec,
            output_root=output_dir,
            patient_config=args.patient_config,
            duration_minutes=args.duration,
            time_step_minutes=args.time_step,
            seed=args.seed,
        )
        for spec in SCENARIOS
    ]

    poster_outputs = generate_results_poster(
        run_dirs=[item["output_dir"] for item in scenario_outputs],
        labels=[item["label"] for item in scenario_outputs],
        output_path=output_dir / "booth_demo_poster.png",
        summary_output_path=output_dir / "booth_demo_poster.json",
        poster_title="288 Decisions. Every Day. We Test Them All.",
        subtitle="Three SDK features in one story: simulate, stress, and protect.",
    )

    ai_outputs: dict[str, str] = {}
    ai_status = "AI preparation was skipped."
    if args.prepare_ai:
        try:
            ai_outputs = prepare_ai_ready_artifacts(scenario_outputs[2]["output_dir"], create_dev_mdmp_cert=True)
            ai_status = "AI-ready artifacts were created for the Supervisor Override scenario."
        except Exception as exc:
            ai_status = f"AI preparation did not block the demo, but it could not finish cleanly: {exc}"

    summary = _build_demo_summary(
        output_dir=output_dir,
        patient_config=args.patient_config,
        duration_minutes=args.duration,
        time_step_minutes=args.time_step,
        seed=args.seed,
        scenario_outputs=scenario_outputs,
        poster_outputs=poster_outputs,
        ai_outputs=ai_outputs,
        ai_status=ai_status,
    )
    _write_json(output_dir / "demo_summary.json", summary)
    _write_text(output_dir / "JURY_TALK_TRACK.md", _build_jury_talk_track(summary))
    _write_text(output_dir / "BEURS_LIVE_DEMO_SCRIPT.txt", _build_live_demo_notes(summary))
    _write_text(output_dir / "run_commands.md", _build_run_commands(summary))

    print("IINTS Live Stage Demo complete.")
    print(f"Patient config: {args.patient_config}")
    print("")
    print("SDK features this script just demonstrated:")
    print("1. run_full(...) -> real run bundles per scenario")
    print("2. generate_results_poster(...) -> one poster from those bundles")
    print("3. prepare_ai_ready_artifacts(...) -> optional AI-ready supervisor case")
    print("")
    print("What to show next:")
    print(f"1. Poster: {poster_outputs['poster_png']}")
    print(f"2. Jury guide: {output_dir / 'JURY_TALK_TRACK.md'}")
    print(f"3. Live demo script: {output_dir / 'BEURS_LIVE_DEMO_SCRIPT.txt'}")
    print(f"4. Commands: {output_dir / 'run_commands.md'}")
    print("")
    print("Three scenario folders:")
    for scenario in scenario_outputs:
        print(f"- {scenario['label']}: {scenario['output_dir']}")
    print("")
    print("Suggested booth flow:")
    print("- Show the constants and the visible SDK API calls in this file.")
    print("- Run this script once.")
    print("- Open the poster and explain the three panels from left to right.")
    print("- If people want proof, open a scenario folder and show the CSV, PDF, and manifest.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
