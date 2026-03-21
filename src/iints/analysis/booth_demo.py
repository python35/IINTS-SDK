from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from iints.ai.prepare import prepare_ai_ready_artifacts
from iints.analysis.poster import generate_results_poster
from iints.core.algorithms.mock_algorithms import RunawayAIAlgorithm
from iints.core.algorithms.pid_controller import PIDController
from iints.highlevel import run_full


@dataclass(frozen=True)
class BoothScenarioSpec:
    slug: str
    label: str
    headline: str
    jury_takeaway: str
    scenario: dict[str, Any]
    algorithm_factory: Callable[[], Any]


def _scenario_specs() -> list[BoothScenarioSpec]:
    return [
        BoothScenarioSpec(
            slug="01_normal_run",
            label="Normal Run",
            headline="The controller keeps glucose in range during a calm day.",
            jury_takeaway=(
                "This is the control case. It shows the SDK can simulate a realistic closed-loop day "
                "and produce clean, auditable outputs."
            ),
            scenario={
                "scenario_name": "Booth Demo - Normal Run",
                "schema_version": "1.1",
                "scenario_version": "1.0",
                "description": "A calm day with one moderate meal to show normal controller behavior.",
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
            algorithm_factory=PIDController,
        ),
        BoothScenarioSpec(
            slug="02_meal_stress_test",
            label="Meal Stress Test",
            headline="The controller reacts to a harder day with meals and exercise.",
            jury_takeaway=(
                "This is the stress case. We deliberately add larger disturbances and show that the "
                "system remains explainable and clinically readable."
            ),
            scenario={
                "scenario_name": "Booth Demo - Meal Stress Test",
                "schema_version": "1.1",
                "scenario_version": "1.0",
                "description": "Two meal challenges plus exercise to stress the controller.",
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
            algorithm_factory=PIDController,
        ),
        BoothScenarioSpec(
            slug="03_supervisor_override",
            label="Supervisor Override",
            headline="A deliberately unsafe AI is blocked by the safety supervisor.",
            jury_takeaway=(
                "This is the safety case. We intentionally use a bad algorithm so the audience can see "
                "that the supervisor prevents dangerous insulin commands and records why."
            ),
            scenario={
                "scenario_name": "Booth Demo - Supervisor Override",
                "schema_version": "1.1",
                "scenario_version": "1.0",
                "description": "A chaos run that forces unsafe insulin requests during a falling glucose phase.",
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
            algorithm_factory=lambda: RunawayAIAlgorithm(max_bolus=5.0),
        ),
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_jury_brief(
    *,
    output_dir: Path,
    scenario_specs: list[BoothScenarioSpec],
    poster_png: Path,
    poster_summary_json: Path,
    run_outputs: dict[str, dict[str, Any]],
    ai_outputs: dict[str, str],
    ai_status: str,
) -> str:
    lines: list[str] = [
        "# IINTS-AF Booth Demo",
        "",
        "## 30-second intro",
        "",
        (
            "IINTS-AF is a safety-first SDK for testing insulin-delivery algorithms. "
            "In this demo we show three stories: normal control, stress handling, and a safety supervisor "
            "blocking an unsafe AI recommendation."
        ),
        "",
        "## Live flow",
        "",
        "1. Run the script and point out that it creates three full simulation bundles.",
        f"2. Open the poster: `{poster_png}`",
        "3. Walk the jury left to right: normal run, stress test, supervisor override.",
        "4. Show that each run also has results CSV, audit trail, PDF report, and manifest.",
        "5. Optionally show the local AI explanation on the supervisor run.",
        "",
        "## What to say per panel",
        "",
    ]
    for spec in scenario_specs:
        outputs = run_outputs[spec.slug]
        lines.extend(
            [
                f"### {spec.label}",
                "",
                f"- Headline: {spec.headline}",
                f"- Why it matters: {spec.jury_takeaway}",
                f"- Run directory: `{outputs['output_dir']}`",
                f"- Results CSV: `{outputs['results_csv']}`",
                f"- PDF report: `{outputs['report_pdf']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Optional AI step",
            "",
            ai_status,
            "",
            "If local Ollama + Ministral are ready, run:",
            "",
            "```bash",
            "iints ai local-check --model ministral-3:3b",
            f"iints ai report {run_outputs['03_supervisor_override']['output_dir']} --model ministral-3:3b",
            f"iints ai explain {run_outputs['03_supervisor_override']['output_dir']} --model ministral-3:3b",
            "```",
            "",
            "## Key artifacts",
            "",
            f"- Poster PNG: `{poster_png}`",
            f"- Poster summary: `{poster_summary_json}`",
            f"- Demo summary JSON: `{output_dir / 'demo_summary.json'}`",
            f"- Demo commands: `{output_dir / 'run_commands.md'}`",
        ]
    )

    if ai_outputs:
        lines.extend(
            [
                "",
                "### AI-ready artifacts",
                "",
            ]
        )
        for key, value in ai_outputs.items():
            lines.append(f"- {key}: `{value}`")

    return "\n".join(lines) + "\n"


def _build_commands_markdown(
    *,
    output_dir: Path,
    example_script: Path,
    run_outputs: dict[str, dict[str, Any]],
) -> str:
    supervisor_dir = run_outputs["03_supervisor_override"]["output_dir"]
    return (
        "# Booth Demo Commands\n\n"
        "## Run from source tree\n\n"
        "```bash\n"
        f"PYTHONPATH=src python3 {example_script} --output-dir {output_dir}\n"
        "```\n\n"
        "## Run via installed CLI\n\n"
        "```bash\n"
        f"iints demo-booth --output-dir {output_dir}\n"
        "```\n\n"
        "## Optional local AI explanation\n\n"
        "```bash\n"
        "iints ai local-check --model ministral-3:3b\n"
        f"iints ai report {supervisor_dir} --model ministral-3:3b\n"
        f"iints ai explain {supervisor_dir} --model ministral-3:3b\n"
        "```\n"
    )


def _build_live_demo_script_text(
    *,
    output_dir: Path,
    poster_png: Path,
    run_outputs: dict[str, dict[str, Any]],
) -> str:
    return (
        "IINTS-AF BOOTH LIVE DEMO SCRIPT\n"
        "===============================\n\n"
        "1. WHAT CODE TO SHOW FIRST\n"
        "- Show examples/demos/06_booth_demo.py first.\n"
        "  Reason: it is the shortest, clearest orchestration file and shows the whole story in one screen.\n"
        "- If someone asks how it really works, open src/iints/analysis/booth_demo.py.\n"
        "  Reason: that file defines the three scenarios and writes the poster, talk track, and run bundle outputs.\n\n"
        "2. LIVE COMMAND TO RUN\n"
        "- From the repository root, run:\n"
        "  ./scripts/run_booth_demo.sh\n"
        "- Or use the installed CLI:\n"
        "  iints demo-booth --output-dir results/booth_demo\n\n"
        "3. WHAT TO SAY WHILE IT RUNS\n"
        "- IINTS-AF is a safety-first SDK for testing insulin-delivery algorithms before real-world use.\n"
        "- This demo generates three reproducible scenarios: a normal run, a stress test, and a supervisor override case.\n"
        "- Every scenario produces real artifacts: CSV, PDF, audit trail, baseline comparison, and a manifest.\n\n"
        "4. WHAT TO OPEN AFTER THE RUN\n"
        f"- Open the poster: {poster_png}\n"
        f"- Normal run folder: {run_outputs['01_normal_run']['output_dir']}\n"
        f"- Meal stress folder: {run_outputs['02_meal_stress_test']['output_dir']}\n"
        f"- Supervisor folder: {run_outputs['03_supervisor_override']['output_dir']}\n\n"
        "5. JURY WALKTHROUGH\n"
        "- Panel 1: Normal Run\n"
        "  Say: this is the control case. The algorithm keeps glucose in a clinically readable range.\n"
        "- Panel 2: Meal Stress Test\n"
        "  Say: here we add larger disturbances and show that the system stays explainable under stress.\n"
        "- Panel 3: Supervisor Override\n"
        "  Say: here we intentionally use a bad AI policy so the safety supervisor can prove it blocks unsafe insulin.\n\n"
        "6. OPTIONAL AI STEP\n"
        "- If Ollama is ready, run:\n"
        "  iints ai local-check --model ministral-3:3b\n"
        f"  iints ai report {run_outputs['03_supervisor_override']['output_dir']} --model ministral-3:3b\n"
        f"  iints ai explain {run_outputs['03_supervisor_override']['output_dir']} --model ministral-3:3b\n"
        "- Say: the local model explains the result, but only after the SDK has prepared the run artifacts.\n\n"
        "7. IF THE JURY ASKS WHY THIS MATTERS\n"
        "- It is not just a graph generator.\n"
        "- It is a reproducible safety workflow: simulate, stress, audit, visualize, explain.\n"
        "- The supervisor case is the key proof that the SDK is safety-first, not just AI-first.\n\n"
        "8. HANDY FILES\n"
        f"- Poster PNG: {poster_png}\n"
        f"- Jury markdown: {output_dir / 'JURY_TALK_TRACK.md'}\n"
        f"- Demo commands: {output_dir / 'run_commands.md'}\n"
        f"- Demo summary JSON: {output_dir / 'demo_summary.json'}\n"
    )


def build_booth_demo(
    output_dir: str | Path = "./results/booth_demo",
    *,
    duration_minutes: int = 360,
    time_step: int = 5,
    seed: int = 42,
    prepare_ai: bool = True,
    create_dev_mdmp_cert: bool = True,
) -> dict[str, str]:
    """
    Create a fair-ready demo bundle with three scenario runs, a poster, and jury notes.
    """
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    specs = _scenario_specs()
    run_outputs: dict[str, dict[str, Any]] = {}
    run_dirs: list[Path] = []
    labels: list[str] = []

    for spec in specs:
        run_dir = resolved_output / spec.slug
        outputs = run_full(
            algorithm=spec.algorithm_factory(),
            scenario=spec.scenario,
            patient_config="default_patient",
            duration_minutes=duration_minutes,
            time_step=time_step,
            seed=seed,
            output_dir=run_dir,
            enable_profiling=False,
        )
        run_outputs[spec.slug] = {
            "label": spec.label,
            "headline": spec.headline,
            "jury_takeaway": spec.jury_takeaway,
            "output_dir": str(run_dir),
            "results_csv": str(outputs["results_csv"]),
            "report_pdf": str(outputs["report_pdf"]),
            "run_manifest_path": str(outputs["run_manifest_path"]),
        }
        run_dirs.append(run_dir)
        labels.append(spec.label)

    poster_outputs = generate_results_poster(
        run_dirs=run_dirs,
        labels=labels,
        output_path=resolved_output / "booth_demo_poster.png",
        summary_output_path=resolved_output / "booth_demo_poster.json",
        poster_title="288 Decisions. Every Day. We Test Them All.",
        subtitle="Normal control, stress handling, and safety override in one fair-ready story.",
    )

    ai_outputs: dict[str, str] = {}
    ai_status = "AI preparation was skipped."
    if prepare_ai:
        supervisor_dir = run_dirs[-1]
        try:
            ai_outputs = prepare_ai_ready_artifacts(
                supervisor_dir,
                create_dev_mdmp_cert=create_dev_mdmp_cert,
            )
            if "mdmp_cert" in ai_outputs:
                ai_status = (
                    "AI-ready payloads and a local development MDMP certificate were generated for the "
                    "Supervisor Override run."
                )
            else:
                ai_status = "AI-ready payloads were generated for the Supervisor Override run."
        except Exception as exc:
            ai_status = f"AI preparation did not block the demo, but it could not finish cleanly: {exc}"

    summary_payload = {
        "output_dir": str(resolved_output),
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step,
        "seed": seed,
        "poster_png": poster_outputs["poster_png"],
        "poster_summary_json": poster_outputs["summary_json"],
        "ai_status": ai_status,
        "scenarios": [
            {
                "slug": spec.slug,
                "label": spec.label,
                "headline": spec.headline,
                "jury_takeaway": spec.jury_takeaway,
                "scenario_name": spec.scenario["scenario_name"],
                **run_outputs[spec.slug],
            }
            for spec in specs
        ],
    }
    _write_json(resolved_output / "demo_summary.json", summary_payload)

    example_script = Path("examples/demos/06_booth_demo.py")
    commands_markdown = _build_commands_markdown(
        output_dir=resolved_output,
        example_script=example_script,
        run_outputs=run_outputs,
    )
    commands_path = resolved_output / "run_commands.md"
    _write_text(commands_path, commands_markdown)

    live_demo_script_text = _build_live_demo_script_text(
        output_dir=resolved_output,
        poster_png=Path(poster_outputs["poster_png"]),
        run_outputs=run_outputs,
    )
    live_demo_script_path = resolved_output / "BEURS_LIVE_DEMO_SCRIPT.txt"
    _write_text(live_demo_script_path, live_demo_script_text)

    jury_brief = _build_jury_brief(
        output_dir=resolved_output,
        scenario_specs=specs,
        poster_png=Path(poster_outputs["poster_png"]),
        poster_summary_json=Path(poster_outputs["summary_json"]),
        run_outputs=run_outputs,
        ai_outputs=ai_outputs,
        ai_status=ai_status,
    )
    jury_brief_path = resolved_output / "JURY_TALK_TRACK.md"
    _write_text(jury_brief_path, jury_brief)

    artifact_paths: dict[str, str] = {
        "output_dir": str(resolved_output),
        "poster_png": poster_outputs["poster_png"],
        "poster_summary_json": poster_outputs["summary_json"],
        "demo_summary_json": str(resolved_output / "demo_summary.json"),
        "jury_talk_track": str(jury_brief_path),
        "run_commands": str(commands_path),
        "live_demo_script": str(live_demo_script_path),
    }
    for spec in specs:
        artifact_paths[f"{spec.slug}_dir"] = run_outputs[spec.slug]["output_dir"]
    artifact_paths.update(ai_outputs)
    return artifact_paths
