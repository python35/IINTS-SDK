from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from iints.ai.assistant import IINTSAssistant
from iints.ai.backends import DEFAULT_MINISTRAL_MODEL
from iints.ai.prepare import prepare_ai_ready_artifacts
from iints.live_patient.daemon import _start_api_server
from iints.live_patient.edge_ops import summarize_edge_workspace
from iints.live_patient.runtime import (
    LivePatientDaemon,
    PatientRuntimeConfig,
    PatientRuntimeStore,
    get_runtime_scenario_profile,
    is_process_alive,
    list_runtime_scenario_profiles,
)
from iints.live_patient.service_export import service_instructions_text, write_service_artifacts
from iints.live_patient.uno_q import export_uno_q_bridge


app = typer.Typer(help="Persistent digital patient runtime for Raspberry Pi demos and long-running virtual patient studies.")


def _store_for(workspace: Path) -> PatientRuntimeStore:
    return PatientRuntimeStore(workspace.expanduser().resolve() / "patient_state.db")


def _status_for(workspace: Path) -> dict[str, Any]:
    db_path = workspace.expanduser().resolve() / "patient_state.db"
    if not db_path.is_file():
        return {}
    return PatientRuntimeStore(db_path).read_status()


def _queue_command(
    workspace: Path,
    command: str,
    payload: dict[str, Any] | None = None,
    wait_seconds: float = 5.0,
) -> dict[str, Any] | None:
    store = _store_for(workspace)
    command_id = store.enqueue_command(command, payload)
    return store.await_command(command_id, timeout_seconds=wait_seconds)


def _default_output_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / "live_bundle" / "ai" / "realism_review.md"


def _parse_speed(value: str | float) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = str(value).strip().lower()
        if text.endswith("x"):
            text = text[:-1]
        parsed = float(text)
    if parsed <= 0.0:
        raise typer.BadParameter("Speed must be a positive number, for example '60' or '60x'.")
    return parsed


def _resolve_review_inputs(workspace: Path, mdmp_cert: Path | None) -> tuple[Path, Path, Path | None]:
    bundle_dir = workspace.expanduser().resolve() / "live_bundle"
    outputs = prepare_ai_ready_artifacts(bundle_dir, create_dev_mdmp_cert=mdmp_cert is None)
    resolved_cert = mdmp_cert or Path(outputs["mdmp_cert"])
    public_key = Path(outputs["mdmp_public_key"]) if "mdmp_public_key" in outputs else None
    payload_path = Path(outputs.get("review_payload") or (bundle_dir / "ai" / "review_payload.json"))
    return payload_path, resolved_cert, public_key


@app.command("scenarios")
def scenarios() -> None:
    console = Console()
    table = Table(title="Digital Patient Scenario Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Default Seed")
    table.add_column("Warm Start")
    table.add_column("Description", overflow="fold")
    for profile in list_runtime_scenario_profiles():
        warm_start = f"{profile.warm_start_minutes} min" if profile.warm_start_minutes > 0 else "-"
        table.add_row(profile.name, str(profile.default_seed), warm_start, profile.description)
    console.print(table)


@app.command("start")
def start(
    algo: Annotated[Path, typer.Option(help="Path to the insulin algorithm Python file.")],
    patient_config: Annotated[str, typer.Option(help="Patient configuration name or YAML path.")] = "default_patient",
    patient_model: Annotated[str, typer.Option("--patient-model", help="Patient model type: auto, bergman, custom, simglucose.")] = "auto",
    scenario_profile: Annotated[str, typer.Option(help="Live day profile: normal_day, sport_day, bad_carb_count, night_hypo_risk, expo_hot_start.")] = "normal_day",
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
    mode: Annotated[str, typer.Option(help="Clock mode: real-time or demo-time.")] = "demo-time",
    speed: Annotated[str, typer.Option(help="Acceleration factor for demo-time mode. Accepts values like 60 or 60x.")] = "60x",
    api_host: Annotated[str, typer.Option(help="Host for the local FastAPI dashboard service.")] = "127.0.0.1",
    api_port: Annotated[int, typer.Option(help="Port for the local FastAPI dashboard service.")] = 8765,
    seed: Annotated[Optional[int], typer.Option(help="Optional deterministic simulation seed.")] = None,
    foreground: Annotated[bool, typer.Option("--foreground", hidden=True)] = False,
    max_steps: Annotated[Optional[int], typer.Option("--max-steps", hidden=True)] = None,
    reset: Annotated[bool, typer.Option("--reset", hidden=True)] = False,
) -> None:
    console = Console()
    workspace = workspace.expanduser().resolve()
    status = _status_for(workspace)
    pid = status.get("pid")
    if is_process_alive(pid):
        console.print(f"[bold red]A digital patient is already running in {workspace} (pid {pid}).[/bold red]")
        raise typer.Exit(code=1)

    speed_value = _parse_speed(speed)
    profile = get_runtime_scenario_profile(scenario_profile)
    effective_seed = seed if seed is not None else profile.default_seed
    cfg = PatientRuntimeConfig(
        workspace=str(workspace),
        algo_path=str(algo.expanduser().resolve()),
        patient_config=patient_config,
        patient_model_type=patient_model,
        scenario_profile=profile.name,
        mode=mode,
        speed=speed_value,
        api_host=api_host,
        api_port=api_port,
        seed=effective_seed,
    )
    workspace.mkdir(parents=True, exist_ok=True)
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    if foreground:
        daemon = LivePatientDaemon(cfg)
        daemon.install_signal_handlers()
        daemon.bootstrap(reset=reset)
        server, thread = _start_api_server(cfg)
        try:
            daemon.run(max_steps=max_steps)
        finally:
            server.should_exit = True
            thread.join(timeout=2.0)
        console.print(f"[green]Foreground digital patient finished in {workspace}.[/green]")
        return

    with cfg.log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "iints.live_patient.daemon",
                "--config",
                str(cfg.config_path),
                *(["--reset"] if reset else []),
            ],
            stdout=handle,
            stderr=handle,
            start_new_session=True,
        )

    deadline = time.time() + 8.0
    while time.time() < deadline:
        latest_status = _status_for(workspace)
        if latest_status.get("daemon_status") in {"running", "paused", "starting"}:
            break
        time.sleep(0.2)

    details = "\n".join(
        [
            f"Workspace: {workspace}",
            f"Scenario: {profile.name}",
            f"Seed: {effective_seed}",
            f"Dashboard: {cfg.dashboard_url}",
            f"API: {cfg.api_url}",
            f"Mode: {mode} ({speed_value:g}x)" if mode == "demo-time" else f"Mode: {mode}",
            f"Status: iints patient status --workspace {workspace}",
            f"Reset: iints patient expo-reset --workspace {workspace}",
            f"Review: iints patient review --workspace {workspace} --model {DEFAULT_MINISTRAL_MODEL}",
            "Tip: use Raspberry Pi Connect screen sharing to present the dashboard remotely.",
        ]
    )
    console.print(Panel(details, title="Digital Patient Started", border_style="green"))


@app.command("status")
def status(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    workspace = workspace.expanduser().resolve()
    payload = summarize_edge_workspace(workspace)
    if not payload:
        console.print(f"[bold red]No patient runtime found in {workspace}.[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title="IINTS Digital Patient Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    for key in [
        "daemon_status",
        "algorithm_name",
        "simulated_clock",
        "last_glucose_mgdl",
        "last_delivered_insulin_units",
        "last_safety_reason",
        "last_event_summary",
        "mode",
        "speed",
        "scenario_profile",
        "active_seed",
        "api_host",
        "api_port",
        "workspace",
    ]:
        value = payload.get(key)
        if value is None or value == "":
            value = "-"
        table.add_row(key, str(value))
    api_host = payload.get("api_host")
    api_port = payload.get("api_port")
    if api_host and api_port:
        table.add_row("dashboard_url", f"http://{api_host}:{api_port}/dashboard")
        table.add_row("kiosk_url", f"http://{api_host}:{api_port}/kiosk")
        table.add_row("api_url", f"http://{api_host}:{api_port}")
    table.add_row("pid_alive", str(is_process_alive(payload.get("pid"))))
    certification = payload.get("certification") or {}
    review = payload.get("review") or {}
    table.add_row("certification", str(certification.get("label") or "-"))
    table.add_row("review", str(review.get("label") or "-"))
    table.add_row("workspace_size_mb", str(payload.get("workspace_size_mb", "-")))
    console.print(table)


@app.command("inject-meal")
def inject_meal(
    carbs: Annotated[float, typer.Option(help="Carbohydrate amount to inject immediately into the live patient.")],
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    result = _queue_command(workspace, "inject_meal", {"carbs": carbs})
    if result is None or result.get("status") != "done":
        console.print("[bold red]The meal command was not acknowledged in time.[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Meal queued:[/green] {carbs:.0f} g carbs")


@app.command("pause")
def pause(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    result = _queue_command(workspace, "pause")
    if result is None or result.get("status") != "done":
        console.print("[bold red]Pause was not acknowledged in time.[/bold red]")
        raise typer.Exit(code=1)
    console.print("[green]Digital patient paused.[/green]")


@app.command("resume")
def resume(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    result = _queue_command(workspace, "resume")
    if result is None or result.get("status") != "done":
        console.print("[bold red]Resume was not acknowledged in time.[/bold red]")
        raise typer.Exit(code=1)
    console.print("[green]Digital patient resumed.[/green]")


@app.command("expo-reset")
def expo_reset(
    scenario_profile: Annotated[Optional[str], typer.Option(help="Optional profile to load after reset. Defaults to expo_hot_start.")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Optional deterministic seed override for the reset profile.")] = None,
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    payload: dict[str, Any] = {}
    if scenario_profile is not None:
        payload["scenario_profile"] = get_runtime_scenario_profile(scenario_profile).name
    if seed is not None:
        payload["seed"] = seed
    result = _queue_command(workspace, "expo_reset", payload, wait_seconds=8.0)
    if result is None or result.get("status") != "done":
        console.print("[bold red]Expo reset was not acknowledged in time.[/bold red]")
        raise typer.Exit(code=1)
    details = result.get("result", {})
    console.print(
        f"[green]Expo mode reset complete.[/green] profile={details.get('scenario_profile', 'expo_hot_start')} seed={details.get('active_seed', '-')}"
    )


@app.command("stop")
def stop(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    workspace = workspace.expanduser().resolve()
    result = _queue_command(workspace, "stop", wait_seconds=5.0)
    if result is None:
        console.print("[bold red]Stop request timed out.[/bold red]")
        raise typer.Exit(code=1)
    deadline = time.time() + 8.0
    while time.time() < deadline:
        status = _status_for(workspace)
        if not is_process_alive(status.get("pid")):
            console.print("[green]Digital patient stopped.[/green]")
            return
        time.sleep(0.2)
    console.print("[yellow]Stop request acknowledged, but the daemon is still shutting down.[/yellow]")


@app.command("export-service")
def export_service(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
    output: Annotated[Optional[Path], typer.Option(help="Optional output service file path.")] = None,
    service_name: Annotated[str, typer.Option(help="systemd service name without the .service suffix.")] = "iints-digital-patient",
    user_name: Annotated[Optional[str], typer.Option(help="Linux user that should run the service. Defaults to the current shell user.")] = None,
    python_path: Annotated[Optional[Path], typer.Option(help="Python executable used in ExecStart. Defaults to the current interpreter.")] = None,
) -> None:
    console = Console()
    workspace = workspace.expanduser().resolve()
    config_path = workspace / "patient_runtime_config.json"
    if not config_path.is_file():
        console.print(f"[bold red]No patient runtime config found in {workspace}. Start the patient once first.[/bold red]")
        raise typer.Exit(code=1)

    config = PatientRuntimeConfig.from_path(config_path)
    outputs = write_service_artifacts(
        config,
        output_path=(output or (workspace / f"{service_name}.service")),
        service_name=service_name,
        user_name=user_name,
        python_path=python_path,
    )
    console.print(
        Panel(
            service_instructions_text(Path(outputs["service_file"]), service_name),
            title="systemd Install Steps",
            border_style="green",
        )
    )
    console.print(f"[green]Service file:[/green] {outputs['service_file']}")
    console.print(f"[green]Install notes:[/green] {outputs['install_notes']}")


@app.command("export-uno-bridge")
def export_uno_bridge(
    output_dir: Annotated[Path, typer.Option(help="Directory where the UNO Q bridge scaffold should be written.")] = Path("./uno_q_bridge"),
) -> None:
    console = Console()
    outputs = export_uno_q_bridge(output_dir)
    table = Table(title="UNO Q Bridge Scaffold")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", overflow="fold")
    table.add_row("Sketch", outputs["sketch"])
    table.add_row("README", outputs["readme"])
    table.add_row("Protocol", outputs["protocol"])
    console.print(table)


@app.command("hardware-bridge")
def hardware_bridge(
    board: Annotated[str, typer.Option(help="Hardware bridge target. Currently supported: uno_q.")] = "uno_q",
    output_dir: Annotated[Path, typer.Option(help="Directory where the hardware bridge scaffold should be written.")] = Path("./uno_q_bridge"),
) -> None:
    console = Console()
    normalized = board.strip().lower()
    if normalized != "uno_q":
        console.print("[bold red]Only the `uno_q` hardware bridge is currently implemented.[/bold red]")
        raise typer.Exit(code=1)
    outputs = export_uno_q_bridge(output_dir)
    console.print(
        Panel(
            "\n".join(
                [
                    f"Bridge target: {normalized}",
                    f"Sketch: {outputs['sketch']}",
                    f"README: {outputs['readme']}",
                    f"Protocol: {outputs['protocol']}",
                ]
            ),
            title="Hardware Bridge Ready",
            border_style="cyan",
        )
    )


@app.command("kiosk")
def kiosk(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
) -> None:
    console = Console()
    summary = summarize_edge_workspace(workspace)
    if not summary:
        console.print(f"[bold red]No patient runtime found in {workspace}.[/bold red]")
        raise typer.Exit(code=1)

    details = "\n".join(
        [
            f"Kiosk URL: {summary['kiosk_url']}",
            f"Dashboard URL: {summary['dashboard_url']}",
            f"Scenario: {summary.get('scenario_profile', '-')}",
            f"Certification: {(summary.get('certification') or {}).get('label', '-')}",
            f"Review: {(summary.get('review') or {}).get('label', '-')}",
            "Tip: open the kiosk URL on the Pi and present it via Raspberry Pi Connect screen sharing.",
        ]
    )
    console.print(Panel(details, title="Digital Patient Kiosk", border_style="green"))


@app.command("review")
def review(
    workspace: Annotated[Path, typer.Option(help="Workspace directory for the persistent digital patient state.")] = Path("./digital_patient_runtime"),
    model: Annotated[str, typer.Option(help="Local Ollama model used for the realism review.")] = DEFAULT_MINISTRAL_MODEL,
    mdmp_cert: Annotated[Optional[Path], typer.Option(help="Optional MDMP certificate path. If omitted, the live bundle gets a local dev cert.")] = None,
    output: Annotated[Optional[Path], typer.Option(help="Optional output markdown file for the review.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option(help="Optional Ollama base URL override.")] = None,
    minimum_grade: Annotated[str, typer.Option(help="Minimum MDMP grade required before review runs.")] = "research_grade",
) -> None:
    console = Console()
    workspace = workspace.expanduser().resolve()
    try:
        payload_path, cert_path, public_key = _resolve_review_inputs(workspace, mdmp_cert)
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assistant = IINTSAssistant(
            cert_path,
            mode="local",
            model=model,
            minimum_grade=minimum_grade,
            public_key_path=public_key,
            ollama_host=ollama_host,
        )
        response = assistant.review_realism(payload)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    output_path = output or _default_output_path(workspace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text + "\n", encoding="utf-8")
    console.print(Panel(response.text, title="Digital Patient Review", border_style="cyan"))
    console.print(f"[green]Saved:[/green] {output_path}")
