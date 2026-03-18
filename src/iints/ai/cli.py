from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from .assistant import AIResponse, IINTSAssistant
from .backends import DEFAULT_MINISTRAL_MODEL, OllamaBackend
from .model_catalog import list_local_mistral_models
from .prepare import prepare_ai_ready_artifacts


app = typer.Typer(help="Research-only AI assistant commands gated by MDMP certification.")


def _validate_trust_inputs(
    public_key: Path | None,
    trust_store: Path | None,
) -> None:
    if public_key is not None and trust_store is not None:
        raise typer.BadParameter("Use either --public-key or --trust-store, not both.")


def _load_json_payload(path: Path, label: str) -> Any:
    if not path.is_file():
        raise typer.BadParameter(f"{label} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{label} must be valid JSON: {path}") from exc
    return payload


def _default_prepared_payload(task: str, ai_dir: Path) -> Path:
    candidates = {
        "explain": ["step_riskiest.json", "step_latest.json"],
        "trends": ["trends_payload.json"],
        "anomalies": ["anomalies_payload.json"],
        "report": ["report_payload.json"],
    }.get(task, [])
    for filename in candidates:
        candidate = ai_dir / filename
        if candidate.is_file():
            return candidate
    expected = ", ".join(candidates) if candidates else "prepared payload"
    raise typer.BadParameter(
        f"No prepared AI payload found in {ai_dir}. Expected one of: {expected}. "
        "Run `iints ai prepare <run_dir>` first."
    )


def _resolve_cli_inputs(
    *,
    task: str,
    input_path: Path,
    mdmp_cert: Path | None,
    public_key: Path | None,
    trust_store: Path | None,
) -> tuple[Path, Path, Path | None]:
    resolved_input = input_path
    resolved_cert = mdmp_cert
    resolved_public_key = public_key

    if input_path.is_dir():
        ai_dir = input_path / "ai"
        resolved_input = _default_prepared_payload(task, ai_dir)
        if resolved_cert is None:
            candidate_cert = ai_dir / "report.signed.mdmp"
            if candidate_cert.is_file():
                resolved_cert = candidate_cert
        if resolved_public_key is None and trust_store is None:
            candidate_public_key = ai_dir / "keys" / "mdmp_pub_v1.pem"
            if candidate_public_key.is_file():
                resolved_public_key = candidate_public_key

    if resolved_cert is None:
        raise typer.BadParameter(
            "No MDMP certificate provided. Pass --mdmp-cert or run "
            "`iints ai prepare <run_dir>` to generate a local development certificate."
        )

    return resolved_input, resolved_cert, resolved_public_key


def _write_output(path: Path | None, response: AIResponse) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text + "\n", encoding="utf-8")


def _render_response(console: Console, title: str, response: AIResponse, output: Path | None) -> None:
    console.print(Panel(response.text, title=title, border_style="cyan"))
    console.print(
        f"[green]Backend:[/green] {response.backend} | "
        f"[green]Model:[/green] {response.model} | "
        f"[green]MDMP grade:[/green] {response.certification.grade}"
    )
    if output is not None:
        console.print(f"[green]Saved:[/green] {output}")


def _render_local_check(console: Console, status: dict[str, object]) -> None:
    installed = status.get("installed_models", [])
    installed_text = ", ".join(str(item) for item in installed) if isinstance(installed, list) and installed else "none"
    ready = bool(status.get("ready"))
    resolved_model = status.get("resolved_model") or "not found"
    smoke_text = status.get("smoke_test") or "not run"
    console.print(
        Panel(
            "\n".join(
                [
                    f"Endpoint: {status.get('base_url')}",
                    f"Requested model: {status.get('requested_model')}",
                    f"Resolved local model: {resolved_model}",
                    f"Server version: {status.get('server_version') or 'unknown'}",
                    f"Timeout (s): {status.get('timeout_seconds')}",
                    f"Installed models: {installed_text}",
                    (
                        f"Pull command: {status.get('pull_command')}"
                        if status.get("pull_command")
                        else "Pull command: not needed"
                    ),
                    f"Generate smoke-test: {smoke_text}",
                ]
            ),
            title="IINTS AI Local Check",
            border_style="cyan" if ready else "yellow",
        )
    )
    if ready:
        console.print("[green]Local Ollama backend is ready for open Ministral 3 inference.[/green]")
    else:
        console.print("[bold red]Local Ollama backend is reachable, but the requested model is missing.[/bold red]")
    if status.get("version_ok") is False:
        console.print("[bold red]Ollama is too old for the open Ministral 3 runtime.[/bold red]")


@app.command("models")
def models() -> None:
    console = Console()
    table = Table(title="IINTS AI Local Mistral Model Guide")
    table.add_column("Model Tag", style="cyan", no_wrap=True)
    table.add_column("Best For", style="green")
    table.add_column("Approx Download")
    table.add_column("System RAM")
    table.add_column("GPU VRAM")
    table.add_column("Notes", overflow="fold")

    for profile in list_local_mistral_models():
        vram = f"{profile.recommended_vram_gb}+ GB" if profile.recommended_vram_gb is not None else "CPU-only"
        table.add_row(
            profile.tag,
            profile.fit,
            f"{profile.approx_download_gb:.1f} GB",
            f"{profile.recommended_system_ram_gb}+ GB",
            vram,
            profile.notes,
        )

    console.print(table)
    console.print(
        "[dim]Tip:[/dim] start with "
        f"`{DEFAULT_MINISTRAL_MODEL}` unless you know your hardware can comfortably run a larger local model."
    )


@app.command("prepare")
def prepare(
    run_dir: Annotated[Path, typer.Argument(help="Run output directory containing results.csv and run_metadata.json.")],
    create_dev_mdmp_cert: Annotated[
        bool,
        typer.Option(
            "--create-dev-mdmp-cert/--no-create-dev-mdmp-cert",
            help="Generate a local development MDMP certificate and keypair for AI commands.",
        ),
    ] = True,
    grade: Annotated[str, typer.Option(help="Grade to embed in the local development MDMP certificate.")] = "research_grade",
    expires_days: Annotated[int, typer.Option(help="Certificate expiry window in days for local development certs.")] = 30,
    key_dir: Annotated[Optional[Path], typer.Option(help="Optional directory to store the generated local MDMP keypair.")] = None,
) -> None:
    console = Console()
    try:
        outputs = prepare_ai_ready_artifacts(
            run_dir,
            create_dev_mdmp_cert=create_dev_mdmp_cert,
            grade=grade,
            expires_days=expires_days,
            key_dir=key_dir,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title="IINTS AI Prepared Artifacts")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", overflow="fold")
    for key, value in outputs.items():
        table.add_row(key, value)
    console.print(table)
    console.print("[green]Prepared AI payloads are ready.[/green]")
    if "mdmp_cert" in outputs:
        console.print(
            "[green]You can now run:[/green] "
            f"`iints ai report {run_dir}` or `iints ai explain {run_dir}`"
        )


def _build_assistant(
    *,
    mdmp_cert: Path,
    mode: str,
    model: str,
    minimum_grade: str,
    public_key: Path | None,
    trust_store: Path | None,
    ollama_host: str | None,
    timeout_seconds: float,
) -> IINTSAssistant:
    _validate_trust_inputs(public_key, trust_store)
    return IINTSAssistant(
        mdmp_cert,
        mode=mode,
        model=model,
        minimum_grade=minimum_grade,
        public_key_path=public_key,
        trust_store_path=trust_store,
        ollama_host=ollama_host,
        timeout_seconds=timeout_seconds,
    )


@app.command("local-check")
def local_check(
    model: Annotated[str, typer.Option(help="Ollama model name to validate locally.")] = DEFAULT_MINISTRAL_MODEL,
    ollama_host: Annotated[Optional[str], typer.Option(help="Override the Ollama base URL.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="HTTP timeout for Ollama health checks.")] = 120.0,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test/--no-smoke-test",
            help="Run a tiny generation request after health checks to prove the model can actually answer.",
        ),
    ] = True,
) -> None:
    console = Console()
    backend = OllamaBackend(model_name=model, base_url=ollama_host, timeout_seconds=timeout_seconds)
    try:
        if not backend.available():
            console.print(
                "[bold red]Error:[/bold red] "
                f"Could not reach Ollama at {backend.base_url}. Start Ollama and try again."
            )
            raise typer.Exit(code=1)
        status = backend.healthcheck()
        if smoke_test and bool(status.get("ready")):
            smoke = backend.smoke_test()
            status["smoke_test"] = f"OK ({smoke.get('response')})"
        elif smoke_test:
            status["smoke_test"] = "skipped (model not ready)"
        else:
            status["smoke_test"] = "disabled"
        _render_local_check(console, status)
        if not bool(status.get("ready")):
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("explain")
def explain(
    input_json: Annotated[Path, typer.Argument(help="Prepared run directory or JSON file with a single simulation step or decision context.")],
    mdmp_cert: Annotated[Optional[Path], typer.Option(help="Signed MDMP artifact required before AI analysis can run.")] = None,
    mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = "auto",
    model: Annotated[str, typer.Option(help="Ollama model name to use.")] = DEFAULT_MINISTRAL_MODEL,
    minimum_grade: Annotated[str, typer.Option(help="Minimum MDMP grade required to allow analysis.")] = "research_grade",
    public_key: Annotated[Optional[Path], typer.Option(help="Explicit MDMP public key for verification.")] = None,
    trust_store: Annotated[Optional[Path], typer.Option(help="MDMP trust store for verification.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option(help="Override the Ollama base URL.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="HTTP timeout for Ollama generation requests.")] = 120.0,
    output: Annotated[Optional[Path], typer.Option(help="Optional file path to save the explanation.")] = None,
) -> None:
    console = Console()
    try:
        resolved_input, resolved_cert, resolved_public_key = _resolve_cli_inputs(
            task="explain",
            input_path=input_json,
            mdmp_cert=mdmp_cert,
            public_key=public_key,
            trust_store=trust_store,
        )
        payload = _load_json_payload(resolved_input, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=resolved_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=resolved_public_key,
            trust_store=trust_store,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )
        response = assistant.explain_decision(payload)
        _write_output(output, response)
        _render_response(console, "IINTS AI Explain", response, output)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("trends")
def trends(
    input_json: Annotated[Path, typer.Argument(help="Prepared run directory or JSON file with glucose trace data or a run payload.")],
    mdmp_cert: Annotated[Optional[Path], typer.Option(help="Signed MDMP artifact required before AI analysis can run.")] = None,
    mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = "auto",
    model: Annotated[str, typer.Option(help="Ollama model name to use.")] = DEFAULT_MINISTRAL_MODEL,
    minimum_grade: Annotated[str, typer.Option(help="Minimum MDMP grade required to allow analysis.")] = "research_grade",
    public_key: Annotated[Optional[Path], typer.Option(help="Explicit MDMP public key for verification.")] = None,
    trust_store: Annotated[Optional[Path], typer.Option(help="MDMP trust store for verification.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option(help="Override the Ollama base URL.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="HTTP timeout for Ollama generation requests.")] = 120.0,
    output: Annotated[Optional[Path], typer.Option(help="Optional file path to save the analysis.")] = None,
) -> None:
    console = Console()
    try:
        resolved_input, resolved_cert, resolved_public_key = _resolve_cli_inputs(
            task="trends",
            input_path=input_json,
            mdmp_cert=mdmp_cert,
            public_key=public_key,
            trust_store=trust_store,
        )
        payload = _load_json_payload(resolved_input, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=resolved_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=resolved_public_key,
            trust_store=trust_store,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )
        response = assistant.analyze_trends(payload)
        _write_output(output, response)
        _render_response(console, "IINTS AI Trends", response, output)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("anomalies")
def anomalies(
    input_json: Annotated[Path, typer.Argument(help="Prepared run directory or JSON file with simulation results or run summary.")],
    mdmp_cert: Annotated[Optional[Path], typer.Option(help="Signed MDMP artifact required before AI analysis can run.")] = None,
    mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = "auto",
    model: Annotated[str, typer.Option(help="Ollama model name to use.")] = DEFAULT_MINISTRAL_MODEL,
    minimum_grade: Annotated[str, typer.Option(help="Minimum MDMP grade required to allow analysis.")] = "research_grade",
    public_key: Annotated[Optional[Path], typer.Option(help="Explicit MDMP public key for verification.")] = None,
    trust_store: Annotated[Optional[Path], typer.Option(help="MDMP trust store for verification.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option(help="Override the Ollama base URL.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="HTTP timeout for Ollama generation requests.")] = 120.0,
    output: Annotated[Optional[Path], typer.Option(help="Optional file path to save the anomaly summary.")] = None,
) -> None:
    console = Console()
    try:
        resolved_input, resolved_cert, resolved_public_key = _resolve_cli_inputs(
            task="anomalies",
            input_path=input_json,
            mdmp_cert=mdmp_cert,
            public_key=public_key,
            trust_store=trust_store,
        )
        payload = _load_json_payload(resolved_input, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=resolved_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=resolved_public_key,
            trust_store=trust_store,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )
        response = assistant.detect_anomalies(payload)
        _write_output(output, response)
        _render_response(console, "IINTS AI Anomalies", response, output)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)


@app.command("report")
def report(
    input_json: Annotated[Path, typer.Argument(help="Prepared run directory or JSON file with run-level simulation outputs.")],
    mdmp_cert: Annotated[Optional[Path], typer.Option(help="Signed MDMP artifact required before AI analysis can run.")] = None,
    mode: Annotated[str, typer.Option(help="AI backend mode. Use 'local' for Ollama/Ministral.")] = "auto",
    model: Annotated[str, typer.Option(help="Ollama model name to use.")] = DEFAULT_MINISTRAL_MODEL,
    minimum_grade: Annotated[str, typer.Option(help="Minimum MDMP grade required to allow analysis.")] = "research_grade",
    public_key: Annotated[Optional[Path], typer.Option(help="Explicit MDMP public key for verification.")] = None,
    trust_store: Annotated[Optional[Path], typer.Option(help="MDMP trust store for verification.")] = None,
    ollama_host: Annotated[Optional[str], typer.Option(help="Override the Ollama base URL.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="HTTP timeout for Ollama generation requests.")] = 120.0,
    output: Annotated[Optional[Path], typer.Option(help="Optional file path to save the markdown report.")] = None,
) -> None:
    console = Console()
    try:
        resolved_input, resolved_cert, resolved_public_key = _resolve_cli_inputs(
            task="report",
            input_path=input_json,
            mdmp_cert=mdmp_cert,
            public_key=public_key,
            trust_store=trust_store,
        )
        payload = _load_json_payload(resolved_input, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=resolved_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=resolved_public_key,
            trust_store=trust_store,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )
        response = assistant.generate_report(payload)
        _write_output(output, response)
        _render_response(console, "IINTS AI Report", response, output)
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
