from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

from .assistant import AIResponse, IINTSAssistant
from .backends import DEFAULT_MINISTRAL_MODEL, OllamaBackend


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
    input_json: Annotated[Path, typer.Argument(help="JSON file with a single simulation step or decision context.")],
    mdmp_cert: Annotated[Path, typer.Option(help="Signed MDMP artifact required before AI analysis can run.")],
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
        payload = _load_json_payload(input_json, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=mdmp_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=public_key,
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
    input_json: Annotated[Path, typer.Argument(help="JSON file with glucose trace data or a run payload.")],
    mdmp_cert: Annotated[Path, typer.Option(help="Signed MDMP artifact required before AI analysis can run.")],
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
        payload = _load_json_payload(input_json, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=mdmp_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=public_key,
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
    input_json: Annotated[Path, typer.Argument(help="JSON file with simulation results or run summary.")],
    mdmp_cert: Annotated[Path, typer.Option(help="Signed MDMP artifact required before AI analysis can run.")],
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
        payload = _load_json_payload(input_json, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=mdmp_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=public_key,
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
    input_json: Annotated[Path, typer.Argument(help="JSON file with run-level simulation outputs.")],
    mdmp_cert: Annotated[Path, typer.Option(help="Signed MDMP artifact required before AI analysis can run.")],
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
        payload = _load_json_payload(input_json, "Input JSON")
        assistant = _build_assistant(
            mdmp_cert=mdmp_cert,
            mode=mode,
            model=model,
            minimum_grade=minimum_grade,
            public_key=public_key,
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
