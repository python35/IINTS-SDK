"""Typer CLI subcommands for IINTS Simulation SDK 2.0 & IINTS Research Lab (`iints lab`)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
import typer

from iints.research import (
    analyze_research_results,
    approve_research_plan,
    execute_research_plan,
    generate_verifiable_report_bundle,
    list_research_templates,
    load_research_plan,
    propose_research_plan,
    save_research_plan,
)
from iints.sdk2 import list_forecasters, run_single_experiment
from iints.sdk2.patients import available_simulator_backends

app = typer.Typer(
    help="IINTS Research Lab & SDK 2.0 — Pre-registered reproducible forecasting & sensor-fault benchmarks."
)
console = Console()


@app.command("templates")
def cmd_templates() -> None:
    """List pre-built scientific research templates and available simulators/forecasters."""
    table = Table(title="IINTS Research Lab Templates")
    table.add_column("Template ID", style="cyan", no_wrap=True)
    table.add_column("Direction", style="green")
    table.add_column("Research Question")
    for item in list_research_templates():
        table.add_row(item["template_id"], item["direction"], item["question"])
    console.print(table)
    console.print(f"[bold]Simulator backends:[/bold] {', '.join(available_simulator_backends())}")
    console.print(f"[bold]Registered forecasters:[/bold] {', '.join(list_forecasters())}")


@app.command("design")
def cmd_design(
    template: str = typer.Option("study_001_cgm_dropout", "--template", "-t", help="Research template ID."),
    out: Path = typer.Option(Path("results/study_001/plan.yaml"), "--out", "-o", help="Output YAML plan path."),
    profiles: int = typer.Option(100, "--profiles", "-n", help="Number of virtual patient profiles."),
    seed: int = typer.Option(42, "--seed", help="Master deterministic seed."),
    approve: bool = typer.Option(False, "--approve", help="Immediately approve and lock the plan."),
    reviewer: str = typer.Option("principal_investigator", "--reviewer", help="Reviewer name when --approve is set."),
) -> None:
    """Propose a pre-registered ResearchPlan YAML (and optionally approve/lock it)."""
    plan = propose_research_plan(template, n_profiles=profiles, seed=seed)
    if approve:
        plan = approve_research_plan(plan, reviewer=reviewer)
    save_research_plan(plan, out)
    console.print(
        f"[bold green]Saved plan:[/bold green] {out} | status=[bold]{plan.status}[/bold] | "
        f"cells={plan.total_simulation_cells()}"
    )


@app.command("approve")
def cmd_approve(
    plan_path: Path = typer.Argument(..., help="Path to proposed ResearchPlan YAML."),
    reviewer: str = typer.Option(..., "--reviewer", "-r", help="Researcher approving the design."),
) -> None:
    """Approve and cryptographically lock a proposed ResearchPlan YAML."""
    plan = load_research_plan(plan_path)
    approved = approve_research_plan(plan, reviewer=reviewer)
    save_research_plan(approved, plan_path)
    console.print(
        f"[bold green]Approved:[/bold green] {plan_path} | reviewer={approved.approved_by} | "
        f"lock={approved.plan_lock_sha256}"
    )


@app.command("run")
def cmd_run(
    plan_path: Path = typer.Argument(..., help="Path to APPROVED ResearchPlan YAML."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Artifact output directory."),
    workers: int = typer.Option(2, "--workers", "-w", help="Parallel worker threads."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore existing checkpoints and rerun from scratch."),
) -> None:
    """Execute an approved ResearchPlan, run paired statistics, and emit a verifiable report."""
    plan = load_research_plan(plan_path)
    dest = out_dir or plan_path.parent
    run_out = execute_research_plan(plan, dest, max_workers=workers, resume=not no_resume)
    analysis = analyze_research_results(run_out["cell_metrics"], plan)
    bundle = generate_verifiable_report_bundle(run_out, analysis)
    console.print(f"[bold green]Completed {plan.plan_id}![/bold green] Report: {bundle['report_md']}")


@app.command("single")
def cmd_single(
    spec_yaml: Path = typer.Argument(..., help="Path to single SDK 2.0 ExperimentSpec YAML."),
    out_dir: Path = typer.Option(Path("results/single_run"), "--out-dir", "-o", help="Output directory."),
) -> None:
    """Run a single declarative SDK 2.0 ExperimentSpec YAML."""
    res = run_single_experiment(spec_yaml, output_dir=out_dir)
    console.print(res["metrics"].to_string(index=False))
    console.print(f"[bold green]Artifacts & provenance written to:[/bold green] {out_dir}")


@app.command("studio")
def cmd_studio(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    workspace: Path = typer.Option(Path("results/research_studio"), "--workspace"),
) -> None:
    """Launch the interactive IINTS Research Studio 2.0 web interface."""
    import uvicorn
    from iints.studio.app import create_studio_app

    console.print(f"[bold cyan]Starting IINTS Research Studio 2.0 at http://{host}:{port}[/bold cyan]")
    uvicorn.run(create_studio_app(workspace), host=host, port=port)
