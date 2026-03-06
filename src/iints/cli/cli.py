import typer  # type: ignore
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional, cast
from dataclasses import asdict
from typing_extensions import Annotated
from pydantic import ValidationError
import os
import importlib.util
import importlib
import sys
import json
import tempfile
import time
import yaml # Added for Virtual Patient Registry
import pandas as pd # Added for DataFrame in benchmark results
import numpy as np

from rich.console import Console  # type: ignore # For pretty printing
from rich.table import Table  # type: ignore # For comparison table
from rich.panel import Panel  # type: ignore # For nicer auto-doc output

import iints # Import the top-level SDK package
from iints.analysis.baseline import run_baseline_comparison, write_baseline_comparison
from iints.api.registry import list_algorithm_plugins
from iints.core.patient.profile import PatientProfile
from iints.core.safety import SafetyConfig
from iints.scenarios import ScenarioGeneratorConfig, generate_random_scenario
from iints.data.nightscout import NightscoutConfig, import_nightscout
from iints.data.tidepool import TidepoolClient
from iints.data.importer import (
    export_demo_csv,
    export_standard_csv,
    guess_column_mapping,
    import_cgm_dataframe,
    load_demo_dataframe,
    scenario_from_csv,
    scenario_from_dataframe,
)
from iints.data.registry import (
    load_dataset_registry,
    get_dataset,
    fetch_dataset,
    DatasetFetchError,
    DatasetRegistryError,
)
from iints.data.contracts import load_contract_yaml
from iints.data.runner import (
    ContractRunner,
    MDMP_GRADE_ORDER,
    mdmp_grade_meets_minimum,
)
from iints.data.mdmp_visualizer import build_mdmp_dashboard_html
from iints.data.synthetic_mirror import generate_synthetic_mirror
from iints.utils.run_io import (
    build_run_metadata,
    build_run_manifest,
    generate_run_id,
    maybe_sign_manifest,
    resolve_output_dir,
    resolve_seed,
    write_json,
)
from iints.validation import (
    apply_contract_to_config,
    build_stress_events,
    compute_run_metrics,
    evaluate_expected_ranges,
    evaluate_run,
    format_validation_error,
    load_golden_benchmark_pack,
    load_contract_spec,
    load_validation_profiles,
    run_deterministic_replay_check,
    load_scenario,
    load_patient_config,
    load_patient_config_by_name,
    migrate_scenario_dict,
    scenario_to_payloads,
    scenario_warnings,
    verify_safety_contract,
    validate_patient_config_dict,
    validate_scenario_dict,
)


app = typer.Typer(help="IINTS-AF SDK CLI - Intelligent Insulin Titration System for Artificial Pancreas research.")
docs_app = typer.Typer(help="Generate documentation and technical summaries for IINTS-AF components.")
presets_app = typer.Typer(help="Clinic-safe presets and quickstart runs.")
profiles_app = typer.Typer(help="Patient profiles and physiological presets.")
data_app = typer.Typer(help="Official datasets and data packs.")
mdmp_app = typer.Typer(help="MDMP protocol commands (separate namespace).")
scenarios_app = typer.Typer(help="Scenario generation and utilities.")
algorithms_app = typer.Typer(help="Algorithm registry and plugins.")
app.add_typer(docs_app, name="docs")
app.add_typer(presets_app, name="presets")
app.add_typer(profiles_app, name="profiles")
app.add_typer(data_app, name="data")
app.add_typer(mdmp_app, name="mdmp")
app.add_typer(scenarios_app, name="scenarios")
app.add_typer(algorithms_app, name="algorithms")

def _load_algorithm_instance(algo: Path, console: Console) -> iints.InsulinAlgorithm:
    if not algo.is_file():
        console.print(f"[bold red]Error: Algorithm file '{algo}' not found.[/bold red]")
        raise typer.Exit(code=1)

    module_name = algo.stem
    spec = importlib.util.spec_from_file_location(module_name, algo)
    if spec is None:
        console.print(f"[bold red]Error: Could not load module spec for {algo}[/bold red]")
        raise typer.Exit(code=1)

    module = importlib.util.module_from_spec(spec)
    module.iints = iints # type: ignore
    sys.modules[module_name] = module
    try:
        if spec.loader:
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Could not load module loader for {algo}")
    except Exception as e:
        console.print(f"[bold red]Error loading algorithm module {algo}: {e}[/bold red]")
        raise typer.Exit(code=1)

    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            return obj()

    console.print(f"[bold red]Error: No subclass of InsulinAlgorithm found in {algo}[/bold red]")
    raise typer.Exit(code=1)


def _load_algorithm_instance_silent(algo: Path) -> iints.InsulinAlgorithm:
    if not algo.is_file():
        raise FileNotFoundError(f"Algorithm file '{algo}' not found.")
    module_name = algo.stem
    spec = importlib.util.spec_from_file_location(module_name, algo)
    if spec is None:
        raise ImportError(f"Could not load module spec for {algo}")
    module = importlib.util.module_from_spec(spec)
    module.iints = iints  # type: ignore
    sys.modules[module_name] = module
    if spec.loader:
        spec.loader.exec_module(module)
    else:
        raise ImportError(f"Could not load module loader for {algo}")
    for _, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            return obj()
    raise ImportError(f"No subclass of InsulinAlgorithm found in {algo}")

def _load_presets() -> List[Dict[str, Any]]:
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        content = files("iints.presets").joinpath("presets.json").read_text()
    else:
        from importlib import resources
        content = resources.read_text("iints.presets", "presets.json")
    return json.loads(content)


def _load_evidence_sources() -> Dict[str, Any]:
    if sys.version_info >= (3, 9):
        from importlib.resources import files

        content = files("iints.presets").joinpath("evidence_sources.yaml").read_text()
    else:
        from importlib import resources

        content = resources.read_text("iints.presets", "evidence_sources.yaml")
    raw = yaml.safe_load(content) or {}
    if not isinstance(raw, dict):
        raise ValueError("evidence_sources.yaml must contain a top-level mapping")
    return raw


def _filtered_evidence_sources(category: Optional[str] = None) -> Dict[str, Any]:
    payload = _load_evidence_sources()
    rows = payload.get("sources", [])
    if not isinstance(rows, list):
        raise ValueError("evidence_sources.yaml must contain a 'sources' list")
    selected: List[Dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        if category and str(entry.get("category", "")).strip().lower() != category.strip().lower():
            continue
        selected.append(entry)
    return {
        "version": payload.get("version", 1),
        "updated_utc": payload.get("updated_utc"),
        "category": category,
        "count": len(selected),
        "sources": selected,
    }


def _write_certification_summary(
    output_dir: Path,
    outputs: Dict[str, Any],
    profile: str,
    report: Dict[str, Any],
    source_manifest_path: Optional[Path],
) -> Path:
    checks = report.get("checks", [])
    failed_checks = [check for check in checks if not bool(check.get("passed", False))]
    status = "PASS" if bool(report.get("passed", False)) else "FAIL"
    required_passed = report.get("required_checks_passed", 0)
    required_total = report.get("required_checks_total", 0)

    lines = [
        "# IINTS-AF Certification Bundle",
        "",
        f"- Status: **{status}** ({required_passed}/{required_total} required checks passed)",
        f"- Validation profile: `{profile}`",
        f"- Results CSV: `{outputs.get('results_csv', '')}`",
        f"- Validation report: `validation_report.json`",
        f"- Run manifest: `{outputs.get('run_manifest_path', '')}`",
    ]
    if source_manifest_path is not None:
        lines.append(f"- Source manifest: `{source_manifest_path}`")
    if failed_checks:
        lines.append("")
        lines.append("## Failed Checks")
        for check in failed_checks:
            lines.append(
                f"- `{check.get('label', check.get('metric', 'metric'))}`: "
                f"value={check.get('value')} threshold {check.get('op')} {check.get('threshold')}"
            )
    else:
        lines.append("")
        lines.append("All required checks passed.")

    lines.extend(
        [
            "",
            "## Next Steps",
            "1. Open `clinical_report.pdf` for visual review.",
            "2. Inspect `validation_report.json` for gate-level details.",
            "3. Archive `run_manifest.json`, `run_metadata.json`, and `sources_manifest.json` (if present).",
        ]
    )

    summary_path = output_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def _get_preset(name: str) -> Dict[str, Any]:
    presets = _load_presets()
    for preset in presets:
        if preset.get("name") == name:
            return preset
    raise KeyError(name)


def _parse_column_mapping(items: List[str], console: Console) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            console.print(f"[bold red]Invalid mapping '{item}'. Use key=value.[/bold red]")
            raise typer.Exit(code=1)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            console.print(f"[bold red]Invalid mapping '{item}'. Use key=value.[/bold red]")
            raise typer.Exit(code=1)
        mapping[key] = value
    return mapping


def _load_json_dict(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _parse_float_csv(values: str, label: str) -> List[float]:
    parsed: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid float in {label}: '{token}'") from exc
    if not parsed:
        raise ValueError(f"{label} cannot be empty")
    return parsed


def _load_predictor_service_from_path(predictor_path: Optional[Path], console: Console) -> Optional[object]:
    if predictor_path is None:
        return None
    if not predictor_path.is_file():
        console.print(f"[bold red]Predictor checkpoint not found: {predictor_path}[/bold red]")
        raise typer.Exit(code=1)
    try:
        from iints.research.predictor import load_predictor_service

        return load_predictor_service(predictor_path)
    except Exception as exc:
        console.print(
            "[bold red]Could not load predictor service. Install research extras "
            "(`pip install iints-sdk-python35[research]`) and verify the checkpoint format.[/bold red]"
        )
        console.print(f"[red]Details:[/red] {exc}")
        raise typer.Exit(code=1)


def _build_safety_config_from_options(**kwargs: Any) -> Optional[SafetyConfig]:
    if all(value is None for value in kwargs.values()):
        return None
    config = SafetyConfig()
    for key, value in kwargs.items():
        if value is not None:
            setattr(config, key, value)
    return config


def _build_safety_config_from_dict(values: Optional[Dict[str, Any]]) -> Optional[SafetyConfig]:
    if not values:
        return None
    config = SafetyConfig()
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _run_parallel_job(job: Dict[str, Any]) -> Dict[str, Any]:
    algo_path = Path(job["algo"])
    scenario_path = Path(job["scenario_path"])
    output_dir = Path(job["output_dir"])
    patient_config = job["patient_config"]
    safety_config = _build_safety_config_from_dict(job.get("safety_overrides"))
    algorithm_instance = _load_algorithm_instance_silent(algo_path)
    predictor = None
    predictor_path = job.get("predictor_path")
    if predictor_path:
        from iints.research.predictor import load_predictor_service

        predictor = load_predictor_service(Path(predictor_path))

    outputs = iints.run_simulation(
        algorithm=algorithm_instance,
        scenario=str(scenario_path),
        patient_config=patient_config,
        duration_minutes=int(job["duration_minutes"]),
        time_step=int(job["time_step"]),
        seed=job.get("seed"),
        output_dir=output_dir,
        compare_baselines=bool(job.get("compare_baselines")),
        export_audit=bool(job.get("export_audit")),
        generate_report=bool(job.get("generate_report")),
        safety_config=safety_config,
        predictor=predictor,
    )
    outputs.pop("results", None)
    safety_report = outputs.get("safety_report", {})
    return {
        "scenario": scenario_path.stem,
        "patient": job["patient_label"],
        "output_dir": str(output_dir),
        "results_csv": outputs.get("results_csv"),
        "report_pdf": outputs.get("report_pdf"),
        "run_manifest": outputs.get("run_manifest_path"),
        "terminated_early": safety_report.get("terminated_early", False),
        "total_violations": safety_report.get("total_violations", 0),
        "error": "",
    }

@app.command()
def evaluate(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    population: Annotated[int, typer.Option(help="Number of virtual patients to simulate")] = 100,
    patient_config_name: Annotated[str, typer.Option("--patient-config", help="Base patient configuration name")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option("--patient-config-path", help="Path to base patient config YAML")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option("--scenario", help="Path to scenario JSON")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Time step in minutes")] = 5,
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory")] = None,
    max_workers: Annotated[Optional[int], typer.Option(help="Max parallel workers (default: all cores)")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for reproducibility")] = None,
    patient_model: Annotated[str, typer.Option("--patient-model", help="Patient model: auto, bergman, custom, simglucose")] = "auto",
):
    """
    Run a Monte Carlo population evaluation of an algorithm.

    Generates N virtual patients with physiological variation, runs each
    through the simulator in parallel, and reports aggregate TIR, hypo-risk,
    and Safety Index with 95% confidence intervals.

    Example:
        iints evaluate --algo my_algo.py --population 500 --seed 42
    """
    console = Console()
    console.print(f"[bold blue]IINTS-AF Population Evaluation[/bold blue]")
    console.print(f"  Algorithm:       [green]{algo.name}[/green]")
    console.print(f"  Population size: [cyan]{population}[/cyan]")
    console.print(f"  Patient model:   [cyan]{patient_model}[/cyan]")
    console.print(f"  Duration:        {duration} min")
    console.print()

    # Validate algo file exists
    _load_algorithm_instance(algo, console)

    patient_config: Union[str, Path] = str(patient_config_path) if patient_config_path else patient_config_name
    scenario = str(scenario_path) if scenario_path else None

    from iints.highlevel import run_population

    with console.status("[bold green]Running population evaluation...", spinner="dots"):
        results = run_population(
            algo_path=str(algo),
            n_patients=population,
            scenario=scenario,
            patient_config=patient_config,
            duration_minutes=duration,
            time_step=time_step,
            seed=seed,
            output_dir=str(output_dir) if output_dir else None,
            max_workers=max_workers,
            patient_model_type=patient_model,
        )

    report = results["population_report"]
    agg = report["aggregate_metrics"]
    safety_agg = report["aggregate_safety"]

    # --- Results table ---
    table = Table(title=f"Population Evaluation Results (N={population})")
    table.add_column("Metric", style="cyan", min_width=25)
    table.add_column("Mean", justify="right", style="green")
    table.add_column("95% CI", justify="right", style="yellow")
    table.add_column("Std", justify="right", style="dim")

    _METRIC_DISPLAY = {
        "tir_70_180": "TIR 70-180 mg/dL (%)",
        "tir_below_70": "Time <70 mg/dL (%)",
        "tir_below_54": "Time <54 mg/dL (%)",
        "tir_above_180": "Time >180 mg/dL (%)",
        "mean_glucose": "Mean Glucose (mg/dL)",
        "cv": "Coefficient of Variation (%)",
        "gmi": "Glucose Management Indicator (%)",
    }

    for metric_key, stats in agg.items():
        label = _METRIC_DISPLAY.get(metric_key, metric_key)
        table.add_row(
            label,
            f"{stats['mean']:.1f}",
            f"[{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]",
            f"{stats['std']:.1f}",
        )

    if "safety_index" in safety_agg:
        si = safety_agg["safety_index"]
        table.add_row(
            "[bold]Safety Index[/bold]",
            f"[bold]{si['mean']:.1f}[/bold]",
            f"[{si['ci_lower']:.1f}, {si['ci_upper']:.1f}]",
            f"{si['std']:.1f}",
        )

    console.print(table)

    # --- Grade distribution ---
    if "grade_distribution" in safety_agg:
        console.print()
        grade_table = Table(title="Safety Grade Distribution")
        grade_table.add_column("Grade", style="bold")
        grade_table.add_column("Count", justify="right")
        grade_table.add_column("Percentage", justify="right")
        for grade in ["A", "B", "C", "D", "F"]:
            count = safety_agg["grade_distribution"].get(grade, 0)
            pct = count / population * 100 if population else 0
            grade_table.add_row(grade, str(count), f"{pct:.1f}%")
        console.print(grade_table)

    etr = safety_agg.get("early_termination_rate")
    if etr is not None and etr > 0:
        console.print(f"\n[yellow]Early termination rate: {etr * 100:.1f}%[/yellow]")

    console.print(f"\n[green]Results saved to:[/green] {results['output_dir']}")
    console.print(f"  - population_summary.csv")
    console.print(f"  - population_report.json")
    console.print(f"  - population_report.pdf")


@app.command("doctor")
def doctor(
    smoke_run: Annotated[bool, typer.Option(help="Run a short deterministic smoke simulation")] = False,
    smoke_duration: Annotated[int, typer.Option(help="Smoke simulation duration in minutes")] = 30,
):
    """
    Environment and installation health-check for developers.
    """
    console = Console()
    required_checks: List[Tuple[str, bool, str]] = []

    py_ok = sys.version_info >= (3, 10)
    required_checks.append(
        ("Python >= 3.10", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    )

    package_checks = [
        ("pandas", True, "core"),
        ("numpy", True, "core"),
        ("typer", True, "core"),
        ("rich", True, "core"),
        ("yaml", True, "core"),
        ("torch", False, "research"),
        ("onnx", False, "research"),
        ("onnxscript", False, "research"),
        ("h5py", False, "research"),
    ]

    table = Table(title="IINTS Doctor", show_lines=False)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail", style="dim")

    for check_name, passed, detail in required_checks:
        table.add_row(check_name, "[green]OK[/green]" if passed else "[red]FAIL[/red]", detail)

    for module_name, required, category in package_checks:
        available = _module_available(module_name)
        status = "[green]OK[/green]" if available else ("[red]MISSING[/red]" if required else "[yellow]optional[/yellow]")
        extra = f"{category} dependency"
        table.add_row(f"Module: {module_name}", status, extra)
        if required:
            required_checks.append((f"Module: {module_name}", available, extra))

    try:
        presets = _load_presets()
        presets_ok = len(presets) > 0
        table.add_row("Preset bundle", "[green]OK[/green]" if presets_ok else "[red]FAIL[/red]", f"{len(presets)} presets loaded")
        required_checks.append(("Preset bundle", presets_ok, f"{len(presets)} presets"))
    except Exception as exc:
        table.add_row("Preset bundle", "[red]FAIL[/red]", str(exc))
        required_checks.append(("Preset bundle", False, str(exc)))

    try:
        profiles = load_validation_profiles()
        profiles_ok = len(profiles) > 0
        table.add_row(
            "Validation profiles",
            "[green]OK[/green]" if profiles_ok else "[red]FAIL[/red]",
            ", ".join(sorted(profiles.keys())) if profiles else "none",
        )
        required_checks.append(("Validation profiles", profiles_ok, f"{len(profiles)} profiles"))
    except Exception as exc:
        table.add_row("Validation profiles", "[red]FAIL[/red]", str(exc))
        required_checks.append(("Validation profiles", False, str(exc)))

    if smoke_run:
        try:
            with tempfile.TemporaryDirectory(prefix="iints_doctor_") as tmp:
                smoke_outputs = iints.run_simulation(
                    algorithm=iints.ConstantDoseAlgorithm(),
                    duration_minutes=max(smoke_duration, 5),
                    output_dir=Path(tmp),
                    compare_baselines=False,
                    export_audit=False,
                    generate_report=False,
                    seed=42,
                )
                results_path = smoke_outputs.get("results_csv", "")
                smoke_ok = bool(results_path) and Path(results_path).is_file()
                detail = f"results: {results_path}" if smoke_ok else "simulation output missing"
                table.add_row("Smoke simulation", "[green]OK[/green]" if smoke_ok else "[red]FAIL[/red]", detail)
                required_checks.append(("Smoke simulation", smoke_ok, detail))
        except Exception as exc:
            table.add_row("Smoke simulation", "[red]FAIL[/red]", str(exc))
            required_checks.append(("Smoke simulation", False, str(exc)))

    console.print(table)
    if not all(passed for _, passed, _ in required_checks):
        raise typer.Exit(code=1)


@app.command("validation-profiles")
def validation_profiles(
    profiles_path: Annotated[Optional[Path], typer.Option(help="Optional custom profiles YAML")] = None,
):
    """List available run-validation profiles."""
    console = Console()
    try:
        profiles = load_validation_profiles(profiles_path)
    except Exception as exc:
        console.print(f"[bold red]Could not load validation profiles: {exc}[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title="Validation Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Min Duration (min)", justify="right")
    table.add_column("Checks", justify="right")
    table.add_column("Description")
    for profile_id, profile in sorted(profiles.items()):
        table.add_row(
            profile_id,
            str(profile.min_duration_minutes),
            str(len(profile.checks)),
            profile.description,
        )
    console.print(table)


@app.command("replay-check")
def replay_check(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt)")] = None,
    patient_config_name: Annotated[str, typer.Option(help="Patient config name")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Patient config YAML path")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to scenario JSON")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 240,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    seed: Annotated[int, typer.Option(help="Deterministic seed")] = 42,
    repeats: Annotated[int, typer.Option(help="Replay runs to compare")] = 2,
    output_json: Annotated[Optional[Path], typer.Option(help="Optional output JSON path")] = None,
    fail_on_mismatch: Annotated[bool, typer.Option(help="Exit code 1 if replay mismatch is detected")] = True,
):
    """
    Deterministic replay suite: same seed, same config, same output hashes.
    """
    console = Console()
    algorithm_instance = _load_algorithm_instance(algo, console)
    predictor = _load_predictor_service_from_path(predictor_path, console)

    patient_config: Union[str, Path]
    if patient_config_path:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Patient config file not found: {patient_config_path}[/bold red]")
            raise typer.Exit(code=1)
        patient_config = patient_config_path
    else:
        patient_config = patient_config_name

    scenario_payload: Optional[Dict[str, Any]] = None
    if scenario_path:
        if not scenario_path.is_file():
            console.print(f"[bold red]Scenario file not found: {scenario_path}[/bold red]")
            raise typer.Exit(code=1)
        scenario_payload = load_scenario(scenario_path).model_dump()

    result = run_deterministic_replay_check(
        algorithm=algorithm_instance,
        scenario=scenario_payload,
        patient_config=patient_config,
        duration_minutes=duration,
        time_step=time_step,
        seed=seed,
        repeats=max(2, repeats),
        predictor=predictor,
    )

    table = Table(title="Deterministic Replay Check")
    table.add_column("Run", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Results Hash")
    table.add_column("Safety Hash")
    for idx, digest in enumerate(result.digests, start=1):
        table.add_row(str(idx), str(digest.rows), digest.results_hash[:16], digest.safety_hash[:16])
    console.print(table)
    console.print("[green]PASS[/green]" if result.passed else "[red]FAIL[/red]")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result.to_dict(), indent=2))
        console.print(f"[green]Replay report written:[/green] {output_json}")

    if fail_on_mismatch and not result.passed:
        raise typer.Exit(code=1)


@app.command("golden-benchmark")
def golden_benchmark(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt)")] = None,
    pack_path: Annotated[Optional[Path], typer.Option(help="Optional golden benchmark YAML path")] = None,
    output_dir: Annotated[Optional[Path], typer.Option(help="Optional output directory for benchmark runs")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    duration_override: Annotated[Optional[int], typer.Option(help="Optional duration override (minutes)")] = None,
    output_json: Annotated[Optional[Path], typer.Option(help="Write benchmark report JSON")] = None,
    fail_on_check: Annotated[bool, typer.Option(help="Exit code 1 when any scenario fails")] = True,
):
    """
    Run a golden scenario pack and validate metrics against expected ranges.
    """
    console = Console()
    algorithm_instance = _load_algorithm_instance(algo, console)
    predictor = _load_predictor_service_from_path(predictor_path, console)

    try:
        pack = load_golden_benchmark_pack(pack_path)
    except Exception as exc:
        console.print(f"[bold red]Could not load golden benchmark pack: {exc}[/bold red]")
        raise typer.Exit(code=1)

    if not pack.scenarios:
        console.print("[bold red]Golden benchmark pack contains no scenarios.[/bold red]")
        raise typer.Exit(code=1)

    run_root = output_dir
    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    if run_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="iints_golden_")
        run_root = Path(temp_dir.name)
    run_root.mkdir(parents=True, exist_ok=True)

    scenario_results: List[Dict[str, Any]] = []
    all_passed = True

    for idx, scenario_spec in enumerate(pack.scenarios):
        algorithm_instance.reset()
        try:
            preset = _get_preset(scenario_spec.preset)
        except KeyError:
            console.print(f"[bold red]Unknown preset in golden pack: {scenario_spec.preset}[/bold red]")
            if temp_dir is not None:
                temp_dir.cleanup()
            raise typer.Exit(code=1)

        duration_minutes = int(duration_override if duration_override is not None else preset.get("duration_minutes", 720))
        time_step_minutes = int(preset.get("time_step_minutes", 5))
        scenario_output = run_root / f"{idx+1:02d}_{scenario_spec.preset}"

        outputs = iints.run_simulation(
            algorithm=algorithm_instance,
            scenario=preset.get("scenario"),
            patient_config=str(preset.get("patient_config", "default_patient")),
            duration_minutes=duration_minutes,
            time_step=time_step_minutes,
            seed=seed,
            output_dir=scenario_output,
            compare_baselines=False,
            export_audit=False,
            generate_report=False,
            predictor=predictor,
        )

        metrics = compute_run_metrics(
            outputs["results"],
            safety_report=outputs.get("safety_report"),
            duration_minutes=duration_minutes,
        )
        checks = evaluate_expected_ranges(metrics, scenario_spec.expected)
        passed = all(item.get("passed", False) for item in checks.values())
        all_passed = all_passed and passed

        scenario_results.append(
            {
                "preset": scenario_spec.preset,
                "description": scenario_spec.description,
                "passed": passed,
                "metrics": metrics,
                "checks": checks,
                "output_dir": str(scenario_output),
            }
        )

    table = Table(title=f"Golden Benchmark Pack — {pack.name} v{pack.version}")
    table.add_column("Preset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Checks", justify="right")
    table.add_column("Failed Metrics")
    for row in scenario_results:
        failed_metrics = [metric for metric, check in row["checks"].items() if not check.get("passed", False)]
        table.add_row(
            row["preset"],
            "[green]PASS[/green]" if row["passed"] else "[red]FAIL[/red]",
            str(len(row["checks"])),
            ", ".join(failed_metrics) if failed_metrics else "-",
        )
    console.print(table)
    console.print("[green]Golden benchmark PASS[/green]" if all_passed else "[red]Golden benchmark FAIL[/red]")

    payload = {
        "pack": {"name": pack.name, "version": pack.version},
        "seed": seed,
        "duration_override": duration_override,
        "passed": all_passed,
        "scenarios": scenario_results,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Golden benchmark report written:[/green] {output_json}")

    if temp_dir is not None:
        temp_dir.cleanup()

    if fail_on_check and not all_passed:
        raise typer.Exit(code=1)


@app.command("validate-run")
def validate_run(
    results_csv: Annotated[Path, typer.Option(help="Path to simulation results CSV")],
    profile: Annotated[str, typer.Option(help="Validation profile id")] = "research_default",
    safety_report_path: Annotated[Optional[Path], typer.Option(help="Optional safety report JSON")] = None,
    duration_minutes: Annotated[Optional[int], typer.Option(help="Override run duration (minutes)")] = None,
    profiles_path: Annotated[Optional[Path], typer.Option(help="Optional custom profiles YAML")] = None,
    output_json: Annotated[Optional[Path], typer.Option(help="Write validation report JSON")] = None,
    fail_on_check: Annotated[bool, typer.Option(help="Exit with code 1 when required checks fail")] = True,
):
    """
    Validate a simulation run against a profile (TIR/hypo/CV/safety gates).
    """
    console = Console()
    if not results_csv.is_file():
        console.print(f"[bold red]Results file not found: {results_csv}[/bold red]")
        raise typer.Exit(code=1)

    try:
        profiles = load_validation_profiles(profiles_path)
    except Exception as exc:
        console.print(f"[bold red]Could not load validation profiles: {exc}[/bold red]")
        raise typer.Exit(code=1)

    selected_profile = profiles.get(profile)
    if selected_profile is None:
        available = ", ".join(sorted(profiles.keys()))
        console.print(f"[bold red]Unknown profile '{profile}'. Available: {available}[/bold red]")
        raise typer.Exit(code=1)

    safety_report: Optional[Dict[str, Any]] = None
    if safety_report_path is not None:
        if not safety_report_path.is_file():
            console.print(f"[bold red]Safety report not found: {safety_report_path}[/bold red]")
            raise typer.Exit(code=1)
        try:
            safety_report = _load_json_dict(safety_report_path, "safety report")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)

    try:
        results_df = pd.read_csv(results_csv)
    except Exception as exc:
        console.print(f"[bold red]Could not read results CSV: {exc}[/bold red]")
        raise typer.Exit(code=1)

    try:
        report = evaluate_run(
            results_df,
            profile=selected_profile,
            safety_report=safety_report,
            duration_minutes=duration_minutes,
        )
    except Exception as exc:
        console.print(f"[bold red]Run validation failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    summary = Table(title=f"Run Validation — {selected_profile.profile_id}")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Profile description", selected_profile.description)
    summary.add_row(
        "Required checks",
        f"{report.required_checks_passed}/{report.required_checks_total}",
    )
    summary.add_row("Score", f"{report.score:.1f}")
    summary.add_row("Overall", "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]")
    console.print(summary)

    checks_table = Table(title="Validation Checks")
    checks_table.add_column("Status", style="bold")
    checks_table.add_column("Check")
    checks_table.add_column("Value", justify="right")
    checks_table.add_column("Target", justify="right")
    checks_table.add_column("Required", justify="center")
    for check in report.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        value = "n/a" if check.value is None else f"{check.value:.2f}"
        target = f"{check.rule.op} {check.rule.threshold:g}"
        checks_table.add_row(
            status,
            check.rule.label,
            value,
            target,
            "yes" if check.rule.required else "no",
        )
    console.print(checks_table)

    metrics_table = Table(title="Computed Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right")
    for key, metric_value in sorted(report.metrics.items()):
        metrics_table.add_row(key, f"{metric_value:.3f}")
    console.print(metrics_table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"[green]Validation report written:[/green] {output_json}")

    if fail_on_check and not report.passed:
        raise typer.Exit(code=1)


@app.command("contract-verify")
def contract_verify(
    contract_path: Annotated[Optional[Path], typer.Option(help="Optional YAML safety-contract file")] = None,
    glucose_min: Annotated[float, typer.Option(help="Minimum glucose to test (mg/dL)")] = 50.0,
    glucose_max: Annotated[float, typer.Option(help="Maximum glucose to test (mg/dL)")] = 160.0,
    glucose_step: Annotated[float, typer.Option(help="Glucose grid step (mg/dL)")] = 5.0,
    trend_min: Annotated[float, typer.Option(help="Minimum trend to test (mg/dL/min)")] = -5.0,
    trend_max: Annotated[float, typer.Option(help="Maximum trend to test (mg/dL/min)")] = 3.0,
    trend_step: Annotated[float, typer.Option(help="Trend grid step (mg/dL/min)")] = 0.5,
    proposed_doses: Annotated[str, typer.Option(help="Comma-separated proposed insulin doses (U)")] = "0.0,0.5,1.0,2.0,4.0",
    iob_values: Annotated[Optional[str], typer.Option(help="Optional comma-separated current IOB values (U)")] = None,
    output_json: Annotated[Optional[Path], typer.Option(help="Write contract verification report JSON")] = None,
    fail_on_violation: Annotated[bool, typer.Option(help="Exit with code 1 when any contract violation is found")] = True,
):
    """
    Compile and formally verify the deterministic safety contract on a full input grid.
    """
    console = Console()
    try:
        spec = load_contract_spec(contract_path)
    except Exception as exc:
        console.print(f"[bold red]Could not load contract spec: {exc}[/bold red]")
        raise typer.Exit(code=1)

    if glucose_step <= 0 or trend_step <= 0:
        console.print("[bold red]glucose-step and trend-step must be > 0[/bold red]")
        raise typer.Exit(code=1)

    try:
        dose_values = _parse_float_csv(proposed_doses, "proposed-doses")
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)
    iob_grid = None
    if iob_values is not None:
        try:
            iob_grid = _parse_float_csv(iob_values, "iob-values")
        except ValueError as exc:
            console.print(f"[bold red]{exc}[/bold red]")
            raise typer.Exit(code=1)

    glucose_values = np.arange(glucose_min, glucose_max + 1e-9, glucose_step).tolist()
    trend_values = np.arange(trend_min, trend_max + 1e-9, trend_step).tolist()
    report = verify_safety_contract(
        spec,
        glucose_values=glucose_values,
        trend_values=trend_values,
        proposed_doses=dose_values,
        iob_values=iob_grid,
    )

    table = Table(title="Safety Contract Verification")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Contract enabled", str(spec.contract_enabled))
    table.add_row("Glucose threshold", f"{spec.contract_glucose_threshold:.2f} mg/dL")
    table.add_row("Trend threshold", f"{spec.contract_trend_threshold_mgdl_min:.2f} mg/dL/min")
    table.add_row("Max IOB", f"{spec.contract_max_iob_units:.2f} U")
    table.add_row("Max bolus", f"{spec.contract_max_bolus_units:.2f} U")
    table.add_row("Hypo cutoff", f"{spec.contract_hypo_cutoff_mgdl:.2f} mg/dL")
    table.add_row("Total cases", str(report.total_cases))
    table.add_row("Expected blocked cases", str(report.expected_block_cases))
    table.add_row("Blocked cases", str(report.blocked_cases))
    table.add_row("Violations", str(len(report.violations)))
    table.add_row("Status", "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]")
    console.print(table)

    if report.violations:
        violations_table = Table(title="Contract Violations (first 20)")
        violations_table.add_column("Glucose", justify="right")
        violations_table.add_column("Trend", justify="right")
        violations_table.add_column("Proposed", justify="right")
        violations_table.add_column("Approved", justify="right")
        violations_table.add_column("Reason")
        for violation in report.violations[:20]:
            violations_table.add_row(
                f"{violation.glucose_mgdl:.1f}",
                f"{violation.trend_mgdl_min:.2f}",
                f"{violation.proposed_insulin_units:.2f}",
                f"{violation.approved_insulin_units:.2f}",
                violation.reason,
            )
        console.print(violations_table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"[green]Contract report written:[/green] {output_json}")

    if fail_on_violation and not report.passed:
        raise typer.Exit(code=1)


@app.command("certify-run")
def certify_run(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    profile: Annotated[str, typer.Option(help="Validation profile id")] = "research_default",
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt) for dual-guard forecasting")] = None,
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Path to a patient config YAML")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to scenario JSON")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Directory to save outputs")] = Path("results/certified_run"),
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    export_sources: Annotated[bool, typer.Option(help="Write sources_manifest.json with peer-reviewed references.")] = True,
    source_category: Annotated[Optional[str], typer.Option(help="Optional evidence source category filter (guideline, trial, model, ...).")] = None,
    write_summary: Annotated[bool, typer.Option(help="Write SUMMARY.md with key artifacts and next steps.")] = True,
    fail_on_check: Annotated[bool, typer.Option(help="Exit code 1 when validation profile fails")] = True,
):
    """
    One-command certification pipeline: run-full + profile validation + artifacts.
    """
    console = Console()
    algorithm_instance = _load_algorithm_instance(algo, console)
    predictor = _load_predictor_service_from_path(predictor_path, console)

    patient_config: Union[str, Path]
    if patient_config_path is not None:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Patient config file not found: {patient_config_path}[/bold red]")
            raise typer.Exit(code=1)
        patient_config = patient_config_path
    else:
        patient_config = patient_config_name

    outputs = iints.run_full(
        algorithm=algorithm_instance,
        scenario=str(scenario_path) if scenario_path else None,
        patient_config=patient_config,
        duration_minutes=duration,
        time_step=time_step,
        seed=seed,
        output_dir=output_dir,
        predictor=predictor,
    )

    results_csv = Path(outputs["results_csv"])
    safety_summary_path = Path(outputs["audit"]["summary"]) if "audit" in outputs else None

    try:
        profiles = load_validation_profiles()
    except Exception as exc:
        console.print(f"[bold red]Could not load validation profiles: {exc}[/bold red]")
        raise typer.Exit(code=1)
    selected_profile = profiles.get(profile)
    if selected_profile is None:
        console.print(f"[bold red]Unknown profile '{profile}'.[/bold red]")
        raise typer.Exit(code=1)

    results_df = pd.read_csv(results_csv)
    safety_report = _load_json_dict(safety_summary_path, "safety summary") if safety_summary_path else {}
    report = evaluate_run(
        results_df,
        profile=selected_profile,
        safety_report=safety_report,
        duration_minutes=duration,
    )

    validation_path = Path(outputs["output_dir"]) / "validation_report.json"
    report_payload = report.to_dict()
    validation_path.write_text(json.dumps(report_payload, indent=2))
    console.print(f"[green]Validation report:[/green] {validation_path}")
    console.print(f"[bold]{'PASS' if report.passed else 'FAIL'}[/bold] {report.required_checks_passed}/{report.required_checks_total} required checks passed")

    source_manifest_path: Optional[Path] = None
    if export_sources:
        try:
            source_manifest = _filtered_evidence_sources(category=source_category)
            source_manifest_path = Path(outputs["output_dir"]) / "sources_manifest.json"
            source_manifest_path.write_text(json.dumps(source_manifest, indent=2))
            console.print(f"[green]Source manifest:[/green] {source_manifest_path}")
        except Exception as exc:
            console.print(f"[yellow]Could not export source manifest: {exc}[/yellow]")

    if write_summary:
        summary_path = _write_certification_summary(
            Path(outputs["output_dir"]),
            outputs,
            profile,
            report_payload,
            source_manifest_path,
        )
        console.print(f"[green]Summary:[/green] {summary_path}")

    if fail_on_check and not report.passed:
        raise typer.Exit(code=1)


@app.command("study-ready")
def study_ready(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to scenario JSON")] = None,
    output_dir: Annotated[Path, typer.Option(help="Directory to save outputs")] = Path("results/study_ready"),
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    profile: Annotated[str, typer.Option(help="Validation profile id")] = "research_default",
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt)")] = None,
    patient_config_name: Annotated[str, typer.Option(help="Patient configuration name")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Optional patient config YAML path")] = None,
    fail_on_check: Annotated[bool, typer.Option(help="Exit code 1 when validation profile fails")] = True,
):
    """
    Simplified one-command pipeline: run + validate + sources manifest + summary bundle.
    """
    certify_run(
        algo=algo,
        profile=profile,
        predictor_path=predictor_path,
        patient_config_name=patient_config_name,
        patient_config_path=patient_config_path,
        scenario_path=scenario_path,
        duration=duration,
        time_step=time_step,
        output_dir=output_dir,
        seed=seed,
        export_sources=True,
        source_category=None,
        write_summary=True,
        fail_on_check=fail_on_check,
    )


@app.command()
def init(
    project_name: Annotated[str, typer.Option(help="Name of the project directory")] = "my_iints_project",
    template: Annotated[str, typer.Option(help="Project template: research or clinical-trial")] = "research",
):
    """
    Initialize a new IINTS-AF project with a standard folder structure.
    """
    console = Console()
    project_path = Path(project_name)

    if project_path.exists():
        console.print(f"[bold red]Error: Directory '{project_name}' already exists.[/bold red]")
        raise typer.Exit(code=1)

    selected_template = template.strip().lower()
    if selected_template not in {"research", "clinical-trial"}:
        console.print("[bold red]Invalid --template. Use 'research' or 'clinical-trial'.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Initializing IINTS-AF Project: {project_name} ({selected_template})[/bold blue]")

    # Base directories
    (project_path / "algorithms").mkdir(parents=True)
    (project_path / "scenarios").mkdir(parents=True)
    (project_path / "data").mkdir(parents=True)
    (project_path / "results").mkdir(parents=True)

    if selected_template == "clinical-trial":
        (project_path / "contracts").mkdir(parents=True)
        (project_path / "audit").mkdir(parents=True)
        (project_path / "notebooks").mkdir(parents=True)
        (project_path / "reports").mkdir(parents=True)
        (project_path / "data" / "demo").mkdir(parents=True)

    # Copy Default Algorithm
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            algo_content = files("iints.templates").joinpath("default_algorithm.py").read_text()
            scenario_content = files("iints.templates.scenarios").joinpath("example_scenario.json").read_text()
            exercise_content = files("iints.templates.scenarios").joinpath("exercise_stress.json").read_text()
            stacking_content = files("iints.templates.scenarios").joinpath("chaos_insulin_stacking.json").read_text()
            runaway_content = files("iints.templates.scenarios").joinpath("chaos_runaway_ai.json").read_text()
        else:
            from importlib import resources
            algo_content = resources.read_text("iints.templates", "default_algorithm.py")
            scenario_content = resources.read_text("iints.templates.scenarios", "example_scenario.json")
            exercise_content = resources.read_text("iints.templates.scenarios", "exercise_stress.json")
            stacking_content = resources.read_text("iints.templates.scenarios", "chaos_insulin_stacking.json")
            runaway_content = resources.read_text("iints.templates.scenarios", "chaos_runaway_ai.json")
    except Exception as e:
        console.print(f"[bold red]Error reading template files: {e}[/bold red]")
        raise typer.Exit(code=1)

    # Instantiate the default algorithm template
    algo_content = algo_content.replace("{{ALGO_NAME}}", "ExampleAlgorithm")
    algo_content = algo_content.replace("{{AUTHOR_NAME}}", "IINTS User")

    with open(project_path / "algorithms" / "example_algorithm.py", "w") as f:
        f.write(algo_content)

    with open(project_path / "scenarios" / "example_scenario.json", "w") as f:
        f.write(scenario_content)
    with open(project_path / "scenarios" / "exercise_stress.json", "w") as f:
        f.write(exercise_content)
    with open(project_path / "scenarios" / "chaos_insulin_stacking.json", "w") as f:
        f.write(stacking_content)
    with open(project_path / "scenarios" / "chaos_runaway_ai.json", "w") as f:
        f.write(runaway_content)

    if selected_template == "clinical-trial":
        contract_path = project_path / "contracts" / "clinical_mdmp_contract.yaml"
        contract_payload = _build_data_contract_template()
        contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False))

        demo_df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T00:15:00Z",
                    "2026-01-01T00:20:00Z",
                    "2026-01-01T00:25:00Z",
                ],
                "glucose": [118.0, 120.0, 122.0, 124.0, 123.0, 121.0],
                "carbs": [0.0, 0.0, 0.0, 12.0, 0.0, 0.0],
                "insulin": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            }
        )
        demo_path = project_path / "data" / "demo" / "diabetes_cgm.csv"
        demo_df.to_csv(demo_path, index=False)

    # Create README
    if selected_template == "clinical-trial":
        readme_content = f"""# {project_name}

Powered by IINTS-AF SDK (clinical-trial scaffold).

## Structure
- `algorithms/`: Algorithms under evaluation.
- `scenarios/`: Simulation scenarios.
- `contracts/`: MDMP contracts.
- `data/demo/`: Synthetic-safe starter dataset.
- `audit/`: Validation JSON + MDMP dashboards.
- `reports/`: Study outputs for sharing.
- `results/`: Raw simulation outputs.

## MDMP Quick Path
```bash
iints data contract-run contracts/clinical_mdmp_contract.yaml data/demo/diabetes_cgm.csv \\
  --output-json audit/contract_data_report.json \\
  --min-mdmp-grade research_grade --fail-on-noncompliant

iints data mdmp-visualizer audit/contract_data_report.json \\
  --output-html audit/mdmp_dashboard.html
```

## Run Simulation
```bash
iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/example_scenario.json
```
"""
    else:
        readme_content = f"""# {project_name}

Powered by IINTS-AF SDK.

## Structure
- `algorithms/`: Place your custom python algorithms here.
- `scenarios/`: JSON files defining stress test scenarios.
- `data/`: Custom patient data or configuration.
- `results/`: Simulation outputs.

## Getting Started

1. Run the example algorithm:
   ```bash
   iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/example_scenario.json
   ```

2. Create a new algorithm:
   ```bash
   iints new-algo MyNewAlgo --output-dir algorithms/
   ```
"""
    with open(project_path / "README.md", "w") as f:
        f.write(readme_content)

    console.print(f"[green]Project initialized successfully in '{project_name}'[/green]")
    if selected_template == "clinical-trial":
        console.print(
            "To get started:\n"
            f"  cd {project_name}\n"
            "  iints data contract-run contracts/clinical_mdmp_contract.yaml data/demo/diabetes_cgm.csv "
            "--output-json audit/contract_data_report.json\n"
            "  iints data mdmp-visualizer audit/contract_data_report.json --output-html audit/mdmp_dashboard.html"
        )
    else:
        console.print(f"To get started:\n  cd {project_name}\n  iints run --algo algorithms/example_algorithm.py")

@app.command()
def quickstart(
    project_name: Annotated[str, typer.Option(help="Name of the project directory")] = "iints_quickstart",
):
    """
    Create a ready-to-run project using clinic-safe presets and a demo algorithm.
    """
    console = Console()
    project_path = Path(project_name)

    if project_path.exists():
        console.print(f"[bold red]Error: Directory '{project_name}' already exists.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Creating IINTS-AF Quickstart Project: {project_name}[/bold blue]")

    (project_path / "algorithms").mkdir(parents=True)
    (project_path / "scenarios").mkdir(parents=True)
    (project_path / "results").mkdir(parents=True)

    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            algo_content = files("iints.templates").joinpath("default_algorithm.py").read_text()
        else:
            from importlib import resources
            algo_content = resources.read_text("iints.templates", "default_algorithm.py")
    except Exception as e:
        console.print(f"[bold red]Error reading template files: {e}[/bold red]")
        raise typer.Exit(code=1)

    algo_content = algo_content.replace("{{ALGO_NAME}}", "QuickstartAlgorithm")
    algo_content = algo_content.replace("{{AUTHOR_NAME}}", "IINTS User")
    algo_path = project_path / "algorithms" / "example_algorithm.py"
    algo_path.write_text(algo_content)

    try:
        preset = _get_preset("baseline_t1d")
        scenario_path = project_path / "scenarios" / "clinic_safe_baseline.json"
        scenario_path.write_text(json.dumps(preset.get("scenario", {}), indent=2))
    except Exception as e:
        console.print(f"[yellow]Preset scenario not available: {e}[/yellow]")

    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            exercise_content = files("iints.templates.scenarios").joinpath("exercise_stress.json").read_text()
        else:
            from importlib import resources
            exercise_content = resources.read_text("iints.templates.scenarios", "exercise_stress.json")
        (project_path / "scenarios" / "exercise_stress.json").write_text(exercise_content)
    except Exception as e:
        console.print(f"[yellow]Exercise stress scenario not available: {e}[/yellow]")

    readme_content = f"""# {project_name}

Clinic-safe quickstart project powered by IINTS-AF.

## Quickstart

Run a clinic-safe preset:

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Run with the included scenario file:

```bash
iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/clinic_safe_baseline.json --duration 1440
```
"""
    (project_path / "README.md").write_text(readme_content)

    console.print(f"[green]Quickstart project ready in '{project_name}'.[/green]")
    console.print(f"Next:\n  cd {project_name}\n  iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py")
@app.command()
def new_algo(
    name: Annotated[str, typer.Option(help="Name of the new algorithm")],
    author: Annotated[str, typer.Option(help="Author of the algorithm")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save the new algorithm file")] = Path("."),
):
    """
    Creates a new algorithm template file based on the BaseAlgorithm.
    """
    if not output_dir.is_dir():
        typer.echo(f"Error: Output directory '{output_dir}' does not exist.")
        raise typer.Exit(code=1)

    try:
        # Try Python 3.9+ style
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            template_content = files("iints.templates").joinpath("default_algorithm.py").read_text()
        else:
            # Fallback for Python 3.8
            from importlib import resources
            template_content = resources.read_text("iints.templates", "default_algorithm.py")
    except Exception as e:
        typer.echo(f"Error reading template file: {e}")
        raise typer.Exit(code=1)

    # Replace placeholders
    # We expect the template class name to be {{ALGO_NAME}} and author {{AUTHOR_NAME}}
    # But wait, the file I wrote uses {{ALGO_NAME}} as a class name, which is valid syntax only if I replace it.
    # The file on disk is valid python ONLY if those tokens are valid. 
    # Actually, in the previous step I wrote {{ALGO_NAME}} literally into the python file.
    # That makes the template file itself invalid python syntax until replaced. 
    # That's fine for a template file, but might confuse linters. 
    # Ideally it's a .txt or .tmpl, but .py is fine if we accept it's a template.
    
    final_content = template_content.replace("{{ALGO_NAME}}", f"{name}Algorithm")
    final_content = final_content.replace("{{AUTHOR_NAME}}", author)

    output_file = output_dir / f"{name.lower().replace(' ', '_')}_algorithm.py"
    with open(output_file, "w") as f:
        f.write(final_content)
    
    typer.echo(f"Successfully created new algorithm template: {output_file}")


@presets_app.command("list")
def presets_list():
    """List clinic-safe presets."""
    console = Console()
    presets = _load_presets()
    table = Table(title="Clinic-Safe Presets", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Patient Config")
    table.add_column("Duration (min)", justify="right")
    for preset in presets:
        table.add_row(
            preset.get("name", ""),
            preset.get("description", ""),
            preset.get("patient_config", ""),
            str(preset.get("duration_minutes", "")),
        )
    console.print(table)


@presets_app.command("show")
def presets_show(
    name: Annotated[str, typer.Option(help="Preset name (e.g., baseline_t1d)")],
):
    """Show a preset definition."""
    console = Console()
    try:
        preset = _get_preset(name)
    except KeyError:
        console.print(f"[bold red]Error: Unknown preset '{name}'.[/bold red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(preset, indent=2))


@presets_app.command("run")
def presets_run(
    name: Annotated[str, typer.Option(help="Preset name (e.g., baseline_t1d)")],
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt) for dual-guard forecasting")] = None,
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save outputs")] = None,
    compare_baselines: Annotated[bool, typer.Option(help="Run PID and standard pump baselines in the background")] = True,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
    patient_model_type: Annotated[str, typer.Option("--patient-model", help="Patient model: auto, bergman, custom, simglucose")] = "auto",
    sensor_noise_std: Annotated[Optional[float], typer.Option("--sensor-noise-std", help="CGM noise std (mg/dL)")] = None,
    sensor_lag_minutes: Annotated[Optional[int], typer.Option("--sensor-lag-minutes", help="CGM lag (minutes)")] = None,
    sensor_dropout_prob: Annotated[Optional[float], typer.Option("--sensor-dropout-prob", help="CGM dropout probability (0-1)")] = None,
    sensor_bias: Annotated[Optional[float], typer.Option("--sensor-bias", help="CGM bias (mg/dL)")] = None,
    safety_min_glucose: Annotated[Optional[float], typer.Option("--safety-min-glucose", help="Min plausible glucose (mg/dL)")] = None,
    safety_max_glucose: Annotated[Optional[float], typer.Option("--safety-max-glucose", help="Max plausible glucose (mg/dL)")] = None,
    safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option("--safety-max-glucose-delta-per-5-min", help="Max glucose delta per 5 min (mg/dL)")] = None,
    safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hypo-threshold", help="Hypoglycemia threshold (mg/dL)")] = None,
    safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-severe-hypo-threshold", help="Severe hypoglycemia threshold (mg/dL)")] = None,
    safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hyper-threshold", help="Hyperglycemia threshold (mg/dL)")] = None,
    safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option("--safety-max-bolus", help="Max insulin per bolus (U)")] = None,
    safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option("--safety-glucose-rate-alarm", help="Glucose rate alarm (mg/dL/min)")] = None,
    safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option("--safety-max-insulin-per-hour", help="Max insulin per 60 min (U)")] = None,
    safety_max_iob: Annotated[Optional[float], typer.Option("--safety-max-iob", help="Max insulin on board (U)")] = None,
    safety_trend_stop: Annotated[Optional[float], typer.Option("--safety-trend-stop", help="Negative trend cutoff (mg/dL/min)")] = None,
    safety_hypo_cutoff: Annotated[Optional[float], typer.Option("--safety-hypo-cutoff", help="Hard hypo cutoff (mg/dL)")] = None,
    safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option("--safety-critical-glucose", help="Critical glucose threshold (mg/dL)")] = None,
    safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option("--safety-critical-duration", help="Critical glucose duration (minutes)")] = None,
    safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-uncertainty-gate", help="Enable predictor uncertainty gate")] = None,
    safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option("--safety-predictor-max-std", help="Max allowed predictor std (mg/dL) before fallback")] = None,
    safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-ood-gate", help="Enable predictor OOD gate")] = None,
    safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option("--safety-predictor-ood-z", help="Predictor OOD z-score threshold")] = None,
    safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option("--safety-predictor-ood-feature-fraction", help="Max fraction of features allowed OOD")] = None,
):
    """Run a clinic-safe preset with an algorithm and generate outputs."""
    console = Console()
    try:
        preset = _get_preset(name)
    except KeyError:
        console.print(f"[bold red]Error: Unknown preset '{name}'.[/bold red]")
        raise typer.Exit(code=1)

    algorithm_instance = _load_algorithm_instance(algo, console)
    console.print(f"Loaded algorithm: [green]{algorithm_instance.get_algorithm_metadata().name}[/green]")
    predictor = _load_predictor_service_from_path(predictor_path, console)

    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)

    try:
        patient_config_name = preset.get("patient_config", "default_patient")
        validated_patient_params = load_patient_config_by_name(patient_config_name).model_dump()
        patient_model = iints.PatientFactory.create_patient(patient_type=patient_model_type, **validated_patient_params)
    except ValidationError as e:
        console.print("[bold red]Patient config validation failed:[/bold red]")
        for line in format_validation_error(e):
            console.print(f"- {line}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error loading patient config {patient_config_name}: {e}[/bold red]")
        raise typer.Exit(code=1)

    duration = int(preset.get("duration_minutes", 720))
    time_step = int(preset.get("time_step_minutes", 5))
    safety_config = _build_safety_config_from_options(
        min_glucose=safety_min_glucose,
        max_glucose=safety_max_glucose,
        max_glucose_delta_per_5_min=safety_max_glucose_delta_per_5_min,
        hypoglycemia_threshold=safety_hypoglycemia_threshold,
        severe_hypoglycemia_threshold=safety_severe_hypoglycemia_threshold,
        hyperglycemia_threshold=safety_hyperglycemia_threshold,
        max_insulin_per_bolus=safety_max_insulin_per_bolus,
        glucose_rate_alarm=safety_glucose_rate_alarm,
        max_insulin_per_hour=safety_max_insulin_per_hour,
        max_iob=safety_max_iob,
        trend_stop=safety_trend_stop,
        hypo_cutoff=safety_hypo_cutoff,
        critical_glucose_threshold=safety_critical_glucose_threshold,
        critical_glucose_duration_minutes=safety_critical_glucose_duration_minutes,
        predictor_uncertainty_gate_enabled=safety_predictor_uncertainty_gate_enabled,
        predictor_uncertainty_max_std_mgdl=safety_predictor_uncertainty_max_std_mgdl,
        predictor_ood_gate_enabled=safety_predictor_ood_gate_enabled,
        predictor_ood_zscore_threshold=safety_predictor_ood_zscore_threshold,
        predictor_ood_max_feature_fraction=safety_predictor_ood_max_feature_fraction,
    )

    sensor_model = None
    if any(v is not None for v in (sensor_noise_std, sensor_lag_minutes, sensor_dropout_prob, sensor_bias)):
        sensor_model = iints.SensorModel(
            noise_std=float(sensor_noise_std or 0.0),
            lag_minutes=int(sensor_lag_minutes or 0),
            dropout_prob=float(sensor_dropout_prob or 0.0),
            bias=float(sensor_bias or 0.0),
            seed=resolved_seed,
        )
    elif patient_model_type == "auto":
        sensor_model = iints.SensorModel(
            noise_std=7.0,
            lag_minutes=10,
            dropout_prob=0.0,
            bias=0.0,
            seed=resolved_seed,
        )

    simulator_kwargs: Dict[str, Any] = {
        "patient_model": patient_model,
        "algorithm": algorithm_instance,
        "time_step": time_step,
        "safety_config": safety_config,
        "predictor": predictor,
    }
    if sensor_model is not None:
        simulator_kwargs["sensor_model"] = sensor_model
    simulator_kwargs["seed"] = resolved_seed
    if safety_config is None:
        safety_config = SafetyConfig()
        if "critical_glucose_threshold" in preset:
            safety_config.critical_glucose_threshold = float(preset["critical_glucose_threshold"])
        if "critical_glucose_duration_minutes" in preset:
            safety_config.critical_glucose_duration_minutes = int(preset["critical_glucose_duration_minutes"])
        simulator_kwargs["safety_config"] = safety_config
    simulator = iints.Simulator(**simulator_kwargs)

    scenario_payload = preset.get("scenario", {})
    try:
        scenario_model = validate_scenario_dict(scenario_payload)
    except ValidationError as e:
        console.print("[bold red]Preset scenario validation failed:[/bold red]")
        for line in format_validation_error(e):
            console.print(f"- {line}")
        raise typer.Exit(code=1)

    stress_event_payloads = scenario_to_payloads(scenario_model)
    for warning in scenario_warnings(scenario_model):
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    for event in build_stress_events(stress_event_payloads):
        simulator.add_stress_event(event)

    output_dir = resolve_output_dir(output_dir, run_id)
    results_df, safety_report = simulator.run_batch(duration)

    scenario_payload = scenario_model.model_dump() if scenario_model else None
    config_payload: Dict[str, Any] = {
        "run_type": "preset",
        "preset_name": name,
        "algorithm": {
            "class": f"{algorithm_instance.__class__.__module__}.{algorithm_instance.__class__.__name__}",
            "metadata": algorithm_instance.get_algorithm_metadata().to_dict(),
        },
        "patient_config": validated_patient_params,
        "scenario": scenario_payload,
        "duration_minutes": duration,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
        "compare_baselines": compare_baselines,
        "export_audit": True,
        "generate_report": True,
        "safety_config": asdict(safety_config),
        "predictor_path": str(predictor_path) if predictor_path else None,
    }
    config_path = output_dir / "config.json"
    write_json(config_path, config_payload)
    run_metadata = build_run_metadata(run_id, resolved_seed, config_payload, output_dir)
    run_metadata_path = output_dir / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    results_file = output_dir / "results.csv"
    results_df.to_csv(results_file, index=False)
    console.print(f"Results saved to: {results_file}")
    console.print(f"Run metadata: {run_metadata_path}")

    audit_dir = output_dir / "audit"
    try:
        audit_paths = simulator.export_audit_trail(results_df, output_dir=str(audit_dir))
        console.print(f"Audit trail: {audit_paths}")
    except Exception as e:
        console.print(f"[yellow]Audit export skipped: {e}[/yellow]")

    if compare_baselines:
        comparison = run_baseline_comparison(
            patient_params=validated_patient_params,
            stress_event_payloads=stress_event_payloads,
            duration=duration,
            time_step=time_step,
            primary_label=algorithm_instance.get_algorithm_metadata().name,
            primary_results=results_df,
            primary_safety=safety_report,
            seed=resolved_seed,
        )
        safety_report["baseline_comparison"] = comparison
        baseline_paths = write_baseline_comparison(comparison, output_dir / "baseline")
        console.print(f"Baseline comparison saved to: {baseline_paths}")

    report_path = output_dir / "report.pdf"
    iints.generate_report(results_df, str(report_path), safety_report)
    console.print(f"PDF report saved to: {report_path}")

    manifest_files = {
        "config": config_path,
        "run_metadata": run_metadata_path,
        "results_csv": results_file,
        "report_pdf": report_path,
    }
    if compare_baselines:
        manifest_files["baseline_json"] = output_dir / "baseline" / "baseline_comparison.json"
        manifest_files["baseline_csv"] = output_dir / "baseline" / "baseline_comparison.csv"
    audit_summary_path = output_dir / "audit" / "audit_summary.json"
    if audit_summary_path.exists():
        manifest_files["audit_summary"] = audit_summary_path
    run_manifest = build_run_manifest(output_dir, manifest_files)
    run_manifest_path = output_dir / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)
    console.print(f"Run manifest: {run_manifest_path}")
    signature_path = maybe_sign_manifest(run_manifest_path)
    if signature_path:
        console.print(f"Run manifest signature: {signature_path}")


@presets_app.command("create")
def presets_create(
    name: Annotated[str, typer.Option(help="Preset name (snake_case)")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for preset files")] = Path("./presets"),
    initial_glucose: Annotated[float, typer.Option(help="Initial glucose (mg/dL)")] = 140.0,
    basal_insulin_rate: Annotated[float, typer.Option(help="Basal insulin rate (U/hr)")] = 0.5,
    insulin_sensitivity: Annotated[float, typer.Option(help="Insulin sensitivity (mg/dL per U)")] = 50.0,
    carb_factor: Annotated[float, typer.Option(help="Carb factor (g per U)")] = 10.0,
):
    """
    Generate a clinic-safe preset scaffold (patient YAML + scenario JSON).
    """
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_config_name = f"clinic_safe_{name}"
    patient_yaml_path = output_dir / f"{patient_config_name}.yaml"
    scenario_path = output_dir / f"{name}.json"

    patient_yaml = (
        f"basal_insulin_rate: {basal_insulin_rate}\n"
        f"insulin_sensitivity: {insulin_sensitivity}\n"
        f"carb_factor: {carb_factor}\n"
        "glucose_decay_rate: 0.03\n"
        f"initial_glucose: {initial_glucose}\n"
        "glucose_absorption_rate: 0.03\n"
        "insulin_action_duration: 300.0\n"
        "insulin_peak_time: 75.0\n"
        "meal_mismatch_epsilon: 1.0\n"
    )
    patient_yaml_path.write_text(patient_yaml)

    scenario = {
        "scenario_name": f"Clinic Safe {name.replace('_', ' ').title()}",
        "schema_version": "1.1",
        "scenario_version": "1.0",
        "stress_events": [
            {"start_time": 60, "event_type": "meal", "value": 45, "absorption_delay_minutes": 15, "duration": 60},
            {"start_time": 360, "event_type": "meal", "value": 60, "absorption_delay_minutes": 20, "duration": 90},
            {"start_time": 720, "event_type": "meal", "value": 70, "absorption_delay_minutes": 15, "duration": 90},
        ],
    }
    scenario_path.write_text(json.dumps(scenario, indent=2))

    preset_snippet = {
        "name": name,
        "description": "Custom clinic-safe preset (generated).",
        "patient_config": patient_config_name,
        "duration_minutes": 1440,
        "time_step_minutes": 5,
        "critical_glucose_threshold": 40.0,
        "critical_glucose_duration_minutes": 30,
        "scenario": scenario,
    }

    console.print(f"[green]Preset files created:[/green] {patient_yaml_path} , {scenario_path}")
    console.print("[bold]Add this preset to presets.json if you want it built-in:[/bold]")
    console.print_json(json.dumps(preset_snippet, indent=2))


@app.command("run-wizard")
def run_wizard():
    """Interactive wizard to run a preset quickly."""
    console = Console()
    presets = _load_presets()
    preset_names = [preset.get("name", "") for preset in presets if preset.get("name")]
    if not preset_names:
        console.print("[bold red]No presets available.[/bold red]")
        raise typer.Exit(code=1)

    preset_choice = typer.prompt("Preset name", default=preset_names[0])
    algo_path = Path(typer.prompt("Algorithm path", default="algorithms/example_algorithm.py"))
    seed_input = typer.prompt("Seed (blank for auto)", default="", show_default=False)
    seed = int(seed_input) if seed_input.strip() else None
    output_dir_input = typer.prompt("Output directory (blank for default)", default="", show_default=False)
    output_dir = Path(output_dir_input) if output_dir_input.strip() else None

    presets_run(
        name=preset_choice,
        algo=algo_path,
        predictor_path=None,
        output_dir=output_dir,
        compare_baselines=True,
        seed=seed,
        safety_min_glucose=None,
        safety_max_glucose=None,
        safety_max_glucose_delta_per_5_min=None,
        safety_hypoglycemia_threshold=None,
        safety_severe_hypoglycemia_threshold=None,
        safety_hyperglycemia_threshold=None,
        safety_max_insulin_per_bolus=None,
        safety_glucose_rate_alarm=None,
        safety_max_insulin_per_hour=None,
        safety_max_iob=None,
        safety_trend_stop=None,
        safety_hypo_cutoff=None,
        safety_critical_glucose_threshold=None,
        safety_critical_glucose_duration_minutes=None,
    )


@profiles_app.command("create")
def profiles_create(
    name: Annotated[str, typer.Option(help="Profile name (file stem)")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for the profile YAML")] = Path("./patient_profiles"),
    isf: Annotated[float, typer.Option(help="Insulin Sensitivity Factor (mg/dL per unit)")] = 50.0,
    icr: Annotated[float, typer.Option(help="Insulin-to-carb ratio (grams per unit)")] = 10.0,
    basal_rate: Annotated[float, typer.Option(help="Basal insulin rate (U/hr)")] = 0.8,
    initial_glucose: Annotated[float, typer.Option(help="Initial glucose (mg/dL)")] = 120.0,
    dawn_strength: Annotated[float, typer.Option(help="Dawn phenomenon strength (mg/dL per hour)")] = 0.0,
    dawn_start: Annotated[float, typer.Option(help="Dawn phenomenon start hour (0-23)")] = 4.0,
    dawn_end: Annotated[float, typer.Option(help="Dawn phenomenon end hour (0-24)")] = 8.0,
):
    """Create a patient profile YAML you can pass to --patient-config-path."""
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.yaml"

    profile = PatientProfile(
        isf=isf,
        icr=icr,
        basal_rate=basal_rate,
        initial_glucose=initial_glucose,
        dawn_phenomenon_strength=dawn_strength,
        dawn_start_hour=dawn_start,
        dawn_end_hour=dawn_end,
    )
    with output_path.open("w") as handle:
        yaml.safe_dump(profile.to_patient_config(), handle, sort_keys=False)

    console.print(f"[green]Patient profile saved:[/green] {output_path}")
    console.print("Use it with:")
    console.print(f"  iints run --algo algorithms/example_algorithm.py --patient-config-path {output_path}")


@scenarios_app.command("generate")
def scenarios_generate(
    name: Annotated[str, typer.Option(help="Scenario name")] = "Generated Scenario",
    output_path: Annotated[Path, typer.Option(help="Output JSON path")] = Path("./scenarios/generated_scenario.json"),
    duration_minutes: Annotated[int, typer.Option(help="Scenario duration in minutes")] = 1440,
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    meal_count: Annotated[int, typer.Option(help="Number of meal events")] = 3,
    meal_min_grams: Annotated[float, typer.Option(help="Min meal size (g carbs)")] = 30.0,
    meal_max_grams: Annotated[float, typer.Option(help="Max meal size (g carbs)")] = 80.0,
    exercise_count: Annotated[int, typer.Option(help="Number of exercise events")] = 0,
    sensor_error_count: Annotated[int, typer.Option(help="Number of sensor error events")] = 0,
):
    """Generate a random stress-test scenario JSON."""
    config = ScenarioGeneratorConfig(
        name=name,
        duration_minutes=duration_minutes,
        seed=seed,
        meal_count=meal_count,
        meal_min_grams=meal_min_grams,
        meal_max_grams=meal_max_grams,
        exercise_count=exercise_count,
        sensor_error_count=sensor_error_count,
    )
    scenario = generate_random_scenario(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenario, indent=2))
    typer.echo(f"Scenario saved: {output_path}")


@scenarios_app.command("wizard")
def scenarios_wizard():
    """Interactive scenario generator."""
    name = typer.prompt("Scenario name", default="Generated Scenario")
    duration_minutes = int(typer.prompt("Duration (minutes)", default="1440"))
    meal_count = int(typer.prompt("Meal events", default="3"))
    meal_min = float(typer.prompt("Meal min grams", default="30"))
    meal_max = float(typer.prompt("Meal max grams", default="80"))
    exercise_count = int(typer.prompt("Exercise events", default="0"))
    sensor_error_count = int(typer.prompt("Sensor error events", default="0"))
    output_path = Path(typer.prompt("Output JSON path", default="scenarios/generated_scenario.json"))

    config = ScenarioGeneratorConfig(
        name=name,
        duration_minutes=duration_minutes,
        meal_count=meal_count,
        meal_min_grams=meal_min,
        meal_max_grams=meal_max,
        exercise_count=exercise_count,
        sensor_error_count=sensor_error_count,
    )
    scenario = generate_random_scenario(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenario, indent=2))
    typer.echo(f"Scenario saved: {output_path}")


@scenarios_app.command("migrate")
def scenarios_migrate(
    input_path: Annotated[Path, typer.Argument(help="Scenario JSON to migrate")],
    output_path: Annotated[Optional[Path], typer.Option(help="Output path (default: overwrite input)")] = None,
):
    """Migrate a scenario JSON to the latest schema version."""
    console = Console()
    if not input_path.is_file():
        console.print(f"[bold red]Error: Scenario '{input_path}' not found.[/bold red]")
        raise typer.Exit(code=1)
    try:
        data = json.loads(input_path.read_text())
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Invalid JSON: {exc}[/bold red]")
        raise typer.Exit(code=1)
    migrated = migrate_scenario_dict(data)
    target = output_path or input_path
    target.write_text(json.dumps(migrated, indent=2))
    console.print(f"[green]Migrated scenario saved to {target}[/green]")
@app.command()
def run(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt) for dual-guard forecasting")] = None,
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default_patient' or 'patient_559_config')")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Path to a patient config YAML (overrides --patient-config-name)")] = None,
    scenario_path: Annotated[
        Optional[Path],
        typer.Option(
            "--scenario",
            "--scenario-path",
            help="Path to the scenario JSON file (e.g., scenarios/example_scenario.json)",
        ),
    ] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720, # 12 hours
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save simulation results")] = None,
    compare_baselines: Annotated[bool, typer.Option(help="Run PID and standard pump baselines in the background")] = True,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
    patient_model_type: Annotated[str, typer.Option("--patient-model", help="Patient model: auto, bergman, custom, simglucose")] = "auto",
    sensor_noise_std: Annotated[Optional[float], typer.Option("--sensor-noise-std", help="CGM noise std (mg/dL)")] = None,
    sensor_lag_minutes: Annotated[Optional[int], typer.Option("--sensor-lag-minutes", help="CGM lag (minutes)")] = None,
    sensor_dropout_prob: Annotated[Optional[float], typer.Option("--sensor-dropout-prob", help="CGM dropout probability (0-1)")] = None,
    sensor_bias: Annotated[Optional[float], typer.Option("--sensor-bias", help="CGM bias (mg/dL)")] = None,
    safety_min_glucose: Annotated[Optional[float], typer.Option("--safety-min-glucose", help="Min plausible glucose (mg/dL)")] = None,
    safety_max_glucose: Annotated[Optional[float], typer.Option("--safety-max-glucose", help="Max plausible glucose (mg/dL)")] = None,
    safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option("--safety-max-glucose-delta-per-5-min", help="Max glucose delta per 5 min (mg/dL)")] = None,
    safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hypo-threshold", help="Hypoglycemia threshold (mg/dL)")] = None,
    safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-severe-hypo-threshold", help="Severe hypoglycemia threshold (mg/dL)")] = None,
    safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hyper-threshold", help="Hyperglycemia threshold (mg/dL)")] = None,
    safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option("--safety-max-bolus", help="Max insulin per bolus (U)")] = None,
    safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option("--safety-glucose-rate-alarm", help="Glucose rate alarm (mg/dL/min)")] = None,
    safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option("--safety-max-insulin-per-hour", help="Max insulin per 60 min (U)")] = None,
    safety_max_iob: Annotated[Optional[float], typer.Option("--safety-max-iob", help="Max insulin on board (U)")] = None,
    safety_trend_stop: Annotated[Optional[float], typer.Option("--safety-trend-stop", help="Negative trend cutoff (mg/dL/min)")] = None,
    safety_hypo_cutoff: Annotated[Optional[float], typer.Option("--safety-hypo-cutoff", help="Hard hypo cutoff (mg/dL)")] = None,
    safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option("--safety-critical-glucose", help="Critical glucose threshold (mg/dL)")] = None,
    safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option("--safety-critical-duration", help="Critical glucose duration (minutes)")] = None,
    safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-uncertainty-gate", help="Enable predictor uncertainty gate")] = None,
    safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option("--safety-predictor-max-std", help="Max allowed predictor std (mg/dL) before fallback")] = None,
    safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-ood-gate", help="Enable predictor OOD gate")] = None,
    safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option("--safety-predictor-ood-z", help="Predictor OOD z-score threshold")] = None,
    safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option("--safety-predictor-ood-feature-fraction", help="Max fraction of features allowed OOD")] = None,
):
    """
    Run an IINTS-AF simulation using a specified algorithm and patient configuration.
    """
    console = Console() # Define console locally for each command to prevent Rich issues
    console.print(f"[bold blue]Starting IINTS-AF simulation with algorithm: {algo.name}[/bold blue]")
    console.print(f"Patient configuration: {patient_config_name}")
    console.print(f"Simulation duration: {duration} minutes, time step: {time_step} minutes")

    # 1. Load Algorithm
    if not algo.is_file():
        console.print(f"[bold red]Error: Algorithm file '{algo}' not found.[/bold red]")
        raise typer.Exit(code=1)
    
    module_name = algo.stem
    # Use importlib.util to load the module directly from the file path
    spec = importlib.util.spec_from_file_location(module_name, algo)
    if spec is None:
        console.print(f"[bold red]Error: Could not load module spec for {algo}[/bold red]")
        raise typer.Exit(code=1)
    
    module = importlib.util.module_from_spec(spec)
    
    # Manually inject 'iints' package into the loaded module's global namespace
    # This ensures 'from iints import ...' works within the algorithm file
    module.iints = iints # type: ignore
    
    sys.modules[module_name] = module
    try:
        if spec.loader: # Ensure loader is not None
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Could not load module loader for {algo}")
    except Exception as e:
        console.print(f"[bold red]Error loading algorithm module {algo}: {e}[/bold red]")
        raise typer.Exit(code=1)

    algorithm_instance = None
    for name_in_module, obj in module.__dict__.items(): # Renamed 'name' to 'name_in_module' to avoid conflict
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            algorithm_instance = obj() # Instantiate the algorithm
            console.print(f"Loaded algorithm: [green]{algorithm_instance.get_algorithm_metadata().name}[/green]")
            break
    
    if algorithm_instance is None:
        console.print(f"[bold red]Error: No subclass of InsulinAlgorithm found in {algo}[/bold red]")
        raise typer.Exit(code=1)
    predictor = _load_predictor_service_from_path(predictor_path, console)
    
    # 2. Get Device
    device_manager = iints.DeviceManager()
    device = device_manager.get_device()
    console.print(f"Using compute device: [blue]{device}[/blue]")

    # 3. Instantiate Patient Model
    try:
        if patient_config_path:
            if not patient_config_path.is_file():
                console.print(f"[bold red]Error: Patient config file '{patient_config_path}' not found.[/bold red]")
                raise typer.Exit(code=1)
            validated_patient_params = load_patient_config(patient_config_path).model_dump()
            patient_label = patient_config_path.stem
        else:
            validated_patient_params = load_patient_config_by_name(patient_config_name).model_dump()
            patient_label = patient_config_name

        patient_model = iints.PatientFactory.create_patient(patient_type=patient_model_type, **validated_patient_params)
        console.print(
            f"Using patient model: {patient_model.__class__.__name__} "
            f"({patient_model_type}) with config [cyan]{patient_label}[/cyan]"
        )
    except ValidationError as e:
        console.print("[bold red]Patient config validation failed:[/bold red]")
        for line in format_validation_error(e):
            console.print(f"- {line}")
        raise typer.Exit(code=1)
    except TypeError as e:
        console.print(f"[bold red]Error instantiating PatientModel with parameters from {patient_config_name}: {e}[/bold red]")
        console.print("[bold red]Please check that patient configuration keys match PatientModel constructor arguments.[/bold red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error loading patient config {patient_config_name}: {e}[/bold red]")
        raise typer.Exit(code=1)


    # 4. Load Scenario Data (if provided)
    stress_events = []
    stress_event_payloads: List[Dict[str, Any]] = []
    scenario_model = None
    scenario_payload: Optional[Dict[str, Any]] = None
    if scenario_path:
        if not scenario_path.is_file():
            console.print(f"[bold red]Error: Scenario file '{scenario_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        
        try:
            scenario_model = load_scenario(scenario_path)
            scenario_payload = scenario_model.model_dump()
            stress_event_payloads = scenario_to_payloads(scenario_model)
            stress_events = build_stress_events(stress_event_payloads)
            console.print(
                f"Loaded {len(stress_events)} stress events from scenario: [magenta]{scenario_path.name}[/magenta]"
            )
            for warning in scenario_warnings(scenario_model):
                console.print(f"[yellow]Warning:[/yellow] {warning}")
        except ValidationError as e:
            console.print("[bold red]Scenario validation failed:[/bold red]")
            for line in format_validation_error(e):
                console.print(f"- {line}")
            raise typer.Exit(code=1)
    
    # 5. Run Simulation
    safety_config = _build_safety_config_from_options(
        min_glucose=safety_min_glucose,
        max_glucose=safety_max_glucose,
        max_glucose_delta_per_5_min=safety_max_glucose_delta_per_5_min,
        hypoglycemia_threshold=safety_hypoglycemia_threshold,
        severe_hypoglycemia_threshold=safety_severe_hypoglycemia_threshold,
        hyperglycemia_threshold=safety_hyperglycemia_threshold,
        max_insulin_per_bolus=safety_max_insulin_per_bolus,
        glucose_rate_alarm=safety_glucose_rate_alarm,
        max_insulin_per_hour=safety_max_insulin_per_hour,
        max_iob=safety_max_iob,
        trend_stop=safety_trend_stop,
        hypo_cutoff=safety_hypo_cutoff,
        critical_glucose_threshold=safety_critical_glucose_threshold,
        critical_glucose_duration_minutes=safety_critical_glucose_duration_minutes,
        predictor_uncertainty_gate_enabled=safety_predictor_uncertainty_gate_enabled,
        predictor_uncertainty_max_std_mgdl=safety_predictor_uncertainty_max_std_mgdl,
        predictor_ood_gate_enabled=safety_predictor_ood_gate_enabled,
        predictor_ood_zscore_threshold=safety_predictor_ood_zscore_threshold,
        predictor_ood_max_feature_fraction=safety_predictor_ood_max_feature_fraction,
    )

    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)
    output_dir = resolve_output_dir(output_dir, run_id)

    effective_safety_config = safety_config or SafetyConfig()
    sensor_model = None
    if any(v is not None for v in (sensor_noise_std, sensor_lag_minutes, sensor_dropout_prob, sensor_bias)):
        sensor_model = iints.SensorModel(
            noise_std=float(sensor_noise_std or 0.0),
            lag_minutes=int(sensor_lag_minutes or 0),
            dropout_prob=float(sensor_dropout_prob or 0.0),
            bias=float(sensor_bias or 0.0),
            seed=resolved_seed,
        )
    elif patient_model_type == "auto":
        sensor_model = iints.SensorModel(
            noise_std=7.0,
            lag_minutes=10,
            dropout_prob=0.0,
            bias=0.0,
            seed=resolved_seed,
        )

    simulator = iints.Simulator(
        patient_model=patient_model,
        algorithm=algorithm_instance,
        time_step=time_step,
        seed=resolved_seed,
        safety_config=effective_safety_config,
        sensor_model=sensor_model,
        predictor=predictor,
    )
    
    for event in stress_events:
        simulator.add_stress_event(event)

    simulation_results_df, safety_report = simulator.run_batch(duration)
    
    # 6. Output Results
    config_payload: Dict[str, Any] = {
        "run_type": "single",
        "algorithm": {
            "class": f"{algorithm_instance.__class__.__module__}.{algorithm_instance.__class__.__name__}",
            "metadata": algorithm_instance.get_algorithm_metadata().to_dict(),
        },
        "patient_config": validated_patient_params,
        "scenario": scenario_payload,
        "duration_minutes": duration,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
        "compare_baselines": compare_baselines,
        "export_audit": False,
        "generate_report": True,
        "safety_config": asdict(effective_safety_config),
        "predictor_path": str(predictor_path) if predictor_path else None,
    }
    config_path = output_dir / "config.json"
    write_json(config_path, config_payload)
    run_metadata = build_run_metadata(run_id, resolved_seed, config_payload, output_dir)
    run_metadata_path = output_dir / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    results_file = output_dir / "results.csv"
    
    simulation_results_df.to_csv(results_file, index=False)
    
    console.print(f"\nSimulation completed. Results saved to: {results_file}")
    console.print(f"Run metadata: {run_metadata_path}")
    console.print("\n--- Safety Report ---")
    for key, value in safety_report.items():
        console.print(f"{key}: {value}")
    
    console.print("\nDisplaying head of simulation results:")
    console.print(Panel(str(simulation_results_df.head()))) # Use Panel for rich output

    if compare_baselines:
        comparison = run_baseline_comparison(
            patient_params=validated_patient_params,
            stress_event_payloads=stress_event_payloads,
            duration=duration,
            time_step=time_step,
            primary_label=algorithm_instance.get_algorithm_metadata().name,
            primary_results=simulation_results_df,
            primary_safety=safety_report,
            seed=resolved_seed,
        )
        safety_report["baseline_comparison"] = comparison
        baseline_paths = write_baseline_comparison(comparison, output_dir / "baseline")
        console.print(f"Baseline comparison saved to: {baseline_paths}")

    # Generate full report (using the new iints.generate_report function)
    report_output_path = output_dir / "report.pdf"
    iints.generate_report(simulation_results_df, str(report_output_path), safety_report)

    manifest_files = {
        "config": config_path,
        "run_metadata": run_metadata_path,
        "results_csv": results_file,
        "report_pdf": report_output_path,
    }
    if compare_baselines:
        manifest_files["baseline_json"] = output_dir / "baseline" / "baseline_comparison.json"
        manifest_files["baseline_csv"] = output_dir / "baseline" / "baseline_comparison.csv"
    run_manifest = build_run_manifest(output_dir, manifest_files)
    run_manifest_path = output_dir / "run_manifest.json"
    write_json(run_manifest_path, run_manifest)
    console.print(f"Run manifest: {run_manifest_path}")


@app.command("run-full")
def run_full(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt) for dual-guard forecasting")] = None,
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default_patient')")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Path to a patient config YAML (overrides --patient-config-name)")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to the scenario JSON file")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save results + audit + report")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
    safety_min_glucose: Annotated[Optional[float], typer.Option("--safety-min-glucose", help="Min plausible glucose (mg/dL)")] = None,
    safety_max_glucose: Annotated[Optional[float], typer.Option("--safety-max-glucose", help="Max plausible glucose (mg/dL)")] = None,
    safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option("--safety-max-glucose-delta-per-5-min", help="Max glucose delta per 5 min (mg/dL)")] = None,
    safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hypo-threshold", help="Hypoglycemia threshold (mg/dL)")] = None,
    safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-severe-hypo-threshold", help="Severe hypoglycemia threshold (mg/dL)")] = None,
    safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hyper-threshold", help="Hyperglycemia threshold (mg/dL)")] = None,
    safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option("--safety-max-bolus", help="Max insulin per bolus (U)")] = None,
    safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option("--safety-glucose-rate-alarm", help="Glucose rate alarm (mg/dL/min)")] = None,
    safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option("--safety-max-insulin-per-hour", help="Max insulin per 60 min (U)")] = None,
    safety_max_iob: Annotated[Optional[float], typer.Option("--safety-max-iob", help="Max insulin on board (U)")] = None,
    safety_trend_stop: Annotated[Optional[float], typer.Option("--safety-trend-stop", help="Negative trend cutoff (mg/dL/min)")] = None,
    safety_hypo_cutoff: Annotated[Optional[float], typer.Option("--safety-hypo-cutoff", help="Hard hypo cutoff (mg/dL)")] = None,
    safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option("--safety-critical-glucose", help="Critical glucose threshold (mg/dL)")] = None,
    safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option("--safety-critical-duration", help="Critical glucose duration (minutes)")] = None,
    safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-uncertainty-gate", help="Enable predictor uncertainty gate")] = None,
    safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option("--safety-predictor-max-std", help="Max allowed predictor std (mg/dL) before fallback")] = None,
    safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-ood-gate", help="Enable predictor OOD gate")] = None,
    safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option("--safety-predictor-ood-z", help="Predictor OOD z-score threshold")] = None,
    safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option("--safety-predictor-ood-feature-fraction", help="Max fraction of features allowed OOD")] = None,
):
    """One-line runner: results CSV + audit + PDF + baseline comparison."""
    console = Console()
    algorithm_instance = _load_algorithm_instance(algo, console)
    predictor = _load_predictor_service_from_path(predictor_path, console)

    patient_config: Union[str, Path]
    if patient_config_path:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Error: Patient config file '{patient_config_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        patient_config = patient_config_path
    else:
        patient_config = patient_config_name

    safety_config = _build_safety_config_from_options(
        min_glucose=safety_min_glucose,
        max_glucose=safety_max_glucose,
        max_glucose_delta_per_5_min=safety_max_glucose_delta_per_5_min,
        hypoglycemia_threshold=safety_hypoglycemia_threshold,
        severe_hypoglycemia_threshold=safety_severe_hypoglycemia_threshold,
        hyperglycemia_threshold=safety_hyperglycemia_threshold,
        max_insulin_per_bolus=safety_max_insulin_per_bolus,
        glucose_rate_alarm=safety_glucose_rate_alarm,
        max_insulin_per_hour=safety_max_insulin_per_hour,
        max_iob=safety_max_iob,
        trend_stop=safety_trend_stop,
        hypo_cutoff=safety_hypo_cutoff,
        critical_glucose_threshold=safety_critical_glucose_threshold,
        critical_glucose_duration_minutes=safety_critical_glucose_duration_minutes,
        predictor_uncertainty_gate_enabled=safety_predictor_uncertainty_gate_enabled,
        predictor_uncertainty_max_std_mgdl=safety_predictor_uncertainty_max_std_mgdl,
        predictor_ood_gate_enabled=safety_predictor_ood_gate_enabled,
        predictor_ood_zscore_threshold=safety_predictor_ood_zscore_threshold,
        predictor_ood_max_feature_fraction=safety_predictor_ood_max_feature_fraction,
    )

    outputs = iints.run_full(
        algorithm=algorithm_instance,
        scenario=str(scenario_path) if scenario_path else None,
        patient_config=patient_config,
        duration_minutes=duration,
        time_step=time_step,
        seed=seed,
        output_dir=output_dir,
        safety_config=safety_config,
        predictor=predictor,
    )

    console.print("[green]Run complete.[/green]")
    if "results_csv" in outputs:
        console.print(f"Results CSV: {outputs['results_csv']}")
    if "report_pdf" in outputs:
        console.print(f"Report PDF: {outputs['report_pdf']}")
    if "audit" in outputs:
        console.print(f"Audit: {outputs['audit']}")
    if "baseline_files" in outputs:
        console.print(f"Baseline files: {outputs['baseline_files']}")
    if "run_metadata_path" in outputs:
        console.print(f"Run metadata: {outputs['run_metadata_path']}")
    if "run_manifest_path" in outputs:
        console.print(f"Run manifest: {outputs['run_manifest_path']}")
    if "profiling_path" in outputs:
        console.print(f"Profiling report: {outputs['profiling_path']}")
    if "run_manifest_signature" in outputs:
        console.print(f"Run manifest signature: {outputs['run_manifest_signature']}")


@app.command("run-parallel")
def run_parallel(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt) for dual-guard forecasting")] = None,
    scenarios_dir: Annotated[Optional[Path], typer.Option(help="Directory with scenario JSON files")] = None,
    scenario_paths: Annotated[List[Path], typer.Option("--scenario-path", help="Scenario JSON path (repeatable)")] = [],
    patient_config_name: Annotated[str, typer.Option(help="Patient config name")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Patient config YAML path")] = None,
    patient_configs_dir: Annotated[Optional[Path], typer.Option(help="Directory of patient YAML configs")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Root directory for batch outputs")] = Path("./results/batch"),
    max_workers: Annotated[Optional[int], typer.Option(help="Max parallel workers")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Base seed for deterministic runs")] = None,
    compare_baselines: Annotated[bool, typer.Option(help="Run PID + standard pump baselines")] = False,
    export_audit: Annotated[bool, typer.Option(help="Export audit trails")] = False,
    generate_report: Annotated[bool, typer.Option(help="Generate PDF reports")] = False,
    safety_min_glucose: Annotated[Optional[float], typer.Option("--safety-min-glucose")] = None,
    safety_max_glucose: Annotated[Optional[float], typer.Option("--safety-max-glucose")] = None,
    safety_max_glucose_delta_per_5_min: Annotated[Optional[float], typer.Option("--safety-max-glucose-delta-per-5-min")] = None,
    safety_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hypo-threshold")] = None,
    safety_severe_hypoglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-severe-hypo-threshold")] = None,
    safety_hyperglycemia_threshold: Annotated[Optional[float], typer.Option("--safety-hyper-threshold")] = None,
    safety_max_insulin_per_bolus: Annotated[Optional[float], typer.Option("--safety-max-bolus")] = None,
    safety_glucose_rate_alarm: Annotated[Optional[float], typer.Option("--safety-glucose-rate-alarm")] = None,
    safety_max_insulin_per_hour: Annotated[Optional[float], typer.Option("--safety-max-insulin-per-hour")] = None,
    safety_max_iob: Annotated[Optional[float], typer.Option("--safety-max-iob")] = None,
    safety_trend_stop: Annotated[Optional[float], typer.Option("--safety-trend-stop")] = None,
    safety_hypo_cutoff: Annotated[Optional[float], typer.Option("--safety-hypo-cutoff")] = None,
    safety_critical_glucose_threshold: Annotated[Optional[float], typer.Option("--safety-critical-glucose")] = None,
    safety_critical_glucose_duration_minutes: Annotated[Optional[int], typer.Option("--safety-critical-duration")] = None,
    safety_predictor_uncertainty_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-uncertainty-gate")] = None,
    safety_predictor_uncertainty_max_std_mgdl: Annotated[Optional[float], typer.Option("--safety-predictor-max-std")] = None,
    safety_predictor_ood_gate_enabled: Annotated[Optional[bool], typer.Option("--safety-predictor-ood-gate")] = None,
    safety_predictor_ood_zscore_threshold: Annotated[Optional[float], typer.Option("--safety-predictor-ood-z")] = None,
    safety_predictor_ood_max_feature_fraction: Annotated[Optional[float], typer.Option("--safety-predictor-ood-feature-fraction")] = None,
):
    """Run many scenarios in parallel across CPU cores."""
    console = Console()
    base_seed = resolve_seed(seed)
    if predictor_path is not None:
        _load_predictor_service_from_path(predictor_path, console)
    if scenarios_dir is None and not scenario_paths:
        console.print("[bold red]Provide --scenarios-dir or at least one --scenario-path.[/bold red]")
        raise typer.Exit(code=1)

    scenarios: List[Path] = []
    if scenarios_dir:
        if not scenarios_dir.is_dir():
            console.print(f"[bold red]Scenarios directory not found: {scenarios_dir}[/bold red]")
            raise typer.Exit(code=1)
        scenarios.extend(sorted(scenarios_dir.glob("*.json")))
    scenarios.extend(scenario_paths)
    scenarios = [path for path in scenarios if path.is_file()]
    if not scenarios:
        console.print("[bold red]No scenario JSON files found.[/bold red]")
        raise typer.Exit(code=1)

    patient_configs: List[Union[str, Path]] = []
    patient_labels: List[str] = []
    if patient_configs_dir:
        if not patient_configs_dir.is_dir():
            console.print(f"[bold red]Patient config directory not found: {patient_configs_dir}[/bold red]")
            raise typer.Exit(code=1)
        for path in sorted(patient_configs_dir.glob("*.yaml")):
            patient_configs.append(path)
            patient_labels.append(path.stem)
    elif patient_config_path:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Patient config file not found: {patient_config_path}[/bold red]")
            raise typer.Exit(code=1)
        patient_configs.append(patient_config_path)
        patient_labels.append(patient_config_path.stem)
    else:
        patient_configs.append(patient_config_name)
        patient_labels.append(patient_config_name)

    safety_overrides = {
        "min_glucose": safety_min_glucose,
        "max_glucose": safety_max_glucose,
        "max_glucose_delta_per_5_min": safety_max_glucose_delta_per_5_min,
        "hypoglycemia_threshold": safety_hypoglycemia_threshold,
        "severe_hypoglycemia_threshold": safety_severe_hypoglycemia_threshold,
        "hyperglycemia_threshold": safety_hyperglycemia_threshold,
        "max_insulin_per_bolus": safety_max_insulin_per_bolus,
        "glucose_rate_alarm": safety_glucose_rate_alarm,
        "max_insulin_per_hour": safety_max_insulin_per_hour,
        "max_iob": safety_max_iob,
        "trend_stop": safety_trend_stop,
        "hypo_cutoff": safety_hypo_cutoff,
        "critical_glucose_threshold": safety_critical_glucose_threshold,
        "critical_glucose_duration_minutes": safety_critical_glucose_duration_minutes,
        "predictor_uncertainty_gate_enabled": safety_predictor_uncertainty_gate_enabled,
        "predictor_uncertainty_max_std_mgdl": safety_predictor_uncertainty_max_std_mgdl,
        "predictor_ood_gate_enabled": safety_predictor_ood_gate_enabled,
        "predictor_ood_zscore_threshold": safety_predictor_ood_zscore_threshold,
        "predictor_ood_max_feature_fraction": safety_predictor_ood_max_feature_fraction,
    }
    safety_overrides = {k: v for k, v in safety_overrides.items() if v is not None}

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Dict[str, Any]] = []
    idx = 0
    for scenario in scenarios:
        for patient_config, patient_label in zip(patient_configs, patient_labels):
            job_seed = base_seed + idx
            run_output_dir = output_dir / f"{scenario.stem}__{patient_label}"
            jobs.append(
                {
                    "algo": str(algo),
                    "scenario_path": str(scenario),
                    "patient_config": patient_config,
                    "patient_label": patient_label,
                    "output_dir": str(run_output_dir),
                    "duration_minutes": duration,
                    "time_step": time_step,
                    "seed": job_seed,
                    "compare_baselines": compare_baselines,
                    "export_audit": export_audit,
                    "generate_report": generate_report,
                    "safety_overrides": safety_overrides,
                    "predictor_path": str(predictor_path) if predictor_path else None,
                }
            )
            idx += 1

    console.print(f"Launching {len(jobs)} parallel jobs...")
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_parallel_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "scenario": Path(job["scenario_path"]).stem,
                    "patient": job["patient_label"],
                    "output_dir": job["output_dir"],
                    "results_csv": "",
                    "report_pdf": "",
                    "terminated_early": False,
                    "total_violations": 0,
                    "error": str(exc),
                }
            results.append(result)

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    console.print(f"[green]Batch summary saved:[/green] {summary_path}")


@app.command("scorecard")
def scorecard(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    profile: Annotated[str, typer.Option(help="Validation profile id")] = "research_default",
    predictor_path: Annotated[Optional[Path], typer.Option("--predictor", help="Optional predictor checkpoint (.pt)")] = None,
    presets_csv: Annotated[Optional[str], typer.Option(help="Comma-separated preset names (default: all bundled presets)")] = None,
    output_dir: Annotated[Path, typer.Option(help="Output directory for scorecard artifacts")] = Path("results/scorecard"),
    seed: Annotated[Optional[int], typer.Option(help="Base random seed")] = 42,
):
    """
    Run a scenario bank and generate a validation scorecard.
    """
    console = Console()
    algorithm_template = _load_algorithm_instance(algo, console)
    predictor = _load_predictor_service_from_path(predictor_path, console)
    try:
        profiles = load_validation_profiles()
    except Exception as exc:
        console.print(f"[bold red]Could not load validation profiles: {exc}[/bold red]")
        raise typer.Exit(code=1)
    selected_profile = profiles.get(profile)
    if selected_profile is None:
        console.print(f"[bold red]Unknown profile '{profile}'.[/bold red]")
        raise typer.Exit(code=1)

    all_presets = _load_presets()
    selected_names: Optional[set[str]] = None
    if presets_csv:
        selected_names = {name.strip() for name in presets_csv.split(",") if name.strip()}
    presets_to_run = [
        preset for preset in all_presets
        if selected_names is None or preset.get("name") in selected_names
    ]
    if not presets_to_run:
        console.print("[bold red]No presets selected for scorecard.[/bold red]")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_base = resolve_seed(seed)
    rows: List[Dict[str, Any]] = []
    for idx, preset in enumerate(presets_to_run):
        preset_name = str(preset.get("name", f"preset_{idx}"))
        scenario_payload = preset.get("scenario")
        if not isinstance(scenario_payload, dict):
            console.print(f"[yellow]Skipping {preset_name}: no scenario payload[/yellow]")
            continue

        run_output_dir = output_dir / preset_name
        algorithm_instance = algorithm_template.__class__()
        outputs = iints.run_simulation(
            algorithm=algorithm_instance,
            scenario=scenario_payload,
            patient_config=str(preset.get("patient_config", "default_patient")),
            duration_minutes=int(preset.get("duration_minutes", 720)),
            time_step=int(preset.get("time_step_minutes", 5)),
            seed=seed_base + idx,
            output_dir=run_output_dir,
            compare_baselines=False,
            export_audit=True,
            generate_report=False,
            predictor=predictor,
        )
        results_df = outputs["results"]
        safety_report = outputs.get("safety_report", {})
        validation = evaluate_run(
            results_df,
            profile=selected_profile,
            safety_report=safety_report,
            duration_minutes=int(preset.get("duration_minutes", 720)),
        )

        rows.append(
            {
                "preset": preset_name,
                "passed": validation.passed,
                "required_checks_passed": validation.required_checks_passed,
                "required_checks_total": validation.required_checks_total,
                "validation_score": validation.score,
                "tir_70_180": validation.metrics.get("tir_70_180"),
                "tir_below_70": validation.metrics.get("tir_below_70"),
                "tir_below_54": validation.metrics.get("tir_below_54"),
                "cv": validation.metrics.get("cv"),
                "safety_index": validation.metrics.get("safety_index"),
                "supervisor_interventions_per_hour": validation.metrics.get("supervisor_interventions_per_hour"),
                "output_dir": str(run_output_dir),
            }
        )

        (run_output_dir / "validation_report.json").write_text(json.dumps(validation.to_dict(), indent=2))

    if not rows:
        console.print("[bold red]Scorecard produced no rows.[/bold red]")
        raise typer.Exit(code=1)

    scorecard_df = pd.DataFrame(rows).sort_values(by=["passed", "validation_score"], ascending=[False, False])
    scorecard_csv = output_dir / "scorecard.csv"
    scorecard_json = output_dir / "scorecard.json"
    scorecard_df.to_csv(scorecard_csv, index=False)
    scorecard_json.write_text(json.dumps(rows, indent=2))

    table = Table(title=f"Scenario Bank Scorecard ({selected_profile.profile_id})")
    table.add_column("Preset", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Checks", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("TIR", justify="right")
    table.add_column("T<70", justify="right")
    for row in rows:
        table.add_row(
            str(row["preset"]),
            "[green]PASS[/green]" if bool(row["passed"]) else "[red]FAIL[/red]",
            f"{int(row['required_checks_passed'])}/{int(row['required_checks_total'])}",
            f"{float(row['validation_score']):.1f}",
            f"{float(row['tir_70_180']):.1f}",
            f"{float(row['tir_below_70']):.1f}",
        )
    console.print(table)
    console.print(f"[green]Scorecard CSV:[/green] {scorecard_csv}")
    console.print(f"[green]Scorecard JSON:[/green] {scorecard_json}")

@app.command()
def report(
    results_csv: Annotated[Path, typer.Option(help="Path to a simulation results CSV")],
    output_path: Annotated[Path, typer.Option(help="Output PDF path")] = Path("./results/clinical_report.pdf"),
    safety_report_path: Annotated[Optional[Path], typer.Option(help="Optional safety report JSON path")] = None,
    audit_output_dir: Annotated[Optional[Path], typer.Option(help="Optional audit output directory")] = None,
    bundle_dir: Annotated[Optional[Path], typer.Option(help="If set, write PDF + plots + audit into this folder")] = None,
):
    """Generate a clinical PDF report (and optional audit summary) from a results CSV."""
    console = Console()
    if not results_csv.is_file():
        console.print(f"[bold red]Error: Results file '{results_csv}' not found.[/bold red]")
        raise typer.Exit(code=1)

    results_df = pd.read_csv(results_csv)
    safety_report: Dict[str, Any] = {}
    if safety_report_path:
        if not safety_report_path.is_file():
            console.print(f"[bold red]Error: Safety report file '{safety_report_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        safety_report = json.loads(safety_report_path.read_text())

    if bundle_dir:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        output_path = bundle_dir / "clinical_report.pdf"
        audit_output_dir = bundle_dir / "audit"
        plots_dir = bundle_dir / "plots"
        generator = iints.ClinicalReportGenerator()
        generator.export_plots(results_df, str(plots_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iints.generate_report(results_df, str(output_path), safety_report)
    console.print(f"PDF report saved to: [link=file://{output_path}]{output_path}[/link]")

    if audit_output_dir:
        audit_output_dir.mkdir(parents=True, exist_ok=True)
        audit_columns = [
            "time_minutes",
            "glucose_actual_mgdl",
            "glucose_to_algo_mgdl",
            "algo_recommended_insulin_units",
            "delivered_insulin_units",
            "safety_reason",
            "safety_triggered",
            "supervisor_latency_ms",
        ]
        available = [c for c in audit_columns if c in results_df.columns]
        audit_df = results_df[available].copy()

        jsonl_path = audit_output_dir / "audit_trail.jsonl"
        csv_path = audit_output_dir / "audit_trail.csv"
        summary_path = audit_output_dir / "audit_summary.json"

        audit_df.to_json(jsonl_path, orient="records", lines=True)
        audit_df.to_csv(csv_path, index=False)

        overrides = audit_df[audit_df.get("safety_triggered", False) == True] if "safety_triggered" in audit_df.columns else audit_df.iloc[0:0]
        reasons = overrides["safety_reason"].value_counts().to_dict() if "safety_reason" in overrides.columns else {}
        summary = {
            "total_steps": int(len(audit_df)),
            "total_overrides": int(len(overrides)),
            "top_reasons": reasons,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        console.print(f"Audit exports saved to: {audit_output_dir}")


@app.command()
def validate(
    scenario_path: Annotated[Path, typer.Option(help="Path to a scenario JSON file")],
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Optional patient config YAML to validate")] = None,
):
    """Validate a scenario JSON for out-of-range values and missing keys."""
    console = Console()
    if not scenario_path.is_file():
        console.print(f"[bold red]Error: Scenario file '{scenario_path}' not found.[/bold red]")
        raise typer.Exit(code=1)

    try:
        scenario = json.loads(scenario_path.read_text())
        scenario_model = validate_scenario_dict(scenario)
        for warning in scenario_warnings(scenario_model):
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error: Invalid JSON - {e}[/bold red]")
        raise typer.Exit(code=1)
    except ValidationError as e:
        console.print("[bold red]Scenario validation failed:[/bold red]")
        for line in format_validation_error(e):
            console.print(f"- {line}")
        raise typer.Exit(code=1)

    if patient_config_path:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Error: Patient config '{patient_config_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        try:
            patient_config = yaml.safe_load(patient_config_path.read_text())
            validate_patient_config_dict(patient_config)
        except yaml.YAMLError as e:
            console.print(f"[bold red]Error: Invalid YAML - {e}[/bold red]")
            raise typer.Exit(code=1)
        except ValidationError as e:
            console.print("[bold red]Patient config validation failed:[/bold red]")
            for line in format_validation_error(e):
                console.print(f"- {line}")
            raise typer.Exit(code=1)

    console.print("[green]Scenario validation passed.[/green]")


@data_app.command("list")
def data_list():
    """List official datasets and access requirements."""
    console = Console()
    datasets = load_dataset_registry()
    table = Table(title="IINTS-AF Official Datasets", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="green")
    table.add_column("Name", style="white")
    table.add_column("Access", style="magenta")
    table.add_column("Source", style="yellow")
    for entry in datasets:
        table.add_row(
            entry.get("id", ""),
            entry.get("name", ""),
            entry.get("access", ""),
            entry.get("source", ""),
        )
    console.print(table)


@data_app.command("info")
def data_info(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id (see `iints data list`)")],
):
    """Show metadata and access info for a dataset."""
    console = Console()
    try:
        dataset = get_dataset(dataset_id)
    except DatasetRegistryError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(dataset, indent=2))
    citation = dataset.get("citation", {})
    if citation:
        text = citation.get("text")
        bibtex = citation.get("bibtex")
        if text:
            console.print("\n[bold]Citation (text)[/bold]")
            console.print(text)
        if bibtex:
            console.print("\n[bold]Citation (BibTeX)[/bold]")
            console.print(bibtex)


@data_app.command("cite")
def data_cite(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id (see `iints data list`)")],
):
    """Print BibTeX citation for a dataset."""
    console = Console()
    try:
        dataset = get_dataset(dataset_id)
    except DatasetRegistryError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    citation = dataset.get("citation", {})
    bibtex = citation.get("bibtex")
    text = citation.get("text")
    if bibtex:
        console.print(bibtex)
        return
    if text:
        console.print(text)
        return
    console.print("[yellow]No citation available for this dataset.[/yellow]")


@data_app.command("fetch")
def data_fetch(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id (see `iints data list`)")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory (default: data_packs/official/<id>)")] = None,
    extract: Annotated[bool, typer.Option(help="Extract zip files if present")] = True,
    verify: Annotated[bool, typer.Option(help="Verify SHA-256 if available and emit SHA256SUMS.txt")] = True,
):
    """Download a dataset (public-download only)."""
    console = Console()
    try:
        dataset = get_dataset(dataset_id)
    except DatasetRegistryError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = Path("data_packs") / "official" / dataset_id
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    access = dataset.get("access", "manual")
    landing = dataset.get("landing_page", "")
    if access in {"request", "manual"}:
        console.print("[yellow]Manual download required for this dataset.[/yellow]")
        if landing:
            console.print(f"Source: {landing}")
        console.print("After downloading, place files in:")
        console.print(f"  {output_dir}")
        return

    try:
        downloaded = fetch_dataset(dataset_id, output_dir=output_dir, extract=extract, verify=verify)
        console.print(f"[green]Downloaded {len(downloaded)} file(s) to {output_dir}[/green]")
    except DatasetFetchError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)


def _build_data_contract_template() -> Dict[str, Any]:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "security": "PII_ENCRYPTED",
                "metadata": {
                    "required_columns": ["timestamp", "glucose"],
                    "column_types": {
                        "glucose": "float",
                    },
                    "ranges": {
                        "glucose": {"min": 40, "max": 400},
                    },
                    "unit_conversions": {
                        "glucose": {"from": "mmol/L", "to": "mg/dL"},
                    },
                },
            }
        ],
        "processes": [
            {
                "name": "GlucoseData",
                "input_stream": "PatientHealth.glucose",
                "features": [
                    {
                        "name": "rolling_avg_15m",
                        "operation": "moving_average",
                        "source": "PatientHealth.glucose",
                        "window": "15m",
                    }
                ],
                "labels": [
                    {
                        "name": "hyper_event",
                        "expression": "glucose > 180",
                        "classes": ["Critical", "Normal"],
                    }
                ],
                "validations": [
                    {
                        "expression": "glucose is not null and glucose > 20",
                        "on_fail": "DISCARD_AND_LOG",
                    }
                ],
            }
        ],
    }


@data_app.command("contract-template")
def data_contract_template(
    output_path: Annotated[Path, typer.Option(help="Where to write the starter contract YAML")] = Path("data_contract.yaml"),
):
    """Write a starter data contract template for model-ready validation."""
    console = Console()
    template = _build_data_contract_template()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(template, sort_keys=False))
    console.print(f"[green]Contract template written:[/green] {output_path}")


@data_app.command("contract-run")
def data_contract_run(
    contract_path: Annotated[Path, typer.Argument(help="Path to contract YAML")],
    input_csv: Annotated[Path, typer.Argument(help="Path to input CSV")],
    output_json: Annotated[Optional[Path], typer.Option(help="Optional output report JSON path")] = None,
    apply_builtin_transforms: Annotated[bool, typer.Option(help="Apply built-in unit conversion transforms from the contract")] = True,
    fail_on_noncompliant: Annotated[bool, typer.Option(help="Exit code 1 when compliance checks fail")] = False,
    min_mdmp_grade: Annotated[
        Optional[str],
        typer.Option(help="Optional MDMP grade gate (draft, research_grade, clinical_grade)"),
    ] = None,
):
    """Validate a dataset against a model-ready contract and compute compliance score."""
    console = Console()
    if not contract_path.is_file():
        console.print(f"[bold red]Contract file not found: {contract_path}[/bold red]")
        raise typer.Exit(code=1)
    if not input_csv.is_file():
        console.print(f"[bold red]Input CSV not found: {input_csv}[/bold red]")
        raise typer.Exit(code=1)

    try:
        contract = load_contract_yaml(contract_path)
        df = pd.read_csv(input_csv)
        runner = ContractRunner(contract)
        report = runner.run(df, apply_builtin_transforms=apply_builtin_transforms)
    except Exception as exc:
        console.print(f"[bold red]Contract runner failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    summary = Table(title="Data Contract Compliance")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    summary.add_row("Rows", str(report.row_count))
    summary.add_row("Compliance", f"{report.compliance_score:.2f}%")
    summary.add_row("Status", "[green]PASS[/green]" if report.is_compliant else "[red]FAIL[/red]")
    summary.add_row("MDMP grade", report.mdmp_grade)
    summary.add_row("MDMP protocol", report.mdmp_protocol_version)
    summary.add_row(
        "Certified",
        "yes" if report.certified_for_medical_research else "no",
    )
    summary.add_row("Contract fingerprint", report.contract_fingerprint_sha256[:16] + "...")
    summary.add_row("Dataset fingerprint", report.dataset_fingerprint_sha256[:16] + "...")
    console.print(summary)

    checks_table = Table(title="Checks")
    checks_table.add_column("Check", style="green")
    checks_table.add_column("Passed")
    checks_table.add_column("Failed rows", justify="right")
    checks_table.add_column("Detail")
    for check in report.checks:
        checks_table.add_row(
            check.name,
            "[green]yes[/green]" if check.passed else "[red]no[/red]",
            str(check.failed_rows),
            check.detail,
        )
    console.print(checks_table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"[green]Contract report written:[/green] {output_json}")

    if min_mdmp_grade is not None:
        normalized = min_mdmp_grade.strip().lower()
        if normalized not in MDMP_GRADE_ORDER:
            allowed = ", ".join(MDMP_GRADE_ORDER)
            console.print(f"[bold red]Invalid --min-mdmp-grade value: {min_mdmp_grade}. Use one of: {allowed}[/bold red]")
            raise typer.Exit(code=1)
        if not mdmp_grade_meets_minimum(report.mdmp_grade, normalized):
            console.print(
                f"[bold red]MDMP gate failed:[/bold red] got {report.mdmp_grade}, requires at least {normalized}"
            )
            raise typer.Exit(code=1)

    if fail_on_noncompliant and not report.is_compliant:
        raise typer.Exit(code=1)


@data_app.command("mdmp-visualizer")
def data_mdmp_visualizer(
    report_json: Annotated[Path, typer.Argument(help="Path to contract-run JSON report")],
    output_html: Annotated[Path, typer.Option(help="Output HTML path")] = Path("results/mdmp_dashboard.html"),
    title: Annotated[str, typer.Option(help="Dashboard title")] = "IINTS MDMP Certification Dashboard",
):
    """Build a single-file interactive MDMP certification dashboard."""
    console = Console()

    if not report_json.is_file():
        console.print(f"[bold red]Report JSON not found: {report_json}[/bold red]")
        raise typer.Exit(code=1)

    try:
        payload = json.loads(report_json.read_text())
        html_text = build_mdmp_dashboard_html(payload, title=title)
    except Exception as exc:
        console.print(f"[bold red]Could not build MDMP visualizer: {exc}[/bold red]")
        raise typer.Exit(code=1)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text)
    console.print(f"[green]MDMP dashboard written:[/green] {output_html}")


@data_app.command("synthetic-mirror")
def data_synthetic_mirror(
    input_csv: Annotated[Path, typer.Argument(help="Source CSV (validated real dataset)")],
    contract_path: Annotated[Path, typer.Argument(help="Contract YAML path used as schema/range guard")],
    output_csv: Annotated[Path, typer.Option(help="Output synthetic CSV path")] = Path("data/synthetic_mirror.csv"),
    output_json: Annotated[Optional[Path], typer.Option(help="Optional synthetic mirror report JSON")] = Path("results/synthetic_mirror_report.json"),
    rows: Annotated[Optional[int], typer.Option(help="Optional number of rows to generate")] = None,
    seed: Annotated[int, typer.Option(help="Random seed for deterministic synthesis")] = 42,
    noise_scale: Annotated[float, typer.Option(help="Numeric perturbation scale as fraction of source std-dev")] = 0.05,
    min_mdmp_grade: Annotated[
        Optional[str],
        typer.Option(help="Optional MDMP grade gate for generated synthetic dataset"),
    ] = "research_grade",
    fail_on_noncompliant: Annotated[bool, typer.Option(help="Exit code 1 when generated dataset fails compliance")] = True,
):
    """Generate a privacy-safe synthetic mirror dataset and validate it against MDMP contract gates."""
    console = Console()
    if not input_csv.is_file():
        console.print(f"[bold red]Input CSV not found: {input_csv}[/bold red]")
        raise typer.Exit(code=1)
    if not contract_path.is_file():
        console.print(f"[bold red]Contract file not found: {contract_path}[/bold red]")
        raise typer.Exit(code=1)

    try:
        source_df = pd.read_csv(input_csv)
        contract = load_contract_yaml(contract_path)
        synthetic_df, artifact = generate_synthetic_mirror(
            source_df,
            contract,
            rows=rows,
            seed=seed,
            noise_scale=noise_scale,
        )
    except Exception as exc:
        console.print(f"[bold red]Synthetic mirror generation failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_csv, index=False)
    console.print(f"[green]Synthetic mirror CSV written:[/green] {output_csv}")

    summary = Table(title="Synthetic Mirror Summary")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    summary.add_row("Source rows", str(artifact.summary.get("source_rows", 0)))
    summary.add_row("Synthetic rows", str(artifact.summary.get("synthetic_rows", 0)))
    summary.add_row("Noise scale", str(artifact.summary.get("noise_scale", 0.0)))
    summary.add_row("MDMP grade", artifact.validation.mdmp_grade)
    summary.add_row("Compliance", f"{artifact.validation.compliance_score:.2f}%")
    summary.add_row("Status", "[green]PASS[/green]" if artifact.validation.is_compliant else "[red]FAIL[/red]")
    console.print(summary)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(artifact.to_dict(), indent=2))
        console.print(f"[green]Synthetic mirror report written:[/green] {output_json}")

    if min_mdmp_grade is not None:
        normalized = min_mdmp_grade.strip().lower()
        if normalized not in MDMP_GRADE_ORDER:
            allowed = ", ".join(MDMP_GRADE_ORDER)
            console.print(f"[bold red]Invalid --min-mdmp-grade value: {min_mdmp_grade}. Use one of: {allowed}[/bold red]")
            raise typer.Exit(code=1)
        if not mdmp_grade_meets_minimum(artifact.validation.mdmp_grade, normalized):
            console.print(
                f"[bold red]MDMP gate failed:[/bold red] got {artifact.validation.mdmp_grade}, requires at least {normalized}"
            )
            raise typer.Exit(code=1)

    if fail_on_noncompliant and not artifact.validation.is_compliant:
        raise typer.Exit(code=1)


@mdmp_app.command("template")
def mdmp_template(
    output_path: Annotated[Path, typer.Option(help="Where to write the MDMP contract YAML")] = Path("mdmp_contract.yaml"),
):
    """Write an MDMP contract template (preferred MDMP namespace)."""
    data_contract_template(output_path=output_path)


@mdmp_app.command("validate")
def mdmp_validate(
    contract_path: Annotated[Path, typer.Argument(help="Path to MDMP contract YAML")],
    input_csv: Annotated[Path, typer.Argument(help="Path to input CSV")],
    output_json: Annotated[Optional[Path], typer.Option(help="Optional output report JSON path")] = None,
    apply_builtin_transforms: Annotated[
        bool,
        typer.Option(help="Apply built-in unit conversion transforms (default: off for explicit MDMP operation)"),
    ] = False,
    fail_on_noncompliant: Annotated[bool, typer.Option(help="Exit code 1 when compliance checks fail")] = False,
    min_mdmp_grade: Annotated[
        Optional[str],
        typer.Option(help="Optional MDMP grade gate (draft, research_grade, clinical_grade)"),
    ] = None,
):
    """Validate dataset against MDMP contract (preferred MDMP namespace)."""
    data_contract_run(
        contract_path=contract_path,
        input_csv=input_csv,
        output_json=output_json,
        apply_builtin_transforms=apply_builtin_transforms,
        fail_on_noncompliant=fail_on_noncompliant,
        min_mdmp_grade=min_mdmp_grade,
    )


@mdmp_app.command("visualizer")
def mdmp_visualizer(
    report_json: Annotated[Path, typer.Argument(help="Path to MDMP validation report JSON")],
    output_html: Annotated[Path, typer.Option(help="Output HTML path")] = Path("results/mdmp_dashboard.html"),
    title: Annotated[str, typer.Option(help="Dashboard title")] = "IINTS MDMP Certification Dashboard",
):
    """Generate MDMP dashboard from report JSON (preferred MDMP namespace)."""
    data_mdmp_visualizer(report_json=report_json, output_html=output_html, title=title)


@mdmp_app.command("synthetic-mirror")
def mdmp_synthetic_mirror(
    input_csv: Annotated[Path, typer.Argument(help="Source CSV")],
    contract_path: Annotated[Path, typer.Argument(help="MDMP contract YAML")],
    output_csv: Annotated[Path, typer.Option(help="Output synthetic CSV path")] = Path("data/synthetic_mirror.csv"),
    output_json: Annotated[Optional[Path], typer.Option(help="Optional synthetic mirror report JSON")] = Path("results/synthetic_mirror_report.json"),
    rows: Annotated[Optional[int], typer.Option(help="Optional number of rows to generate")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    noise_scale: Annotated[float, typer.Option(help="Numeric perturbation scale")] = 0.05,
    min_mdmp_grade: Annotated[Optional[str], typer.Option(help="Optional MDMP grade gate")] = "research_grade",
    fail_on_noncompliant: Annotated[bool, typer.Option(help="Exit code 1 when generated dataset fails compliance")] = True,
):
    """Generate synthetic mirror dataset (preferred MDMP namespace)."""
    data_synthetic_mirror(
        input_csv=input_csv,
        contract_path=contract_path,
        output_csv=output_csv,
        output_json=output_json,
        rows=rows,
        seed=seed,
        noise_scale=noise_scale,
        min_mdmp_grade=min_mdmp_grade,
        fail_on_noncompliant=fail_on_noncompliant,
    )


@app.command("sources")
def sources(
    category: Annotated[Optional[str], typer.Option(help="Filter by source category (guideline, trial, model, dataset, ...).")] = None,
    output_json: Annotated[Optional[Path], typer.Option(help="Optional JSON output path.")] = None,
):
    """List evidence sources used to ground simulation defaults and evaluation targets."""
    console = Console()

    try:
        payload = _filtered_evidence_sources(category=category)
    except Exception as exc:
        console.print(f"[bold red]Could not load evidence sources: {exc}[/bold red]")
        raise typer.Exit(code=1)

    filtered = payload.get("sources", [])
    if not isinstance(filtered, list):
        console.print("[bold red]Invalid evidence sources payload[/bold red]")
        raise typer.Exit(code=1)

    if not filtered:
        if category:
            console.print(f"[yellow]No sources found for category '{category}'.[/yellow]")
        else:
            console.print("[yellow]No evidence sources found.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="IINTS-AF Evidence Sources", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="green")
    table.add_column("Category", style="magenta")
    table.add_column("Component", style="white")
    table.add_column("Year", style="yellow")
    table.add_column("Title", style="white")
    table.add_column("DOI / URL", style="blue")
    for entry in filtered:
        citation = str(entry.get("citation", ""))
        year = ""
        for token in citation.replace(";", " ").replace(".", " ").split():
            if len(token) == 4 and token.isdigit():
                year = token
                break
        doi = str(entry.get("doi", "")).strip()
        url = str(entry.get("url", "")).strip()
        table.add_row(
            str(entry.get("id", "")),
            str(entry.get("category", "")),
            str(entry.get("component", "")),
            year,
            str(entry.get("title", "")),
            doi if doi else url,
        )
    console.print(table)
    console.print("[dim]See docs/EVIDENCE_BASE.md for mapping, assumptions, and implementation notes.[/dim]")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Saved source manifest: {output_json}[/green]")


# ---------------------------------------------------------------------------
# P1-4: Research pipeline CLI commands
# ---------------------------------------------------------------------------

research_app = typer.Typer(help="Research pipeline: dataset preparation and quality reporting.")
app.add_typer(research_app, name="research")


@research_app.command("prepare-azt1d")
def research_prepare_azt1d(
    input_dir: Annotated[Path, typer.Option(help="Root directory containing AZT1D Subject folders")] = Path("data_packs/public/azt1d/AZT1D 2025/CGM Records"),
    output: Annotated[Path, typer.Option(help="Output dataset path (CSV or Parquet)")] = Path("data_packs/public/azt1d/processed/azt1d_merged.csv"),
    report: Annotated[Path, typer.Option(help="Quality report output path")] = Path("data_packs/public/azt1d/quality_report.json"),
    time_step: Annotated[int, typer.Option(help="Expected CGM sample interval (minutes)")] = 5,
    max_gap_multiplier: Annotated[float, typer.Option(help="Segment-break gap multiplier")] = 2.5,
    dia_minutes: Annotated[float, typer.Option(help="Insulin action duration (minutes)")] = 240.0,
    peak_minutes: Annotated[float, typer.Option(help="IOB peak time (minutes, OpenAPS bilinear)")] = 75.0,
    carb_absorb_minutes: Annotated[float, typer.Option(help="Carb absorption duration (minutes)")] = 120.0,
    max_basal: Annotated[float, typer.Option(help="Clip basal values above this (U/hr)")] = 20.0,
    max_bolus: Annotated[float, typer.Option(help="Clip bolus values above this (U)")] = 30.0,
    max_carbs: Annotated[float, typer.Option(help="Clip carb grams above this")] = 200.0,
    basal_is_rate: Annotated[bool, typer.Option(help="Treat Basal column as U/hr (convert to U/step)")] = True,
):
    """
    Prepare the AZT1D CGM dataset for LSTM predictor training.

    Reads per-subject CSVs, applies basal-rate conversion (U/hr → U/step),
    derives IOB/COB using the OpenAPS bilinear model, adds time-of-day
    cyclical features, and writes the merged dataset plus a quality report.

    Example
    -------
    iints research prepare-azt1d --input-dir data_packs/public/azt1d/... --output merged.parquet
    """
    console = Console()
    if not input_dir.exists():
        console.print(f"[bold red]Input directory not found: {input_dir}[/bold red]")
        raise typer.Exit(code=1)

    import subprocess, sys  # noqa: E401
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_azt1d.py"),
        "--input", str(input_dir),
        "--output", str(output),
        "--report", str(report),
        "--time-step", str(time_step),
        "--max-gap-multiplier", str(max_gap_multiplier),
        "--dia-minutes", str(dia_minutes),
        "--peak-minutes", str(peak_minutes),
        "--carb-absorb-minutes", str(carb_absorb_minutes),
        "--max-basal", str(max_basal),
        "--max-bolus", str(max_bolus),
        "--max-carbs", str(max_carbs),
    ]
    if not basal_is_rate:
        cmd.append("--no-basal-is-rate")
    try:
        import importlib.util as _ilu
        # Prefer in-process execution when the research script is importable
        spec = _ilu.spec_from_file_location(
            "_prepare_azt1d",
            Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_azt1d.py",
        )
        if spec is not None and spec.loader is not None:
            import sys as _sys
            _old_argv = _sys.argv[:]
            _sys.argv = cmd[1:]  # strip python interpreter
            try:
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                mod.main()  # type: ignore[attr-defined]
            finally:
                _sys.argv = _old_argv
        else:
            result = subprocess.run(cmd, check=True)
    except Exception as exc:
        console.print(f"[bold red]prepare-azt1d failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Dataset written to:[/green] {output}")
    console.print(f"[green]Quality report   :[/green] {report}")


@research_app.command("prepare-ohio")
def research_prepare_ohio(
    input_dir: Annotated[Path, typer.Option(help="Root directory containing OhioT1DM patient_* folders")] = Path("data_packs/public/ohio_t1dm"),
    output: Annotated[Path, typer.Option(help="Output dataset path (CSV or Parquet)")] = Path("data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv"),
    report: Annotated[Path, typer.Option(help="Quality report output path")] = Path("data_packs/public/ohio_t1dm/quality_report.json"),
    time_step: Annotated[int, typer.Option(help="Expected CGM sample interval (minutes)")] = 5,
    max_gap_multiplier: Annotated[float, typer.Option(help="Segment-break gap multiplier")] = 2.5,
    dia_minutes: Annotated[float, typer.Option(help="Insulin action duration (minutes)")] = 240.0,
    peak_minutes: Annotated[float, typer.Option(help="IOB peak time (minutes, OpenAPS bilinear)")] = 75.0,
    carb_absorb_minutes: Annotated[float, typer.Option(help="Carb absorption duration (minutes)")] = 120.0,
    max_insulin: Annotated[float, typer.Option(help="Clip insulin units above this")] = 30.0,
    max_carbs: Annotated[float, typer.Option(help="Clip carb grams above this")] = 200.0,
    icr_default: Annotated[float, typer.Option(help="Fallback ICR (g/U)")] = 10.0,
    isf_default: Annotated[float, typer.Option(help="Fallback ISF (mg/dL per U)")] = 50.0,
    basal_default: Annotated[float, typer.Option(help="Fallback basal rate (U/hr)")] = 0.0,
    meal_window_min: Annotated[float, typer.Option(help="Meal→insulin matching window (minutes)")] = 30.0,
    isf_window_min: Annotated[float, typer.Option(help="ISF estimation window (minutes)")] = 60.0,
    min_meal_carbs: Annotated[float, typer.Option(help="Minimum carbs to consider a meal (g)")] = 5.0,
    min_bolus: Annotated[float, typer.Option(help="Minimum insulin to consider a bolus (U)")] = 0.1,
):
    """
    Prepare the OhioT1DM dataset for LSTM predictor training.

    Reads per-patient CSVs, derives IOB/COB using the OpenAPS bilinear model,
    estimates effective ISF/ICR/basal per subject, adds time-of-day features,
    and writes the merged dataset plus a quality report.

    Example
    -------
    iints research prepare-ohio --input-dir data_packs/public/ohio_t1dm --output ohio.parquet
    """
    console = Console()
    if not input_dir.exists():
        console.print(f"[bold red]Input directory not found: {input_dir}[/bold red]")
        raise typer.Exit(code=1)

    import subprocess, sys  # noqa: E401
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_ohio_t1dm.py"),
        "--input", str(input_dir),
        "--output", str(output),
        "--report", str(report),
        "--time-step", str(time_step),
        "--max-gap-multiplier", str(max_gap_multiplier),
        "--dia-minutes", str(dia_minutes),
        "--peak-minutes", str(peak_minutes),
        "--carb-absorb-minutes", str(carb_absorb_minutes),
        "--max-insulin", str(max_insulin),
        "--max-carbs", str(max_carbs),
        "--icr-default", str(icr_default),
        "--isf-default", str(isf_default),
        "--basal-default", str(basal_default),
        "--meal-window-min", str(meal_window_min),
        "--isf-window-min", str(isf_window_min),
        "--min-meal-carbs", str(min_meal_carbs),
        "--min-bolus", str(min_bolus),
    ]
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "_prepare_ohio_t1dm",
            Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_ohio_t1dm.py",
        )
        if spec is not None and spec.loader is not None:
            import sys as _sys
            _old_argv = _sys.argv[:]
            _sys.argv = cmd[1:]
            try:
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                mod.main()  # type: ignore[attr-defined]
            finally:
                _sys.argv = _old_argv
        else:
            subprocess.run(cmd, check=True)
    except Exception as exc:
        console.print(f"[bold red]prepare-ohio failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Dataset written to:[/green] {output}")
    console.print(f"[green]Quality report   :[/green] {report}")


@research_app.command("prepare-hupa")
def research_prepare_hupa(
    input_dir: Annotated[Path, typer.Option(help="Root directory containing HUPA-UCM CSV files")] = Path("data_packs/public/hupa_ucm"),
    output: Annotated[Path, typer.Option(help="Output dataset path (CSV or Parquet)")] = Path("data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv"),
    report: Annotated[Path, typer.Option(help="Quality report output path")] = Path("data_packs/public/hupa_ucm/quality_report.json"),
    time_step: Annotated[int, typer.Option(help="Expected CGM sample interval (minutes)")] = 5,
    max_gap_multiplier: Annotated[float, typer.Option(help="Segment-break gap multiplier")] = 2.5,
    dia_minutes: Annotated[float, typer.Option(help="Insulin action duration (minutes)")] = 240.0,
    peak_minutes: Annotated[float, typer.Option(help="IOB peak time (minutes, OpenAPS bilinear)")] = 75.0,
    carb_absorb_minutes: Annotated[float, typer.Option(help="Carb absorption duration (minutes)")] = 120.0,
    max_insulin: Annotated[float, typer.Option(help="Clip insulin units above this")] = 30.0,
    max_carbs: Annotated[float, typer.Option(help="Clip carb grams above this")] = 200.0,
    carb_serving_grams: Annotated[float, typer.Option(help="Carb serving size (g) for carb_input")] = 10.0,
    basal_is_rate: Annotated[bool, typer.Option(help="Treat basal_rate as U/hr (convert to U/step)")] = False,
    icr_default: Annotated[float, typer.Option(help="Fallback ICR (g/U)")] = 10.0,
    isf_default: Annotated[float, typer.Option(help="Fallback ISF (mg/dL per U)")] = 50.0,
    basal_default: Annotated[float, typer.Option(help="Fallback basal rate (U/hr)")] = 0.0,
    meal_window_min: Annotated[float, typer.Option(help="Meal→insulin matching window (minutes)")] = 30.0,
    isf_window_min: Annotated[float, typer.Option(help="ISF estimation window (minutes)")] = 60.0,
    min_meal_carbs: Annotated[float, typer.Option(help="Minimum carbs to consider a meal (g)")] = 5.0,
    min_bolus: Annotated[float, typer.Option(help="Minimum insulin to consider a bolus (U)")] = 0.1,
):
    """
    Prepare the HUPA-UCM dataset for LSTM predictor training.

    Parses per-patient CSVs, derives IOB/COB, estimates ISF/ICR/basal per
    subject, and writes the merged dataset plus a quality report.

    Example
    -------
    iints research prepare-hupa --input-dir data_packs/public/hupa_ucm --output hupa.parquet
    """
    console = Console()
    if not input_dir.exists():
        console.print(f"[bold red]Input directory not found: {input_dir}[/bold red]")
        raise typer.Exit(code=1)

    import subprocess, sys  # noqa: E401
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_hupa_ucm.py"),
        "--input", str(input_dir),
        "--output", str(output),
        "--report", str(report),
        "--time-step", str(time_step),
        "--max-gap-multiplier", str(max_gap_multiplier),
        "--dia-minutes", str(dia_minutes),
        "--peak-minutes", str(peak_minutes),
        "--carb-absorb-minutes", str(carb_absorb_minutes),
        "--max-insulin", str(max_insulin),
        "--max-carbs", str(max_carbs),
        "--carb-serving-grams", str(carb_serving_grams),
        "--icr-default", str(icr_default),
        "--isf-default", str(isf_default),
        "--basal-default", str(basal_default),
        "--meal-window-min", str(meal_window_min),
        "--isf-window-min", str(isf_window_min),
        "--min-meal-carbs", str(min_meal_carbs),
        "--min-bolus", str(min_bolus),
    ]
    if basal_is_rate:
        cmd.append("--basal-is-rate")
    else:
        cmd.append("--no-basal-is-rate")
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "_prepare_hupa_ucm",
            Path(__file__).parent.parent.parent.parent.parent / "research" / "prepare_hupa_ucm.py",
        )
        if spec is not None and spec.loader is not None:
            import sys as _sys
            _old_argv = _sys.argv[:]
            _sys.argv = cmd[1:]
            try:
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                mod.main()  # type: ignore[attr-defined]
            finally:
                _sys.argv = _old_argv
        else:
            subprocess.run(cmd, check=True)
    except Exception as exc:
        console.print(f"[bold red]prepare-hupa failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Dataset written to:[/green] {output}")
    console.print(f"[green]Quality report   :[/green] {report}")


@research_app.command("quality")
def research_quality(
    report: Annotated[Path, typer.Option(help="Path to quality_report.json produced by prepare-azt1d")] = Path("data_packs/public/azt1d/quality_report.json"),
):
    """
    Display a quality report produced by `iints research prepare-azt1d`.

    Shows dataset summary statistics, subject count, glucose range, and
    pipeline parameters in a formatted table.

    Example
    -------
    iints research quality --report data_packs/public/azt1d/quality_report.json
    """
    console = Console()
    if not report.exists():
        console.print(f"[bold red]Report not found: {report}[/bold red]")
        console.print("Run [bold]iints research prepare-azt1d[/bold] first.")
        raise typer.Exit(code=1)

    try:
        data = json.loads(report.read_text())
    except Exception as exc:
        console.print(f"[bold red]Failed to parse report: {exc}[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title=f"AZT1D Dataset Quality Report", show_header=True, header_style="bold cyan")
    table.add_column("Field", style="green")
    table.add_column("Value", style="white")

    display_keys = [
        ("source", "Source directory"),
        ("records_total", "Total records"),
        ("subjects", "Subject count"),
        ("subject_ids", "Subject IDs"),
        ("start_time", "Start time"),
        ("end_time", "End time"),
        ("glucose_mean", "Glucose mean (mg/dL)"),
        ("glucose_std", "Glucose std (mg/dL)"),
        ("glucose_min", "Glucose min (mg/dL)"),
        ("glucose_max", "Glucose max (mg/dL)"),
        ("insulin_mean", "Insulin mean (U/step)"),
        ("carb_mean", "Carb mean (g/step)"),
        ("iob_model", "IOB model"),
        ("basal_is_rate", "Basal as U/hr"),
        ("time_step_minutes", "Time step (min)"),
        ("dia_minutes", "DIA (min)"),
        ("peak_minutes", "Peak time (min)"),
        ("carb_absorb_minutes", "Carb absorb (min)"),
    ]

    for key, label in display_keys:
        if key in data:
            val = data[key]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            elif isinstance(val, float):
                val = f"{val:.3f}"
            else:
                val = str(val)
            table.add_row(label, val)

    console.print(table)


@research_app.command("export-onnx")
def research_export_onnx(
    model: Annotated[Path, typer.Option(help="Predictor checkpoint (.pt)")] = Path("models/hupa_finetuned_v2/predictor.pt"),
    out: Annotated[Path, typer.Option(help="Output ONNX file path")] = Path("models/predictor.onnx"),
):
    """
    Export a trained predictor to ONNX for edge/Jetson deployment.
    """
    console = Console()
    if not model.exists():
        console.print(f"[bold red]Model not found: {model}[/bold red]")
        raise typer.Exit(code=1)

    import subprocess, sys  # noqa: E401
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent.parent.parent.parent / "research" / "export_predictor.py"),
        "--model", str(model),
        "--out", str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        console.print(f"[bold red]export-onnx failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[green]ONNX written to:[/green] {out}")


@research_app.command("audit-split")
def research_audit_split(
    data: Annotated[Path, typer.Option(help="Prepared dataset path (CSV/Parquet)")],
    history_steps: Annotated[int, typer.Option(help="History window length")] = 48,
    horizon_steps: Annotated[int, typer.Option(help="Forecast horizon length")] = 6,
    feature_columns_csv: Annotated[str, typer.Option(help="Comma-separated feature columns")] = "glucose_actual_mgdl,patient_iob_units,patient_cob_grams,effective_isf,effective_icr,effective_basal_rate_u_per_hr,glucose_trend_mgdl_min",
    target_column: Annotated[str, typer.Option(help="Target column")] = "glucose_actual_mgdl",
    subject_column: Annotated[str, typer.Option(help="Subject ID column")] = "subject_id",
    segment_column: Annotated[Optional[str], typer.Option(help="Segment column (optional)")] = "segment_id",
    output_json: Annotated[Optional[Path], typer.Option(help="Write audit report JSON")] = None,
):
    """
    Audit subject-level leakage and window overlap for train/val/test splits.
    """
    console = Console()
    if not data.exists():
        console.print(f"[bold red]Dataset not found: {data}[/bold red]")
        raise typer.Exit(code=1)

    from iints.research.dataset import load_dataset
    from iints.research.audit import audit_subject_split_and_leakage

    df = load_dataset(data)
    feature_columns = [col.strip() for col in feature_columns_csv.split(",") if col.strip()]
    report = audit_subject_split_and_leakage(
        df,
        history_steps=history_steps,
        horizon_steps=horizon_steps,
        feature_columns=feature_columns,
        target_column=target_column,
        subject_column=subject_column,
        segment_column=segment_column,
    )

    table = Table(title="Leakage & Split Audit")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Leakage free", "[green]yes[/green]" if report["leakage_free"] else "[red]no[/red]")
    table.add_row("Train subjects", str(report["subject_counts"]["train"]))
    table.add_row("Val subjects", str(report["subject_counts"]["val"]))
    table.add_row("Test subjects", str(report["subject_counts"]["test"]))
    table.add_row("Train windows", str(report["sequence_counts"]["train"]))
    table.add_row("Val windows", str(report["sequence_counts"]["val"]))
    table.add_row("Test windows", str(report["sequence_counts"]["test"]))
    table.add_row("Overlap train/val", str(report["sequence_overlap_counts"]["train_val"]))
    table.add_row("Overlap train/test", str(report["sequence_overlap_counts"]["train_test"]))
    table.add_row("Overlap val/test", str(report["sequence_overlap_counts"]["val_test"]))
    console.print(table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Audit report written:[/green] {output_json}")

    if not report["leakage_free"]:
        raise typer.Exit(code=1)


@research_app.command("evaluate-forecast")
def research_evaluate_forecast(
    input_csv: Annotated[Path, typer.Option(help="CSV with observed/predicted columns")],
    observed_column: Annotated[str, typer.Option(help="Observed glucose column")] = "glucose_actual_mgdl",
    predicted_column: Annotated[str, typer.Option(help="Predicted glucose column")] = "predicted_glucose_ai_30min",
    predicted_std_column: Annotated[Optional[str], typer.Option(help="Optional prediction std column")] = "predictor_uncertainty_std_mgdl",
    gate_profile: Annotated[Optional[str], typer.Option(help="Optional calibration gate profile id")] = None,
    gate_profiles_path: Annotated[Optional[Path], typer.Option(help="Optional calibration gate profiles YAML path")] = None,
    fail_on_gate: Annotated[bool, typer.Option(help="Exit code 1 when calibration gate fails")] = False,
    output_json: Annotated[Optional[Path], typer.Option(help="Write metrics JSON")] = None,
):
    """
    Calibration-first forecast evaluation (global, band-wise, and alarm quality).
    """
    console = Console()
    if not input_csv.exists():
        console.print(f"[bold red]Input CSV not found: {input_csv}[/bold red]")
        raise typer.Exit(code=1)
    df = pd.read_csv(input_csv)
    for col in (observed_column, predicted_column):
        if col not in df.columns:
            console.print(f"[bold red]Missing column: {col}[/bold red]")
            raise typer.Exit(code=1)

    from iints.research.evaluation import forecast_error_report

    observed = df[observed_column].to_numpy(dtype=float)
    predicted = df[predicted_column].to_numpy(dtype=float)
    predicted_std = None
    if predicted_std_column and predicted_std_column in df.columns:
        predicted_std = df[predicted_std_column].to_numpy(dtype=float)

    report = forecast_error_report(observed, predicted, predicted_std)

    table = Table(title="Forecast Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Samples", str(report["n"]))
    table.add_row("MAE", f"{report['mae']:.3f}")
    table.add_row("RMSE", f"{report['rmse']:.3f}")
    table.add_row("Bias", f"{report['bias']:.3f}")
    table.add_row("Within ±10 mg/dL (%)", f"{report['within_10_mgdl_pct']:.2f}")
    table.add_row("Within ±20 mg/dL (%)", f"{report['within_20_mgdl_pct']:.2f}")
    table.add_row("False hypo alarm (%)", f"{report['false_hypo_alarm_rate_pct']:.2f}")
    table.add_row("Missed hypo (%)", f"{report['missed_hypo_rate_pct']:.2f}")
    if "interval_95_coverage_pct" in report:
        table.add_row("95% interval coverage (%)", f"{report['interval_95_coverage_pct']:.2f}")
        table.add_row("Mean predicted std (mg/dL)", f"{report['mean_predicted_std_mgdl']:.3f}")
    console.print(table)

    band_table = Table(title="Band-wise Error")
    band_table.add_column("Band", style="cyan")
    band_table.add_column("Count", justify="right")
    band_table.add_column("MAE", justify="right")
    band_table.add_column("RMSE", justify="right")
    band_table.add_column("Bias", justify="right")
    for band, values in report["band_metrics"].items():
        band_table.add_row(
            band,
            f"{values['count']:.0f}",
            f"{values['mae']:.3f}" if not np.isnan(values["mae"]) else "n/a",
            f"{values['rmse']:.3f}" if not np.isnan(values["rmse"]) else "n/a",
            f"{values['bias']:.3f}" if not np.isnan(values["bias"]) else "n/a",
        )
    console.print(band_table)

    gate_failed = False
    if gate_profile:
        from iints.research.calibration_gate import (
            evaluate_calibration_gate,
            load_calibration_gate_profiles,
        )

        try:
            profiles = load_calibration_gate_profiles(gate_profiles_path)
        except Exception as exc:
            console.print(f"[bold red]Could not load calibration gate profiles: {exc}[/bold red]")
            raise typer.Exit(code=1)
        gate = profiles.get(gate_profile)
        if gate is None:
            available = ", ".join(sorted(profiles.keys()))
            console.print(f"[bold red]Unknown calibration gate '{gate_profile}'. Available: {available}[/bold red]")
            raise typer.Exit(code=1)

        checks = evaluate_calibration_gate(report, gate)
        gate_passed = all(check.get("passed", False) for check in checks.values()) if checks else True
        gate_failed = not gate_passed
        report["calibration_gate"] = {
            "profile": gate_profile,
            "passed": gate_passed,
            "checks": checks,
        }

        gate_table = Table(title=f"Calibration Gate — {gate_profile}")
        gate_table.add_column("Status", style="bold")
        gate_table.add_column("Metric", style="cyan")
        gate_table.add_column("Value", justify="right")
        gate_table.add_column("Threshold", justify="right")
        gate_table.add_column("Reason")
        for metric, check in checks.items():
            status = "[green]PASS[/green]" if check.get("passed", False) else "[red]FAIL[/red]"
            value = "n/a" if check.get("value") is None else f"{float(check['value']):.3f}"
            threshold = "n/a" if check.get("threshold") is None else f"{float(check['threshold']):.3f}"
            gate_table.add_row(status, metric, value, threshold, str(check.get("reason", "")))
        console.print(gate_table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Forecast metrics written:[/green] {output_json}")

    if fail_on_gate and gate_failed:
        raise typer.Exit(code=1)


@research_app.command("parity-check")
def research_parity_check(
    model: Annotated[Path, typer.Option(help="Predictor checkpoint (.pt)")],
    onnx: Annotated[Path, typer.Option(help="Exported ONNX model path")],
    samples: Annotated[int, typer.Option(help="Random sample count for parity check")] = 64,
    tolerance: Annotated[float, typer.Option(help="Maximum allowed absolute error")] = 1e-3,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
    output_json: Annotated[Optional[Path], typer.Option(help="Write parity report JSON")] = None,
):
    """
    Check Torch vs ONNX parity and report edge latency.
    """
    console = Console()
    if not model.exists():
        console.print(f"[bold red]Model not found: {model}[/bold red]")
        raise typer.Exit(code=1)
    if not onnx.exists():
        console.print(f"[bold red]ONNX model not found: {onnx}[/bold red]")
        raise typer.Exit(code=1)

    try:
        import onnxruntime as ort
    except Exception as exc:
        console.print("[bold red]onnxruntime is required for parity-check.[/bold red]")
        console.print(f"[red]Details:[/red] {exc}")
        raise typer.Exit(code=1)

    from iints.research.predictor import load_predictor_service

    predictor = load_predictor_service(model)
    feature_count = len(predictor.feature_columns)
    history_steps = int(predictor.history_steps)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, history_steps, feature_count)).astype(np.float32)

    t0 = time.perf_counter()
    torch_pred = predictor.predict(X)
    torch_latency_ms = (time.perf_counter() - t0) * 1000.0

    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    t1 = time.perf_counter()
    onnx_pred = session.run(None, {input_name: X})[0]
    onnx_latency_ms = (time.perf_counter() - t1) * 1000.0

    diff = np.abs(torch_pred - onnx_pred)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    passed = max_abs <= tolerance

    report = {
        "samples": samples,
        "tolerance": tolerance,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "torch_latency_ms": torch_latency_ms,
        "onnx_latency_ms": onnx_latency_ms,
        "passed": passed,
    }

    table = Table(title="Torch/ONNX Parity")
    table.add_column("Field", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Samples", str(samples))
    table.add_row("Tolerance", f"{tolerance:.6f}")
    table.add_row("Max abs diff", f"{max_abs:.6f}")
    table.add_row("Mean abs diff", f"{mean_abs:.6f}")
    table.add_row("Torch latency (ms)", f"{torch_latency_ms:.2f}")
    table.add_row("ONNX latency (ms)", f"{onnx_latency_ms:.2f}")
    table.add_row("Status", "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
    console.print(table)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Parity report written:[/green] {output_json}")

    if not passed:
        raise typer.Exit(code=1)


@research_app.command("registry-list")
def research_registry_list(
    registry: Annotated[Path, typer.Option(help="Path to model registry JSON")] = Path("models/registry.json"),
    stage: Annotated[Optional[str], typer.Option(help="Optional stage filter (candidate/validated/production/archived)")] = None,
    limit: Annotated[int, typer.Option(help="Max rows to print")] = 30,
):
    """List model runs from the research registry."""
    console = Console()
    from iints.research.model_registry import list_registry

    rows = list_registry(registry)
    if stage:
        rows = [row for row in rows if str(row.get("stage", "candidate")) == stage]

    if not rows:
        console.print(f"[yellow]No registry entries found at {registry}.[/yellow]")
        return

    rows = rows[-limit:] if limit > 0 else rows
    rows = list(reversed(rows))

    table = Table(title=f"Model Registry — {registry}")
    table.add_column("Run ID", style="cyan")
    table.add_column("Stage", style="bold")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("Timestamp")
    table.add_column("Model Path")
    for row in rows:
        table.add_row(
            str(row.get("run_id", "")),
            str(row.get("stage", "candidate")),
            "n/a" if row.get("test_rmse") is None else f"{float(row['test_rmse']):.3f}",
            "n/a" if row.get("test_mae") is None else f"{float(row['test_mae']):.3f}",
            str(row.get("timestamp_utc", "")),
            str(row.get("model_path", "")),
        )
    console.print(table)


@research_app.command("registry-promote")
def research_registry_promote(
    registry: Annotated[Path, typer.Option(help="Path to model registry JSON")] = Path("models/registry.json"),
    run_id: Annotated[str, typer.Option(help="Run ID to promote")]= "",
    stage: Annotated[str, typer.Option(help="Target stage (validated/production/archived/candidate)")] = "validated",
    force: Annotated[bool, typer.Option(help="Allow production promotion without validated stage")] = False,
    output_json: Annotated[Optional[Path], typer.Option(help="Write promotion result JSON")] = None,
):
    """Promote a model run through candidate -> validated -> production stages."""
    console = Console()
    from iints.research.model_registry import ModelStage, promote_registry_run

    allowed = {"candidate", "validated", "production", "archived"}
    if stage not in allowed:
        console.print(f"[bold red]Invalid stage '{stage}'. Allowed: {', '.join(sorted(allowed))}[/bold red]")
        raise typer.Exit(code=1)
    if not run_id.strip():
        console.print("[bold red]--run-id is required[/bold red]")
        raise typer.Exit(code=1)

    stage_literal = cast(ModelStage, stage)
    result = promote_registry_run(registry, run_id=run_id.strip(), stage=stage_literal, force=force)
    payload = result.to_dict()
    console.print_json(json.dumps(payload, indent=2))

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]Promotion result written:[/green] {output_json}")

    if not result.updated:
        raise typer.Exit(code=1)


@app.command("import-data")
def import_data(
    input_csv: Annotated[Path, typer.Option(help="Path to CGM CSV file")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for scenario + standard CSV")] = Path("./results/imported"),
    data_format: Annotated[str, typer.Option(help="Data format preset: generic, dexcom, libre")] = "generic",
    scenario_name: Annotated[str, typer.Option(help="Scenario name")] = "Imported CGM Scenario",
    scenario_version: Annotated[str, typer.Option(help="Scenario version")] = "1.0",
    time_unit: Annotated[str, typer.Option(help="Timestamp unit: minutes or seconds")] = "minutes",
    carb_threshold: Annotated[float, typer.Option(help="Minimum carbs (g) to create a meal event")] = 0.1,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Optional output scenario path")] = None,
    data_path: Annotated[Optional[Path], typer.Option(help="Optional output standard CSV path")] = None,
    mapping: Annotated[List[str], typer.Option("--map", help="Column mapping key=value (e.g., timestamp=Time, glucose=SGV)")] = [],
):
    """Import real-world CGM CSV into the IINTS standard schema + scenario JSON."""
    console = Console()
    if not input_csv.is_file():
        console.print(f"[bold red]Error: Input CSV '{input_csv}' not found.[/bold red]")
        raise typer.Exit(code=1)

    column_map = _parse_column_mapping(mapping, console)

    try:
        result = scenario_from_csv(
            input_csv,
            scenario_name=scenario_name,
            scenario_version=scenario_version,
            data_format=data_format,
            column_map=column_map or None,
            time_unit=time_unit,
            carb_threshold=carb_threshold,
        )
    except Exception as e:
        console.print(f"[bold red]Import failed: {e}[/bold red]")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = scenario_path or output_dir / "scenario.json"
    data_path = data_path or output_dir / "cgm_standard.csv"

    scenario_path.write_text(json.dumps(result.scenario, indent=2))
    export_standard_csv(result.dataframe, data_path)

    console.print(f"[green]Scenario saved:[/green] {scenario_path}")
    console.print(f"[green]Standard CSV saved:[/green] {data_path}")
    console.print(f"Rows: {len(result.dataframe)} | Meal events: {len(result.scenario.get('stress_events', []))}")


@app.command("import-wizard")
def import_wizard():
    """Interactive wizard to import real-world CGM CSVs."""
    console = Console()
    input_csv = Path(typer.prompt("Path to CGM CSV"))
    if not input_csv.is_file():
        console.print(f"[bold red]Error: Input CSV '{input_csv}' not found.[/bold red]")
        raise typer.Exit(code=1)

    data_format = typer.prompt("Data format (generic/dexcom/libre)", default="generic")

    header = pd.read_csv(input_csv, nrows=1)
    columns = list(header.columns)
    guesses = guess_column_mapping(columns, data_format=data_format)

    console.print(f"[bold]Detected columns:[/bold] {', '.join(columns)}")
    try:
        preview = pd.read_csv(input_csv, nrows=5)
        console.print(Panel(preview.to_string(index=False), title="Preview (first 5 rows)"))
    except Exception as exc:
        console.print(f"[yellow]Preview unavailable:[/yellow] {exc}")

    ts_col = typer.prompt("Timestamp column", default=guesses.get("timestamp") or columns[0])
    glucose_col = typer.prompt("Glucose column", default=guesses.get("glucose") or "")
    carbs_col = typer.prompt("Carbs column (optional)", default=guesses.get("carbs") or "")
    insulin_col = typer.prompt("Insulin column (optional)", default=guesses.get("insulin") or "")

    mapping = {
        "timestamp": ts_col.strip(),
        "glucose": glucose_col.strip(),
        "carbs": carbs_col.strip(),
        "insulin": insulin_col.strip(),
    }
    mapping = {k: v for k, v in mapping.items() if v}

    time_unit = typer.prompt("Timestamp unit (minutes/seconds)", default="minutes")
    scenario_name = typer.prompt("Scenario name", default="Imported CGM Scenario")
    output_dir = Path(typer.prompt("Output directory", default="results/imported"))

    try:
        result = scenario_from_csv(
            input_csv,
            scenario_name=scenario_name,
            data_format=data_format,
            column_map=mapping or None,
            time_unit=time_unit,
        )
    except Exception as exc:
        console.print(f"[bold red]Import failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / "scenario.json"
    data_path = output_dir / "cgm_standard.csv"

    scenario_path.write_text(json.dumps(result.scenario, indent=2))
    export_standard_csv(result.dataframe, data_path)

    console.print(f"[green]Scenario saved:[/green] {scenario_path}")
    console.print(f"[green]Standard CSV saved:[/green] {data_path}")
    console.print(f"Rows: {len(result.dataframe)} | Meal events: {len(result.scenario.get('stress_events', []))}")


@app.command("import-demo")
def import_demo(
    output_dir: Annotated[Path, typer.Option(help="Output directory for scenario + CSV")] = Path("./results/demo_import"),
    scenario_name: Annotated[str, typer.Option(help="Scenario name")] = "Demo CGM Scenario",
    export_raw: Annotated[bool, typer.Option(help="Export the raw demo CSV into output dir")] = True,
):
    """Generate a ready-to-run scenario from the bundled demo CGM data pack."""
    console = Console()
    demo_df = load_demo_dataframe()
    standard_df = import_cgm_dataframe(demo_df, data_format="generic", source="demo")
    scenario = scenario_from_dataframe(standard_df, scenario_name=scenario_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / "scenario.json"
    data_path = output_dir / "cgm_standard.csv"

    scenario_path.write_text(json.dumps(scenario, indent=2))
    export_standard_csv(standard_df, data_path)
    if export_raw:
        export_demo_csv(output_dir / "demo_cgm.csv")

    console.print(f"[green]Scenario saved:[/green] {scenario_path}")
    console.print(f"[green]Standard CSV saved:[/green] {data_path}")
    if export_raw:
        console.print(f"[green]Raw demo CSV saved:[/green] {output_dir / 'demo_cgm.csv'}")


@app.command("import-nightscout")
def import_nightscout_cmd(
    url: Annotated[str, typer.Option(help="Nightscout base URL")],
    output_dir: Annotated[Path, typer.Option(help="Output directory for scenario + CSV")] = Path("./results/nightscout_import"),
    api_secret: Annotated[Optional[str], typer.Option(help="API secret (if required)")] = None,
    token: Annotated[Optional[str], typer.Option(help="API token (if required)")] = None,
    start: Annotated[Optional[str], typer.Option(help="Start time (ISO string)")] = None,
    end: Annotated[Optional[str], typer.Option(help="End time (ISO string)")] = None,
    limit: Annotated[Optional[int], typer.Option(help="Limit number of entries")] = None,
    scenario_name: Annotated[str, typer.Option(help="Scenario name")] = "Nightscout Import",
):
    """Import CGM entries from Nightscout into a scenario + standard CSV."""
    console = Console()
    config = NightscoutConfig(
        url=url,
        api_secret=api_secret,
        token=token,
        start=start,
        end=end,
        limit=limit,
    )
    try:
        result = import_nightscout(config, scenario_name=scenario_name)
    except ImportError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Nightscout import failed: {exc}[/bold red]")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = output_dir / "scenario.json"
    data_path = output_dir / "cgm_standard.csv"
    scenario_path.write_text(json.dumps(result.scenario, indent=2))
    export_standard_csv(result.dataframe, data_path)
    console.print(f"[green]Scenario saved:[/green] {scenario_path}")
    console.print(f"[green]Standard CSV saved:[/green] {data_path}")


@app.command("import-tidepool")
def import_tidepool_cmd(
    base_url: Annotated[str, typer.Option(help="Tidepool API base URL")] = "https://api.tidepool.org",
    token: Annotated[Optional[str], typer.Option(help="Bearer token")] = None,
):
    """Skeleton Tidepool client for future cloud imports."""
    console = Console()
    client = TidepoolClient(base_url=base_url, token=token)
    try:
        _ = client._headers()
    except Exception as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1)
    console.print("[yellow]Tidepool client skeleton is initialized. Auth flow and endpoints are TODO.[/yellow]")

@app.command("check-deps")
def check_deps():
    """Check optional dependencies and report readiness."""
    console = Console()

    def has(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    checks = [
        ("Simulator", True, "Core simulation engine"),
        ("Metrics", has("numpy") and has("pandas"), "Clinical metrics"),
        ("Reporting", has("matplotlib") and has("fpdf"), "PDF reports"),
        ("Validation", has("pydantic"), "Schema validation"),
        ("Deep Learning", has("torch"), "AI models (install with `pip install iints[torch]`)"),
    ]

    table = Table(title="IINTS Dependency Check", show_lines=False)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes")

    for name, ok, notes in checks:
        status = "OK" if ok else "Missing"
        table.add_row(name, status, notes)

    console.print(table)


@algorithms_app.command("list")
def algorithms_list():
    """List available algorithm plugins and built-ins."""
    console = Console()
    entries = list_algorithm_plugins()
    table = Table(title="IINTS Algorithms", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Source", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Class", style="white")
    for entry in entries:
        table.add_row(
            entry.name,
            entry.source,
            entry.status,
            entry.class_path,
        )
    console.print(table)


@algorithms_app.command("info")
def algorithms_info(
    name: Annotated[str, typer.Argument(help="Algorithm display name")],
):
    """Show metadata for a specific algorithm."""
    console = Console()
    entries = list_algorithm_plugins()
    matches = [entry for entry in entries if entry.name.lower() == name.lower()]
    if not matches:
        console.print(f"[bold red]Algorithm '{name}' not found.[/bold red]")
        raise typer.Exit(code=1)
    entry = matches[0]
    console.print(f"[bold]Name:[/bold] {entry.name}")
    console.print(f"[bold]Source:[/bold] {entry.source}")
    console.print(f"[bold]Status:[/bold] {entry.status}")
    console.print(f"[bold]Class:[/bold] {entry.class_path}")
    if entry.error:
        console.print(f"[bold red]Error:[/bold red] {entry.error}")
    if entry.metadata:
        console.print_json(json.dumps(entry.metadata.to_dict(), indent=2))
@docs_app.command("algo")
def docs_algo(
    algo_path: Annotated[Path, typer.Option(help="Path to the algorithm Python file to document")],
):
    """
    Generates a technical summary (auto-documentation) for a specified InsulinAlgorithm.
    """
    console = Console()
    console.print(f"[bold blue]Generating Auto-Documentation for Algorithm: {algo_path.name}[/bold blue]")

    if not algo_path.is_file():
        console.print(f"[bold red]Error: Algorithm file '{algo_path}' not found.[/bold red]")
        raise typer.Exit(code=1)
    
    # Load Algorithm dynamically
    algorithm_instance = None
    module_name = algo_path.stem
    spec = importlib.util.spec_from_file_location(module_name, algo_path)
    if spec is None:
        console.print(f"[bold red]Error: Could not load module spec for {algo_path}[/bold red]")
        raise typer.Exit(code=1)
    
    module = importlib.util.module_from_spec(spec)
    module.iints = iints # type: ignore # Inject iints package
    sys.modules[module_name] = module
    try:
        if spec.loader: # Ensure loader is not None
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Could not load module loader for {algo_path}")
    except Exception as e:
        console.print(f"[bold red]Error loading algorithm module {algo_path}: {e}[/bold red]")
        raise typer.Exit(code=1)

    algorithm_class = None
    for name_in_module, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            algorithm_class = obj
            algorithm_instance = obj() # Instantiate to get metadata
            break
    
    if algorithm_instance is None:
        console.print(f"[bold red]Error: No subclass of InsulinAlgorithm found in {algo_path}[/bold red]")
        raise typer.Exit(code=1)
    
    # Extract Metadata
    metadata = algorithm_instance.get_algorithm_metadata()

    # Ensure algorithm_class is not None (it shouldn't be if algorithm_instance is not None)
    assert algorithm_class is not None

    # Extract class docstring
    class_doc = algorithm_class.__doc__ if algorithm_class.__doc__ else "No class docstring available."

    # Extract predict_insulin docstring
    predict_insulin_doc = algorithm_class.predict_insulin.__doc__ if algorithm_class.predict_insulin.__doc__ else "No docstring for predict_insulin method."

    console.print(Panel(
        f"[bold blue]Algorithm Overview[/bold blue]\n\n"
        f"[green]Name:[/green] {metadata.name}\n"
        f"[green]Version:[/green] {metadata.version}\n"
        f"[green]Author:[/green] {metadata.author}\n"
        f"[green]Type:[/green] {metadata.algorithm_type}\n"
        f"[green]Description:[/green] {metadata.description}\n"
        f"[green]Requires Training:[/green] {metadata.requires_training}\n"
        f"[green]Supported Scenarios:[/green] {', '.join(metadata.supported_scenarios)}\n\n"
        f"[bold blue]Class Documentation[/bold blue]\n"
        f"{class_doc}\n\n"
        f"[bold blue]predict_insulin Method Documentation[/bold blue]\n"
        f"{predict_insulin_doc}",
        title=f"Auto-Doc: {metadata.name}",
        border_style="blue"
    ))

@app.command()
def benchmark(
    algo_to_benchmark: Annotated[Path, typer.Option(help="Path to the AI algorithm Python file to benchmark")],
    # standard_pump_config: Annotated[str, typer.Option(help="Name of the standard pump patient config (e.g., 'default')")] = "default", # This will be loaded implicitly for the standard pump
    patient_configs_dir: Annotated[Path, typer.Option(help="Directory containing patient configuration YAML files")] = Path("src/iints/data/virtual_patients"),
    scenarios_dir: Annotated[Path, typer.Option(help="Directory containing scenario JSON files")] = Path("scenarios"),
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes for each run")] = 720, # 12 hours
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save all benchmark results")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Base seed for deterministic runs")] = None,
):
    """
    Runs a series of simulations to benchmark an AI algorithm against a standard pump
    across multiple patient configurations and scenarios.
    """
    console = Console()
    resolved_seed = resolve_seed(seed)
    run_id = generate_run_id(resolved_seed)
    output_dir = resolve_output_dir(output_dir, run_id)
    console.print(f"[bold blue]Starting IINTS-AF Benchmark Suite[/bold blue]")
    console.print(f"AI Algorithm: [green]{algo_to_benchmark.name}[/green]")
    # console.print(f"Standard Pump Config: [yellow]{standard_pump_config}[/yellow]") # Removed, as standard pump uses patient params
    console.print(f"Patient Configs from: [cyan]{patient_configs_dir}[/cyan]")
    console.print(f"Scenarios from: [magenta]{scenarios_dir}[/magenta]")
    console.print(f"Duration: {duration} min, Time Step: {time_step} min")
    console.print(f"Run ID: {run_id}")
    console.print(f"Output directory: {output_dir}")

    if not algo_to_benchmark.is_file():
        console.print(f"[bold red]Error: AI Algorithm file '{algo_to_benchmark}' not found.[/bold red]")
        raise typer.Exit(code=1)
    if not patient_configs_dir.is_dir():
        console.print(f"[bold red]Error: Patient configurations directory '{patient_configs_dir}' not found.[/bold red]")
        raise typer.Exit(code=1)
    if not scenarios_dir.is_dir():
        console.print(f"[bold red]Error: Scenarios directory '{scenarios_dir}' not found.[/bold red]")
        raise typer.Exit(code=1)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "run_type": "benchmark",
        "run_id": run_id,
        "algorithm_path": str(algo_to_benchmark),
        "patient_configs_dir": str(patient_configs_dir),
        "scenarios_dir": str(scenarios_dir),
        "duration_minutes": duration,
        "time_step_minutes": time_step,
        "seed": resolved_seed,
    }
    config_path = output_dir / "config.json"
    write_json(config_path, config_payload)
    run_metadata = build_run_metadata(run_id, resolved_seed, config_payload, output_dir)
    run_metadata_path = output_dir / "run_metadata.json"
    write_json(run_metadata_path, run_metadata)

    # Load AI Algorithm
    ai_algo_instance = None
    module_name_ai = algo_to_benchmark.stem
    spec_ai = importlib.util.spec_from_file_location(module_name_ai, algo_to_benchmark)
    if spec_ai is None:
        console.print(f"[bold red]Error: Could not load module spec for AI algorithm {algo_to_benchmark}[/bold red]")
        raise typer.Exit(code=1)
    module_ai = importlib.util.module_from_spec(spec_ai)
    module_ai.iints = iints # type: ignore # Inject iints package
    sys.modules[module_name_ai] = module_ai
    try:
        if spec_ai.loader: # Ensure loader is not None
            spec_ai.loader.exec_module(module_ai)
        else:
            raise ImportError(f"Could not load module loader for AI algorithm {algo_to_benchmark}")
    except Exception as e:
        console.print(f"[bold red]Error loading AI algorithm module {algo_to_benchmark}: {e}[/bold red]")
        raise typer.Exit(code=1)

    for name_in_module, obj in module_ai.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, iints.InsulinAlgorithm) and obj is not iints.InsulinAlgorithm:
            ai_algo_instance = obj()
            break
    if ai_algo_instance is None:
        console.print(f"[bold red]Error: No subclass of InsulinAlgorithm found in AI algorithm {algo_to_benchmark}[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"Loaded AI Algorithm: [green]{ai_algo_instance.get_algorithm_metadata().name}[/green]")

    # Get compute device
    device_manager = iints.DeviceManager()
    device = device_manager.get_device()
    console.print(f"Using compute device: [blue]{device}[/blue]")

    # Collect patient configurations and scenarios
    patient_config_files = list(patient_configs_dir.glob("*.yaml"))
    scenario_files = list(scenarios_dir.glob("*.json"))

    if not patient_config_files:
        console.print(f"[bold red]Error: No patient configuration files found in '{patient_configs_dir}'[/bold red]")
        raise typer.Exit(code=1)
    if not scenario_files:
        console.print(f"[bold red]Error: No scenario files found in '{scenarios_dir}'[/bold red]")
        raise typer.Exit(code=1)

    benchmark_results = []
    run_index = 0

    # Iterate through patients and scenarios
    for patient_config_file in patient_config_files:
        patient_config_name = patient_config_file.stem
        try:
            with open(patient_config_file, 'r') as f:
                patient_params = yaml.safe_load(f)
            console.print(f"\n[bold underline]Benchmarking Patient: {patient_config_name}[/bold underline]")
        except yaml.YAMLError as e:
            console.print(f"[bold red]Error parsing patient configuration '{patient_config_file.name}': {e}[/bold red]")
            continue # Skip this patient

        for scenario_file in scenario_files:
            scenario_name = scenario_file.stem
            console.print(f"  [bold]Scenario: {scenario_name}[/bold]")
            
            # Load Scenario Data (validated)
            try:
                scenario_model = load_scenario(scenario_file)
                payloads = scenario_to_payloads(scenario_model)
                stress_events = build_stress_events(payloads)
            except ValidationError as e:
                console.print(f"[bold red]  Scenario validation failed: {scenario_file.name}[/bold red]")
                for line in format_validation_error(e):
                    console.print(f"  - {line}")
                continue
            except Exception as e:
                console.print(f"[bold red]  Error loading scenario '{scenario_file.name}': {e}[/bold red]")
                continue # Skip this scenario

            job_seed = resolved_seed + run_index
            # --- Run AI Algorithm Simulation ---
            console.print(f"    Running [green]{ai_algo_instance.get_algorithm_metadata().name}[/green]...")
            patient_model_ai = iints.PatientModel(**patient_params) # New instance for each run
            simulator_ai = iints.Simulator(
                patient_model=patient_model_ai,
                algorithm=ai_algo_instance,
                time_step=time_step,
                seed=job_seed,
            )
            for event in stress_events:
                simulator_ai.add_stress_event(event)
            
            try:
                results_df_ai, safety_report_ai = simulator_ai.run_batch(duration)
                metrics_ai = iints.generate_benchmark_metrics(results_df_ai)
            except Exception as e:
                console.print(f"[bold red]      AI Simulation failed: {e}[/bold red]")
                # Provide dummy metrics for failed simulations to allow table generation
                metrics_ai = {"TIR (%)": float('nan'), "Hypoglycemia (<70 mg/dL) (%)": float('nan'),
                              "Hyperglycemia (>180 mg/dL) (%)": float('nan'), "Avg Glucose (mg/dL)": float('nan')}
                safety_report_ai = {'num_violations': float('nan')} # Dummy report

            # --- Run Standard Pump Algorithm Simulation ---
            console.print(f"    Running [yellow]Standard Pump[/yellow]...") # Use name directly
            # The standard pump also needs patient-specific parameters for ISF, ICR, basal rate, etc.
            # We'll pass the patient_params directly to the StandardPumpAlgorithm constructor.
            standard_pump_algo_instance = iints.StandardPumpAlgorithm(settings=patient_params) 
            patient_model_std = iints.PatientModel(**patient_params) # New instance for each run
            simulator_std = iints.Simulator(
                patient_model=patient_model_std,
                algorithm=standard_pump_algo_instance,
                time_step=time_step,
                seed=job_seed,
            )
            for event in stress_events:
                simulator_std.add_stress_event(event)
            
            try:
                results_df_std, safety_report_std = simulator_std.run_batch(duration)
                metrics_std = iints.generate_benchmark_metrics(results_df_std)
            except Exception as e:
                console.print(f"[bold red]      Standard Pump Simulation failed: {e}[/bold red]")
                # Provide dummy metrics for failed simulations
                metrics_std = {"TIR (%)": float('nan'), "Hypoglycemia (<70 mg/dL) (%)": float('nan'),
                               "Hyperglycemia (>180 mg/dL) (%)": float('nan'), "Avg Glucose (mg/dL)": float('nan')}
                safety_report_std = {'num_violations': float('nan')} # Dummy report


            # Store results
            benchmark_results.append({
                "run_id": run_id,
                "seed": job_seed,
                "Patient": patient_config_name,
                "Scenario": scenario_name,
                "AI Algo": ai_algo_instance.get_algorithm_metadata().name,
                **{f"AI {k}": v for k, v in metrics_ai.items()},
                **{f"AI Safety Violations": safety_report_ai.get('total_violations', float('nan'))},
                "Standard Algo": standard_pump_algo_instance.get_algorithm_metadata().name,
                **{f"Std {k}": v for k, v in metrics_std.items()},
                **{f"Std Safety Violations": safety_report_std.get('total_violations', float('nan'))},
            })
            run_index += 1
    
    console.print("\n[bold green]Benchmark Suite Completed![/bold green]")

    # Print Comparison Table
    if benchmark_results:
        results_df = pd.DataFrame(benchmark_results)
        
        table = Table(title="IINTS-AF Benchmark Results", show_header=True, header_style="bold magenta")
        
        # Add columns dynamically
        table.add_column("Patient", style="cyan", no_wrap=True)
        table.add_column("Scenario", style="cyan", no_wrap=True)
        
        # Assuming AI Algo and Standard Algo names are consistent across results
        ai_algo_name = benchmark_results[0]["AI Algo"]
        std_algo_name = benchmark_results[0]["Standard Algo"]

        # Get a sample of metric keys (excluding 'AI Algo', 'Standard Algo')
        # Fix: Filter out non-metric keys like 'AI Algo'
        sample_metrics_keys_raw = [k.replace('AI ', '') for k in benchmark_results[0].keys() if k.startswith('AI ') and 'Algo' not in k and 'Violations' not in k]

        for metric_name_raw in sample_metrics_keys_raw:
            table.add_column(f"{ai_algo_name} {metric_name_raw}", style="green")
            table.add_column(f"{std_algo_name} {metric_name_raw}", style="yellow")
        
        table.add_column(f"{ai_algo_name} Safety Violations", style="red")
        table.add_column(f"{std_algo_name} Safety Violations", style="red")

        for _, row in results_df.iterrows():
            row_data = [str(row["Patient"]), str(row["Scenario"])]
            for metric_name_raw in sample_metrics_keys_raw:
                ai_val = row[f'AI {metric_name_raw}']
                std_val = row[f'Std {metric_name_raw}']
                
                ai_formatted = f"{ai_val:.2f}%" if "%" in metric_name_raw and not pd.isna(ai_val) else (f"{ai_val:.2f}" if not pd.isna(ai_val) else "N/A")
                std_formatted = f"{std_val:.2f}%" if "%" in metric_name_raw and not pd.isna(std_val) else (f"{std_val:.2f}" if not pd.isna(std_val) else "N/A")
                
                row_data.append(ai_formatted)
                row_data.append(std_formatted)
            
            ai_safety_violations = row['AI Safety Violations']
            std_safety_violations = row['Std Safety Violations']
            row_data.append(f"{ai_safety_violations:.0f}" if not pd.isna(ai_safety_violations) else "N/A")
            row_data.append(f"{std_safety_violations:.0f}" if not pd.isna(std_safety_violations) else "N/A")
            table.add_row(*row_data)

        console.print(table)

        results_csv = output_dir / "benchmark_summary.csv"
        results_df.to_csv(results_csv, index=False)
        console.print(f"[green]Benchmark summary saved:[/green] {results_csv}")

        manifest_files = {
            "config": config_path,
            "run_metadata": run_metadata_path,
            "benchmark_summary": results_csv,
        }
        run_manifest = build_run_manifest(output_dir, manifest_files)
        run_manifest_path = output_dir / "run_manifest.json"
        write_json(run_manifest_path, run_manifest)
        console.print(f"Run manifest: {run_manifest_path}")
        signature_path = maybe_sign_manifest(run_manifest_path)
        if signature_path:
            console.print(f"Run manifest signature: {signature_path}")
    else:
        console.print("[yellow]No benchmark results were generated.[/yellow]")
