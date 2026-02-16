import typer  # type: ignore
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional
from typing_extensions import Annotated
from pydantic import ValidationError
import os
import importlib.util
import sys
import json
import yaml # Added for Virtual Patient Registry
import pandas as pd # Added for DataFrame in benchmark results

from rich.console import Console  # type: ignore # For pretty printing
from rich.table import Table  # type: ignore # For comparison table
from rich.panel import Panel  # type: ignore # For nicer auto-doc output

import iints # Import the top-level SDK package
from iints.analysis.baseline import run_baseline_comparison, write_baseline_comparison
from iints.core.patient.profile import PatientProfile
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
from iints.validation import (
    build_stress_events,
    format_validation_error,
    load_scenario,
    load_patient_config,
    load_patient_config_by_name,
    scenario_to_payloads,
    scenario_warnings,
    validate_patient_config_dict,
    validate_scenario_dict,
)


app = typer.Typer(help="IINTS-AF SDK CLI - Intelligent Insulin Titration System for Artificial Pancreas research.")
docs_app = typer.Typer(help="Generate documentation and technical summaries for IINTS-AF components.")
presets_app = typer.Typer(help="Clinic-safe presets and quickstart runs.")
profiles_app = typer.Typer(help="Patient profiles and physiological presets.")
data_app = typer.Typer(help="Official datasets and data packs.")
app.add_typer(docs_app, name="docs")
app.add_typer(presets_app, name="presets")
app.add_typer(profiles_app, name="profiles")
app.add_typer(data_app, name="data")

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

def _load_presets() -> List[Dict[str, Any]]:
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        content = files("iints.presets").joinpath("presets.json").read_text()
    else:
        from importlib import resources
        content = resources.read_text("iints.presets", "presets.json")
    return json.loads(content)

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

 

@app.command()
def init(
    project_name: Annotated[str, typer.Option(help="Name of the project directory")] = "my_iints_project",
):
    """
    Initialize a new IINTS-AF research project with a standard folder structure.
    """
    console = Console()
    project_path = Path(project_name)
    
    if project_path.exists():
         console.print(f"[bold red]Error: Directory '{project_name}' already exists.[/bold red]")
         raise typer.Exit(code=1)
    
    console.print(f"[bold blue]Initializing IINTS-AF Project: {project_name}[/bold blue]")
    
    # Create Directories
    (project_path / "algorithms").mkdir(parents=True)
    (project_path / "scenarios").mkdir(parents=True)
    (project_path / "data").mkdir(parents=True)
    (project_path / "results").mkdir(parents=True)
    
    # Copy Default Algorithm
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            algo_content = files("iints.templates").joinpath("default_algorithm.py").read_text()
            scenario_content = files("iints.templates.scenarios").joinpath("example_scenario.json").read_text()
        else:
            from importlib import resources
            algo_content = resources.read_text("iints.templates", "default_algorithm.py")
            scenario_content = resources.read_text("iints.templates.scenarios", "example_scenario.json")
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
        
    # Create README
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
    output_dir: Annotated[Optional[Path], typer.Option(help="Directory to save outputs")] = None,
    compare_baselines: Annotated[bool, typer.Option(help="Run PID and standard pump baselines in the background")] = True,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
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

    try:
        patient_config_name = preset.get("patient_config", "default_patient")
        validated_patient_params = load_patient_config_by_name(patient_config_name).model_dump()
        patient_model = iints.PatientModel(**validated_patient_params)
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
    simulator_kwargs: Dict[str, Any] = {
        "patient_model": patient_model,
        "algorithm": algorithm_instance,
        "time_step": time_step,
    }
    if seed is not None:
        simulator_kwargs["seed"] = seed
    if "critical_glucose_threshold" in preset:
        simulator_kwargs["critical_glucose_threshold"] = float(preset["critical_glucose_threshold"])
    if "critical_glucose_duration_minutes" in preset:
        simulator_kwargs["critical_glucose_duration_minutes"] = int(preset["critical_glucose_duration_minutes"])
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

    if output_dir is None:
        output_dir = algo.parent / "results" / "presets"
    output_dir = output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df, safety_report = simulator.run_batch(duration)

    algo_name = algorithm_instance.get_algorithm_metadata().name.replace(" ", "_").lower()
    results_file = output_dir / f"preset_{name}_{algo_name}.csv"
    results_df.to_csv(results_file, index=False)
    console.print(f"Results saved to: [link=file://{results_file}]{results_file}[/link]")

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
            seed=seed,
        )
        safety_report["baseline_comparison"] = comparison
        baseline_paths = write_baseline_comparison(comparison, output_dir / "baseline")
        console.print(f"Baseline comparison saved to: {baseline_paths}")

    report_path = output_dir / f"preset_{name}_{algo_name}.pdf"
    iints.generate_report(results_df, str(report_path), safety_report)
    console.print(f"PDF report saved to: [link=file://{report_path}]{report_path}[/link]")


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
@app.command()
def run(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default_patient' or 'patient_559_config')")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Path to a patient config YAML (overrides --patient-config-name)")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to the scenario JSON file (e.g., scenarios/example_scenario.json)")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720, # 12 hours
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Directory to save simulation results")] = Path("./results/data"),
    compare_baselines: Annotated[bool, typer.Option(help="Run PID and standard pump baselines in the background")] = True,
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
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

        patient_model = iints.PatientModel(**validated_patient_params)
        console.print(f"Using patient model: {patient_model.__class__.__name__} with config [cyan]{patient_label}[/cyan]")
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
    if scenario_path:
        if not scenario_path.is_file():
            console.print(f"[bold red]Error: Scenario file '{scenario_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        
        try:
            scenario_model = load_scenario(scenario_path)
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
    simulator = iints.Simulator(
        patient_model=patient_model,
        algorithm=algorithm_instance,
        time_step=time_step,
        seed=seed,
    )
    
    for event in stress_events:
        simulator.add_stress_event(event)

    simulation_results_df, safety_report = simulator.run_batch(duration)
    
    # 6. Output Results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct a more robust filename
    algo_name_for_filename = algorithm_instance.get_algorithm_metadata().name.replace(' ', '_').lower()
    scenario_name_for_filename = Path(scenario_path).stem if scenario_path else 'no_scenario'
    
    results_file = output_dir / f"sim_results_{algo_name_for_filename}_patient_{patient_label}_scenario_{scenario_name_for_filename}_duration_{duration}m.csv"
    
    simulation_results_df.to_csv(results_file, index=False)
    
    console.print(f"\nSimulation completed. Results saved to: [link=file://{results_file}]{results_file}[/link]") # Formatted link
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
            seed=seed,
        )
        safety_report["baseline_comparison"] = comparison
        baseline_paths = write_baseline_comparison(comparison, output_dir / "baseline")
        console.print(f"Baseline comparison saved to: {baseline_paths}")

    # Generate full report (using the new iints.generate_report function)
    report_output_path = output_dir / f"sim_report_{algo_name_for_filename}_patient_{patient_config_name}_scenario_{scenario_name_for_filename}_duration_{duration}m.pdf"
    iints.generate_report(simulation_results_df, str(report_output_path), safety_report)


@app.command("run-full")
def run_full(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default_patient')")] = "default_patient",
    patient_config_path: Annotated[Optional[Path], typer.Option(help="Path to a patient config YAML (overrides --patient-config-name)")] = None,
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to the scenario JSON file")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720,
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Directory to save results + audit + report")] = Path("./results/run_full"),
    seed: Annotated[Optional[int], typer.Option(help="Random seed for deterministic runs")] = None,
):
    """One-line runner: results CSV + audit + PDF + baseline comparison."""
    console = Console()
    algorithm_instance = _load_algorithm_instance(algo, console)

    patient_config: Union[str, Path]
    if patient_config_path:
        if not patient_config_path.is_file():
            console.print(f"[bold red]Error: Patient config file '{patient_config_path}' not found.[/bold red]")
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
    dataset_id: Annotated[str, typer.Option(help="Dataset id (see `iints data list`)")],
):
    """Show metadata and access info for a dataset."""
    console = Console()
    try:
        dataset = get_dataset(dataset_id)
    except DatasetRegistryError as e:
        console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(dataset, indent=2))


@data_app.command("fetch")
def data_fetch(
    dataset_id: Annotated[str, typer.Option(help="Dataset id (see `iints data list`)")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory (default: data_packs/official/<id>)")] = None,
    extract: Annotated[bool, typer.Option(help="Extract zip files if present")] = True,
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
        downloaded = fetch_dataset(dataset_id, output_dir=output_dir, extract=extract)
        console.print(f"[green]Downloaded {len(downloaded)} file(s) to {output_dir}[/green]")
    except DatasetFetchError as e:
        console.print(f"[bold red]{e}[/bold red]")
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
    output_dir: Annotated[Path, typer.Option(help="Directory to save all benchmark results")] = Path("./results/benchmarks"),
):
    """
    Runs a series of simulations to benchmark an AI algorithm against a standard pump
    across multiple patient configurations and scenarios.
    """
    console = Console()
    console.print(f"[bold blue]Starting IINTS-AF Benchmark Suite[/bold blue]")
    console.print(f"AI Algorithm: [green]{algo_to_benchmark.name}[/green]")
    # console.print(f"Standard Pump Config: [yellow]{standard_pump_config}[/yellow]") # Removed, as standard pump uses patient params
    console.print(f"Patient Configs from: [cyan]{patient_configs_dir}[/cyan]")
    console.print(f"Scenarios from: [magenta]{scenarios_dir}[/magenta]")
    console.print(f"Duration: {duration} min, Time Step: {time_step} min")

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
            
            # Load Scenario Data
            stress_events = []
            try:
                with open(scenario_file, 'r') as f:
                    scenario_config = json.load(f)
                for event_data in scenario_config.get("stress_events", []):
                    event = iints.StressEvent(
                        start_time=event_data['start_time'],
                        event_type=event_data['event_type'],
                        value=event_data.get('value'),
                        reported_value=event_data.get('reported_value'),
                        absorption_delay_minutes=event_data.get('absorption_delay_minutes', 0),
                        duration=event_data.get('duration', 0)
                    )
                    stress_events.append(event)
            except (json.JSONDecodeError, KeyError) as e:
                console.print(f"[bold red]  Error loading scenario '{scenario_file.name}': {e}[/bold red]")
                continue # Skip this scenario

            # --- Run AI Algorithm Simulation ---
            console.print(f"    Running [green]{ai_algo_instance.get_algorithm_metadata().name}[/green]...")
            patient_model_ai = iints.PatientModel(**patient_params) # New instance for each run
            simulator_ai = iints.Simulator(patient_model=patient_model_ai, algorithm=ai_algo_instance, time_step=time_step)
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
            simulator_std = iints.Simulator(patient_model=patient_model_std, algorithm=standard_pump_algo_instance, time_step=time_step)
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
                "Patient": patient_config_name,
                "Scenario": scenario_name,
                "AI Algo": ai_algo_instance.get_algorithm_metadata().name,
                **{f"AI {k}": v for k, v in metrics_ai.items()},
                **{f"AI Safety Violations": safety_report_ai.get('total_violations', float('nan'))},
                "Standard Algo": standard_pump_algo_instance.get_algorithm_metadata().name,
                **{f"Std {k}": v for k, v in metrics_std.items()},
                **{f"Std Safety Violations": safety_report_std.get('total_violations', float('nan'))},
            })
    
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
