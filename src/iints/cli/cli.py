import typer  # type: ignore
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional
from typing_extensions import Annotated
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


app = typer.Typer(help="IINTS-AF SDK CLI - Intelligent Insulin Titration System for Artificial Pancreas research.")
docs_app = typer.Typer(help="Generate documentation and technical summaries for IINTS-AF components.")
app.add_typer(docs_app, name="docs")

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

    template_content = f"""from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmResult, AlgorithmMetadata
from typing import Dict, Any

class {name}Algorithm(iints.InsulinAlgorithm):
    def __init__(self, settings: Dict[str, Any] = None):
        super().__init__(settings)
        self.set_algorithm_metadata(iints.AlgorithmMetadata(
            name="{name}",
            author="{author}",
            description="A new custom insulin algorithm.",
            algorithm_type="rule_based" # Change as appropriate
        ))
        # Initialize any specific state or parameters for your algorithm here

    def predict_insulin(self, data: iints.AlgorithmInput) -> Dict[str, Any]:
        # --- YOUR ALGORITHM LOGIC GOES HERE ---
        # This is a basic placeholder. Implement your actual insulin prediction logic.

        # Example: Deliver 0.1 units if glucose is above 120 mg/dL
        total_insulin = 0.0
        bolus_insulin = 0.0
        basal_insulin = 0.0
        correction_bolus = 0.0
        meal_bolus = 0.0

        if data.current_glucose > 120:
            correction_bolus = (data.current_glucose - 120) / self.isf / 5 # Example: 1 unit per 50 mg/dL above 120
            total_insulin += correction_bolus
            self._log_reason(f"Correcting high glucose", "glucose_level", data.current_glucose, f"Delivered {{correction_bolus:.2f}} units to reduce {{data.current_glucose}} mg/dL")

        if data.carb_intake > 0:
            meal_bolus = data.carb_intake / self.icr
            total_insulin += meal_bolus
            self._log_reason(f"Meal intake detected", "carb_intake", data.carb_intake, f"Delivered {{meal_bolus:.2f}} units for {{data.carb_intake}}g carbs")

        # Simulate basal rate (e.g., a continuous small delivery)
        # For simplicity, let's assume a fixed basal delivery over the time step
        # You might integrate this with your overall basal strategy
        # basal_insulin = 0.01 * data.time_step # Example: 0.01 units per minute basal
        # total_insulin += basal_insulin
        self._log_reason(f"Maintaining basal rate", "basal", data.time_step, f"Delivered {{basal_insulin:.2f}} units basal")


        # Ensure no negative insulin delivery
        total_insulin = max(0.0, total_insulin)

        # Store important decisions in the why_log for transparency
        self._log_reason(f"Final insulin decision: {{total_insulin:.2f}} units", "decision", total_insulin)

        return {{ 
            "total_insulin_delivered": total_insulin,
            "bolus_insulin": bolus_insulin, # You might differentiate between meal and correction bolus here
            "basal_insulin": basal_insulin,
            "correction_bolus": correction_bolus,
            "meal_bolus": meal_bolus,
        }}
"""

    output_file = output_dir / f"{name.lower().replace(' ', '_')}_algorithm.py"
    with open(output_file, "w") as f:
        f.write(template_content)
    
    typer.echo(f"Successfully created new algorithm template: {output_file}")


@app.command()
def run(
    algo: Annotated[Path, typer.Option(help="Path to the algorithm Python file")],
    patient_config_name: Annotated[str, typer.Option(help="Name of the patient configuration (e.g., 'default' or 'patient_559_config')")] = "default",
    scenario_path: Annotated[Optional[Path], typer.Option(help="Path to the scenario JSON file (e.g., scenarios/example_scenario.json)")] = None,
    duration: Annotated[int, typer.Option(help="Simulation duration in minutes")] = 720, # 12 hours
    time_step: Annotated[int, typer.Option(help="Simulation time step in minutes")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Directory to save simulation results")] = Path("./results/data"),
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
    patient_config_path = Path(f"src/iints/data/virtual_patients/{patient_config_name}.yaml")
    if not patient_config_path.is_file():
        console.print(f"[bold red]Error: Patient configuration file '{patient_config_path}' not found.[/bold red]")
        console.print(f"[bold red]Please ensure '{patient_config_name}.yaml' exists in src/iints/data/virtual_patients/[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(patient_config_path, 'r') as f:
            patient_params = yaml.safe_load(f)
        patient_model = iints.PatientModel(**patient_params)
        console.print(f"Using patient model: {patient_model.__class__.__name__} with config from [cyan]{patient_config_name}.yaml[/cyan]")
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing patient configuration file {patient_config_path}: {e}[/bold red]")
        raise typer.Exit(code=1)
    except TypeError as e:
        console.print(f"[bold red]Error instantiating PatientModel with parameters from {patient_config_path}: {e}[/bold red]")
        console.print("[bold red]Please check that patient configuration keys match PatientModel constructor arguments.[/bold red]")
        raise typer.Exit(code=1)


    # 4. Load Scenario Data (if provided)
    stress_events = []
    if scenario_path:
        if not scenario_path.is_file():
            console.print(f"[bold red]Error: Scenario file '{scenario_path}' not found.[/bold red]")
            raise typer.Exit(code=1)
        
        try:
            with open(scenario_path, 'r') as f:
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
            console.print(f"Loaded {len(stress_events)} stress events from scenario: [magenta]{scenario_path.name}[/magenta]")

        except json.JSONDecodeError as e:
            console.print(f"[bold red]Error parsing scenario JSON file {scenario_path}: {e}[/bold red]")
            raise typer.Exit(code=1)
        except KeyError as e:
            console.print(f"[bold red]Error: Missing key in scenario event data: {e}[/bold red]")
            raise typer.Exit(code=1)
    
    # 5. Run Simulation
    simulator = iints.Simulator(patient_model=patient_model, algorithm=algorithm_instance, time_step=time_step)
    
    for event in stress_events:
        simulator.add_stress_event(event)

    simulation_results_df, safety_report = simulator.run_batch(duration)
    
    # 6. Output Results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct a more robust filename
    algo_name_for_filename = algorithm_instance.get_algorithm_metadata().name.replace(' ', '_').lower()
    scenario_name_for_filename = Path(scenario_path).stem if scenario_path else 'no_scenario'
    
    results_file = output_dir / f"sim_results_{algo_name_for_filename}_patient_{patient_config_name}_scenario_{scenario_name_for_filename}_duration_{duration}m.csv"
    
    simulation_results_df.to_csv(results_file, index=False)
    
    console.print(f"\nSimulation completed. Results saved to: [link file://{results_file}]{results_file}[/link]") # Formatted link
    console.print("\n--- Safety Report ---")
    for key, value in safety_report.items():
        console.print(f"{key}: {value}")
    
    console.print("\nDisplaying head of simulation results:")
    console.print(Panel(str(simulation_results_df.head()))) # Use Panel for rich output

    # Generate full report (using the new iints.generate_report function)
    report_output_path = output_dir / f"sim_report_{algo_name_for_filename}_patient_{patient_config_name}_scenario_{scenario_name_for_filename}_duration_{duration}m.pdf"
    iints.generate_report(simulation_results_df, str(report_output_path))


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