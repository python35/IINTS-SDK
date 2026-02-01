import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.text import Text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.universal_parser import UniversalParser, StandardDataPack
from src.algorithm.battle_runner import BattleRunner
from src.algorithm.pid_controller import PIDController
from src.algorithm.correction_bolus import CorrectionBolus
from src.algorithm.hybrid_algorithm import HybridInsulinAlgorithm
from src.simulation.simulator import StressEvent

console = Console()

def get_file_path_from_user() -> Path:
    """Prompts the user for a CSV file path and validates it."""
    while True:
        file_path_str = Prompt.ask(
            "[bold cyan]Enter the path to your CSV or JSON data file[/bold cyan]"
        )
        file_path = Path(file_path_str)
        if file_path.is_file():
            return file_path
        else:
            console.print(f"[bold red]Error: File not found at '{file_path_str}'. Please try again.[/bold red]")

def get_what_if_event_from_user(data_pack: StandardDataPack) -> StressEvent:
    """Prompts the user to define a 'what-if' event."""
    console.print("\n[bold blue]Define a 'What-If' Event:[/bold blue]")
    
    # Determine the max time in the dataset to guide the user
    max_time_minutes = data_pack.data['timestamp'].max()
    
    event_type = Prompt.ask(
        "[bold green]Select event type[/bold green]",
        choices=["meal", "bolus"],
        default="meal"
    )
    
    event_time = IntPrompt.ask(
        f"[bold green]Enter event time in minutes (0 - {int(max_time_minutes)})[/bold green]",
        default=60,
        choices=[str(i * 10) for i in range(int(max_time_minutes / 10) + 1)] # Provide some common intervals
    )
    
    event_value = FloatPrompt.ask(
        f"[bold green]Enter value for {event_type} (e.g., grams for meal, units for bolus)[/bold green]",
        default=30.0 if event_type == "meal" else 1.0
    )
    
    console.print(f"  [italic yellow]What-If Event: {event_type} of {event_value} at {event_time} minutes[/italic yellow]")
    return StressEvent(start_time=event_time, event_type=event_type, value=event_value)

def main():
    console.print(Panel(
        "[bold green]IINTS-AF Manual Scenario Editor[/bold green]\n"
        "Define 'What-If' scenarios with your own data and compare them!",
        title="[bold blue]Welcome[/bold blue]",
        expand=False
    ))

    # 1. Get file path and parse data
    file_path = get_file_path_from_user()
    parser = UniversalParser(auto_validate=True)
    console.print(f"[dim]Parsing {file_path}...[/dim]")
    parse_result = parser.parse(str(file_path))

    if not parse_result.success:
        console.print(f"[bold red]Failed to parse data: {parse_result.errors}[/bold red]")
        for warn in parse_result.warnings:
            console.print(f"[yellow]Warning: {warn}[/yellow]")
        return

    original_data_pack = parse_result.data_pack
    console.print(Panel(
        f"[bold green]Data Loaded Successfully![/bold green]\n"
        f"Data Points: {original_data_pack.data_points}\n"
        f"Duration: {original_data_pack.duration_hours:.1f} hours\n"
        f"Confidence Score: {original_data_pack.confidence_score:.1%}\n"
        f"Quality Summary: {original_data_pack.quality_report.summary if original_data_pack.quality_report else 'N/A'}",
        title="[bold blue]Original Data Summary[/bold blue]",
        expand=False
    ))
    
    # Show warnings if any
    if parse_result.warnings:
        console.print("\n[bold yellow]Data Parsing Warnings:[/bold yellow]")
        for warn in parse_result.warnings:
            console.print(f"- {warn}")

    # 2. Define "What-If" event
    what_if_event = get_what_if_event_from_user(original_data_pack)

    # 3. Prepare algorithms for BattleRunner
    console.print("\n[bold blue]Select AI Algorithms for Comparison:[/bold blue]")
    available_algorithms = {
        "pid": PIDController(),
        "correction_bolus": CorrectionBolus(),
        "hybrid": HybridInsulinAlgorithm(),
        # Add other algorithms as needed
    }
    
    selected_algo_keys_str = Prompt.ask(
        "[bold green]Enter algorithm names to compare (comma-separated, e.g., pid,hybrid)[/bold green]",
        default="hybrid"
    )
    selected_algo_keys = [k.strip().lower() for k in selected_algo_keys_str.split(',')]
    
    algorithms_to_use = {}
    for key in selected_algo_keys:
        if key in available_algorithms:
            algorithms_to_use[key.title()] = available_algorithms[key]
        else:
            console.print(f"[bold red]Algorithm '{key}' not found. Skipping.[/bold red]")

    if not algorithms_to_use:
        console.print("[bold red]No valid algorithms selected. Aborting.[/bold red]")
        return

    # 4. Run BattleRunner for Original Scenario
    console.print("\n[bold blue]Running Battle for Original Scenario...[/bold blue]")
    original_battle_runner = BattleRunner(
        algorithms=algorithms_to_use,
        patient_data=original_data_pack.data,
        scenario_name="Original"
    )
    original_report, _ = original_battle_runner.run_battle()
    console.print(Panel(
        f"[bold green]Original Scenario Results[/bold green]\n"
        f"Winner: {original_report['winner']}\n"
        f"TIR (Winner): {original_report['rankings'][0]['tir']:.1f}%",
        title="[bold blue]Battle Report[/bold blue]",
        expand=False
    ))

    # 5. Run BattleRunner for What-If Scenario
    console.print("\n[bold blue]Running Battle for What-If Scenario...[/bold blue]")
    
    # Create a copy of the original data to modify for the what-if scenario
    what_if_patient_data = original_data_pack.data.copy()
    
    # Apply the what-if event to the data for the BattleRunner
    stress_events_for_what_if = [what_if_event]
    
    what_if_battle_runner = BattleRunner(
        algorithms=algorithms_to_use,
        patient_data=what_if_patient_data, # Use a potentially modified patient data if needed
        stress_events=stress_events_for_what_if,
        scenario_name="What-If"
    )
    what_if_report, _ = what_if_battle_runner.run_battle()
    console.print(Panel(
        f"[bold green]What-If Scenario Results[/bold green]\n"
        f"Winner: {what_if_report['winner']}\n"
        f"TIR (Winner): {what_if_report['rankings'][0]['tir']:.1f}%",
        title="[bold blue]Battle Report[/bold blue]",
        expand=False
    ))

    console.print("\n[bold magenta]Manual Scenario Analysis Complete![/bold magenta]")
    console.print("Review the battle reports above for comparison.")


if __name__ == "__main__":
    main()
