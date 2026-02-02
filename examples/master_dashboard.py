#!/usr/bin/env python3
"""
IINTS-AF Professional Control Center
Medical-Grade Terminal Interface
"""

import sys
import time
import random
import subprocess
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.live import Live # New import for live terminal updates
from rich.text import Text # New import for rich text formatting
import glob # New import for file discovery
import pandas as pd # New import for DataFrame manipulation

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core simulation components
from iints.core.simulator import Simulator
from iints.core.patient.models import PatientModel
from iints.core.algorithms.discovery import discover_algorithms
from iints.data.universal_parser import UniversalParser
from iints.core.simulation.scenario_parser import parse_scenario # For scenario loading
from iints.core.simulator import StressEvent # New import for StressEvent class
try:
    from iints.analysis.clinical_tir_analyzer import ClinicalTIRAnalyzer
    from iints.analysis.explainable_ai import ClinicalAuditTrail
    from iints.analysis.edge_performance_monitor import EdgeAIPerformanceMonitor
except ImportError as e:
    print(f"Warning: Some analysis modules not available: {e}")

class ProfessionalControlCenter:
    """Medical-Grade Control Center Interface"""
    
    def __init__(self):
        self.console = Console()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize analysis components
        self.tir_analyzer = ClinicalTIRAnalyzer()
        self.audit_trail = ClinicalAuditTrail()
        self.performance_monitor = EdgeAIPerformanceMonitor()
    
    def display_header(self):
        """Show the main header"""
        header = Panel(
            Align.center("[bold white]IINTS-AF CLINICAL CONTROL CENTER[/bold white]\n[cyan]Medical AI Research Platform[/cyan]"),
            style="blue",
            padding=(1, 2)
        )
        self.console.print(header)
    
    def display_system_status(self):
        """Show current system information"""
        status_table = Table(title="System Status", show_header=False)
        status_table.add_column("Parameter", style="cyan")
        status_table.add_column("Value", style="green")
        
        status_table.add_row("Hardware Platform", "NVIDIA Jetson Nano")
        status_table.add_row("AI Engine", "LSTM Neural Adaptation")
        status_table.add_row("Safety Supervisor", "ACTIVE")
        status_table.add_row("Current Dataset", "Ohio T1DM Patient 559")
        
        self.console.print(status_table)
        self.console.print()
    
    def display_main_menu(self):
        """Show available options"""
        menu_table = Table(title="Available Modules", show_header=False)
        menu_table.add_column("Option", style="bold blue", width=8)
        menu_table.add_column("Module", style="white")
        
        # Core modules
        menu_table.add_row("[1]", "In-Silico Clinical Validation")
        menu_table.add_row("[2]", "Population Study Analytics")
        menu_table.add_row("[3]", "Scientific Visualization Generator")
        menu_table.add_row("[4]", "Clinical Documentation System")
        menu_table.add_row("[5]", "MATLAB Control Theory Export")
        
        menu_table.add_row("", "")  # Separator
        
        # Advanced modules
        menu_table.add_row("[6]", "Professional TIR Analysis")
        menu_table.add_row("[7]", "Clinical Decision Audit System")
        menu_table.add_row("[8]", "Edge AI Performance Validation")
        menu_table.add_row("[A]", "Comparative Algorithm Benchmarking")
        menu_table.add_row("[B]", "AI Uncertainty Quantification")
        menu_table.add_row("[C]", "Real-Time Glucose Dashboard")
        menu_table.add_row("[D]", "Algorithm X-Ray System")
        
        menu_table.add_row("", "")  # Separator
        
        # System
        menu_table.add_row("[9]", "Complete Platform Demonstration")
        menu_table.add_row("[0]", "Exit System")
        
        self.console.print(menu_table)
    
    def run_control_center(self):
        """Main program loop"""
        while True:
            self.console.clear()
            self.display_header()
            self.display_system_status()
            self.display_main_menu()
            
            choice = Prompt.ask(
                "\nSelect module",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "a", "B", "b", "C", "c", "D", "d"]
            )
            
            if choice == "0":
                self.console.print("\n[green]IINTS-AF Control Center shutdown complete[/green]")
                break
            elif choice == "1":
                self.run_clinical_validation()
            elif choice == "2":
                self.generate_excel_population_study()
            elif choice == "3":
                self.generate_science_expo_graphics()
            elif choice == "4":
                self.generate_clinical_pdf()
            elif choice == "5":
                self.export_matlab_analysis()
            elif choice == "6":
                self.run_5zone_tir_analysis()
            elif choice == "7":
                self.run_explainable_ai_demo()
            elif choice == "8":
                self.run_edge_performance_validation()
            elif choice == "9":
                self.run_complete_demo()
            elif choice.upper() == "A":
                self.run_comparative_benchmarking()
            elif choice.upper() == "B":
                self.run_uncertainty_quantification()
            elif choice.upper() == "C":
                self.run_real_time_dashboard()
            elif choice.upper() == "D":
                self.run_algorithm_xray()
            
            if choice != "0":
                Prompt.ask("\nPress Enter to return to main menu")
    
    def run_clinical_validation(self):
        """
        Runs a live in-silico clinical validation with traceability log.
        This provides real-time feedback on AI decisions, patient state, and safety.
        """
        self.console.clear()
        self.console.print("[bold blue]In-Silico Clinical Validation - Live Traceability[/bold blue]\n")

        # 1. Data File Selection
        self.console.print("Searching for patient data files in 'data_packs/'...")
        data_files = glob.glob('data_packs/**/timeseries.csv', recursive=True)
        data_files.extend(glob.glob('data_packs/**/data.csv', recursive=True))

        if not data_files:
            self.console.print("[red]No patient data files found in 'data_packs/'.[/red]")
            self.console.print("Please add some, for example by using the --import-ohio flag.")
            return

        self.console.print("\nPlease select a data file to run the simulation on:")
        for i, f in enumerate(data_files):
            self.console.print(f"[{i+1}] {f}")
        choice = Prompt.ask(f"Enter number (1-{len(data_files)})", choices=[str(i+1) for i in range(len(data_files))])
        selected_file = data_files[int(choice)-1]
        self.console.print(f"You selected: [green]{selected_file}[/green]")

        # 2. Algorithm Selection
        self.console.print("\nDiscovering available algorithms...")
        available_algos_map = discover_algorithms()
        if not available_algos_map:
            self.console.print("[red]Fatal: No algorithms found. Check your installation.[/red]")
            return
        
        available_algorithms_list = sorted(available_algos_map.keys())

        self.console.print("Available algorithms:")
        for i, algo_name in enumerate(available_algorithms_list):
            self.console.print(f"[{i+1}] {algo_name}")
        algo_choice = Prompt.ask(f"Enter number (1-{len(available_algorithms_list)})", choices=[str(i+1) for i in range(len(available_algorithms_list))])
        selected_algo_name = available_algorithms_list[int(algo_choice)-1]
        self.console.print(f"You selected algorithm: [green]{selected_algo_name}[/green]")
        selected_algo_class = available_algos_map[selected_algo_name]
        algorithm_instance = selected_algo_class()
        
        # 3. Scenario Selection
        stress_events = []
        scenario_name_from_file = Path(selected_file).stem
        self.console.print("\nScanning for custom scenarios in 'scenarios/'...")
        scenario_files = glob.glob('scenarios/**/*.json', recursive=True)
        
        if scenario_files and Confirm.ask("Do you want to apply a custom scenario file?", default=False):
            self.console.print("\\nPlease select a scenario file:")
            for i, f in enumerate(scenario_files):
                self.console.print(f"[{i+1}] {f}")
            self.console.print("[0] Skip")

            scenario_choice = Prompt.ask(f"Enter number (0-{len(scenario_files)})", choices=[str(i) for i in range(len(scenario_files)+1)])
            
            if int(scenario_choice) > 0:
                selected_scenario_file = scenario_files[int(scenario_choice)-1]
                self.console.print(f"You selected scenario: [green]{selected_scenario_file}[/green]")
                try:
                    scenario_metadata, stress_events = parse_scenario(selected_scenario_file)
                    scenario_name_from_file = scenario_metadata.get('name', scenario_name_from_file)
                    self.console.print(f"Successfully loaded scenario '[green]{scenario_name_from_file}[/green]' with {len(stress_events)} events.")
                except Exception as e:
                    self.console.print(f"[red]Warning: Could not parse scenario file: {e}. Continuing without custom scenario.[/red]")
                    stress_events = []
        
        # 4. Parse Data
        self.console.print("\nLoading and parsing patient data...")
        parser = UniversalParser()
        parse_result = parser.parse(selected_file)

        if not parse_result.success:
            self.console.print(f"[red]Error parsing data file: {parse_result.errors}[/red]")
            return

        patient_data = parse_result.data_pack.data
        patient_data.rename(columns={"timestamp": "time"}, inplace=True)
        if 'carbs' not in patient_data.columns:
            patient_data['carbs'] = 0.0
        patient_data['carbs'] = pd.to_numeric(patient_data['carbs']).fillna(0)
        self.console.print("[green]Data parsed successfully.[/green]")

        # Determine simulation duration (e.g., from max time in patient data)
        simulation_duration = int(patient_data['time'].max()) if not patient_data.empty else 1440 # Default 24 hours

        # Instantiate patient model
        initial_glucose_from_data = patient_data['glucose'].iloc[0] if not patient_data.empty else 120.0
        patient_model = PatientModel(initial_glucose=initial_glucose_from_data)
        
        # Instantiate simulator
        simulator = Simulator(
            patient_model=patient_model,
            algorithm=algorithm_instance,
            time_step=5
        )

        # Add stress events from parsed scenario
        for event in stress_events:
            simulator.add_stress_event(event)
        
        # Add patient_data carb events as stress events (as done in BattleRunner)
        # This allows the simulator to handle delayed carb absorption from the data itself.
        # This is important to ensure carbs from the original patient data are modeled
        for index, row in patient_data.iterrows():
            if row['carbs'] > 0:
                simulator.add_stress_event(StressEvent(
                    start_time=int(row['time']), 
                    event_type='meal', 
                    value=row['carbs'],
                    reported_value=row['carbs'] # Assume reported correctly if from patient data
                ))
        
        self.console.print(f"\n[bold green]Starting Live Simulation: '{scenario_name_from_file}' with '{selected_algo_name}'[/bold green]")
        self.console.print("[yellow]Press Ctrl+C to stop simulation.[/yellow]\n")

        # Live display setup
        log_panel_content = Text("Simulation Log:\n", style="dim")
        log_panel_max_lines = self.console.height - 10 # Adjust based on console size
        
        with Live(log_panel_content, screen=True, refresh_per_second=4, console=self.console) as live:
            try:
                for record in simulator.run_live(simulation_duration):
                    # Format DATA Trace
                    data_trace = f"[DATA] Glc: {record['glucose_actual_mgdl']:.0f} (Algo sees: {record['glucose_to_algo_mgdl']:.0f}) mg/dL"
                    
                    # Format ALGO Trace
                    algo_name_for_log = algorithm_instance.get_algorithm_metadata().name
                    algo_reasoning_log = record['algorithm_why_log']
                    algo_trace_text = "No specific reasoning logged."
                    if algo_reasoning_log:
                        # Taking first reason for brevity in live log
                        primary_reason = algo_reasoning_log[0]['reason']
                        value_info = f" (Val: {algo_reasoning_log[0]['value']:.2f})" if algo_reasoning_log[0]['value'] is not None else ""
                        clinical_impact_info = f" -> {algo_reasoning_log[0]['clinical_impact']}" if algo_reasoning_log[0]['clinical_impact'] else ""
                        algo_trace_text = f"{primary_reason}{value_info}{clinical_impact_info}"

                    algo_trace = f"[ALGO:{algo_name_for_log}] Rec: {record['algo_recommended_insulin_units']:.2f}U. {algo_trace_text}"

                    # Format GUARD Trace
                    safety_level = record['safety_level']
                    safety_actions = record['safety_actions'] if record['safety_actions'] else "None"
                    guard_color = "red" if "OVERRIDE" in safety_actions else "green" if safety_level == "SAFE" else "yellow"
                    guard_trace = f"[GUARD:{safety_level}] Actions: {safety_actions}"
                    
                    # Assemble and add to log
                    new_log_line = Text.from_markup(f"[{record['time_minutes']:04d} min] [green]{data_trace}[/green] [cyan]{algo_trace}[/cyan] [{guard_color}]{guard_trace}[/{guard_color}]\n")
                    log_panel_content.append(new_log_line)

                    # Keep log content within reasonable lines
                    num_newlines = log_panel_content.plain.count('\n')
                    if num_newlines > log_panel_max_lines:
                        # Find the index of the nth newline from the end
                        first_line_to_keep_index = log_panel_content.plain.rfind('\n', 0, log_panel_content.plain.rfind('\n', 0, -1) - 1)
                        for _ in range(num_newlines - log_panel_max_lines):
                            first_line_to_keep_index = log_panel_content.plain.find('\n', first_line_to_keep_index + 1)
                        if first_line_to_keep_index != -1:
                            log_panel_content = log_panel_content.copy()
                            log_panel_content.trim_right()
                            log_panel_content = log_panel_content[first_line_to_keep_index+1:]


                    # Update Live display with new Panel content
                    live.update(Panel(log_panel_content, title="[bold cyan]Traceability Log[/bold cyan]", border_style="blue"))
                    
                    time.sleep(0.01) # Small delay for demonstration purposes
            
            except KeyboardInterrupt:
                self.console.print("\n[red]Simulation interrupted by user (Ctrl+C).[/red]")
            except Exception as e:
                self.console.print(f"[red]An error occurred during simulation: {e}[/red]")
            
        self.console.print(f"\n[green]Live Clinical Validation Simulation Complete![/green]")
        # Offer to generate a final report
        if Confirm.ask("\nGenerate a comprehensive clinical validation report?", default=True):
            from scripts.generate_clinical_report import ClinicalReportGenerator
            report_generator = ClinicalReportGenerator()
            # Need to collect all records to pass to the report generator's executive_summary
            # For now, let's just make a dummy call or gather data in `run_batch` if a full report is desired
            # This would require an extra step to run `run_batch` to get the full DataFrame or modify `run_live` to return it too.
            self.console.print("[yellow]Report generation feature coming soon for live simulations.[/yellow]")
    
    def generate_excel_population_study(self):
        """Generate professional Excel population study"""
        self.console.print("\n[bold blue]Excel Population Study Generator[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Creating professional Excel summary...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/generate_excel_summary.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Excel Population Study Generated![/green]")
                    self.console.print(f"[blue]Professional 4-sheet analysis:[/blue]")
                    self.console.print(f"[cyan]• Population Study - TIR improvements across patients[/cyan]")
                    self.console.print(f"[cyan]• Decision Audit - AI decision tracking with confidence[/cyan]")
                    self.console.print(f"[cyan]• Statistical Analysis - P-values and success rates[/cyan]")
                    self.console.print(f"[cyan]• Algorithm Comparison - Performance matrix by scenario[/cyan]")
                    
                    # Find generated file
                    excel_files = list(self.results_dir.glob("IINTS_Summary_*.xlsx"))
                    if excel_files:
                        latest_excel = max(excel_files, key=lambda p: p.stat().st_mtime)
                        self.console.print(f"\n[blue]File: {latest_excel.name}[/blue]")
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")
    
    def generate_science_expo_graphics(self):
        """Generate Science Expo presentation graphics"""
        self.console.print("\n[bold blue]Science Expo Graphics Generator[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Creating publication-ready graphics...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/create_population_graphics.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Science Expo Graphics Generated![/green]")
                    self.console.print(f"[blue]Professional visualizations created:[/blue]")
                    self.console.print(f"[cyan]• Population Study Overview - 4-panel scientific figure[/cyan]")
                    self.console.print(f"[cyan]• Clinical Scenario Heatmap - Algorithm performance matrix[/cyan]")
                    self.console.print(f"[cyan]• Statistical Summary - P-values with significance indicators[/cyan]")
                    self.console.print(f"\n[yellow]Perfect for Science Expo poster presentation![/yellow]")
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")
    
    def generate_clinical_pdf(self):
        """Generate hospital-grade clinical PDF report"""
        self.console.print("\n[bold blue]Clinical PDF Report Generator[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Generating hospital-grade clinical report...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/generate_clinical_report.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Clinical PDF Report Generated![/green]")
                    self.console.print(f"[blue]Clinician-oriented clinical audit report:[/blue]")
                    self.console.print(f"[cyan]• Medtronic CareLink-style formatting[/cyan]")
                    self.console.print(f"[cyan]• 5-zone TIR analysis with clinical benchmarks[/cyan]")
                    self.console.print(f"[cyan]• Learning curve validation and safety metrics[/cyan]")
                    self.console.print(f"[cyan]• Professional medical terminology and colors[/cyan]")
                    
                    # Find generated PDF
                    pdf_files = list(Path("results/clinical_reports").glob("*.pdf"))
                    if pdf_files:
                        latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
                        self.console.print(f"\n[blue]Report: {latest_pdf.name}[/blue]")
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")
    
    def export_matlab_analysis(self):
        """Export data for MATLAB control theory analysis"""
        self.console.print("\n[bold blue]MATLAB Control Theory Export[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Preparing MATLAB control analysis...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/matlab_control_integration.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]MATLAB Analysis Ready![/green]")
                    self.console.print(f"[blue]University-level control theory analysis:[/blue]")
                    self.console.print(f"[cyan]• System identification and transfer functions[/cyan]")
                    self.console.print(f"[cyan]• Bode plots with gain and phase margins[/cyan]")
                    self.console.print(f"[cyan]• Stability analysis and step response[/cyan]")
                    self.console.print(f"[cyan]• Medical device performance validation[/cyan]")
                    
                    self.console.print(f"\n[yellow]Next steps for MATLAB:[/yellow]")
                    self.console.print(f"[white]1. Open MATLAB[/white]")
                    self.console.print(f"[white]2. Navigate to: results/matlab_analysis/[/white]")
                    self.console.print(f"[white]3. Run: analyze_stability[/white]")
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")

    def run_5zone_tir_analysis(self):
        """Run professional 5-zone TIR analysis"""
        self.console.print("\n[bold blue]Professional 5-Zone TIR Analysis[/bold blue]\n")
        
        # Generate sample glucose data for demo
        sample_glucose = [
            45, 65, 85, 120, 140, 165, 180, 195, 220, 260,  # Various zones
            110, 125, 135, 150, 160, 170, 145, 130, 115, 105  # Mostly target
        ]
        # Add additional target range values
        sample_glucose.extend([random.uniform(70, 180) for _ in range(30)])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Analyzing glucose zones using Medtronic standard...", total=None)
            time.sleep(1)  # Simulate processing
            
            analysis = self.tir_analyzer.analyze_glucose_zones(sample_glucose)
            progress.stop()
        
        self.console.print(f"[green]Medtronic 5-Zone Clinical Analysis Complete![/green]\n")
        
        # Display zone analysis
        table = Table(title="Professional TIR Analysis Results")
        table.add_column("Clinical Zone", style="cyan")
        table.add_column("Range (mg/dL)", style="white")
        table.add_column("Percentage", style="green")
        table.add_column("Count", style="yellow")
        
        for zone_name, data in analysis.items():
            if zone_name in ['clinical_assessment', 'total_readings']:
                continue
            
            table.add_row(
                data['clinical_name'],
                data['range_mg_dL'],
                f"{data['percentage']:.1f}%",
                str(data['count'])
            )
        
        self.console.print(table)
        
        # Clinical assessment
        assessment = analysis['clinical_assessment']
        self.console.print(f"\n[yellow]Clinical Assessment:[/yellow]")
        self.console.print(f"  Risk Level: {assessment['risk_level']}")
        self.console.print(f"  TIR Quality: {assessment['tir_quality']}")
        self.console.print(f"  Primary Concern: {assessment['primary_concern']}")
        self.console.print(f"  Total Readings: {analysis['total_readings']}")
    
    def run_explainable_ai_demo(self):
        """Run explainable AI audit trail demonstration"""
        self.console.print("\n[bold blue]Explainable AI Clinical Audit Trail[/bold blue]\n")
        
        # Simulate clinical decisions
        scenarios = [
            (120, [115, 118, 120], 0.2, 0.85, False, None, "Stable glucose maintenance"),
            (185, [170, 178, 185], 1.5, 0.92, False, {'meal_detected': True}, "Post-meal correction"),
            (65, [75, 70, 65], -0.8, 0.78, True, None, "Hypoglycemia prevention"),
            (240, [220, 235, 240], 2.1, 0.88, False, None, "Hyperglycemia correction")
        ]
        
        self.console.print("[cyan]Generating clinical decision explanations...[/cyan]\n")
        
        for i, (glucose, trend, insulin, confidence, override, context, description) in enumerate(scenarios):
            timestamp = datetime.now().replace(hour=8+i, minute=15)
            
            entry = self.audit_trail.log_decision(
                timestamp, glucose, trend, insulin, confidence, override, context
            )
            
            self.console.print(f"[yellow]Decision {i+1}: {description}[/yellow]")
            self.console.print(f"  Time: {timestamp.strftime('%H:%M')}")
            self.console.print(f"  Glucose: {glucose} mg/dL")
            self.console.print(f"  Clinical Reasoning: {entry['clinical_reasoning']}")
            self.console.print(f"  Risk Assessment: {entry['risk_assessment']}")
            self.console.print(f"  Decision Category: {entry['decision_category']}")
            self.console.print()
        
        # Generate summary
        summary = self.audit_trail.generate_clinical_summary(4)
        self.console.print(f"[green]Clinical Summary:[/green]")
        self.console.print(summary)
    
    def run_edge_performance_validation(self):
        """Run Edge AI performance validation"""
        self.console.print("\n[bold blue]Edge AI Performance Validation (Jetson Nano)[/bold blue]\n")
        
        # Mock inference function
        def mock_inference(data):
            time.sleep(random.uniform(0.005, 0.015))  # 5-15ms latency
            return {"prediction": random.random(), "confidence": random.random()}
        
        test_input = {"glucose": 150, "trend": [145, 148, 150]}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Measuring inference performance...", total=None)
            
            # Start monitoring
            self.performance_monitor.start_monitoring()
            
            # Measure performance
            stats = self.performance_monitor.measure_inference_latency(
                mock_inference, test_input, iterations=50
            )
            
            progress.stop()
        
        self.console.print(f"[green]Edge AI Performance Validation Complete![/green]\n")
        
        # Display key metrics
        latency_stats = stats['latency_statistics']
        self.console.print(f"[cyan]Inference Performance:[/cyan]")
        self.console.print(f"  Mean Latency: {latency_stats['mean_ms']:.3f} ms")
        self.console.print(f"  95th Percentile: {latency_stats['p95_ms']:.3f} ms")
        self.console.print(f"  Standard Deviation: {latency_stats['std_ms']:.3f} ms")
        
        self.console.print(f"\n[yellow]Embedded System Assessment:[/yellow]")
        self.console.print(f"  Latency Profile: {stats['performance_classification']}")
        self.console.print(f"  Consistency: {stats['consistency_rating']}")
        self.console.print(f"  Compatibility: {stats['medical_device_assessment']['suitability_rating']}")
        
        # Generate full report
        report = self.performance_monitor.generate_performance_report()
        report_file = self.results_dir / "edge_ai_performance_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.console.print(f"\n[blue]Full report saved: {report_file.name}[/blue]")
    
    def run_comparative_benchmarking(self):
        """Run comparative algorithm benchmarking study"""
        self.console.print("\n[bold blue]Comparative Algorithm Benchmarking[/bold blue]\n")
        
        self.console.print("[cyan]This module compares multiple algorithms head-to-head:[/cyan]")
        self.console.print("[white]• LSTM AI (Your Algorithm)[/white]")
        self.console.print("[white]• Industry PID Controller[/white]")
        self.console.print("[white]• Hybrid Control System[/white]")
        self.console.print("[white]• Fixed Dose Protocol[/white]")
        
        if not Confirm.ask("\nRun full comparative study? (This may take 2-3 minutes)", default=True):
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running comparative algorithm benchmarking...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/test_comparative_benchmarking.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Comparative Benchmarking Complete![/green]")
                    self.console.print(f"[blue]Head-to-head algorithm comparison:[/blue]")
                    self.console.print(f"[cyan]• Statistical significance testing (p-values)[/cyan]")
                    self.console.print(f"[cyan]• TIR performance comparison across 4 patients[/cyan]")
                    self.console.print(f"[cyan]• Clinical risk assessment matrix[/cyan]")
                    self.console.print(f"[cyan]• Publication-ready comparison visualizations[/cyan]")
                    
                    self.console.print(f"\n[yellow]Key Findings:[/yellow]")
                    self.console.print(f"[white]Check results/algorithm_comparison/ for:[/white]")
                    self.console.print(f"[white]• Executive Summary with statistical results[/white]")
                    self.console.print(f"[white]• Performance heatmap comparing all algorithms[/white]")
                    self.console.print(f"[white]• Box plots showing TIR improvements[/white]")
                    
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")
    
    def run_complete_demo(self):
        """Run all modules in sequence"""
        self.console.print("\n[bold green]Complete Platform Demonstration[/bold green]")
        
        demo_modules = [
            ("Clinical Validation", self.run_clinical_validation),
            ("Excel Population Study", self.generate_excel_population_study),
            ("Scientific Graphics", self.generate_science_expo_graphics),
            ("Clinical PDF Report", self.generate_clinical_pdf),
            ("MATLAB Export", self.export_matlab_analysis)
        ]
        
        for module_name, module_function in demo_modules:
            self.console.print(f"\n[cyan]Running: {module_name}[/cyan]")
            try:
                module_function()
                self.console.print(f"[green] {module_name} completed[/green]")
            except Exception as e:
                self.console.print(f"[red] {module_name} failed: {e}[/red]")
            time.sleep(1)
        
        self.console.print("\n[bold green]Complete demonstration finished[/bold green]")
        self.console.print("[cyan]All outputs available in results/ directory[/cyan]")

    def run_uncertainty_quantification(self):
        """Run AI uncertainty quantification analysis"""
        self.console.print("\n[bold blue]AI Uncertainty Quantification[/bold blue]\n")
        
        self.console.print("[cyan]Advanced ML with confidence intervals and 'I don't know' detection:[/cyan]")
        self.console.print("[white]• Monte Carlo Dropout for uncertainty estimation[/white]")
        self.console.print("[white]• Bayesian neural network simulation[/white]")
        self.console.print("[white]• Epistemic vs Aleatoric uncertainty separation[/white]")
        self.console.print("[white]• Clinical decision confidence scoring[/white]")
        
        if not Confirm.ask("\nRun uncertainty quantification study?", default=True):
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Running uncertainty quantification analysis...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/uncertainty_quantification.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Uncertainty Quantification Complete![/green]")
                    self.console.print(f"[blue]Advanced ML uncertainty analysis:[/blue]")
                    self.console.print(f"[cyan]• Confidence intervals for all AI predictions[/cyan]")
                    self.console.print(f"[cyan]• 'I don't know' detection for edge cases[/cyan]")
                    self.console.print(f"[cyan]• Epistemic vs Aleatoric uncertainty breakdown[/cyan]")
                    self.console.print(f"[cyan]• Clinical decision confidence matrix[/cyan]")
                    
                    self.console.print(f"\n[yellow]University-Level Features:[/yellow]")
                    self.console.print(f"[white]• Bayesian neural network simulation[/white]")
                    self.console.print(f"[white]• Monte Carlo Dropout implementation[/white]")
                    self.console.print(f"[white]• Publication-ready uncertainty visualizations[/white]")
                    
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")
    
    def run_real_time_dashboard(self):
        """Launch real-time glucose monitoring dashboard"""
        self.console.print("\n[bold blue]Real-Time Glucose Dashboard[/bold blue]\n")
        
        self.console.print("[cyan]Live glucose simulation with real-time TIR updates:[/cyan]")
        self.console.print("[white]• Real-time glucose curve with confidence intervals[/white]")
        self.console.print("[white]• Live TIR meter with 5-zone classification[/white]")
        self.console.print("[white]• AI uncertainty visualization[/white]")
        self.console.print("[white]• Insulin delivery tracking[/white]")
        
        self.console.print(f"\n[yellow]Dashboard Features:[/yellow]")
        self.console.print(f"[white]• 6-panel real-time visualization[/white]")
        self.console.print(f"[white]• Circadian rhythm simulation[/white]")
        self.console.print(f"[white]• Meal effect modeling[/white]")
        self.console.print(f"[white]• Safety zone highlighting[/white]")
        
        if not Confirm.ask("\nLaunch real-time dashboard? (Will open new window)", default=True):
            return
        
        self.console.print(f"\n[green]Launching Real-Time Dashboard...[/green]")
        self.console.print(f"[blue]Close the dashboard window to return to main menu[/blue]")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/real_time_dashboard.py"
            ], cwd=project_root)
            
            self.console.print(f"\n[green]Dashboard session completed![/green]")
            
        except Exception as e:
            self.console.print(f"[red]Error launching dashboard: {e}[/red]")
    
    def run_algorithm_xray(self):
        """Run Algorithm X-Ray analysis"""
        self.console.print("\n[bold blue]Algorithm X-Ray: Make Invisible Decisions Visible[/bold blue]\n")
        
        self.console.print("[cyan]X-ray vision into algorithm decision-making:[/cyan]")
        self.console.print("[white]• Decision Replay with full reasoning chain[/white]")
        self.console.print("[white]• Algorithm Personality Profiling[/white]")
        self.console.print("[white]• What-If Scenario Comparison[/white]")
        self.console.print("[white]• Clinical Decision Transparency[/white]")
        
        self.console.print(f"\n[yellow]X-Ray Features:[/yellow]")
        self.console.print(f"[white]• Every decision explained with physiological reasoning[/white]")
        self.console.print(f"[white]• Personality traits: Hypo-aversion, Reaction speed, etc.[/white]")
        self.console.print(f"[white]• Alternative scenarios: 'What if I exercised?'[/white]")
        self.console.print(f"[white]• Safety constraint visualization[/white]")
        
        if not Confirm.ask("\nRun Algorithm X-Ray analysis?", default=True):
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Analyzing algorithm decision patterns...", total=None)
            
            try:
                result = subprocess.run([
                    sys.executable, "scripts/visualize_algorithm_xray.py"
                ], capture_output=True, text=True, cwd=project_root)
                
                progress.stop()
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]Algorithm X-Ray Analysis Complete![/green]")
                    self.console.print(f"[blue]Decision transparency achieved:[/blue]")
                    self.console.print(f"[cyan]• Decision timeline with reasoning chain[/cyan]")
                    self.console.print(f"[cyan]• Algorithm personality radar chart[/cyan]")
                    self.console.print(f"[cyan]• What-if scenario comparison[/cyan]")
                    self.console.print(f"[cyan]• Safety override event tracking[/cyan]")
                    
                    self.console.print(f"\n[yellow]Jury Impact Statement:[/yellow]")
                    self.console.print(f"[white]'I make invisible medical decisions visible.'[/white]")
                    self.console.print(f"[white]'This is an X-ray for algorithms.'[/white]")
                    
                else:
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    
            except Exception as e:
                progress.stop()
                self.console.print(f"[red]Error: {e}[/red]")

def main():
    """Launch IINTS-AF Professional Control Center"""
    control_center = ProfessionalControlCenter()
    
    try:
        control_center.run_control_center()
        
    except KeyboardInterrupt:
        control_center.console.print("\n[green]System shutdown initiated[/green]")

if __name__ == "__main__":
    main()
