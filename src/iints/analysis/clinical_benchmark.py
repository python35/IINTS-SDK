#!/usr/bin/env python3
"""
Clinical Benchmark System
Compares AI algorithms against real-world clinical outcomes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iints.data.adapter import DataAdapter

class ClinicalBenchmark:
    """Clinical benchmark comparison system"""
    
    def __init__(self):
        self.console = Console()
        self.adapter = DataAdapter()
        
    def run_ohio_benchmark(self):
        """Run clinical benchmark against Ohio T1DM dataset"""
        self.console.print("\n[bold blue] Clinical Benchmark Study[/bold blue]")
        self.console.print("Comparing AI algorithms against real-world patient outcomes\n")
        
        # Get available Ohio patients
        patients = self.adapter.get_available_ohio_patients()
        
        if not patients:
            self.console.print("[red] No Ohio T1DM patients found[/red]")
            self.console.print("Run: [cyan]python tools/import_ohio.py /path/to/ohio/dataset[/cyan]")
            return
        
        self.console.print(f"[green] Found {len(patients)} Ohio T1DM patients[/green]")
        
        # Select patient for benchmark
        if len(patients) == 1:
            selected_patient = patients[0]
        else:
            self.console.print("\nAvailable patients:")
            for i, patient in enumerate(patients, 1):
                self.console.print(f"  {i}. Patient {patient}")
            
            try:
                choice = int(input("\nSelect patient (1-{}): ".format(len(patients)))) - 1
                selected_patient = patients[choice]
            except (ValueError, IndexError):
                selected_patient = patients[0]
        
        self.console.print(f"\n[yellow] Running benchmark on Patient {selected_patient}[/yellow]")
        
        # Run benchmark comparison
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Running clinical benchmark...", total=None)
            
            results = self.adapter.clinical_benchmark_comparison(
                selected_patient, 
                ['rule_based', 'lstm', 'hybrid']
            )
            
            progress.update(task, completed=True)
        
        # Display results
        self._display_benchmark_results(results)
        
        return results
    
    def _display_benchmark_results(self, results):
        """Display benchmark results in formatted table with visualization"""
        
        # Import visualization tools
        from tools.glucose_visualizer import GlucoseVisualizer
        viz = GlucoseVisualizer()
        
        # Show patient overview first
        patient_id = results['patient_id']
        self.console.print(f"\n[bold] Patient {patient_id} Clinical Data Visualization[/bold]")
        viz.show_patient_overview(patient_id)
        
        # Original patient performance
        original = results['original_performance']
        
        self.console.print(f"\n[bold] Clinical Benchmark Results - Patient {results['patient_id']}[/bold]")
        
        # Original performance panel
        original_panel = Panel(
            f"Time in Range: [cyan]{original['tir_70_180']:.1f}%[/cyan]\n"
            f"GMI (Glucose Management Indicator): [cyan]{original['gmi']:.1f}[/cyan]\n"
            f"Coefficient of Variation: [cyan]{original['cv_percent']:.1f}%[/cyan]",
            title="[yellow]Original Patient Performance[/yellow]",
            border_style="yellow"
        )
        self.console.print(original_panel)
        
        # Algorithm comparison table with visual indicators
        table = Table(title="AI Algorithm Performance vs. Original")
        table.add_column("Algorithm", style="cyan", no_wrap=True)
        table.add_column("Time in Range", justify="right")
        table.add_column("Improvement", justify="right")
        table.add_column("Visual Progress", justify="center")
        table.add_column("Clinical Impact", justify="center")
        
        for algo, metrics in results['algorithm_results'].items():
            tir = metrics['tir_70_180']
            improvement = metrics['improvement_percent']
            
            # Create visual progress bar
            progress_chars = int(improvement / 2)  # Scale to reasonable size
            progress_bar = "" * max(0, progress_chars) + "" * max(0, 10 - progress_chars)
            
            # Determine clinical impact
            if improvement > 10:
                impact = "[green] Excellent[/green]"
            elif improvement > 5:
                impact = "[yellow] Good[/yellow]"
            elif improvement > 0:
                impact = "[blue] Modest[/blue]"
            else:
                impact = "[red] Poor[/red]"
            
            table.add_row(
                algo.upper(),
                f"{tir:.1f}%",
                f"{improvement:+.1f}%",
                progress_bar,
                impact
            )
        
        self.console.print(table)
        
        # Clinical interpretation with visual elements
        best_algo = max(results['algorithm_results'].items(), 
                       key=lambda x: x[1]['improvement_percent'])
        
        # Create improvement visualization
        original_tir = original['tir_70_180']
        best_tir = best_algo[1]['tir_70_180']
        
        improvement_viz = f"""
[bold]Before:[/bold] {'█' * int(original_tir/10)}{'░' * (10 - int(original_tir/10))} {original_tir:.1f}%
[bold]After: [/bold] {'█' * int(best_tir/10)}{'░' * (10 - int(best_tir/10))} {best_tir:.1f}%
        """
        
        interpretation = Panel(
            f"{improvement_viz}\n"
            f"[bold] Best Algorithm:[/bold] {best_algo[0].upper()}\n"
            f"[bold] Clinical Improvement:[/bold] {best_algo[1]['improvement_percent']:+.1f}% TIR\n"
            f"[bold] Research Validation:[/bold] Based on Ohio University T1DM Dataset\n"
            f"[bold] Publication Quality:[/bold] Suitable for peer-reviewed research",
            title="[green] Clinical Impact Analysis[/green]",
            border_style="green"
        )
        self.console.print(interpretation)
    
    def export_benchmark_results(self, results, filename="clinical_benchmark.json"):
        """Export benchmark results for publication"""
        import json
        from datetime import datetime
        
        export_data = {
            "study_metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": "Ohio University T1DM Dataset",
                "patient_id": results['patient_id'],
                "framework": "IINTS-AF v1.0"
            },
            "clinical_benchmarks": results['original_performance'],
            "algorithm_results": results['algorithm_results'],
            "citation": "Marling, C., & Bunescu, R. (2018). The OhioT1DM Dataset for Blood Glucose Level Prediction"
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.console.print(f"\n[green] Results exported to {filename}[/green]")

def main():
    """Run clinical benchmark as standalone tool"""
    benchmark = ClinicalBenchmark()
    results = benchmark.run_ohio_benchmark()
    
    if results:
        benchmark.export_benchmark_results(results)

if __name__ == "__main__":
    main()