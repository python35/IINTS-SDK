#!/usr/bin/env python3
"""
Terminal Glucose Visualization
Professional charts for Science-Expo presentation
"""

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from pathlib import Path

class GlucoseVisualizer:
    """Professional glucose visualization for terminal"""
    
    def __init__(self):
        self.console = Console()
        
    def create_ascii_chart(self, values, width=60, height=15):
        """Create ASCII chart for glucose values"""
        if not values or len(values) == 0:
            return ["No data available"]
            
        min_val = min(values)
        max_val = max(values)
        
        # Normalize values to chart height
        if max_val == min_val:
            normalized = [height // 2] * len(values)
        else:
            normalized = [int((v - min_val) / (max_val - min_val) * (height - 1)) for v in values]
        
        # Create chart grid
        chart = []
        for row in range(height - 1, -1, -1):
            line = ""
            for col in range(min(width, len(normalized))):
                if normalized[col] == row:
                    # Color code glucose values
                    if values[col] < 70:
                        line += "L"  # Low
                    elif values[col] > 180:
                        line += "H"  # High
                    else:
                        line += "N"  # Normal                elif normalized[col] > row:
                    line += "│"
                else:
                    line += " "
            
            # Add y-axis labels
            if row == height - 1:
                y_val = int(max_val)
            elif row == 0:
                y_val = int(min_val)
            elif row == height // 2:
                y_val = int((max_val + min_val) / 2)
            else:
                y_val = ""
                
            chart.append(f"{str(y_val):>3} {line}")
        
        return chart
    
    def show_patient_overview(self, patient_id):
        """Show comprehensive patient overview"""
        patient_path = Path(f"data_packs/public/ohio_t1dm/patient_{patient_id}")
        
        if not patient_path.exists():
            self.console.print(f"[red]Patient {patient_id} not found[/red]")
            return
            
        # Load data
        df = pd.read_csv(patient_path / "timeseries.csv")
        glucose_values = df['glucose_mg_dl'].tolist()
        
        # Calculate metrics
        tir = ((df['glucose_mg_dl'] >= 70) & (df['glucose_mg_dl'] <= 180)).mean() * 100
        mean_glucose = df['glucose_mg_dl'].mean()
        cv = df['glucose_mg_dl'].std() / mean_glucose * 100
        
        # Create overview table
        overview_table = Table(show_header=False, box=None)
        overview_table.add_column("Metric", style="cyan", width=20)
        overview_table.add_column("Value", style="white", width=15)
        overview_table.add_column("Status", style="white", width=15)
        
        # Time in Range
        tir_status = "Excellent" if tir > 70 else "Needs Improvement" if tir > 50 else "Poor"
        overview_table.add_row("Time in Range", f"{tir:.1f}%", tir_status)
        
        # Mean Glucose
        mg_status = "Good" if 70 <= mean_glucose <= 154 else "Elevated"
        overview_table.add_row("Mean Glucose", f"{mean_glucose:.1f} mg/dL", mg_status)
        
        # Coefficient of Variation
        cv_status = "Stable" if cv < 36 else "Variable"
        overview_table.add_row("Variability (CV)", f"{cv:.1f}%", cv_status)
        
        # Hypoglycemia
        hypo_percent = (df['glucose_mg_dl'] < 70).mean() * 100
        hypo_status = "Safe" if hypo_percent < 4 else "Concerning"
        overview_table.add_row("Hypoglycemia", f"{hypo_percent:.1f}%", hypo_status)
        
        # Hyperglycemia
        hyper_percent = (df['glucose_mg_dl'] > 250).mean() * 100
        hyper_status = "Controlled" if hyper_percent < 5 else "Uncontrolled"
        overview_table.add_row("Severe Highs", f"{hyper_percent:.1f}%", hyper_status)
        
        # Display overview
        overview_panel = Panel(
            overview_table,
            title=f"[bold blue]Ohio T1DM Patient {patient_id} - Clinical Overview[/bold blue]",
            border_style="blue"
        )
        
        # Create glucose chart
        chart_lines = self.create_ascii_chart(glucose_values[-120:])  # Last 120 readings
        chart_text = "\n".join(chart_lines)
        
        chart_panel = Panel(
            chart_text,
            title="[bold green]Glucose Trend (Last 10 Hours)[/bold green]",
            border_style="green"
        )
        
        # Display side by side
        columns = Columns([overview_panel, chart_panel], equal=True)
        self.console.print(columns)
        
        # Show insulin and meal events
        insulin_events = df[df['insulin_units'] > 0]
        meal_events = df[df['carbs_grams'] > 0]
        
        if not insulin_events.empty or not meal_events.empty:
            events_table = Table(title="Recent Events")
            events_table.add_column("Time", style="cyan")
            events_table.add_column("Event", style="white")
            events_table.add_column("Value", style="yellow")
            events_table.add_column("Glucose", style="white")
            
            # Add insulin events
            for _, row in insulin_events.tail(3).iterrows():
                time_str = row['timestamp'].split('T')[1][:5]
                events_table.add_row(
                    time_str,
                    "Insulin",
                    f"{row['insulin_units']:.1f}U",
                    f"{row['glucose_mg_dl']:.0f} mg/dL"
                )
            
            # Add meal events
            for _, row in meal_events.tail(3).iterrows():
                time_str = row['timestamp'].split('T')[1][:5]
                events_table.add_row(
                    time_str,
                    "Meal",
                    f"{row['carbs_grams']:.0f}g carbs",
                    f"{row['glucose_mg_dl']:.0f} mg/dL"
                )
            
            self.console.print(events_table)
    
    def compare_patients(self, patient_ids):
        """Compare multiple patients side by side"""
        comparison_table = Table(title="Ohio T1DM Patient Comparison")
        comparison_table.add_column("Metric", style="cyan")
        
        patient_data = {}
        
        for patient_id in patient_ids:
            patient_path = Path(f"data_packs/public/ohio_t1dm/patient_{patient_id}")
            if patient_path.exists():
                df = pd.read_csv(patient_path / "timeseries.csv")
                
                tir = ((df['glucose_mg_dl'] >= 70) & (df['glucose_mg_dl'] <= 180)).mean() * 100
                mean_glucose = df['glucose_mg_dl'].mean()
                cv = df['glucose_mg_dl'].std() / mean_glucose * 100
                hypo = (df['glucose_mg_dl'] < 70).mean() * 100
                
                patient_data[patient_id] = {
                    'tir': tir,
                    'mean_glucose': mean_glucose,
                    'cv': cv,
                    'hypo': hypo
                }
                
                comparison_table.add_column(f"Patient {patient_id}", justify="right")
        
        # Add rows
        comparison_table.add_row("Time in Range", *[f"{data['tir']:.1f}%" for data in patient_data.values()])
        comparison_table.add_row("Mean Glucose", *[f"{data['mean_glucose']:.0f} mg/dL" for data in patient_data.values()])
        comparison_table.add_row("Variability (CV)", *[f"{data['cv']:.1f}%" for data in patient_data.values()])
        comparison_table.add_row("Hypoglycemia", *[f"{data['hypo']:.1f}%" for data in patient_data.values()])
        
        self.console.print(comparison_table)
        
        # Show best/worst performers
        if patient_data:
            best_tir = max(patient_data.items(), key=lambda x: x[1]['tir'])
            worst_tir = min(patient_data.items(), key=lambda x: x[1]['tir'])
            
            summary_panel = Panel(
                f"[green]Best Control: Patient {best_tir[0]} ({best_tir[1]['tir']:.1f}% TIR)[/green]\n"
                f"[red]Needs Help: Patient {worst_tir[0]} ({worst_tir[1]['tir']:.1f}% TIR)[/red]\n"
                f"[blue]Population Range: {worst_tir[1]['tir']:.1f}% - {best_tir[1]['tir']:.1f}% TIR[/blue]",
                title="Population Summary",
                border_style="yellow"
            )
            self.console.print(summary_panel)

def main():
    """Demo glucose visualization"""
    viz = GlucoseVisualizer()
    
    # Show individual patient
    viz.show_patient_overview("559")
    
    print("\n" + "="*80 + "\n")
    
    # Compare patients
    viz.compare_patients(["559", "563"])

if __name__ == "__main__":
    main()