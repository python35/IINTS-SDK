"""Engine for testing tissue-specific insulin resistance on pump algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.simulator import Simulator, StressEvent
from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus

console = Console()

class TissueStressor:
    """Runs comparative multi-scale simulations for tissue-specific insulin resistance."""

    @staticmethod
    def run_stress_test(
        muscle_scalar: float,
        liver_scalar: float,
        output_dir: Path
    ) -> tuple[Path, dict[str, Any]]:
        """
        Runs a simulation comparing Baseline vs Hepatic Resistance vs Peripheral Resistance.
        
        Args:
            muscle_scalar: The scalar for muscle sensitivity (0.0 to 1.0).
            liver_scalar: The scalar for liver sensitivity (0.0 to 1.0).
            output_dir: Where to save the Plotly HTML output.
            
        Returns:
            A tuple of (html_path, metadata_dict).
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise RuntimeError("Plotly is required for tissue stressor. Install with: pip install plotly")

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # We will simulate a standard meal scenario
        duration_minutes = 24 * 60
        basal_rate = 0.8
        
        # 1. Baseline
        baseline_patient = HovorkaPatientModel(
            basal_insulin_rate=basal_rate,
            muscle_sensitivity_scalar=1.0,
            liver_sensitivity_scalar=1.0
        )
        baseline_controller = FixedBasalBolus({"fixed_basal_rate": basal_rate, "carb_ratio": 10.0, "correction_factor": 50.0, "target_glucose": 120.0})
        baseline_sim = Simulator(patient_model=baseline_patient, algorithm=baseline_controller)  # type: ignore[arg-type]
        baseline_sim.add_stress_event(StressEvent(start_time=8 * 60, event_type="meal", value=60.0))
        baseline_sim.add_stress_event(StressEvent(start_time=13 * 60, event_type="meal", value=80.0))
        baseline_sim.add_stress_event(StressEvent(start_time=19 * 60, event_type="meal", value=70.0))
        baseline_results, _ = baseline_sim.run(duration_minutes=duration_minutes)
        
        # 2. Hepatic Resistance (Liver only)
        hepatic_patient = HovorkaPatientModel(
            basal_insulin_rate=basal_rate,
            muscle_sensitivity_scalar=1.0,
            liver_sensitivity_scalar=liver_scalar
        )
        hepatic_controller = FixedBasalBolus({"fixed_basal_rate": basal_rate, "carb_ratio": 10.0, "correction_factor": 50.0, "target_glucose": 120.0})
        hepatic_sim = Simulator(patient_model=hepatic_patient, algorithm=hepatic_controller)  # type: ignore[arg-type]
        hepatic_sim.add_stress_event(StressEvent(start_time=8 * 60, event_type="meal", value=60.0))
        hepatic_sim.add_stress_event(StressEvent(start_time=13 * 60, event_type="meal", value=80.0))
        hepatic_sim.add_stress_event(StressEvent(start_time=19 * 60, event_type="meal", value=70.0))
        hepatic_results, _ = hepatic_sim.run(duration_minutes=duration_minutes)

        # 3. Peripheral Resistance (Muscle only)
        peripheral_patient = HovorkaPatientModel(
            basal_insulin_rate=basal_rate,
            muscle_sensitivity_scalar=muscle_scalar,
            liver_sensitivity_scalar=1.0
        )
        peripheral_controller = FixedBasalBolus({"fixed_basal_rate": basal_rate, "carb_ratio": 10.0, "correction_factor": 50.0, "target_glucose": 120.0})
        peripheral_sim = Simulator(patient_model=peripheral_patient, algorithm=peripheral_controller)  # type: ignore[arg-type]
        peripheral_sim.add_stress_event(StressEvent(start_time=8 * 60, event_type="meal", value=60.0))
        peripheral_sim.add_stress_event(StressEvent(start_time=13 * 60, event_type="meal", value=80.0))
        peripheral_sim.add_stress_event(StressEvent(start_time=19 * 60, event_type="meal", value=70.0))
        peripheral_results, _ = peripheral_sim.run(duration_minutes=duration_minutes)

        # Plotting
        fig = go.Figure()

        time_axis = [t / 60.0 for t in baseline_results["time_minutes"]]

        fig.add_trace(go.Scatter(
            x=time_axis, y=baseline_results["glucose_actual_mgdl"],
            mode="lines", name="Baseline (100% / 100%)",
            line=dict(color="green", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis, y=hepatic_results["glucose_actual_mgdl"],
            mode="lines", name=f"Hepatic Resistance (Liver {int(liver_scalar*100)}%)",
            line=dict(color="orange", width=2, dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=time_axis, y=peripheral_results["glucose_actual_mgdl"],
            mode="lines", name=f"Peripheral Resistance (Muscle {int(muscle_scalar*100)}%)",
            line=dict(color="red", width=2, dash="dot")
        ))

        fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Hypo")
        fig.add_hline(y=180, line_dash="dot", line_color="orange", annotation_text="Hyper")

        fig.update_layout(
            title=f"Pump Algorithm Stress Test: GTEx Tissue-Specific Resistance",
            xaxis_title="Time (Hours)",
            yaxis_title="Blood Glucose (mg/dL)",
            template="plotly_white",
            height=600,
            hovermode="x unified"
        )

        html_path = output_dir / f"tissue_stress_M{int(muscle_scalar*100)}_L{int(liver_scalar*100)}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        
        console.print(f"[green]Saved Tissue Stressor plot to {html_path}[/green]")
        
        metadata = {
            "muscle": muscle_scalar,
            "liver": liver_scalar,
            "html_path": str(html_path)
        }
        return html_path, metadata
