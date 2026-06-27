from pathlib import Path
import os
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.simulator import Simulator
from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus

# Mock database of known mutations for the hybrid approach
KNOWN_MUTATIONS = {
    # Severe mutations (Donohue syndrome, Rabson-Mendenhall)
    "V938M": {"scalar": 0.1, "desc": "Severe Donohue syndrome (90% loss of function)", "residue": 938},
    "R1174W": {"scalar": 0.15, "desc": "Severe insulin resistance (85% loss of function)", "residue": 1174},
    "A1135E": {"scalar": 0.2, "desc": "Rabson-Mendenhall syndrome (80% loss of function)", "residue": 1135},
    
    # Moderate mutations (Type A insulin resistance)
    "D1150E": {"scalar": 0.4, "desc": "Moderate Type A resistance (60% loss of function)", "residue": 1150},
    "P1178L": {"scalar": 0.6, "desc": "Mild Type A resistance (40% loss of function)", "residue": 1178},
    
    # Benign / Polymorphisms
    "H1058C": {"scalar": 0.95, "desc": "Benign polymorphism (5% loss of function)", "residue": 1058},
}

class GenomicsEngine:
    """
    Bridges the gap between molecular mutations (structural biology) 
    and patient-level glycemic simulations (systems biology).
    """

    @staticmethod
    def evaluate_mutation(gene: str, variant: str) -> Dict[str, Any]:
        """
        Translates a string like 'INSR V938M' into a functional scalar.
        In a full implementation, this queries the Ensembl VEP or AlphaGenome API.
        """
        variant = variant.upper().strip()
        if gene.upper() != "INSR":
            return {"scalar": 1.0, "desc": f"Gene {gene} not supported yet.", "residue": None}
            
        if variant in KNOWN_MUTATIONS:
            return KNOWN_MUTATIONS[variant]
            
        # Fallback for unknown mutations
        return {"scalar": 0.5, "desc": "Unknown mutation (Assumed 50% loss of function)", "residue": None}

    @staticmethod
    def run_multi_scale_simulation(gene: str, variant: str, out_dir: Path) -> Tuple[Path, Dict[str, Any]]:
        """
        Runs a comparative simulation: Healthy Baseline vs. Mutated Patient.
        Returns the path to the interactive HTML plot.
        """
        mutation_data = GenomicsEngine.evaluate_mutation(gene, variant)
        scalar = mutation_data["scalar"]
        
        # 1. Setup Healthy Baseline
        healthy_patient = HovorkaPatientModel(
            body_weight=70.0,
            initial_glucose=100.0,
            basal_insulin_rate=1.0,
            insulin_sensitivity=50.0,
            molecular_affinity_scalar=1.0
        )
        healthy_algo = FixedBasalBolus(basal_rate=1.0)
        healthy_sim = Simulator(healthy_patient, healthy_algo)
        healthy_sim.add_meal(time_min=60, carbs_g=60.0) # Standard meal test
        healthy_sim.run(duration_minutes=360)
        
        # 2. Setup Mutated Patient
        mutated_patient = HovorkaPatientModel(
            body_weight=70.0,
            initial_glucose=100.0,
            basal_insulin_rate=1.0,
            insulin_sensitivity=50.0,
            molecular_affinity_scalar=scalar
        )
        mutated_algo = FixedBasalBolus(basal_rate=1.0)
        mutated_sim = Simulator(mutated_patient, mutated_algo)
        mutated_sim.add_meal(time_min=60, carbs_g=60.0)
        mutated_sim.run(duration_minutes=360)
        
        # 3. Extract Data
        t_healthy = [r["time"] for r in healthy_sim.history]
        g_healthy = [r["glucose"] for r in healthy_sim.history]
        
        t_mutated = [r["time"] for r in mutated_sim.history]
        g_mutated = [r["glucose"] for r in mutated_sim.history]
        
        # 4. Generate Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_healthy, y=g_healthy, 
            mode='lines', 
            name='Healthy Baseline (100% Affinity)',
            line=dict(color='blue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=t_mutated, y=g_mutated, 
            mode='lines', 
            name=f'Mutated: {variant} ({int(scalar*100)}% Affinity)',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.add_vline(x=60, line_width=2, line_dash="dash", line_color="green", annotation_text="Meal (60g)")
        
        fig.update_layout(
            title=f"Multi-Scale Coupling: Impact of {gene} {variant} on Systemic Glycemia",
            xaxis_title="Time (minutes)",
            yaxis_title="Blood Glucose (mg/dL)",
            plot_bgcolor='white',
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"multiscale_{gene}_{variant}.html"
        fig.write_html(str(html_path))
        
        return html_path, mutation_data
