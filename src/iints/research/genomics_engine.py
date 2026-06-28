from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Tuple, cast

import pandas as pd

from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.simulator import Simulator, StressEvent


# Fast local mutation map used for deterministic demos and offline CI.
KNOWN_MUTATIONS: dict[str, dict[str, Any]] = {
    # Severe mutations (Donohue syndrome, Rabson-Mendenhall)
    "V938M": {"scalar": 0.1, "desc": "Severe Donohue syndrome (90% loss of function)", "residue": 938},
    "R1174W": {"scalar": 0.15, "desc": "Severe insulin resistance (85% loss of function)", "residue": 1174},
    "A1135E": {"scalar": 0.2, "desc": "Rabson-Mendenhall syndrome (80% loss of function)", "residue": 1135},
    # Moderate mutations (Type A insulin resistance)
    "D1150E": {"scalar": 0.4, "desc": "Moderate Type A resistance (60% loss of function)", "residue": 1150},
    "P1178L": {"scalar": 0.6, "desc": "Mild Type A resistance (40% loss of function)", "residue": 1178},
    # Benign / polymorphisms
    "H1058C": {"scalar": 0.95, "desc": "Benign polymorphism (5% loss of function)", "residue": 1058},
}


def _extract_glucose_trace(results: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Return time/glucose arrays from current or legacy simulator outputs."""

    time_column = "time_minutes" if "time_minutes" in results.columns else "time"
    glucose_column = "glucose_actual_mgdl" if "glucose_actual_mgdl" in results.columns else "glucose"
    missing = [column for column in (time_column, glucose_column) if column not in results.columns]
    if missing:
        raise ValueError(f"Simulator output is missing required columns: {missing}")
    return (
        [float(value) for value in results[time_column].tolist()],
        [float(value) for value in results[glucose_column].tolist()],
    )


def _plotly_graph_objects() -> Any:
    """Import Plotly only when the interactive graph feature is used."""

    try:
        return importlib.import_module("plotly.graph_objects")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The genomics multi-scale HTML plot requires Plotly. Install the "
            "research or desktop extra, for example: "
            'python -m pip install -U "iints-sdk-python35[research]"'
        ) from exc


class GenomicsEngine:
    """Bridge molecular mutation examples to patient-level glycemic simulations."""

    @staticmethod
    def evaluate_mutation(gene: str, variant: str) -> dict[str, Any]:
        """Translate a variant such as ``INSR V938M`` into a functional scalar."""

        variant = variant.upper().strip()
        if gene.upper() != "INSR":
            return {"scalar": 1.0, "desc": f"Gene {gene} not supported yet.", "residue": None}
        if variant in KNOWN_MUTATIONS:
            return dict(KNOWN_MUTATIONS[variant])
        return {"scalar": 0.5, "desc": "Unknown mutation (assumed 50% loss of function)", "residue": None}

    @staticmethod
    def run_multi_scale_simulation(
        gene: str,
        variant: str,
        out_dir: Path,
        *,
        duration_minutes: int = 360,
    ) -> Tuple[Path, dict[str, Any]]:
        """Run healthy-vs-mutated Hovorka simulations and write an HTML plot."""

        go = _plotly_graph_objects()
        normalized_variant = variant.upper().strip()
        mutation_data = GenomicsEngine.evaluate_mutation(gene, normalized_variant)
        scalar = float(mutation_data["scalar"])

        healthy_patient = HovorkaPatientModel(
            initial_glucose=100.0,
            basal_insulin_rate=1.0,
            insulin_sensitivity=50.0,
            molecular_affinity_scalar=1.0,
        )
        healthy_algo = FixedBasalBolus(
            {
                "fixed_basal_rate": 1.0,
                "carb_ratio": 10.0,
                "correction_factor": 50.0,
                "target_glucose": 120.0,
            }
        )
        healthy_sim = Simulator(healthy_patient, healthy_algo)  # type: ignore[arg-type]
        healthy_sim.add_stress_event(StressEvent(start_time=60, event_type="meal", value=60.0))
        healthy_results, _ = healthy_sim.run(duration_minutes=duration_minutes)

        mutated_patient = HovorkaPatientModel(
            initial_glucose=100.0,
            basal_insulin_rate=1.0,
            insulin_sensitivity=50.0,
            molecular_affinity_scalar=scalar,
        )
        mutated_algo = FixedBasalBolus(
            {
                "fixed_basal_rate": 1.0,
                "carb_ratio": 10.0,
                "correction_factor": 50.0,
                "target_glucose": 120.0,
            }
        )
        mutated_sim = Simulator(mutated_patient, mutated_algo)  # type: ignore[arg-type]
        mutated_sim.add_stress_event(StressEvent(start_time=60, event_type="meal", value=60.0))
        mutated_results, _ = mutated_sim.run(duration_minutes=duration_minutes)

        t_healthy, g_healthy = _extract_glucose_trace(cast(pd.DataFrame, healthy_results))
        t_mutated, g_mutated = _extract_glucose_trace(cast(pd.DataFrame, mutated_results))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t_healthy,
                y=g_healthy,
                mode="lines",
                name="Healthy baseline (100% affinity)",
                line=dict(color="blue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_mutated,
                y=g_mutated,
                mode="lines",
                name=f"Mutated: {normalized_variant} ({int(scalar * 100)}% affinity)",
                line=dict(color="red", width=3, dash="dash"),
            )
        )
        fig.add_vline(
            x=60,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text="Meal (60g)",
        )
        fig.update_layout(
            title=f"Multi-scale coupling: impact of {gene.upper()} {normalized_variant} on systemic glycemia",
            xaxis_title="Time (minutes)",
            yaxis_title="Blood glucose (mg/dL)",
            plot_bgcolor="white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"multiscale_{gene.upper()}_{normalized_variant}.html"
        fig.write_html(str(html_path))

        result_data = dict(mutation_data)
        result_data["html_path"] = str(html_path)
        return html_path, result_data
