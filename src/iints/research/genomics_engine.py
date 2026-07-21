from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Tuple, cast

import pandas as pd

from iints.core.algorithms.fixed_basal_bolus import FixedBasalBolus
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.core.simulator import Simulator, StressEvent


# Explicit scenario assumptions used for deterministic demonstrations. These
# retained-function scalars are not clinical estimates for the named variants.
KNOWN_MUTATIONS: dict[str, dict[str, Any]] = {
    "V938M": {"scalar": 0.10, "residue": 938},
    "R1174W": {"scalar": 0.15, "residue": 1174},
    "A1135E": {"scalar": 0.20, "residue": 1135},
    "D1150E": {"scalar": 0.40, "residue": 1150},
    "P1178L": {"scalar": 0.60, "residue": 1178},
    "H1058C": {"scalar": 0.95, "residue": 1058},
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
        """Return an explicit demo assumption plus external evidence context.

        Unknown variants deliberately receive no functional scalar. AlphaFold
        pLDDT and ClinVar classification cannot quantify retained receptor
        function, so callers must provide validated functional evidence before
        running a physiological comparison.
        """
        import re
        from iints.research.alphafold_engine import AlphaFoldGenomicsEngine
        from iints.research.clinvar_engine import ClinVarEngine

        normalized_gene = gene.upper().strip()
        variant = variant.upper().strip()

        # Check known mutations first for fast/offline execution
        if normalized_gene == "INSR" and variant in KNOWN_MUTATIONS:
            result = dict(KNOWN_MUTATIONS[variant])
            retained_percent = int(round(float(result["scalar"]) * 100))
            result.update(
                {
                    "supported": True,
                    "physiological_simulation_allowed": True,
                    "evidence_type": "illustrative_scenario_assumption",
                    "functional_scalar_provenance": "versioned SDK scenario assumption",
                    "desc": (
                        f"Illustrative INSR scenario assuming {retained_percent}% retained "
                        "receptor function. This is not a clinical estimate for the variant."
                    ),
                }
            )
            return result

        # Determine UniProt ID (INSR -> P06213)
        uniprot_id = "P06213" if normalized_gene == "INSR" else gene.strip()

        # Extract residue index from variant (e.g. V938M -> 938)
        match = re.search(r"\d+", variant)
        if not match:
            return {
                "scalar": None,
                "supported": False,
                "physiological_simulation_allowed": False,
                "evidence_type": "insufficient_evidence",
                "desc": "Unknown mutation format; no functional effect was inferred.",
                "residue": None,
            }

        residue_idx = int(match.group())

        clinvar_result = ClinVarEngine.lookup_variant(normalized_gene, variant)
        af_result = AlphaFoldGenomicsEngine.evaluate_plddt_impact(uniprot_id, residue_idx)

        description_parts: list[str] = []
        if clinvar_result.get("found"):
            description_parts.append(
                "ClinVar context: "
                f"{clinvar_result.get('aggregate_classification', 'not_available')}. "
                "This condition-specific classification is not a quantitative functional assay."
            )
        else:
            description_parts.append(str(clinvar_result.get("warning", "No ClinVar context available.")))

        if "error" in af_result:
            description_parts.append(f"AlphaFold structural lookup failed: {af_result['error']}")
            structural_context: dict[str, Any] | None = None
        else:
            description_parts.append(
                f"AlphaFold pLDDT {af_result['plddt']}: {af_result['conclusion']}"
            )
            structural_context = af_result

        description_parts.append(
            "REJECTED: no physiological effect is simulated without an explicit, "
            "quantitative functional scalar and provenance."
        )
        if clinvar_result.get("found") and structural_context is not None:
            evidence_type = "classification_and_structural_context_only"
        elif clinvar_result.get("found"):
            evidence_type = "clinical_classification_context_only"
        elif structural_context is not None:
            evidence_type = "structural_context_only"
        else:
            evidence_type = "insufficient_evidence"

        return {
            "scalar": None,
            "supported": False,
            "physiological_simulation_allowed": False,
            "evidence_type": evidence_type,
            "desc": " ".join(description_parts),
            "residue": residue_idx,
            "clinvar_context": clinvar_result,
            "structural_context": structural_context,
        }

    @staticmethod
    def run_multi_scale_simulation(
        gene: str,
        variant: str,
        out_dir: Path,
        *,
        duration_minutes: int = 360,
        seed: int = 42,
    ) -> Tuple[Path, dict[str, Any]]:
        """Run a reference-vs-assumption comparison and write an HTML plot."""

        normalized_variant = variant.upper().strip()
        mutation_data = GenomicsEngine.evaluate_mutation(gene, normalized_variant)
        raw_scalar = mutation_data.get("scalar")
        if raw_scalar is None:
            raise ValueError(
                f"No validated functional scalar is available for {gene.upper()} "
                f"{normalized_variant}. AlphaFold pLDDT is structural confidence, not "
                "mutation severity, and ClinVar classification is not a quantitative "
                "effect size; no physiological simulation was generated."
            )
        scalar = float(raw_scalar)
        go = _plotly_graph_objects()

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
        healthy_sim = Simulator(healthy_patient, healthy_algo, seed=seed)  # type: ignore[arg-type]
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
        mutated_sim = Simulator(mutated_patient, mutated_algo, seed=seed)  # type: ignore[arg-type]
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
                name="Reference scenario (100% retained function)",
                line=dict(color="blue", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t_mutated,
                y=g_mutated,
                mode="lines",
                name=f"Variant hypothesis: {normalized_variant} ({int(scalar * 100)}% retained)",
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
            title=(
                f"Illustrative multi-scale hypothesis: {gene.upper()} "
                f"{normalized_variant} (seed {seed})"
            ),
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
        result_data["seed"] = seed
        result_data["research_only"] = True
        result_data["functional_scalar_is_assumption"] = True
        return html_path, result_data
