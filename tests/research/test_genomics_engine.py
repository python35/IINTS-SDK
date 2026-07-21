from __future__ import annotations

import pandas as pd
import pytest

from iints.research.genomics_engine import GenomicsEngine, _extract_glucose_trace, _plotly_graph_objects


def test_extract_glucose_trace_accepts_current_simulator_columns() -> None:
    results = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [100.0, 105.5, 112.0],
        }
    )

    time, glucose = _extract_glucose_trace(results)

    assert time == [0.0, 5.0, 10.0]
    assert glucose == [100.0, 105.5, 112.0]


def test_extract_glucose_trace_accepts_legacy_columns() -> None:
    results = pd.DataFrame({"time": [0, 5], "glucose": [90, 95]})

    time, glucose = _extract_glucose_trace(results)

    assert time == [0.0, 5.0]
    assert glucose == [90.0, 95.0]


def test_extract_glucose_trace_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        _extract_glucose_trace(pd.DataFrame({"timestamp": [0], "value": [100]}))


def test_genomics_engine_known_mutation_metadata_is_deterministic() -> None:
    data = GenomicsEngine.evaluate_mutation("INSR", "v938m")

    assert data["scalar"] == 0.1
    assert data["residue"] == 938
    assert data["supported"] is True
    assert data["evidence_type"] == "illustrative_scenario_assumption"
    assert "not a clinical estimate" in data["desc"]


def test_unknown_mutation_never_infers_function_from_plddt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "iints.research.clinvar_engine.ClinVarEngine.lookup_variant",
        lambda *_args: {
            "found": False,
            "query_status": "not_found",
            "aggregate_classification": "not_available",
            "warning": "No exact ClinVar record.",
            "supports_quantitative_functional_scalar": False,
        },
    )
    monkeypatch.setattr(
        "iints.research.alphafold_engine.AlphaFoldGenomicsEngine.evaluate_plddt_impact",
        lambda *_args: {
            "plddt": 98.0,
            "confidence_band": "very_high",
            "supports_functional_inference": False,
            "conclusion": "Structural confidence only.",
        },
    )

    data = GenomicsEngine.evaluate_mutation("INSR", "A999V")

    assert data["scalar"] is None
    assert data["supported"] is False
    assert data["evidence_type"] == "structural_context_only"
    assert data["physiological_simulation_allowed"] is False
    assert "REJECTED" in data["desc"]


def test_pathogenic_clinvar_context_does_not_create_a_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "iints.research.clinvar_engine.ClinVarEngine.lookup_variant",
        lambda *_args: {
            "found": True,
            "query_status": "matched",
            "aggregate_classification": "pathogenic",
            "classifications": ["Pathogenic"],
            "supports_quantitative_functional_scalar": False,
        },
    )
    monkeypatch.setattr(
        "iints.research.alphafold_engine.AlphaFoldGenomicsEngine.evaluate_plddt_impact",
        lambda *_args: {
            "plddt": 91.0,
            "confidence_band": "very_high",
            "supports_functional_inference": False,
            "conclusion": "Local structure confidence only.",
        },
    )

    data = GenomicsEngine.evaluate_mutation("INSR", "A999V")

    assert data["scalar"] is None
    assert data["supported"] is False
    assert data["evidence_type"] == "classification_and_structural_context_only"
    assert data["clinvar_context"]["aggregate_classification"] == "pathogenic"


def test_unknown_mutation_is_not_simulated_without_functional_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        GenomicsEngine,
        "evaluate_mutation",
        staticmethod(
            lambda *_args: {
                "scalar": None,
                "supported": False,
                "desc": "No functional evidence.",
            }
        ),
    )

    with pytest.raises(ValueError, match="pLDDT is structural confidence"):
        GenomicsEngine.run_multi_scale_simulation("INSR", "A999V", tmp_path)


def test_plotly_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> object:
        if name == "plotly.graph_objects":
            raise ModuleNotFoundError(name)
        raise AssertionError(name)

    monkeypatch.setattr("iints.research.genomics_engine.importlib.import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="requires Plotly"):
        _plotly_graph_objects()
