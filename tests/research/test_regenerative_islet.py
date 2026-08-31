from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest

from iints.research.regenerative_islet import (
    build_regenerative_evidence_plan,
    compare_regenerative_islet_proteomics,
    get_regenerative_protein_panel,
    load_regenerative_protein_panels,
)


def _write_comparison_dataset(
    path: Path,
    *,
    panel: str = "beta_cell_identity_and_function",
    test_source: str = "study-1",
    reference_source: str = "study-1",
) -> None:
    targets = get_regenerative_protein_panel(panel).targets
    rows: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        reference_value = 100.0 + target_index
        test_value = reference_value * (2.0 if target.gene_symbol == "INS" else 1.0)
        for replicate in range(3):
            rows.extend(
                [
                    {
                        "gene_symbol": target.gene_symbol,
                        "group": "sc_islet",
                        "sample_id": f"sc-{replicate}",
                        "value": test_value + replicate - 1,
                        "unit": "normalized_intensity",
                        "scale": "linear",
                        "source_id": test_source,
                        "batch_id": f"batch-{replicate}",
                    },
                    {
                        "gene_symbol": target.gene_symbol,
                        "group": "primary_islet",
                        "sample_id": f"primary-{replicate}",
                        "value": reference_value + replicate - 1,
                        "unit": "normalized_intensity",
                        "scale": "linear",
                        "source_id": reference_source,
                        "batch_id": f"batch-{replicate}",
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_bundled_regenerative_panels_cover_function_immunity_stress_and_safety() -> None:
    panels = load_regenerative_protein_panels()

    assert set(panels) == {
        "beta_cell_identity_and_function",
        "immune_visibility_and_evasion",
        "stress_survival_and_graft_support",
        "residual_pluripotency_and_growth_safety",
    }
    assert {target.gene_symbol for target in panels["beta_cell_identity_and_function"].targets} >= {
        "INS",
        "PDX1",
        "NKX6-1",
        "MAFA",
        "GCK",
        "ABCC8",
        "KCNJ11",
    }
    assert {target.gene_symbol for target in panels["residual_pluripotency_and_growth_safety"].targets} >= {
        "POU5F1",
        "NANOG",
        "SOX2",
        "MKI67",
    }


def test_evidence_plan_preserves_non_causal_boundaries() -> None:
    plan = build_regenerative_evidence_plan("immune_visibility_and_evasion")

    assert plan.scientific_boundaries["research_only"] is True
    assert plan.scientific_boundaries["automatic_physiology_mapping_allowed"] is False
    assert plan.scientific_boundaries["alphafold_is_structure_confidence_only"] is True
    assert "predicted_structure" not in plan.source_matrix or "AlphaFold Protein Structure Database" in plan.source_matrix["predicted_structure"]
    assert any("cure claim" in item for item in plan.forbidden_inferences)
    assert "immune_challenge_assay" in plan.source_matrix
    assert "tumorigenicity_assay" in plan.source_matrix


def test_unknown_panel_fails_with_available_choices() -> None:
    with pytest.raises(KeyError, match="unknown regenerative protein panel"):
        get_regenerative_protein_panel("magic_cure_score")


def test_registry_rejects_direct_protein_to_physiology_mapping(tmp_path) -> None:
    bundled_path = resources.files("iints").joinpath(
        "data", "regenerative_protein_panels.json"
    )
    payload = json.loads(bundled_path.read_text(encoding="utf-8"))
    payload["panels"]["beta_cell_identity_and_function"]["targets"][0][
        "direct_physiology_mapping"
    ] = True
    registry = tmp_path / "unsafe_registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="direct physiology mapping is forbidden"):
        load_regenerative_protein_panels(registry)


def test_registry_rejects_weakened_alphafold_boundary(tmp_path) -> None:
    bundled_path = resources.files("iints").joinpath(
        "data", "regenerative_protein_panels.json"
    )
    payload = json.loads(bundled_path.read_text(encoding="utf-8"))
    payload["scientific_boundaries"]["alphafold_is_structure_confidence_only"] = False
    registry = tmp_path / "unsafe_registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="alphafold_is_structure_confidence_only"):
        load_regenerative_protein_panels(registry)


def test_proteomics_comparator_writes_descriptive_evidence_bundle(tmp_path) -> None:
    dataset = tmp_path / "proteomics.csv"
    _write_comparison_dataset(dataset)

    result = compare_regenerative_islet_proteomics(
        dataset,
        tmp_path / "comparison",
        panel_keys=["beta_cell_identity_and_function"],
        normalization_note="Joint median normalization within study-1.",
        bootstrap_samples=100,
        seed=7,
    )

    assert result.status == "ready_for_descriptive_review"
    assert result.observed_target_count == result.target_count == 10
    assert result.comparison_csv.is_file()
    assert result.report_json.is_file()
    assert result.report_md.is_file()

    comparison = pd.read_csv(result.comparison_csv)
    insulin = comparison.loc[comparison["gene_symbol"] == "INS"].iloc[0]
    assert insulin["log2_difference"] == pytest.approx(1.0, abs=0.02)
    assert insulin["status"] == "descriptive_estimate"

    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["automatic_physiology_mapping_performed"] is False
    assert report["equivalence_claim_performed"] is False
    assert report["comparison_design"] == "within-source"
    assert report["input"]["sha256"]


def test_proteomics_comparator_marks_cross_source_design_for_review(tmp_path) -> None:
    dataset = tmp_path / "proteomics.csv"
    _write_comparison_dataset(
        dataset,
        test_source="sc-study",
        reference_source="primary-study",
    )

    result = compare_regenerative_islet_proteomics(
        dataset,
        tmp_path / "comparison",
        panel_keys=["beta_cell_identity_and_function"],
        normalization_note="Separate studies; no validated bridge normalization.",
        bootstrap_samples=0,
    )

    assert result.status == "review_required"
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["comparison_design"] == "cross-source"
    assert any("batch and platform effects" in warning for warning in report["warnings"])


def test_proteomics_comparator_rejects_incompatible_units(tmp_path) -> None:
    dataset = tmp_path / "proteomics.csv"
    _write_comparison_dataset(dataset)
    frame = pd.read_csv(dataset)
    frame.loc[
        (frame["gene_symbol"] == "INS") & (frame["group"] == "primary_islet"),
        "unit",
    ] = "copies_per_cell"
    frame.to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="incompatible units or scales"):
        compare_regenerative_islet_proteomics(
            dataset,
            tmp_path / "comparison",
            panel_keys=["beta_cell_identity_and_function"],
        )


def test_proteomics_comparator_rejects_peptide_level_pseudoreplication(tmp_path) -> None:
    dataset = tmp_path / "proteomics.csv"
    _write_comparison_dataset(dataset)
    frame = pd.read_csv(dataset)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="aggregate peptide rows first"):
        compare_regenerative_islet_proteomics(
            dataset,
            tmp_path / "comparison",
            panel_keys=["beta_cell_identity_and_function"],
        )
