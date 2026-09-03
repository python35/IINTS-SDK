from __future__ import annotations

from pathlib import Path

from iints.core.formula_registry import (
    FORMULA_REGISTRY_VERSION,
    FORMULAS,
    formula_context_for_ai,
    formula_registry_dict,
    formula_registry_markdown,
    get_formula,
)


def test_formula_registry_has_exactly_16_static_formulas() -> None:
    assert FORMULA_REGISTRY_VERSION == "iints-formula-registry-v6"
    assert len(FORMULAS) == 16
    assert len({formula.formula_id for formula in FORMULAS}) == 16


def test_formula_specs_have_sources_units_and_implementation_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for formula in FORMULAS:
        assert formula.formula_id.startswith("F")
        assert formula.canonical_expression
        assert formula.latex_expression
        assert "```" not in formula.latex_expression
        assert formula.solved_or_runtime_form
        assert formula.units
        assert formula.literature_basis
        assert formula.evidence_class in {"canonical", "adapted", "heuristic"}
        assert not any("wikipedia.org" in source for source in formula.literature_basis)
        assert all(source.startswith(("https://", "http://")) for source in formula.literature_basis)
        assert "never derive" in formula.ai_policy.lower()
        for implementation in formula.implementation_paths:
            path_text = implementation.split(":", 1)[0]
            assert (repo_root / path_text).exists(), implementation


def test_formula_registry_dict_is_non_authoritative_for_ai() -> None:
    payload = formula_registry_dict()

    assert payload["formula_count"] == 16
    assert payload["ai_numeric_authority"] is False
    assert len(payload["formulas"]) == 16


def test_get_formula_returns_single_formula_by_id() -> None:
    formula = get_formula("F15_CGM_ISF_OBSERVATION")

    assert formula.title == "CGM blood-to-ISF lag and deterministic observation equation"
    assert formula.category == "sensor"


def test_formula_context_for_ai_is_compact_and_immutable() -> None:
    context = formula_context_for_ai()

    assert context["ai_formula_authority"] is False
    assert "immutable context" in str(context["instruction"]).lower()
    assert len(context["formulas"]) == 16
    assert "F01_BERGMAN_GLUCOSE_RHS" in {item["id"] for item in context["formulas"]}  # type: ignore[index]


def test_formula_registry_markdown_lists_all_formulas() -> None:
    markdown = formula_registry_markdown()

    assert "# IINTS-AF Formula Registry" in markdown
    assert markdown.count("$$") == 32
    assert markdown.count("Plain-text runtime notation") == 16
    assert markdown.count("Evidence class: `") == 16
    for formula in FORMULAS:
        assert formula.formula_id in markdown
        assert formula.latex_expression in markdown
