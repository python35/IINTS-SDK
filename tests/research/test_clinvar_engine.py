from __future__ import annotations

import pytest

from iints.research.clinvar_engine import ClinVarEngine, normalize_protein_variant


def _hit(
    *,
    protein: str = "P06213:p.Ala2Gly",
    significance: str = "Benign",
    gene: str = "INSR",
) -> dict[str, object]:
    return {
        "_id": "chr19:g.7293898G>C",
        "clinvar": {
            "gene": {"symbol": gene},
            "hgvs": {"protein": [protein]},
            "rcv": {
                "accession": "RCV000261376",
                "clinical_significance": significance,
                "review_status": "criteria provided, multiple submitters, no conflicts",
                "conditions": {"name": "Donohue syndrome"},
            },
        },
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A2G", "p.Ala2Gly"),
        ("p.Ala2Gly", "p.Ala2Gly"),
        ("W10*", "p.Trp10Ter"),
        ("not-a-variant", None),
    ],
)
def test_normalize_protein_variant(raw: str, expected: str | None) -> None:
    assert normalize_protein_variant(raw) == expected


def test_clinvar_lookup_requires_an_exact_gene_and_protein_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "hits": [
            _hit(),
            _hit(protein="P06213:p.Arg7Trp", significance="Pathogenic"),
            _hit(gene="OTHER"),
        ]
    }
    monkeypatch.setattr(
        ClinVarEngine,
        "_fetch_query",
        staticmethod(lambda _url, *, timeout_seconds: payload),
    )

    result = ClinVarEngine.lookup_variant("INSR", "A2G")

    assert result["found"] is True
    assert result["query_status"] == "matched"
    assert result["aggregate_classification"] == "benign"
    assert result["classifications"] == ["Benign"]
    assert result["supports_quantitative_functional_scalar"] is False


def test_clinvar_pathogenic_label_never_becomes_a_function_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ClinVarEngine,
        "_fetch_query",
        staticmethod(
            lambda _url, *, timeout_seconds: {
                "hits": [_hit(significance="Pathogenic")]
            }
        ),
    )

    result = ClinVarEngine.lookup_variant("INSR", "A2G")

    assert result["aggregate_classification"] == "pathogenic"
    assert result["supports_quantitative_functional_scalar"] is False
    assert "does not quantify" in str(result["warning"])


def test_clinvar_absence_is_not_interpreted_as_benign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ClinVarEngine,
        "_fetch_query",
        staticmethod(lambda _url, *, timeout_seconds: {"hits": []}),
    )

    result = ClinVarEngine.lookup_variant("INSR", "V938M")

    assert result["found"] is False
    assert result["query_status"] == "not_found"
    assert result["aggregate_classification"] == "not_available"
    assert "not evidence" in str(result["warning"])


def test_clinvar_lookup_failure_is_distinct_from_no_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_url: str, *, timeout_seconds: float) -> dict[str, object]:
        raise TimeoutError("offline")

    monkeypatch.setattr(ClinVarEngine, "_fetch_query", staticmethod(fail))

    result = ClinVarEngine.lookup_variant("INSR", "A2G")

    assert result["found"] is False
    assert result["query_status"] == "lookup_error"
    assert result["aggregate_classification"] == "not_available"
