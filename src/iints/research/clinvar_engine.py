from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any


MYVARIANT_QUERY_URL = "https://myvariant.info/v1/query"

_GENE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,31}$")
_ONE_LETTER_PROTEIN = re.compile(r"^(?:P\.)?([A-Z*])(\d+)([A-Z*])$", re.IGNORECASE)
_THREE_LETTER_PROTEIN = re.compile(
    r"^(?:P\.)?([A-Z][A-Z]{2})(\d+)([A-Z][A-Z]{2}|TER)$",
    re.IGNORECASE,
)
_AA3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "*": "Ter",
}


def normalize_protein_variant(variant: str) -> str | None:
    """Return an HGVS-like protein change without inferring transcript context."""

    compact = re.sub(r"\s+", "", variant.strip())
    one_letter = _ONE_LETTER_PROTEIN.fullmatch(compact)
    if one_letter:
        reference, residue, alternate = one_letter.groups()
        if reference.upper() not in _AA3 or alternate.upper() not in _AA3:
            return None
        return f"p.{_AA3[reference.upper()]}{residue}{_AA3[alternate.upper()]}"

    three_letter = _THREE_LETTER_PROTEIN.fullmatch(compact)
    if not three_letter:
        return None
    reference, residue, alternate = three_letter.groups()
    return f"p.{reference.title()}{residue}{alternate.title()}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _classification_group(classifications: list[str]) -> str:
    labels = {value.strip().lower().replace("_", " ") for value in classifications if value.strip()}
    if not labels:
        return "not_provided"
    if any("conflict" in label for label in labels):
        return "conflicting"

    pathogenic = any(label in {"pathogenic", "likely pathogenic"} for label in labels)
    benign = any(label in {"benign", "likely benign"} for label in labels)
    if pathogenic and benign:
        return "conflicting"
    if pathogenic:
        return "pathogenic"
    if benign:
        return "benign"
    if any("uncertain" in label for label in labels):
        return "uncertain_significance"
    return "other"


class ClinVarEngine:
    """Read ClinVar classifications through MyVariant.info without effect inference.

    MyVariant.info is an annotation aggregator, not an NCBI classification
    authority. Returned ClinVar labels are variant-condition classifications;
    they are never converted into a quantitative receptor-function scalar.
    """

    @staticmethod
    def _fetch_query(url: str, *, timeout_seconds: float) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "IINTS-AF-SDK/1.5"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("MyVariant.info returned a non-object JSON payload.")
        return payload

    @classmethod
    def lookup_variant(
        cls,
        gene: str,
        variant: str,
        *,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Look up an exact protein change and return classification context."""

        normalized_gene = gene.strip().upper()
        normalized_protein = normalize_protein_variant(variant)
        common = {
            "source": "ClinVar via MyVariant.info",
            "source_database": "NCBI ClinVar",
            "transport": "MyVariant.info v1 query API",
            "gene": normalized_gene,
            "variant": variant.strip().upper(),
            "normalized_protein_change": normalized_protein,
            "supports_quantitative_functional_scalar": False,
        }
        if not _GENE_PATTERN.fullmatch(normalized_gene):
            return {
                **common,
                "query_status": "invalid_gene",
                "found": False,
                "aggregate_classification": "not_available",
                "warning": "The gene symbol is not valid for a bounded ClinVar lookup.",
            }
        if normalized_protein is None:
            return {
                **common,
                "query_status": "invalid_variant",
                "found": False,
                "aggregate_classification": "not_available",
                "warning": "Use a protein substitution such as V938M or p.Val938Met.",
            }

        query = (
            f"clinvar.gene.symbol:{normalized_gene} AND "
            f"clinvar.hgvs.protein:*{normalized_protein.removeprefix('p.')}*"
        )
        url = f"{MYVARIANT_QUERY_URL}?{urllib.parse.urlencode({'q': query, 'fields': 'clinvar', 'size': 20})}"
        try:
            payload = cls._fetch_query(url, timeout_seconds=timeout_seconds)
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            return {
                **common,
                "query_status": "lookup_error",
                "found": False,
                "aggregate_classification": "not_available",
                "warning": f"ClinVar lookup was unavailable: {exc}",
            }

        exact_hits: list[dict[str, Any]] = []
        for hit in _as_list(payload.get("hits")):
            if not isinstance(hit, dict):
                continue
            clinvar = hit.get("clinvar")
            if not isinstance(clinvar, dict):
                continue
            gene_payload = clinvar.get("gene")
            symbol = gene_payload.get("symbol") if isinstance(gene_payload, dict) else None
            hgvs = clinvar.get("hgvs")
            proteins = _as_list(hgvs.get("protein")) if isinstance(hgvs, dict) else []
            exact_protein = any(
                isinstance(protein, str)
                and (protein == normalized_protein or protein.endswith(f":{normalized_protein}"))
                for protein in proteins
            )
            if str(symbol).upper() == normalized_gene and exact_protein:
                exact_hits.append(hit)

        if not exact_hits:
            return {
                **common,
                "query_status": "not_found",
                "found": False,
                "aggregate_classification": "not_available",
                "classifications": [],
                "warning": (
                    "No exact ClinVar protein-change record was found. Absence is not evidence "
                    "of benignity or loss of function."
                ),
            }

        classifications: list[str] = []
        review_statuses: list[str] = []
        accessions: list[str] = []
        conditions: list[str] = []
        record_ids: list[str] = []
        for hit in exact_hits:
            if hit.get("_id") is not None:
                record_ids.append(str(hit["_id"]))
            clinvar = hit["clinvar"]
            for rcv in _as_list(clinvar.get("rcv")):
                if not isinstance(rcv, dict):
                    continue
                if rcv.get("clinical_significance"):
                    classifications.append(str(rcv["clinical_significance"]))
                if rcv.get("review_status"):
                    review_statuses.append(str(rcv["review_status"]))
                if rcv.get("accession"):
                    accessions.append(str(rcv["accession"]))
                condition = rcv.get("conditions")
                if isinstance(condition, dict) and condition.get("name"):
                    conditions.append(str(condition["name"]))

        return {
            **common,
            "query_status": "matched",
            "found": True,
            "aggregate_classification": _classification_group(classifications),
            "classifications": sorted(set(classifications)),
            "review_statuses": sorted(set(review_statuses)),
            "accessions": sorted(set(accessions)),
            "conditions": sorted(set(conditions)),
            "record_ids": sorted(set(record_ids)),
            "warning": (
                "ClinVar classification is condition-specific context and does not quantify "
                "retained receptor function or justify a simulation scalar."
            ),
        }
