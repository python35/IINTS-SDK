"""Measured BindingDB affinity evidence for molecular-context research.

Binding measurements complement structural predictions; they do not prove
in-vivo pharmacodynamics and are never converted into IINTS patient-model
parameters automatically.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import ssl
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .external_models_common import safe_stem, timestamp_token, utc_now, write_json
from iints.utils.csv_safety import sanitize_csv_mapping


BINDINGDB_API_BASE = "https://bindingdb.org/rest/getLigandsByUniprots"
MAX_BINDINGDB_RESPONSE_BYTES = 25 * 1024 * 1024
BINDING_EVIDENCE_SCHEMA_VERSION = "1.0"
UNIPROT_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{5,15}$")
AFFINITY_PATTERN = re.compile(r"^\s*(<=|>=|<|>|=|~)?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


@dataclass(frozen=True)
class BindingEvidenceResult:
    output_dir: Path
    records_csv: Path
    evidence_json: Path
    report_md: Path
    uniprot_accession: str
    cutoff_nm: int
    record_count: int
    truncated: bool


def _validate_uniprot(value: str) -> str:
    accession = value.strip().upper()
    if not UNIPROT_PATTERN.fullmatch(accession):
        raise ValueError("UniProt accession must be 6-16 uppercase letters/digits with optional hyphens.")
    return accession


def _parse_affinity(value: str) -> tuple[str, float | None]:
    match = AFFINITY_PATTERN.match(value)
    if match is None:
        return "unparsed", None
    relation = match.group(1) or "="
    number = float(match.group(2))
    return relation, number if math.isfinite(number) else None


def _download_bindingdb(url: str, timeout_seconds: int) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "IINTS-AF-SDK research-only BindingDB connector",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds, context=ssl.create_default_context()) as response:
            payload = response.read(MAX_BINDINGDB_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"BindingDB returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach BindingDB with verified TLS: {exc.reason}") from exc
    if len(payload) > MAX_BINDINGDB_RESPONSE_BYTES:
        raise RuntimeError("BindingDB response exceeded the 25 MiB safety limit.")
    return payload


def _binding_records(payload: bytes) -> list[dict[str, Any]]:
    if not payload.strip():
        return []
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"BindingDB returned invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("BindingDB JSON root is not an object.")
    response = decoded.get("getLindsByUniprotsResponse") or decoded.get("getLigandsByUniprotsResponse")
    if response is None:
        raise RuntimeError("BindingDB JSON did not contain the expected UniProt response object.")
    if not isinstance(response, dict):
        raise RuntimeError("BindingDB UniProt response is not an object.")
    affinities = response.get("affinities", [])
    if affinities is None:
        return []
    if not isinstance(affinities, list) or not all(isinstance(row, dict) for row in affinities):
        raise RuntimeError("BindingDB affinities payload is not a list of objects.")
    return affinities


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    affinity_raw = str(record.get("affinity") or "").strip()
    relation, numeric_value = _parse_affinity(affinity_raw)
    return {
        "target_name": str(record.get("query") or "").strip(),
        "bindingdb_monomer_id": str(record.get("monomerid") or "").strip(),
        "smiles": str(record.get("smile") or "").strip(),
        "affinity_type": str(record.get("affinity_type") or "").strip(),
        "affinity_relation": relation,
        "affinity_value_nm": numeric_value,
        "affinity_as_reported": affinity_raw,
        "pmid": str(record.get("pmid") or "").strip(),
        "doi": str(record.get("doi") or "").strip(),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_relation: dict[str, int] = {}
    values: list[float] = []
    publications: set[str] = set()
    for record in records:
        affinity_type = str(record["affinity_type"] or "unknown")
        relation = str(record["affinity_relation"] or "unparsed")
        by_type[affinity_type] = by_type.get(affinity_type, 0) + 1
        by_relation[relation] = by_relation.get(relation, 0) + 1
        if isinstance(record["affinity_value_nm"], float):
            values.append(record["affinity_value_nm"])
        if record["doi"]:
            publications.add(str(record["doi"]))
        elif record["pmid"]:
            publications.add(f"PMID:{record['pmid']}")
    return {
        "record_count": len(records),
        "affinity_types": dict(sorted(by_type.items())),
        "relations": dict(sorted(by_relation.items())),
        "numeric_value_range_nm": {"minimum": min(values), "maximum": max(values)} if values else None,
        "unique_publication_count": len(publications),
    }


def query_bindingdb_uniprot(
    uniprot_accession: str,
    output_dir: Path,
    *,
    cutoff_nm: int = 10_000,
    max_records: int = 5_000,
    timeout_seconds: int = 30,
    _fetcher: Callable[[str, int], bytes] | None = None,
) -> BindingEvidenceResult:
    """Fetch and export measured affinities for one UniProt target."""

    accession = _validate_uniprot(uniprot_accession)
    if cutoff_nm < 1 or cutoff_nm > 1_000_000_000:
        raise ValueError("cutoff_nm must be between 1 and 1,000,000,000 nM.")
    if max_records < 1 or max_records > 100_000:
        raise ValueError("max_records must be between 1 and 100,000.")
    if timeout_seconds < 1 or timeout_seconds > 300:
        raise ValueError("timeout_seconds must be between 1 and 300.")
    query_url = BINDINGDB_API_BASE + "?" + urlencode(
        {"uniprot": accession, "cutoff": cutoff_nm, "response": "application/json"}
    )
    payload = (_fetcher or _download_bindingdb)(query_url, timeout_seconds)
    raw_records = _binding_records(payload)
    truncated = len(raw_records) > max_records
    records = [_normalise_record(record) for record in raw_records[:max_records]]

    output_root = output_dir.expanduser().resolve()
    run_dir = output_root / f"bindingdb_{safe_stem(accession.lower())}_{timestamp_token()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "binding_affinities.csv"
    fieldnames = [
        "target_name",
        "bindingdb_monomer_id",
        "smiles",
        "affinity_type",
        "affinity_relation",
        "affinity_value_nm",
        "affinity_as_reported",
        "pmid",
        "doi",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanitize_csv_mapping(record) for record in records)
    evidence_path = run_dir / "bindingdb_evidence.json"
    summary = _summary(records)
    write_json(
        evidence_path,
        {
            "schema_version": BINDING_EVIDENCE_SCHEMA_VERSION,
            "generated_at_utc": utc_now(),
            "research_only": True,
            "medical_device": False,
            "source": {
                "name": "BindingDB",
                "endpoint": BINDINGDB_API_BASE,
                "request_url": query_url,
                "uniprot_accession": accession,
                "cutoff_nm": cutoff_nm,
            },
            "record_count_before_limit": len(raw_records),
            "record_count_exported": len(records),
            "truncated": truncated,
            "csv_formula_protection": True,
            "summary": summary,
            "records": records,
            "limitations": [
                "Ki, Kd, IC50, and other assay endpoints are not interchangeable.",
                "Affinity measurements depend on assay design, construct, conditions, and curation.",
                "In-vitro affinity does not establish in-vivo efficacy, pharmacokinetics, or dosing.",
                "BindingDB records never alter IINTS patient parameters automatically.",
                "This evidence must not be used for diagnosis or treatment decisions.",
            ],
        },
    )
    report_path = run_dir / "BINDINGDB_REVIEW.md"
    report_path.write_text(
        "\n".join(
            [
                "# IINTS BindingDB Affinity Evidence",
                "",
                f"- UniProt target: `{accession}`",
                f"- API cutoff: `{cutoff_nm} nM`",
                f"- Records exported: `{len(records)}`",
                f"- Records available before local limit: `{len(raw_records)}`",
                f"- Truncated: `{'yes' if truncated else 'no'}`",
                f"- Affinity types: `{json.dumps(summary['affinity_types'], sort_keys=True)}`",
                "",
                "## Interpretation boundary",
                "",
                "BindingDB adds measured ligand-target evidence beside AlphaFold structure predictions. It does not "
                "turn structural confidence into binding affinity, and it does not make Ki, Kd, and IC50 equivalent.",
                "Assay conditions, protein construct, publication context, and censoring signs (`<` or `>`) must be "
                "reviewed record by record before quantitative use.",
                "",
                "No value in this bundle is mapped to insulin sensitivity, PK/PD, pump control, or treatment.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return BindingEvidenceResult(
        output_dir=run_dir,
        records_csv=csv_path,
        evidence_json=evidence_path,
        report_md=report_path,
        uniprot_accession=accession,
        cutoff_nm=cutoff_nm,
        record_count=len(records),
        truncated=truncated,
    )
