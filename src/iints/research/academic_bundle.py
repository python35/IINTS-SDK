"""FAIR, reviewable metadata bundles for completed IINTS research runs.

The exporter deliberately does not copy, upload, or modify experimental data.
It adds machine-readable RO-Crate metadata, a source selection snapshot, and a
small reproducibility audit alongside an existing run directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
from importlib import resources
import json
import mimetypes
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable

import yaml


RO_CRATE_VERSION = "1.2"
RO_CRATE_CONTEXT = f"https://w3id.org/ro/crate/{RO_CRATE_VERSION}/context"
RO_CRATE_PROFILE = f"https://w3id.org/ro/crate/{RO_CRATE_VERSION}"
ACADEMIC_BUNDLE_FORMAT_VERSION = "1.0"

SENSITIVE_HEADER_TOKENS = {
    "address",
    "birth_date",
    "date_of_birth",
    "dob",
    "email",
    "full_name",
    "medical_record_number",
    "mrn",
    "name",
    "patient_name",
    "phone",
    "telephone",
}

LICENSE_URLS = {
    "Apache-2.0": "https://spdx.org/licenses/Apache-2.0.html",
    "CC-BY-4.0": "https://spdx.org/licenses/CC-BY-4.0.html",
    "CC0-1.0": "https://spdx.org/licenses/CC0-1.0.html",
}


@dataclass(frozen=True)
class AcademicBundleResult:
    """Artifacts written by :func:`build_academic_bundle`."""

    run_dir: Path
    ro_crate_metadata: Path
    audit_json: Path
    sources_json: Path
    readme_md: Path
    artifact_count: int
    source_count: int
    readiness_status: str
    readiness_score_pct: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_sha(run_dir: Path) -> str:
    metadata = _safe_json(run_dir / "run_metadata.json")
    recorded = str(metadata.get("git_sha") or "").strip()
    if recorded and recorded != "unknown":
        return recorded
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=run_dir,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _sdk_version(run_dir: Path) -> str:
    recorded = str(_safe_json(run_dir / "run_metadata.json").get("sdk_version") or "").strip()
    if recorded and recorded != "unknown":
        return recorded
    try:
        return importlib_metadata.version("iints-sdk-python35")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _load_evidence_registry() -> dict[str, Any]:
    text = resources.files("iints.presets").joinpath("evidence_sources.yaml").read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("Bundled evidence_sources.yaml is malformed.")
    return payload


def _normalise_source_ids(source_ids: Iterable[str]) -> list[str]:
    return sorted({value.strip() for value in source_ids if value and value.strip()})


def _valid_orcid(orcid: str) -> bool:
    match = re.fullmatch(r"https://orcid\.org/(\d{4})-(\d{4})-(\d{4})-([\dX]{4})", orcid)
    if not match:
        return False
    compact = "".join(match.groups())
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    check_value = (12 - total % 11) % 11
    expected = "X" if check_value == 10 else str(check_value)
    return compact[-1] == expected


def _infer_source_ids(run_dir: Path, available_ids: set[str]) -> list[str]:
    """Conservatively associate core references visible in run artifacts.

    Inferred sources are labelled as associations, not proof that a publication
    validates the run or that every equation from the publication is present.
    """

    inferred: set[str] = set()
    metadata = _safe_json(run_dir / "run_metadata.json")
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    config_text = json.dumps(config, sort_keys=True).lower()

    if (run_dir / "results.csv").is_file():
        inferred.update({"ada_2026_glycemic_goals", "attd_2019_time_in_range"})
    if "hovorka" in config_text:
        inferred.add("hovorka_2004_nmpc_t1d")
    if "bergman" in config_text:
        inferred.add("bergman_1979_minimal_model")
    if any(path.suffix.lower() == ".pdf" and "agp" in path.name.lower() for path in run_dir.glob("*.pdf")):
        inferred.add("idc_2025_agp_report_overview")
    if (run_dir / "mechanistic_run_manifest.json").is_file():
        inferred.update({"sbml_2019_l3v2_core", "libroadrunner_2015"})

    recorded_ids = metadata.get("evidence_source_ids")
    if isinstance(recorded_ids, list):
        inferred.update(str(value) for value in recorded_ids)
    return sorted(inferred.intersection(available_ids))


def _select_sources(
    run_dir: Path,
    registry: dict[str, Any],
    source_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], str]:
    rows = [row for row in registry["sources"] if isinstance(row, dict)]
    by_id = {str(row.get("id")): row for row in rows if row.get("id")}
    requested = _normalise_source_ids(source_ids)
    method = "explicit"
    if not requested:
        requested = _infer_source_ids(run_dir, set(by_id))
        method = "conservative_auto_association"
    unknown = sorted(set(requested).difference(by_id))
    if unknown:
        raise ValueError(f"Unknown evidence source ID(s): {', '.join(unknown)}")
    return [dict(by_id[source_id]) for source_id in requested], method


def _iter_artifacts(run_dir: Path) -> list[Path]:
    artifacts: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(run_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative.name == "ro-crate-metadata.json":
            continue
        artifacts.append(path)
    return artifacts


def _csv_sensitive_headers(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeError, csv.Error):
        return []
    findings: list[str] = []
    for raw in header:
        normalised = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
        if normalised in SENSITIVE_HEADER_TOKENS:
            findings.append(raw.strip())
    return sorted(set(findings))


def _privacy_findings(run_dir: Path, artifacts: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in artifacts:
        relative = path.relative_to(run_dir).as_posix()
        lowered = relative.lower()
        if any(token in lowered for token in ("patient_name", "medical_record", "mrn", "date_of_birth")):
            findings.append(f"sensitive filename marker: {relative}")
        if path.suffix.lower() == ".csv":
            headers = _csv_sensitive_headers(path)
            if headers:
                findings.append(f"sensitive-looking CSV header(s) in {relative}: {', '.join(headers)}")
    return findings


def _check(
    key: str,
    passed: bool,
    severity: str,
    message: str,
) -> dict[str, Any]:
    return {"id": key, "passed": passed, "severity": severity, "message": message}


def _build_audit(
    run_dir: Path,
    artifacts: list[Path],
    sources: list[dict[str, Any]],
    *,
    creator_name: str | None,
    license_id: str,
    git_sha: str,
) -> dict[str, Any]:
    metadata = _safe_json(run_dir / "run_metadata.json")
    run_manifest = _safe_json(run_dir / "run_manifest.json")
    privacy_findings = _privacy_findings(run_dir, artifacts)
    checks = [
        _check("results_present", (run_dir / "results.csv").is_file(), "required", "Run contains results.csv."),
        _check("run_metadata_present", bool(metadata), "required", "Run metadata is readable JSON."),
        _check("seed_recorded", metadata.get("seed") is not None, "required", "Deterministic seed is recorded."),
        _check(
            "configuration_recorded",
            isinstance(metadata.get("config"), dict) and bool(metadata.get("config")),
            "required",
            "Simulation configuration is recorded.",
        ),
        _check("run_manifest_present", bool(run_manifest), "required", "Original run manifest is readable JSON."),
        _check("git_revision_recorded", git_sha != "unknown", "recommended", "Source revision is identifiable."),
        _check(
            "software_environment_recorded",
            bool(metadata.get("python_version"))
            and isinstance(metadata.get("dependencies"), list)
            and bool(metadata.get("dependencies")),
            "recommended",
            "Python runtime and dependency versions are recorded.",
        ),
        _check("evidence_sources_selected", bool(sources), "recommended", "At least one source is associated."),
        _check("creator_identified", bool((creator_name or "").strip()), "recommended", "Research creator is identified."),
        _check(
            "data_license_identified",
            bool(license_id.strip()) and license_id != "NOASSERTION",
            "recommended",
            "A reuse license for the run artifacts is identified.",
        ),
        _check("artifact_inventory_nonempty", bool(artifacts), "required", "At least one artifact can be hashed."),
        _check(
            "privacy_header_review",
            not privacy_findings,
            "review",
            "No obvious direct-identifier filename or CSV-header markers were found.",
        ),
    ]
    required_failed = [item for item in checks if item["severity"] == "required" and not item["passed"]]
    review_failed = [item for item in checks if item["severity"] in {"recommended", "review"} and not item["passed"]]
    passed_count = sum(bool(item["passed"]) for item in checks)
    score = round(passed_count / len(checks) * 100.0, 2) if checks else 0.0
    status = "incomplete" if required_failed else ("needs_review" if review_failed else "ready")
    return {
        "schema_version": ACADEMIC_BUNDLE_FORMAT_VERSION,
        "generated_at_utc": _utc_now(),
        "scope": ".",
        "status": status,
        "score_pct": score,
        "checks": checks,
        "privacy_findings": privacy_findings,
        "limitations": [
            "This is a metadata and reproducibility audit, not peer review or clinical validation.",
            "Source association does not prove that a publication validates this implementation or run.",
            "The privacy check inspects filenames and CSV headers only; a human must review data before sharing.",
            "IINTS-AF is research software and not a medical device or treatment system.",
        ],
    }


def _source_entity(source: dict[str, Any]) -> dict[str, Any]:
    doi = str(source.get("doi") or "").strip()
    url = str(source.get("url") or "").strip()
    identifier = f"https://doi.org/{doi}" if doi else (url or f"#source-{source['id']}")
    entity: dict[str, Any] = {
        "@id": identifier,
        "@type": "ScholarlyArticle",
        "name": str(source.get("title") or source["id"]),
        "identifier": str(source["id"]),
        "citation": str(source.get("citation") or ""),
        "description": str(source.get("rationale") or ""),
    }
    if doi:
        entity["sameAs"] = f"https://doi.org/{doi}"
    return entity


def _artifact_entity(path: Path, run_dir: Path) -> dict[str, Any]:
    relative = path.relative_to(run_dir).as_posix()
    media_type, _ = mimetypes.guess_type(path.name)
    entity: dict[str, Any] = {
        "@id": relative,
        "@type": "File",
        "name": path.name,
        "contentSize": str(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if media_type:
        entity["encodingFormat"] = media_type
    return entity


def _write_bundle_readme(
    path: Path,
    *,
    title: str,
    audit: dict[str, Any],
    sources: list[dict[str, Any]],
    artifacts: list[Path],
) -> None:
    lines = [
        f"# {title} - Academic Bundle",
        "",
        "This directory contains an IINTS-AF research run plus machine-readable metadata for review and reuse.",
        "It is research software output, not a medical record, medical device certificate, or treatment recommendation.",
        "",
        "## Readiness",
        "",
        f"- Status: `{audit['status']}`",
        f"- Audit score: `{audit['score_pct']:.2f}%`",
        f"- Artifacts inventoried: `{len(artifacts)}`",
        f"- Sources associated: `{len(sources)}`",
        "",
        "## Academic Metadata",
        "",
        "- `ro-crate-metadata.json`: RO-Crate 1.2 JSON-LD file and checksum inventory.",
        "- `academic_audit.json`: reproducibility, attribution, source, and privacy-review checks.",
        "- `academic_sources.json`: exact source-registry snapshot and selection method.",
        "- `run_metadata.json`: original seed, SDK/environment, and run configuration when available.",
        "- `run_manifest.json`: original run artifact manifest when available.",
        "",
        "## Review Before Sharing",
        "",
        "1. Resolve every failed or review audit item.",
        "2. Inspect data contents for direct or indirect identifiers.",
        "3. Confirm every selected source supports the exact claim made in a paper.",
        "4. Keep failed runs and exclusions visible in the study protocol.",
        "5. Archive the software version and dependency environment used for final analysis.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_academic_bundle(
    run_dir: Path,
    *,
    title: str | None = None,
    description: str | None = None,
    creator_name: str | None = None,
    creator_orcid: str | None = None,
    license_id: str = "NOASSERTION",
    source_ids: Iterable[str] = (),
) -> AcademicBundleResult:
    """Add a FAIR-oriented academic metadata layer to one completed run.

    No artifact is uploaded or copied. Existing files are read to calculate
    checksums; the four generated metadata files are written in ``run_dir``.
    """

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if creator_orcid and not _valid_orcid(creator_orcid):
        raise ValueError(
            "creator_orcid must be a valid canonical https://orcid.org/0000-0000-0000-0000 identifier."
        )
    if not license_id.strip():
        raise ValueError("license_id cannot be empty.")

    registry = _load_evidence_registry()
    sources, selection_method = _select_sources(run_dir, registry, source_ids)
    artifacts_before_metadata = _iter_artifacts(run_dir)
    git_sha = _git_sha(run_dir)
    audit = _build_audit(
        run_dir,
        artifacts_before_metadata,
        sources,
        creator_name=creator_name,
        license_id=license_id,
        git_sha=git_sha,
    )

    title = (title or run_dir.name or "IINTS-AF research run").strip()
    description = (
        description
        or "Reproducible IINTS-AF diabetes-technology simulation research run. Research only; not a medical device."
    ).strip()

    sources_path = run_dir / "academic_sources.json"
    audit_path = run_dir / "academic_audit.json"
    readme_path = run_dir / "ACADEMIC_BUNDLE.md"
    crate_path = run_dir / "ro-crate-metadata.json"

    registry_text = resources.files("iints.presets").joinpath("evidence_sources.yaml").read_text(encoding="utf-8")
    source_payload = {
        "schema_version": ACADEMIC_BUNDLE_FORMAT_VERSION,
        "generated_at_utc": _utc_now(),
        "registry_version": registry.get("version", 1),
        "registry_updated_utc": registry.get("updated_utc"),
        "registry_sha256": hashlib.sha256(registry_text.encode("utf-8")).hexdigest(),
        "selection_method": selection_method,
        "selection_warning": (
            "Automatically associated sources are candidates for review, not proof of implementation validity."
            if selection_method != "explicit"
            else "Explicit selection still requires claim-level human review."
        ),
        "sources": sources,
    }
    _write_json(sources_path, source_payload)
    _write_json(audit_path, audit)
    _write_bundle_readme(
        readme_path,
        title=title,
        audit=audit,
        sources=sources,
        artifacts=artifacts_before_metadata,
    )

    artifacts = _iter_artifacts(run_dir)
    file_entities = [_artifact_entity(path, run_dir) for path in artifacts]
    source_entities = [_source_entity(source) for source in sources]

    root: dict[str, Any] = {
        "@id": "./",
        "@type": "Dataset",
        "name": title,
        "description": description,
        "dateModified": _utc_now(),
        "license": (
            {"@id": LICENSE_URLS[license_id]}
            if license_id in LICENSE_URLS
            else license_id
        ),
        "hasPart": [{"@id": entity["@id"]} for entity in file_entities],
        "mentions": [{"@id": entity["@id"]} for entity in source_entities],
        "softwareRequirements": {"@id": "#iints-sdk"},
    }
    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "conformsTo": {"@id": RO_CRATE_PROFILE},
            "about": {"@id": "./"},
        },
        root,
        {
            "@id": "#iints-sdk",
            "@type": "SoftwareApplication",
            "name": "IINTS-AF SDK",
            "softwareVersion": _sdk_version(run_dir),
            "codeRepository": "https://github.com/python35/IINTS-SDK",
            "applicationCategory": "ResearchApplication",
            "operatingSystem": platform.platform(),
            "identifier": git_sha,
            "license": {"@id": LICENSE_URLS["Apache-2.0"]},
        },
    ]
    if creator_name and creator_name.strip():
        creator: dict[str, Any] = {"@id": "#creator", "@type": "Person", "name": creator_name.strip()}
        if creator_orcid:
            creator["identifier"] = creator_orcid
            creator["sameAs"] = creator_orcid
        graph.append(creator)
        root["creator"] = {"@id": "#creator"}
    graph.extend(file_entities)
    graph.extend(source_entities)
    _write_json(crate_path, {"@context": RO_CRATE_CONTEXT, "@graph": graph})

    return AcademicBundleResult(
        run_dir=run_dir,
        ro_crate_metadata=crate_path,
        audit_json=audit_path,
        sources_json=sources_path,
        readme_md=readme_path,
        artifact_count=len(artifacts),
        source_count=len(sources),
        readiness_status=str(audit["status"]),
        readiness_score_pct=float(audit["score_pct"]),
    )
