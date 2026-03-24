from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
from urllib.parse import urlparse
import urllib.request

from mdmp_core.crypto import MDMPSigner, MDMPVerifier, normalize_unsigned_payload


REGISTRY_VERSION = "mdmp-registry-v0"
PUBLIC_BUNDLE_OBJECT = "registry_public_bundle"
PUBLIC_BUNDLE_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_fingerprint(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Fingerprint cannot be empty")
    if text.startswith("sha256:"):
        return text
    return f"sha256:{text}"


def _validate_remote_bundle_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Remote bundle URL must use https:// (or http:// if explicitly allowed)")
    if not parsed.netloc:
        raise ValueError("Remote bundle URL must include a host")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")
    if parsed.scheme == "http" and os.getenv("MDMP_ALLOW_INSECURE_HTTP", "0") != "1":
        raise ValueError("http:// URLs are blocked by default; set MDMP_ALLOW_INSECURE_HTTP=1 for local dev")
    return parsed.geturl()


def _default_registry() -> Dict[str, Any]:
    return {
        "version": REGISTRY_VERSION,
        "updated_utc": _now_iso(),
        "records": {},
    }


def build_public_bundle_payload(registry_path: Path) -> Dict[str, Any]:
    payload = load_registry(registry_path)
    records = payload.get("records", {})
    public_records = {
        fp: row
        for fp, row in records.items()
        if isinstance(row, dict) and str(row.get("visibility", "private")).lower() == "public"
    }
    return {
        "spec_version": "1.0",
        "mdmp_object": PUBLIC_BUNDLE_OBJECT,
        "bundle_version": PUBLIC_BUNDLE_VERSION,
        "version": "mdmp-federated-bundle-v0",
        "generated_utc": _now_iso(),
        "source_registry_version": payload.get("version"),
        "records": public_records,
    }


def _normalize_public_bundle_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid bundle format")

    records = payload.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("Bundle 'records' must be a mapping")

    mdmp_object = payload.get("mdmp_object")
    if mdmp_object not in {None, PUBLIC_BUNDLE_OBJECT}:
        raise ValueError("Bundle is not a registry public bundle")

    return {
        "spec_version": str(payload.get("spec_version", "1.0")),
        "mdmp_object": PUBLIC_BUNDLE_OBJECT,
        "bundle_version": str(payload.get("bundle_version", PUBLIC_BUNDLE_VERSION)),
        "version": str(payload.get("version", "mdmp-federated-bundle-v0")),
        "generated_utc": str(payload.get("generated_utc", _now_iso())),
        "source_registry_version": payload.get("source_registry_version"),
        "records": records,
    }


def _import_public_bundle_payload(
    registry_path: Path,
    bundle_payload: Dict[str, Any],
    *,
    source: str = "bundle",
) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    imported = 0
    for fp, row in bundle_payload["records"].items():
        if not isinstance(row, dict):
            continue
        registry["records"][fp] = {
            **row,
            "federated_source": source,
            "federated_imported_utc": _now_iso(),
            "updated_at_utc": _now_iso(),
        }
        imported += 1
    save_registry(registry_path, registry)
    return {"imported": imported, "registry": str(registry_path)}


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return _default_registry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return _default_registry()
    payload.setdefault("version", REGISTRY_VERSION)
    payload.setdefault("updated_utc", _now_iso())
    records = payload.get("records")
    payload["records"] = records if isinstance(records, dict) else {}
    return payload


def save_registry(path: Path, payload: Dict[str, Any]) -> None:
    payload["updated_utc"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def init_registry(path: Path) -> Dict[str, Any]:
    payload = _default_registry()
    save_registry(path, payload)
    return payload


def _extract_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    grade = str(
        report.get(
            "effective_grade",
            report.get("grade", report.get("mdmp_grade", "draft")),
        )
    )
    protocol = str(report.get("protocol_version", report.get("mdmp_protocol_version", "1.0")))
    dataset_fp_raw = report.get("dataset_fingerprint_sha256", "")
    if not dataset_fp_raw:
        raise ValueError("Report does not contain dataset fingerprint")
    dataset_fp = _normalize_fingerprint(str(dataset_fp_raw))

    staleness = report.get("staleness", {})
    if not isinstance(staleness, dict):
        staleness = {}

    status = str(staleness.get("status", "valid"))
    stale_reason = staleness.get("stale_reason")
    expires = staleness.get("expires") or report.get("expires")

    consent_context = report.get("consent_context", {})
    if not isinstance(consent_context, dict):
        consent_context = {}

    return {
        "fingerprint": dataset_fp,
        "grade": grade,
        "grade_reason": report.get("grade_reason"),
        "protocol_version": protocol,
        "schema_name": report.get("schema_name"),
        "schema_version": report.get("schema_version"),
        "schema_industry": report.get("schema_industry"),
        "contract_fingerprint": (
            _normalize_fingerprint(str(report["contract_fingerprint_sha256"]))
            if report.get("contract_fingerprint_sha256")
            else None
        ),
        "row_count": int(report.get("row_count", 0)),
        "consent_verified": bool(
            report.get("consent_verified", report.get("certified_for_medical_research", False))
        ),
        "consent_jurisdiction": consent_context.get("jurisdiction"),
        "consent_expiry": consent_context.get("expiry"),
        "compliance_score": float(report.get("compliance_score", 0.0)),
        "status": status,
        "stale_reason": stale_reason,
        "expires": expires,
    }


def upsert_record(
    registry_path: Path,
    *,
    fingerprint: Optional[str] = None,
    report: Optional[Dict[str, Any]] = None,
    grade: Optional[str] = None,
    source: Optional[str] = None,
    visibility: str = "private",
    used_in_models: Optional[List[str]] = None,
    expires: Optional[str] = None,
    status: Optional[str] = None,
    stale_reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if report is None and not fingerprint:
        raise ValueError("Provide fingerprint or report")

    base: Dict[str, Any] = {}
    if report is not None:
        base.update(_extract_from_report(report))
    if fingerprint is not None:
        base["fingerprint"] = _normalize_fingerprint(fingerprint)
    if grade is not None:
        base["grade"] = grade
    if expires is not None:
        base["expires"] = expires
    if status is not None:
        base["status"] = status
    if stale_reason is not None:
        base["stale_reason"] = stale_reason

    fp = _normalize_fingerprint(str(base.get("fingerprint", "")))

    payload = load_registry(registry_path)
    existing = payload["records"].get(fp, {})
    if not isinstance(existing, dict):
        existing = {}

    merged = {
        "fingerprint": fp,
        "grade": base.get("grade", existing.get("grade", "draft")),
        "grade_reason": base.get("grade_reason", existing.get("grade_reason")),
        "protocol_version": base.get("protocol_version", existing.get("protocol_version", "1.0")),
        "schema_name": base.get("schema_name", existing.get("schema_name")),
        "schema_version": base.get("schema_version", existing.get("schema_version")),
        "schema_industry": base.get("schema_industry", existing.get("schema_industry")),
        "contract_fingerprint": base.get("contract_fingerprint", existing.get("contract_fingerprint")),
        "row_count": int(base.get("row_count", existing.get("row_count", 0))),
        "consent_verified": bool(base.get("consent_verified", existing.get("consent_verified", False))),
        "consent_jurisdiction": base.get("consent_jurisdiction", existing.get("consent_jurisdiction")),
        "consent_expiry": base.get("consent_expiry", existing.get("consent_expiry")),
        "compliance_score": float(base.get("compliance_score", existing.get("compliance_score", 0.0))),
        "status": str(base.get("status", existing.get("status", "valid"))),
        "stale_reason": base.get("stale_reason", existing.get("stale_reason")),
        "expires": base.get("expires", existing.get("expires")),
        "visibility": visibility.strip().lower(),
        "source": source or existing.get("source"),
        "used_in_models": sorted(
            set((used_in_models or []) + list(existing.get("used_in_models", [])))
        ),
        "metadata": metadata or existing.get("metadata", {}),
        "published_at_utc": existing.get("published_at_utc", _now_iso()),
        "updated_at_utc": _now_iso(),
    }
    payload["records"][fp] = merged
    save_registry(registry_path, payload)
    return merged


def lookup_record(registry_path: Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    payload = load_registry(registry_path)
    return payload["records"].get(_normalize_fingerprint(fingerprint))


def list_records(
    registry_path: Path,
    *,
    grade: Optional[str] = None,
    visibility: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    payload = load_registry(registry_path)
    records = list(payload["records"].values())
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if grade and str(record.get("grade", "")).strip().lower() != grade.strip().lower():
            continue
        if visibility and str(record.get("visibility", "")).strip().lower() != visibility.strip().lower():
            continue
        if status and str(record.get("status", "")).strip().lower() != status.strip().lower():
            continue
        filtered.append(record)
    filtered.sort(key=lambda row: str(row.get("updated_at_utc", "")), reverse=True)
    return filtered[: max(0, int(limit))]


def export_public_bundle(registry_path: Path, output_path: Path) -> Dict[str, Any]:
    bundle = build_public_bundle_payload(registry_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def export_signed_public_bundle(
    registry_path: Path,
    output_path: Path,
    *,
    signer: MDMPSigner,
    expires_days: int | None = None,
) -> Dict[str, Any]:
    signed_bundle = signer.sign_card(
        build_public_bundle_payload(registry_path),
        expires_days=expires_days,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(signed_bundle, indent=2), encoding="utf-8")
    return signed_bundle


def import_public_bundle(registry_path: Path, bundle_path: Path, *, source: str = "bundle") -> Dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict):
        raise ValueError("Invalid bundle format")
    bundle_payload = normalize_unsigned_payload(bundle) if "signature" in bundle else bundle
    return _import_public_bundle_payload(
        registry_path,
        _normalize_public_bundle_payload(bundle_payload),
        source=source,
    )


def sync_public_bundle_from_url(
    registry_path: Path,
    url: str,
    *,
    source: str = "remote",
    public_key_path: Path | None = None,
    trust_store_path: Path | None = None,
) -> Dict[str, Any]:
    safe_url = _validate_remote_bundle_url(url)
    with urllib.request.urlopen(safe_url, timeout=30) as response:  # nosec B310
        payload = response.read().decode("utf-8")
    signed_bundle = json.loads(payload)
    if not isinstance(signed_bundle, dict):
        raise ValueError("Remote bundle is invalid")
    if "signature" not in signed_bundle:
        raise ValueError("Remote public bundle must be signed")

    verification = MDMPVerifier(public_key_path, trust_store_path=trust_store_path).verify(signed_bundle)
    if not verification.get("valid"):
        error = verification.get("error", "bundle_signature_invalid")
        raise ValueError(f"Remote public bundle verification failed: {error}")

    bundle_payload = _normalize_public_bundle_payload(normalize_unsigned_payload(signed_bundle))
    return _import_public_bundle_payload(registry_path, bundle_payload, source=source)
