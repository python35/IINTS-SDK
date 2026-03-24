from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from mdmp_core.audit import build_audit_payload


BUNDLE_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{sha256(canonical.encode('utf-8')).hexdigest()}"


def _read_optional_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_audit_bundle(
    *,
    report: dict[str, Any],
    validated_by: str = "unknown",
    lineage: dict[str, Any] | None = None,
    fingerprint: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lineage_payload = lineage or {}
    fingerprint_payload = fingerprint or {}
    registry_payload = registry or {}
    audit_payload = build_audit_payload(
        report=report,
        lineage=lineage_payload,
        fingerprint=fingerprint_payload,
        registry=registry_payload,
        validated_by=validated_by,
    )
    sources = {
        "report": report,
        "lineage": lineage_payload,
        "fingerprint": fingerprint_payload,
        "registry": registry_payload,
    }
    source_hashes = {name: _hash_payload(value) for name, value in sources.items()}
    return {
        "spec_version": "1.0",
        "mdmp_object": "signed_audit_bundle",
        "bundle_version": BUNDLE_VERSION,
        "created_utc": _now_iso(),
        "audit": audit_payload,
        "sources": sources,
        "source_hashes": source_hashes,
        "bundle_hash": _hash_payload(
            {
                "audit": audit_payload,
                "sources": sources,
                "source_hashes": source_hashes,
            }
        ),
    }


def verify_bundle_integrity(bundle_payload: dict[str, Any]) -> dict[str, Any]:
    sources = bundle_payload.get("sources", {})
    source_hashes = bundle_payload.get("source_hashes", {})
    checks: list[dict[str, Any]] = []
    if not isinstance(sources, dict):
        sources = {}
    if not isinstance(source_hashes, dict):
        source_hashes = {}

    for name in ("report", "lineage", "fingerprint", "registry"):
        source_value = sources.get(name, {})
        expected = source_hashes.get(name)
        actual = _hash_payload(source_value)
        checks.append(
            {
                "name": f"source_hash_{name}",
                "passed": expected == actual,
                "detail": f"expected={expected} actual={actual}",
            }
        )

    expected_bundle_hash = bundle_payload.get("bundle_hash")
    actual_bundle_hash = _hash_payload(
        {
            "audit": bundle_payload.get("audit", {}),
            "sources": sources,
            "source_hashes": source_hashes,
        }
    )
    checks.append(
        {
            "name": "bundle_hash",
            "passed": expected_bundle_hash == actual_bundle_hash,
            "detail": f"expected={expected_bundle_hash} actual={actual_bundle_hash}",
        }
    )
    passed = all(bool(c.get("passed")) for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "expected_bundle_hash": expected_bundle_hash,
        "actual_bundle_hash": actual_bundle_hash,
    }


def build_bundle_from_files(
    *,
    report_json: str | Path,
    validated_by: str = "unknown",
    lineage_json: str | Path | None = None,
    fingerprint_json: str | Path | None = None,
    registry_json: str | Path | None = None,
) -> dict[str, Any]:
    report = _read_optional_json(report_json)
    if not report:
        raise ValueError("report_json must be a valid JSON object")
    lineage = _read_optional_json(lineage_json)
    fingerprint = _read_optional_json(fingerprint_json)
    registry = _read_optional_json(registry_json)
    return build_audit_bundle(
        report=report,
        validated_by=validated_by,
        lineage=lineage,
        fingerprint=fingerprint,
        registry=registry,
    )
