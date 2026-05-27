from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict
import json


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_certificate(
    report: Dict[str, Any],
    *,
    issued_by: str,
    level: str = "research_grade",
    certificate_id: str | None = None,
) -> Dict[str, Any]:
    base = {
        "spec_version": "1.0",
        "mdmp_object": "certificate",
        "certificate_id": certificate_id or f"mdmp-cert-{_now_iso()}",
        "issued_utc": _now_iso(),
        "issued_by": issued_by,
        "level": level,
        "dataset_fingerprint": report.get("dataset_fingerprint_sha256"),
        "contract_fingerprint": report.get("contract_fingerprint_sha256"),
        "grade": report.get("effective_grade", report.get("grade")),
        "grade_reason": report.get("grade_reason"),
        "compliance_score": report.get("compliance_score"),
        "consent_verified": report.get("consent_verified"),
        "schema_version": report.get("schema_version"),
        "protocol_version": report.get("protocol_version"),
        "governance_profile": report.get("governance_profile"),
        "eu_ai_pact_readiness": report.get("eu_ai_pact_readiness"),
        "intended_use": report.get("intended_use", "research_only"),
    }
    signature = sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    base["signature_sha256"] = signature
    return base
