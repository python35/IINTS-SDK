from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Dict
import json

import yaml

from mdmp_core.registry import load_registry


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_optional(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _normalized_fp(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if text.startswith("sha256:") else f"sha256:{text}"


def build_audit_payload(
    *,
    report: Dict[str, Any],
    lineage: Dict[str, Any] | None = None,
    fingerprint: Dict[str, Any] | None = None,
    registry: Dict[str, Any] | None = None,
    validated_by: str = "unknown",
) -> Dict[str, Any]:
    lineage = lineage or {}
    fingerprint = fingerprint or {}
    registry = registry or {}

    dataset_fp = _normalized_fp(
        str(report.get("dataset_fingerprint_sha256") or "").strip()
        or str(fingerprint.get("fingerprint") or "").strip()
    )
    contract_fp = _normalized_fp(str(report.get("contract_fingerprint_sha256") or "").strip())

    registry_match = None
    if dataset_fp and isinstance(registry.get("records"), dict):
        row = registry["records"].get(dataset_fp)
        if isinstance(row, dict):
            registry_match = row

    timeline = []
    if report.get("created_utc"):
        timeline.append({"time": report["created_utc"], "event": "validation_completed"})
    if fingerprint.get("created"):
        timeline.append({"time": fingerprint["created"], "event": "fingerprint_recorded"})
    if fingerprint.get("expires"):
        timeline.append({"time": fingerprint["expires"], "event": "fingerprint_expires"})
    model = lineage.get("model", {}) if isinstance(lineage, dict) else {}
    if isinstance(model, dict) and model.get("training_date"):
        timeline.append({"time": model["training_date"], "event": "lineage_card_created"})
    if isinstance(registry_match, dict) and registry_match.get("published_at_utc"):
        timeline.append({"time": registry_match["published_at_utc"], "event": "registry_published"})
    timeline.sort(key=lambda row: str(row.get("time", "")))

    audit_core = {
        "spec_version": "1.0",
        "mdmp_object": "audit_report",
        "generated_utc": _now_iso(),
        "validated_by": validated_by,
        "protocol_version": report.get("protocol_version"),
        "dataset_fingerprint": dataset_fp,
        "contract_fingerprint": contract_fp,
        "grade": report.get("grade"),
        "grade_reason": report.get("grade_reason"),
        "compliance_score": report.get("compliance_score"),
        "consent_verified": report.get("consent_verified"),
        "consent_context": report.get("consent_context", {}),
        "staleness": report.get("staleness", {}),
        "lineage_status": model.get("lineage_status"),
        "lineage_stale_datasets": model.get("stale_datasets", []),
        "registry_status": registry_match.get("status") if isinstance(registry_match, dict) else None,
        "registry_visibility": registry_match.get("visibility") if isinstance(registry_match, dict) else None,
        "timeline": timeline,
    }
    signature = sha256(
        json.dumps(audit_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit_core["audit_fingerprint"] = f"sha256:{signature}"
    return audit_core


def build_audit_html(payload: Dict[str, Any], *, title: str = "MDMP Audit Report") -> str:
    timeline_rows = "\n".join(
        f"<tr><td>{escape(str(row.get('time', '')))}</td><td>{escape(str(row.get('event', '')))}</td></tr>"
        for row in payload.get("timeline", [])
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #0f172a; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: .9rem; }}
    .k {{ color: #475569; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
    .v {{ font-size: 1.1rem; font-weight: 650; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; text-align: left; padding: .55rem; }}
    pre {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: .8rem; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Grade</div><div class=\"v\">{escape(str(payload.get('grade', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Compliance Score</div><div class=\"v\">{escape(str(payload.get('compliance_score', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Consent Verified</div><div class=\"v\">{escape(str(payload.get('consent_verified', '-')))}</div></div>
    <div class=\"card\"><div class=\"k\">Lineage Status</div><div class=\"v\">{escape(str(payload.get('lineage_status', '-')))}</div></div>
  </div>
  <h2>Audit Timeline</h2>
  <table>
    <thead><tr><th>Timestamp (UTC)</th><th>Event</th></tr></thead>
    <tbody>{timeline_rows}</tbody>
  </table>
  <h2>Audit Fingerprint</h2>
  <pre>{escape(str(payload.get("audit_fingerprint", "")))}</pre>
  <h2>Raw Audit JSON</h2>
  <pre>{escape(json.dumps(payload, indent=2))}</pre>
</body>
</html>
"""


def build_audit_from_sources(
    *,
    report_json: Path,
    validated_by: str = "unknown",
    lineage_card: Path | None = None,
    fingerprint_json: Path | None = None,
    registry_json: Path | None = None,
) -> Dict[str, Any]:
    report = _load_optional(report_json)
    lineage = _load_optional(lineage_card)
    fingerprint = _load_optional(fingerprint_json)
    registry = load_registry(registry_json) if registry_json is not None else {}
    return build_audit_payload(
        report=report,
        lineage=lineage,
        fingerprint=fingerprint,
        registry=registry,
        validated_by=validated_by,
    )
