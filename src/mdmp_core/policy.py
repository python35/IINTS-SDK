from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import yaml

from mdmp_core.runner import grade_meets_minimum


POLICY_VERSION = "1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class PolicySpec:
    min_grade: str | None = "research_grade"
    require_consent_verified: bool = True
    require_not_stale: bool = True
    require_not_expired: bool = True
    require_signature_valid: bool = False
    allowed_jurisdictions: tuple[str, ...] = ()
    downgrade_grade_on_fail: str | None = "draft"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": POLICY_VERSION,
            "min_grade": self.min_grade,
            "require_consent_verified": self.require_consent_verified,
            "require_not_stale": self.require_not_stale,
            "require_not_expired": self.require_not_expired,
            "require_signature_valid": self.require_signature_valid,
            "allowed_jurisdictions": list(self.allowed_jurisdictions),
            "downgrade_grade_on_fail": self.downgrade_grade_on_fail,
        }


def parse_policy(payload: dict[str, Any]) -> PolicySpec:
    if not isinstance(payload, dict):
        raise ValueError("Policy must be a mapping")
    return PolicySpec(
        min_grade=(
            str(payload.get("min_grade")).strip()
            if payload.get("min_grade") is not None and str(payload.get("min_grade")).strip() != ""
            else None
        ),
        require_consent_verified=bool(payload.get("require_consent_verified", True)),
        require_not_stale=bool(payload.get("require_not_stale", True)),
        require_not_expired=bool(payload.get("require_not_expired", True)),
        require_signature_valid=bool(payload.get("require_signature_valid", False)),
        allowed_jurisdictions=tuple(str(x).strip() for x in payload.get("allowed_jurisdictions", []) if str(x).strip()),
        downgrade_grade_on_fail=(
            str(payload.get("downgrade_grade_on_fail")).strip()
            if payload.get("downgrade_grade_on_fail") is not None and str(payload.get("downgrade_grade_on_fail")).strip() != ""
            else None
        ),
    )


def load_policy(path: str | Path) -> PolicySpec:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw) or {}
    else:
        payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Policy file must contain a mapping")
    return parse_policy(payload)


def save_policy(path: str | Path, policy: PolicySpec) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = policy.to_dict()
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_policy() -> PolicySpec:
    return PolicySpec()


def evaluate_policy(report_payload: dict[str, Any], policy: PolicySpec) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    actual_grade = str(report_payload.get("effective_grade", report_payload.get("grade", "raw")))
    min_grade_ok = True
    if policy.min_grade:
        min_grade_ok = grade_meets_minimum(actual_grade, policy.min_grade)
    checks.append(
        {
            "name": "min_grade",
            "passed": bool(min_grade_ok),
            "detail": f"actual={actual_grade} minimum={policy.min_grade}",
        }
    )

    consent_verified = bool(report_payload.get("consent_verified", False))
    consent_ok = True if not policy.require_consent_verified else consent_verified
    checks.append(
        {
            "name": "consent_verified",
            "passed": bool(consent_ok),
            "detail": f"required={policy.require_consent_verified} actual={consent_verified}",
        }
    )

    staleness = report_payload.get("staleness", {})
    stale_status = str(staleness.get("status", "valid")).strip().lower()
    staleness_ok = True if not policy.require_not_stale else stale_status != "stale"
    checks.append(
        {
            "name": "not_stale",
            "passed": bool(staleness_ok),
            "detail": f"required={policy.require_not_stale} status={stale_status}",
        }
    )

    expired = bool(report_payload.get("expired", False))
    expiry_ok = True if not policy.require_not_expired else not expired
    checks.append(
        {
            "name": "not_expired",
            "passed": bool(expiry_ok),
            "detail": f"required={policy.require_not_expired} expired={expired}",
        }
    )

    signature_valid_raw = report_payload.get("signature_valid", report_payload.get("valid"))
    signature_valid = bool(signature_valid_raw) if signature_valid_raw is not None else False
    signature_ok = True if not policy.require_signature_valid else signature_valid
    checks.append(
        {
            "name": "signature_valid",
            "passed": bool(signature_ok),
            "detail": f"required={policy.require_signature_valid} actual={signature_valid}",
        }
    )

    jurisdiction = str((report_payload.get("consent_context", {}) or {}).get("jurisdiction", "")).strip()
    jurisdiction_ok = True
    if policy.allowed_jurisdictions:
        jurisdiction_ok = jurisdiction in set(policy.allowed_jurisdictions)
    checks.append(
        {
            "name": "allowed_jurisdiction",
            "passed": bool(jurisdiction_ok),
            "detail": f"allowed={list(policy.allowed_jurisdictions)} actual={jurisdiction}",
        }
    )

    passed = all(bool(c.get("passed")) for c in checks)
    failed = [c["name"] for c in checks if not bool(c.get("passed"))]
    return {
        "policy_version": POLICY_VERSION,
        "evaluated_utc": _now_iso(),
        "passed": passed,
        "failed_checks": failed,
        "checks": checks,
        "policy": policy.to_dict(),
    }


def apply_policy_effects(
    report_payload: dict[str, Any],
    policy_eval: dict[str, Any],
    policy: PolicySpec,
) -> tuple[str, str]:
    current_grade = str(report_payload.get("effective_grade", report_payload.get("grade", "raw")))
    if bool(policy_eval.get("passed")):
        return current_grade, str(report_payload.get("effective_grade_reason", "policy_pass"))
    downgrade = policy.downgrade_grade_on_fail
    if downgrade:
        return downgrade, f"policy_failed:{','.join(policy_eval.get('failed_checks', []))}"
    return current_grade, f"policy_failed_no_downgrade:{','.join(policy_eval.get('failed_checks', []))}"

