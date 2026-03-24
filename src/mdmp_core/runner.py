from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DataContract


MDMP_PROTOCOL_VERSION = "1.0"
MDMP_SPEC_VERSION = "1.0"
MDMP_GRADE_ORDER = ("raw", "draft", "research_grade", "clinical_grade", "ai_ready")


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str
    failed_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "failed_rows": self.failed_rows,
        }


@dataclass(frozen=True)
class ValidationResult:
    created_utc: str
    protocol_version: str
    is_compliant: bool
    compliance_score: float
    grade: str
    contract_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    row_count: int
    checks: list[ValidationCheck]
    consent_verified: bool
    grade_reason: str
    consent_context: dict[str, Any]
    schema_name: str
    schema_version: str
    schema_industry: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": MDMP_SPEC_VERSION,
            "mdmp_object": "validation_report",
            "created_utc": self.created_utc,
            "protocol_version": self.protocol_version,
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "grade": self.grade,
            "contract_fingerprint_sha256": self.contract_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "row_count": self.row_count,
            "consent_verified": self.consent_verified,
            "grade_reason": self.grade_reason,
            "consent_context": self.consent_context,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "schema_industry": self.schema_industry,
            "checks": [c.to_dict() for c in self.checks],
        }


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    normalized = df.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for col in normalized.columns:
        series = normalized[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized[col] = series.astype("datetime64[ns]").astype("int64").astype(str)
        else:
            normalized[col] = series.astype(str).fillna("__NA__")
    hashed = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64, copy=False)
    digest = sha256()
    digest.update(",".join(normalized.columns).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _check_type(series: pd.Series, expected: str) -> bool:
    expected = expected.lower()
    if expected in {"float", "number", "numeric"}:
        return bool(pd.api.types.is_numeric_dtype(series))
    if expected in {"int", "integer"}:
        return bool(pd.api.types.is_integer_dtype(series))
    if expected in {"str", "string"}:
        return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
    if expected in {"datetime", "timestamp"}:
        return bool(pd.api.types.is_datetime64_any_dtype(series) or pd.to_datetime(series, errors="coerce").notna().mean() > 0.95)
    if expected in {"bool", "boolean"}:
        return bool(pd.api.types.is_bool_dtype(series))
    return True


def _parse_iso_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _grade(
    *,
    score: float,
    is_compliant: bool,
    consent_policy_ok: bool,
    consent_window_ok: bool,
) -> tuple[str, str]:
    if not is_compliant:
        return "raw", "non_compliant_schema_or_bounds"
    if not consent_policy_ok:
        return "research_grade", "consent_policy_not_verified"
    if not consent_window_ok:
        return "research_grade", "consent_window_not_valid"
    if score < 75.0:
        return "draft", "low_validation_score"
    if score < 90.0:
        return "research_grade", "medium_validation_score"
    if score < 95.0:
        return "clinical_grade", "high_score_but_not_ai_ready_threshold"
    return "ai_ready", "all_checks_passed_with_valid_consent"


def grade_meets_minimum(actual: str, minimum: str) -> bool:
    try:
        return MDMP_GRADE_ORDER.index(actual) >= MDMP_GRADE_ORDER.index(minimum)
    except ValueError:
        return False


class ContractRunner:
    def __init__(self, contract: DataContract) -> None:
        self.contract = contract

    def run(self, df: pd.DataFrame) -> ValidationResult:
        checks: list[ValidationCheck] = []

        missing = [c.name for c in self.contract.schema.columns if c.required and c.name not in df.columns]
        checks.append(
            ValidationCheck(
                name="schema_columns",
                passed=len(missing) == 0,
                detail="all required columns present" if not missing else f"missing: {missing}",
            )
        )

        type_failures: list[str] = []
        for col in self.contract.schema.columns:
            if col.name not in df.columns:
                continue
            if not _check_type(df[col.name], col.type):
                type_failures.append(f"{col.name}:{col.type}")
        checks.append(
            ValidationCheck(
                name="schema_types",
                passed=len(type_failures) == 0,
                detail="all type checks passed" if not type_failures else f"type mismatch: {type_failures}",
            )
        )

        range_failures = 0
        range_details: list[str] = []
        for col in self.contract.schema.columns:
            if col.bounds is None or col.name not in df.columns:
                continue
            lo, hi = col.bounds
            values = pd.to_numeric(df[col.name], errors="coerce")
            below = int((values < lo).fillna(False).sum())
            above = int((values > hi).fillna(False).sum())
            if below:
                range_failures += below
                range_details.append(f"{col.name}<{lo}:{below}")
            if above:
                range_failures += above
                range_details.append(f"{col.name}>{hi}:{above}")
        checks.append(
            ValidationCheck(
                name="value_bounds",
                passed=range_failures == 0,
                detail="all bounds satisfied" if range_failures == 0 else "; ".join(range_details),
                failed_rows=range_failures,
            )
        )

        consent_policy_ok = bool(self.contract.consent.ai_training_allowed and self.contract.consent.anonymized)
        now = datetime.now(timezone.utc)

        consent_date_valid = True
        consent_date_error: str | None = None
        if self.contract.consent.consent_date:
            try:
                consent_date_valid = _parse_iso_utc(self.contract.consent.consent_date) <= now
            except Exception:
                consent_date_valid = False
                consent_date_error = "invalid_consent_date_format"

        consent_expiry_valid = True
        consent_expiry_error: str | None = None
        if self.contract.consent.expiry:
            try:
                consent_expiry_valid = now <= _parse_iso_utc(self.contract.consent.expiry)
            except Exception:
                consent_expiry_valid = False
                consent_expiry_error = "invalid_expiry_format"

        consent_window_ok = consent_date_valid and consent_expiry_valid
        consent_ok = consent_policy_ok and consent_window_ok

        checks.append(
            ValidationCheck(
                name="consent_policy",
                passed=consent_policy_ok,
                detail=(
                    "ai_training_allowed + anonymized"
                    if consent_policy_ok
                    else "consent requirements failed (need ai_training_allowed=true and anonymized=true)"
                ),
            )
        )

        window_detail = "consent date/expiry window valid"
        if consent_date_error:
            window_detail = consent_date_error
        elif consent_expiry_error:
            window_detail = consent_expiry_error
        elif not consent_date_valid:
            window_detail = "consent_date is in the future"
        elif not consent_expiry_valid:
            window_detail = "consent has expired"

        checks.append(
            ValidationCheck(
                name="consent_window",
                passed=consent_window_ok,
                detail=window_detail,
            )
        )

        passed = sum(1 for c in checks if c.passed)
        score = round((passed / len(checks)) * 100.0, 2) if checks else 0.0
        is_compliant = all(c.passed for c in checks[:3])
        grade, grade_reason = _grade(
            score=score,
            is_compliant=is_compliant,
            consent_policy_ok=consent_policy_ok,
            consent_window_ok=consent_window_ok,
        )

        consent_context = {
            "ai_training_allowed": self.contract.consent.ai_training_allowed,
            "jurisdiction": self.contract.consent.jurisdiction,
            "anonymized": self.contract.consent.anonymized,
            "consent_date": self.contract.consent.consent_date,
            "expiry": self.contract.consent.expiry,
            "legal_basis": self.contract.consent.legal_basis,
            "policy_ok": consent_policy_ok,
            "window_ok": consent_window_ok,
            "consent_date_valid": consent_date_valid,
            "consent_expiry_valid": consent_expiry_valid,
        }

        return ValidationResult(
            created_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            protocol_version=MDMP_PROTOCOL_VERSION,
            is_compliant=is_compliant,
            compliance_score=score,
            grade=grade,
            contract_fingerprint_sha256=self.contract.fingerprint_sha256(),
            dataset_fingerprint_sha256=dataframe_fingerprint(df),
            row_count=int(len(df)),
            checks=checks,
            consent_verified=consent_ok,
            grade_reason=grade_reason,
            consent_context=consent_context,
            schema_name=self.contract.schema.name,
            schema_version=self.contract.schema.version,
            schema_industry=self.contract.schema.industry,
        )
