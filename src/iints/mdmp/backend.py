from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd
import yaml

from iints.data.contracts import load_contract_yaml as _load_iints_contract_yaml
from iints.data.mdmp_visualizer import build_mdmp_dashboard_html as _build_iints_dashboard_html
from iints.data.runner import ContractRunner as _IintsContractRunner


try:  # Optional standalone MDMP backend
    from mdmp_core.contracts import load_contract as _load_external_contract
    from mdmp_core.runner import ContractRunner as _ExternalContractRunner
    from mdmp_core.visualizer import build_dashboard_html as _build_external_dashboard_html

    _EXTERNAL_MDMP_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional install
    _EXTERNAL_MDMP_AVAILABLE = False


MDMP_GRADE_ORDER = ("raw", "draft", "research_grade", "clinical_grade", "ai_ready")


@dataclass(frozen=True)
class MDMPCheckResult:
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
class MDMPValidationResult:
    is_compliant: bool
    compliance_score: float
    mdmp_grade: str
    mdmp_protocol_version: str
    certified_for_medical_research: bool
    contract_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    row_count: int
    checks: List[MDMPCheckResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "mdmp_grade": self.mdmp_grade,
            "mdmp_protocol_version": self.mdmp_protocol_version,
            "certified_for_medical_research": self.certified_for_medical_research,
            "contract_fingerprint_sha256": self.contract_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "row_count": self.row_count,
            "checks": [check.to_dict() for check in self.checks],
        }


def mdmp_grade_meets_minimum(actual_grade: str, minimum_grade: str) -> bool:
    try:
        actual_idx = MDMP_GRADE_ORDER.index(actual_grade)
        minimum_idx = MDMP_GRADE_ORDER.index(minimum_grade)
    except ValueError:
        return False
    return actual_idx >= minimum_idx


def _normalize_grade(grade: str) -> str:
    value = (grade or "").strip().lower()
    if value in MDMP_GRADE_ORDER:
        return value
    # Backward compatibility with legacy/report variants
    aliases = {
        "clinical": "clinical_grade",
        "research": "research_grade",
    }
    return aliases.get(value, "draft")


def _requested_backend() -> str:
    return os.getenv("IINTS_MDMP_BACKEND", "iints").strip().lower()


def active_mdmp_backend() -> str:
    requested = _requested_backend()
    if requested in {"mdmp", "mdmp_core", "external"} and _EXTERNAL_MDMP_AVAILABLE:
        return "mdmp_core"
    if requested == "auto" and _EXTERNAL_MDMP_AVAILABLE:
        return "mdmp_core"
    return "iints"


def load_mdmp_contract(path: Path) -> Any:
    backend = active_mdmp_backend()
    if backend == "mdmp_core":
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and ("streams" in payload or "processes" in payload):
            # Legacy IINTS contract format.
            return _load_iints_contract_yaml(path)
        try:
            return _load_external_contract(path)
        except Exception:
            # Backward compatibility: legacy IINTS contract format.
            return _load_iints_contract_yaml(path)
    return _load_iints_contract_yaml(path)


def _normalize_checks(checks: Iterable[Any]) -> List[MDMPCheckResult]:
    rows: List[MDMPCheckResult] = []
    for check in checks:
        rows.append(
            MDMPCheckResult(
                name=str(getattr(check, "name", "")),
                passed=bool(getattr(check, "passed", False)),
                detail=str(getattr(check, "detail", "")),
                failed_rows=int(getattr(check, "failed_rows", 0) or 0),
            )
        )
    return rows


def run_mdmp_validation(
    contract: Any,
    df: pd.DataFrame,
    *,
    apply_builtin_transforms: bool = True,
) -> MDMPValidationResult:
    backend = active_mdmp_backend()
    if backend == "mdmp_core":
        try:
            raw_result = _ExternalContractRunner(contract).run(df)
            grade = _normalize_grade(str(getattr(raw_result, "grade", "draft")))
            checks = _normalize_checks(getattr(raw_result, "checks", []))
            return MDMPValidationResult(
                is_compliant=bool(getattr(raw_result, "is_compliant", False)),
                compliance_score=float(getattr(raw_result, "compliance_score", 0.0)),
                mdmp_grade=grade,
                mdmp_protocol_version=str(getattr(raw_result, "protocol_version", "1.0")),
                certified_for_medical_research=grade in {"clinical_grade", "ai_ready"},
                contract_fingerprint_sha256=str(getattr(raw_result, "contract_fingerprint_sha256", "")),
                dataset_fingerprint_sha256=str(getattr(raw_result, "dataset_fingerprint_sha256", "")),
                row_count=int(getattr(raw_result, "row_count", len(df))),
                checks=checks,
            )
        except Exception:
            # Backward compatibility for legacy contracts and mixed environments.
            pass

    raw_result = _IintsContractRunner(contract).run(
        df,
        apply_builtin_transforms=apply_builtin_transforms,
    )
    checks = _normalize_checks(raw_result.checks)
    return MDMPValidationResult(
        is_compliant=raw_result.is_compliant,
        compliance_score=raw_result.compliance_score,
        mdmp_grade=_normalize_grade(raw_result.mdmp_grade),
        mdmp_protocol_version=raw_result.mdmp_protocol_version,
        certified_for_medical_research=raw_result.certified_for_medical_research,
        contract_fingerprint_sha256=raw_result.contract_fingerprint_sha256,
        dataset_fingerprint_sha256=raw_result.dataset_fingerprint_sha256,
        row_count=raw_result.row_count,
        checks=checks,
    )


def build_mdmp_dashboard_html(report: dict[str, Any], *, title: str) -> str:
    if active_mdmp_backend() == "mdmp_core":
        return _build_external_dashboard_html(report, title=title)
    return _build_iints_dashboard_html(report, title=title)
