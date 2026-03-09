from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd
import yaml

from iints.data.contracts import load_contract_yaml as _load_iints_contract_yaml
from iints.data.mdmp_visualizer import build_mdmp_dashboard_html as _build_iints_dashboard_html
from iints.data.runner import ContractRunner as _IintsContractRunner


BACKEND_BUILTIN = "iints"
BACKEND_MDMP = "mdmp_core"


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
    return os.getenv("IINTS_MDMP_BACKEND", BACKEND_BUILTIN).strip().lower()


def is_mdmp_available() -> bool:
    try:
        importlib.import_module("mdmp_core")
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_external_symbols() -> tuple[Any, Any, Any]:
    contracts_mod = importlib.import_module("mdmp_core.contracts")
    runner_mod = importlib.import_module("mdmp_core.runner")
    visualizer_mod = importlib.import_module("mdmp_core.visualizer")
    return (
        getattr(contracts_mod, "load_contract"),
        getattr(runner_mod, "ContractRunner"),
        getattr(visualizer_mod, "build_dashboard_html"),
    )


def get_backend() -> str:
    requested = _requested_backend()
    if requested in {BACKEND_MDMP, "mdmp", "external"}:
        if not is_mdmp_available():
            raise ImportError(
                "mdmp_core not found.\n"
                "Install with: pip install iints-sdk-python35[mdmp]\n"
                "or: pip install 'mdmp-protocol>=0.3.1'"
            )
        return BACKEND_MDMP
    if requested == "auto" and is_mdmp_available():
        return BACKEND_MDMP
    return BACKEND_BUILTIN


def active_mdmp_backend() -> str:
    return get_backend()


def load_mdmp_contract(path: Path) -> Any:
    backend = active_mdmp_backend()
    if backend == BACKEND_MDMP:
        load_external_contract, _, _ = _load_external_symbols()
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and ("streams" in payload or "processes" in payload):
            # Legacy IINTS contract format.
            return _load_iints_contract_yaml(path)
        try:
            return load_external_contract(path)
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
    if backend == BACKEND_MDMP:
        _, external_contract_runner, _ = _load_external_symbols()
        try:
            raw_result = external_contract_runner(contract).run(df)
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
    if active_mdmp_backend() == BACKEND_MDMP:
        _, _, external_dashboard_html = _load_external_symbols()
        return external_dashboard_html(report, title=title)
    return _build_iints_dashboard_html(report, title=title)
