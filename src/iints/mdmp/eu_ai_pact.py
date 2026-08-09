from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping

from .backend import mdmp_grade_meets_minimum


CORE_AI_PACT_CONTROLS = (
    "ai_governance_strategy",
    "high_risk_mapping",
    "ai_literacy",
)

HIGH_RISK_READINESS_CONTROLS = (
    "risk_management",
    "data_governance_and_provenance",
    "technical_documentation",
    "logging_and_traceability",
    "human_oversight",
    "transparency_and_limitations",
    "accuracy_robustness_cybersecurity",
    "post_deployment_monitoring",
)

EU_AI_PACT_CONTROL_DESCRIPTIONS = {
    "ai_governance_strategy": "Documented internal AI governance strategy and ownership.",
    "high_risk_mapping": "Mapping of systems that may fall into high-risk AI Act categories.",
    "ai_literacy": "AI literacy/training plan for people building or deploying the system.",
    "risk_management": "Continuous risk-management process with mitigations and residual-risk review.",
    "data_governance_and_provenance": "Dataset provenance, quality gates, consent/access notes, and lineage.",
    "technical_documentation": "Technical documentation sufficient for independent review.",
    "logging_and_traceability": "Audit logs, fingerprints, run manifests, and reproducibility evidence.",
    "human_oversight": "Human oversight boundaries and stop/review responsibilities.",
    "transparency_and_limitations": "Clear intended-use, non-clinical limitations, and user-facing caveats.",
    "accuracy_robustness_cybersecurity": "Accuracy, robustness, adversarial/data-poisoning, and cybersecurity checks.",
    "post_deployment_monitoring": "Monitoring plan for drift, incidents, regressions, and model updates.",
}


@dataclass(frozen=True)
class EUAIPactReadinessResult:
    status: str
    passed: bool
    score: float
    controls: Dict[str, Dict[str, Any]]
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    disclaimer: str = (
        "Research-only readiness review. This is not legal advice, not EU AI Act conformity assessment, "
        "and not medical-device certification."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "score": self.score,
            "controls": self.controls,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "required_actions": self.required_actions,
            "disclaimer": self.disclaimer,
        }


def _nested_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _has_control(payload: Mapping[str, Any], control: str) -> bool:
    for container_key in ("eu_ai_pact", "governance", "controls", "ai_governance"):
        container = _nested_mapping(payload, container_key)
        if bool(container.get(control)):
            return True
    return bool(payload.get(control))


def _mdmp_evidence(payload: Mapping[str, Any]) -> Dict[str, Any]:
    grade = str(payload.get("mdmp_grade", payload.get("grade", payload.get("effective_grade", "draft"))))
    raw_score = float(payload.get("compliance_score", 0.0) or 0.0)
    if not math.isfinite(raw_score) or not 0.0 <= raw_score <= 100.0:
        raise ValueError("compliance_score must be finite and between 0 and 100")
    # MDMP-Core historically emitted a 0..1 fraction while the bundled runner
    # emits a 0..100 percentage. Normalize both before applying one threshold.
    score = raw_score / 100.0 if raw_score > 1.0 else raw_score
    dataset_fp = bool(payload.get("dataset_fingerprint_sha256") or payload.get("dataset_fingerprint"))
    contract_fp = bool(payload.get("contract_fingerprint_sha256") or payload.get("contract_fingerprint"))
    row_count = int(payload.get("row_count", 0) or 0)
    return {
        "grade": grade,
        "score_fraction": score,
        "raw_score": raw_score,
        "dataset_fingerprint_present": dataset_fp,
        "contract_fingerprint_present": contract_fp,
        "row_count": row_count,
        "meets_research_grade": mdmp_grade_meets_minimum(grade, "research_grade"),
    }


def review_eu_ai_pact_readiness(
    payload: Mapping[str, Any],
    *,
    strict: bool = True,
) -> EUAIPactReadinessResult:
    """Review MDMP evidence against an EU AI Pact-inspired governance checklist.

    The checklist intentionally stays stricter than a normal MDMP pass: it asks
    whether the AI research artifact is documented, traceable, externally
    reviewable, and blocked from unsafe interpretation.
    """

    controls: dict[str, dict[str, Any]] = {}
    critical: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    mdmp = _mdmp_evidence(payload)

    required_controls = (*CORE_AI_PACT_CONTROLS, *HIGH_RISK_READINESS_CONTROLS) if strict else CORE_AI_PACT_CONTROLS
    for control in required_controls:
        passed = _has_control(payload, control)
        controls[control] = {
            "passed": passed,
            "description": EU_AI_PACT_CONTROL_DESCRIPTIONS[control],
        }
        if not passed:
            critical.append(f"Missing governance control: {control}.")
            actions.append(f"Add evidence for {control}: {EU_AI_PACT_CONTROL_DESCRIPTIONS[control]}")

    if not mdmp["dataset_fingerprint_present"]:
        critical.append("Missing dataset fingerprint.")
        actions.append("Run MDMP certification so data provenance is cryptographically traceable.")
    if not mdmp["contract_fingerprint_present"]:
        critical.append("Missing contract fingerprint.")
        actions.append("Attach the model-ready contract fingerprint to the evidence bundle.")
    if mdmp["row_count"] <= 0:
        critical.append("No certified rows are recorded in the MDMP payload.")
        actions.append("Certify a non-empty dataset before using it for AI training evidence.")
    if not mdmp["meets_research_grade"]:
        critical.append(f"MDMP grade '{mdmp['grade']}' does not meet research_grade.")
        actions.append("Resolve MDMP checks until the dataset is at least research_grade.")
    if mdmp["score_fraction"] < 0.95:
        warnings.append(
            f"Normalized MDMP compliance score is {mdmp['score_fraction']:.3f}; "
            "strict AI evidence target is >=0.950."
        )
        actions.append("Inspect failing/warning MDMP rows before training or promoting models.")

    controls["mdmp_traceability"] = {
        "passed": (
            mdmp["dataset_fingerprint_present"]
            and mdmp["contract_fingerprint_present"]
            and mdmp["row_count"] > 0
            and mdmp["meets_research_grade"]
        ),
        "description": "Dataset/contract fingerprints, row count, and MDMP grade are present.",
        "evidence": mdmp,
    }

    score = max(0.0, 1.0 - 0.08 * len(critical) - 0.03 * len(warnings))
    if critical:
        status = "blocked"
    elif warnings:
        status = "needs_review"
    else:
        status = "research_ready"
    return EUAIPactReadinessResult(
        status=status,
        passed=status == "research_ready",
        score=round(score, 4),
        controls=controls,
        critical_failures=critical,
        warnings=warnings,
        required_actions=actions,
    )
