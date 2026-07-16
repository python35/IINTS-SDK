from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

RESEARCH_ONLY_NOTICE = (
    "IINTS-AF is research and education software only. It is not a medical device, "
    "not a treatment recommendation system, and must not be used for diagnosis, "
    "insulin/glucagon dosing, treatment decisions, or real-time patient care."
)

_POLICY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "patient_specific_dose_instruction",
        re.compile(
            r"\b(?:take|give|inject|administer|deliver|bolus|dose)\s+"
            r"\d+(?:\.\d+)?\s*(?:u|unit|units|iu|mg|mcg)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "patient_specific_adjustment_instruction",
        re.compile(
            r"\b(?:increase|decrease|raise|lower|change|set)\s+"
            r"(?:the\s+)?(?:basal|bolus|insulin|glucagon|dose|correction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "direct_treatment_advice",
        re.compile(
            r"\b(?:you|the\s+patient)\s+should\s+"
            r"(?:take|inject|administer|increase|decrease|change|correct|treat)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "diagnosis_or_clinical_decision",
        re.compile(
            r"\b(?:diagnose|diagnosis\s+is|clinical\s+decision|treatment\s+plan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regulatory_overclaim",
        re.compile(
            r"\b(?:ce\s+marked|mdr\s+certified|approved\s+medical\s+device|safe\s+for\s+patient\s+use)\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class PolicyGuardResult:
    """Result of checking generated text against the research-only boundary."""

    allowed: bool
    text: str
    violations: tuple[str, ...] = ()


def scan_text_for_policy_violations(text: str) -> tuple[str, ...]:
    """Return policy violation labels found in generated text.

    This is deliberately conservative and intended for post-generation AI text,
    not for scanning documentation that explains the rules themselves.
    """

    found: list[str] = []
    for label, pattern in _POLICY_PATTERNS:
        if pattern.search(text or ""):
            found.append(label)
    return tuple(found)


def guard_ai_output(text: str, *, source: str = "local_ai") -> PolicyGuardResult:
    """Block local AI output that crosses the research-only boundary.

    The SDK may use local LLMs to explain research artifacts, but generated text
    must not become patient-specific clinical advice. When a violation is found,
    the original text is replaced with a safe audit note instead of being shown as
    authoritative output.
    """

    cleaned = (text or "").strip()
    violations = scan_text_for_policy_violations(cleaned)
    if not violations:
        return PolicyGuardResult(allowed=True, text=cleaned, violations=())

    blocked = "\n".join(
        [
            "# Local AI Output Blocked by IINTS Research Policy",
            "",
            RESEARCH_ONLY_NOTICE,
            "",
            f"Source: `{source}`",
            "",
            "The local AI response was not displayed because it appeared to cross the research-only boundary.",
            "Detected policy signals:",
            *[f"- `{violation}`" for violation in violations],
            "",
            "Use deterministic SDK reports, MDMP certificates, run manifests, and reviewer judgement as the source of truth.",
        ]
    )
    return PolicyGuardResult(allowed=False, text=blocked, violations=violations)
