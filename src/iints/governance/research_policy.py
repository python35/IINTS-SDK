from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

RESEARCH_ONLY_NOTICE = (
    "IINTS-AF is research and education software only. It is not a medical device, "
    "not a treatment recommendation system, and must not be used for diagnosis, "
    "insulin/glucagon dosing, treatment decisions, or real-time patient care."
)

_SIMULATION_CONTEXT_PATTERN = re.compile(
    r"\b(?:simulat(?:ed|ion|or)|virtual|research|educational|scenario|algorithm|"
    r"controller|model|trace|run|result|log|retrospective|dataset|csv|benchmark|"
    r"candidate|sandbox|study)\b",
    re.IGNORECASE,
)

_DIRECT_PATIENT_CONTEXT_PATTERN = re.compile(
    r"\b(?:you|your|the\s+patient|patient)\b",
    re.IGNORECASE,
)

_NEGATED_CLAIM_PATTERN = re.compile(
    r"\b(?:not|never|no|must\s+not|is\s+not|isn't|cannot|can't|do\s+not|don't)\b.{0,50}"
    r"(?:ce\s+marked|mdr\s+certified|approved\s+medical\s+device|"
    r"safe\s+for\s+patient\s+use|diagnos(?:e|is)|clinical\s+decision|treatment\s+plan)",
    re.IGNORECASE,
)

_BLOCKING_POLICY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "patient_specific_dose_instruction",
        re.compile(
            r"\b(?:take|give|inject|administer|deliver|bolus|dose)\s+(?:the\s+patient\s+)?"
            r"\d+(?:\.\d+)?\s*(?:u|unit|units|iu|mg|mcg)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "direct_treatment_advice",
        re.compile(
            r"\b(?:you|the\s+patient|patient)\s+(?:should|must|needs?\s+to|has\s+to)\s+"
            r"(?:take|inject|administer|increase|decrease|change|correct|treat|set|"
            r"bolus|dose)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "diagnosis_or_clinical_decision",
        re.compile(
            r"\b(?:diagnosis\s+is|diagnose\s+(?:you|the\s+patient|patient)|"
            r"clinical\s+decision\s+is|treatment\s+plan\s+is)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regulatory_overclaim",
        re.compile(
            r"\b(?:ce\s+marked|mdr\s+certified|approved\s+medical\s+device|"
            r"safe\s+for\s+patient\s+use)\b",
            re.IGNORECASE,
        ),
    ),
)

_WARNING_POLICY_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "candidate_adjustment_language",
        re.compile(
            r"\b(?:increase|decrease|raise|lower|change|set)\s+"
            r"(?:the\s+)?(?:basal|bolus|insulin|glucagon|dose|correction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "clinical_boundary_language",
        re.compile(
            r"\b(?:clinical\s+decision|treatment\s+plan|diagnosis|diagnose)\b",
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
    warnings: tuple[str, ...] = ()
    action: str = "allow"


def _match_window(text: str, match: re.Match[str], *, radius: int = 96) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return text[start:end]


def _is_negated_claim_context(window: str) -> bool:
    return bool(_NEGATED_CLAIM_PATTERN.search(window))


def _is_simulation_dose_context(window: str) -> bool:
    return bool(_SIMULATION_CONTEXT_PATTERN.search(window)) and not bool(
        _DIRECT_PATIENT_CONTEXT_PATTERN.search(window)
    )


def scan_text_for_policy_violations(text: str) -> tuple[str, ...]:
    """Return policy violation labels found in generated text.

    This scanner is intended for post-generation AI text. It blocks only hard
    research-boundary failures: direct patient-specific dosing/treatment,
    direct diagnostic decisions, or regulatory overclaims. Simulation and
    retrospective research descriptions such as "the controller delivered 2
    units in the scenario" are allowed because they document an experiment
    rather than instructing real-world care.
    """

    found: list[str] = []
    value = text or ""
    for label, pattern in _BLOCKING_POLICY_PATTERNS:
        for match in pattern.finditer(value):
            window = _match_window(value, match)
            if label == "patient_specific_dose_instruction" and _is_simulation_dose_context(window):
                continue
            if label in {"diagnosis_or_clinical_decision", "regulatory_overclaim"} and _is_negated_claim_context(
                window
            ):
                continue
            found.append(label)
            break
    return tuple(found)


def scan_text_for_policy_warnings(text: str) -> tuple[str, ...]:
    """Return non-blocking research-boundary warnings found in generated text."""

    found: list[str] = []
    value = text or ""
    for label, pattern in _WARNING_POLICY_PATTERNS:
        for match in pattern.finditer(value):
            window = _match_window(value, match)
            if _is_negated_claim_context(window):
                continue
            found.append(label)
            break
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
    warnings = tuple(warning for warning in scan_text_for_policy_warnings(cleaned) if warning not in violations)
    if not violations:
        if not warnings:
            return PolicyGuardResult(allowed=True, text=cleaned, violations=(), warnings=(), action="allow")
        note = (
            "\n\nResearch-only boundary note: this output contains language that can sound like "
            "clinical adjustment language. Treat it only as simulation/research interpretation; "
            "deterministic SDK reports and qualified reviewer judgement remain the source of truth."
        )
        return PolicyGuardResult(
            allowed=True,
            text=cleaned + note,
            violations=(),
            warnings=warnings,
            action="warn",
        )

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
    return PolicyGuardResult(allowed=False, text=blocked, violations=violations, warnings=warnings, action="block")
