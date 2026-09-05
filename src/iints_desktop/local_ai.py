from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from iints.ai.backends.ollama import DEFAULT_MINISTRAL_MODEL, OllamaBackend
from iints.governance import RESEARCH_ONLY_NOTICE, guard_ai_output

from iints_desktop.results import build_ai_result_context, load_results_preview

RECOMMENDED_OLLAMA_MODELS = (
    DEFAULT_MINISTRAL_MODEL,
    "mistral-small:latest",
    "llama3.1:8b",
    "qwen2.5:7b",
    "gemma3:4b",
    "hf.co/devanshamin/PubMedDiabetes-LLM-Predictions",
)


@dataclass(frozen=True)
class LocalAIStatus:
    available: bool
    message: str
    resolved_model: str | None = None


@dataclass(frozen=True)
class LocalAIAnswer:
    answer: str
    model: str
    context_used: bool
    policy_violations: tuple[str, ...] = ()
    policy_warnings: tuple[str, ...] = ()
    policy_action: str = "allow"
    numeric_claim_warnings: tuple[str, ...] = ()
    deterministic_metrics: dict[str, str] | None = None
    suppressed_line_count: int = 0
    suppressed_advice_line_count: int = 0
    interpretation_restricted: bool = False


@dataclass(frozen=True)
class LocalAIStartResult:
    available: bool
    message: str
    resolved_model: str | None = None
    started_process: bool = False
    pulled_model: bool = False


SYSTEM_PROMPT = """You are a conservative research-results summarizer for the IINTS-AF diabetes simulation SDK.

REQUIRED METHOD:
1. Treat only the block labelled AUTHORITATIVE DETERMINISTIC FACTS as quantitative evidence.
2. Copy numeric values exactly. Never calculate, estimate, interpolate, convert units, invent confidence intervals, or infer missing physiological parameters.
3. If a requested metric is absent, write "not computed in the attached deterministic summary".
4. Clearly separate observations from interpretation. Label every causal or physiological explanation as a hypothesis that requires validation.
5. Do not infer insulin sensitivity, carbohydrate ratio, basal rate, pharmacokinetics, clinical risk, or treatment quality unless that exact metric is supplied.
6. A safety-triggered sample count is not an intervention count: adjacent sampled rows may describe one sustained episode. Do not infer instability, glucagon use, or causal mechanism from that count alone.
7. Totals must never be divided into ratios or converted into rates. Duration must not be converted into other units.
8. Use these exact section headings and concise bullets:
  Deterministic Facts
  Interpretation
  Limitations
  Next Checks
9. Acknowledge that this is simulated research output. It is not a medical device. Do not provide diagnosis, insulin dosing, or treatment advice.
10. Boundary: {research_only_notice}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT.format(research_only_notice=RESEARCH_ONLY_NOTICE)


def _common_ollama_candidates() -> list[Path]:
    system = platform.system().lower()
    candidates: list[Path] = []
    if system == "darwin":
        candidates.extend(
            [
                Path("/usr/local/bin/ollama"),
                Path("/opt/homebrew/bin/ollama"),
                Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            ]
        )
    elif system == "windows":
        local_app = os.getenv("LOCALAPPDATA")
        program_files = os.getenv("ProgramFiles")
        if local_app:
            candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
        if program_files:
            candidates.append(Path(program_files) / "Ollama" / "ollama.exe")
    else:
        candidates.extend(
            [
                Path("/usr/local/bin/ollama"),
                Path("/usr/bin/ollama"),
                Path.home() / ".local" / "bin" / "ollama",
            ]
        )
    return candidates


def resolve_ollama_executable(extra_candidates: list[Path] | None = None) -> Path | None:
    """Find the local Ollama executable without requiring users to edit PATH."""

    discovered = shutil.which("ollama")
    if discovered:
        return Path(discovered)

    candidates = [*(extra_candidates or []), *_common_ollama_candidates()]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _startup_flags() -> int:
    if platform.system().lower() != "windows":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | int(
        getattr(subprocess, "DETACHED_PROCESS", 0)
    )


def _ollama_environment(base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    return env


def _start_ollama_process(executable: Path, base_url: str) -> None:
    subprocess.Popen(  # noqa: S603 - executable is resolved from PATH/common install locations.
        [str(executable), "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_ollama_environment(base_url),
        start_new_session=platform.system().lower() != "windows",
        creationflags=_startup_flags(),
    )


def _pull_model(executable: Path, model: str, base_url: str, *, timeout_seconds: float) -> None:
    completed = subprocess.run(  # noqa: S603 - executable is resolved from PATH/common install locations.
        [str(executable), "pull", model],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_ollama_environment(base_url),
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "model pull failed").strip()
        raise RuntimeError(f"Could not download local Ollama model '{model}': {detail}")


def _wait_for_ollama(backend: OllamaBackend, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if backend.available():
            return True
        time.sleep(0.75)
    return backend.available()


def start_local_ai_stack(
    *,
    model: str = DEFAULT_MINISTRAL_MODEL,
    host: str | None = None,
    startup_timeout_seconds: float = 45.0,
    pull_timeout_seconds: float = 1800.0,
    pull_missing_model: bool = True,
) -> LocalAIStartResult:
    """Start Ollama when possible and make the requested local model ready."""

    backend = OllamaBackend(model_name=model, base_url=host, timeout_seconds=20.0, num_predict=128)
    parsed = urlparse(backend.base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return LocalAIStartResult(
            available=False,
            message="Start Local AI only manages a local Ollama server. Use Check Ollama for remote endpoints.",
        )

    started_process = False
    executable = resolve_ollama_executable()
    if not backend.available():
        if executable is None:
            return LocalAIStartResult(
                available=False,
                message=(
                    "Ollama is not installed or not on PATH. Install Ollama once, then this button can "
                    "start it automatically and prepare the model."
                ),
            )
        _start_ollama_process(executable, backend.base_url)
        started_process = True
        if not _wait_for_ollama(backend, timeout_seconds=startup_timeout_seconds):
            return LocalAIStartResult(
                available=False,
                started_process=started_process,
                message=f"Started Ollama, but it did not become reachable at {backend.base_url} in time.",
            )

    pulled_model = False
    try:
        resolved = backend.ensure_model_ready()
    except RuntimeError as exc:
        if not pull_missing_model or executable is None:
            return LocalAIStartResult(
                available=False,
                started_process=started_process,
                message=str(exc),
            )
        _pull_model(executable, model, backend.base_url, timeout_seconds=pull_timeout_seconds)
        pulled_model = True
        resolved = backend.ensure_model_ready()

    parts = [f"Local AI ready: {resolved}"]
    if started_process:
        parts.append("Ollama was started automatically")
    if pulled_model:
        parts.append("model downloaded")
    return LocalAIStartResult(
        available=True,
        message="; ".join(parts) + ".",
        resolved_model=resolved,
        started_process=started_process,
        pulled_model=pulled_model,
    )


def check_local_ai(*, model: str = DEFAULT_MINISTRAL_MODEL, host: str | None = None) -> LocalAIStatus:
    backend = OllamaBackend(model_name=model, base_url=host, timeout_seconds=20.0, num_predict=128)
    if not backend.available():
        return LocalAIStatus(
            available=False,
            message=f"Ollama is not reachable at {backend.base_url}. Click Start Local AI or start Ollama manually.",
        )
    try:
        resolved = backend.ensure_model_ready()
    except Exception as exc:
        return LocalAIStatus(available=False, message=str(exc))
    return LocalAIStatus(available=True, message=f"Ready: {resolved}", resolved_model=resolved)


def list_local_ai_models(*, host: str | None = None) -> list[str]:
    """Return installed Ollama models, falling back to curated recommendations."""

    backend = OllamaBackend(model_name=DEFAULT_MINISTRAL_MODEL, base_url=host, timeout_seconds=3.0, num_predict=128)
    if not backend.available():
        return list(RECOMMENDED_OLLAMA_MODELS)
    discovered = backend.list_models()
    merged: list[str] = []
    for model in [*discovered, *RECOMMENDED_OLLAMA_MODELS]:
        if model and model not in merged:
            merged.append(model)
    return merged


def format_ai_answer(text: str) -> str:
    """Make local LLM output easier to read in the desktop text panel."""

    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    replacements = {
        "**": "",
        "__": "",
        "### ": "",
        "## ": "",
        "# ": "",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    lines: list[str] = []
    previous_blank = False
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip().replace("`", "")
        line = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", line)
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        previous_blank = False
        if line.startswith(("- ", "* ")):
            line = "• " + line[2:].strip()
        if line.startswith(("---", "___", "***")):
            continue
        if line.lower().strip("= ").rstrip(":") in {
            "deterministic facts",
            "interpretation",
            "limitations",
            "next checks",
            "clinical overview",
            "biomathematical observations",
            "algorithmic behavior",
            "conclusions",
        }:
            line = f"\n{line.strip('= ').rstrip(':')}"
        lines.append(line)
    return "\n".join(lines).strip()


_NUMERIC_CLAIM_PATTERN = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?%?")

_TREATMENT_ADJUSTMENT_PATTERN = re.compile(
    r"\b(?:adjust(?:ed|ing)?|increase|decrease|raise|lower|change|modify|"
    r"optimi[sz]e|tune|higher|split)\b(?:\s+[A-Za-z][A-Za-z/-]*){0,5}\s+"
    r"(?:basal|bolus|insulin|glucagon|dose|dosing|correction)\b|"
    r"\b(?:basal|bolus|insulin|glucagon|dose|dosing|correction)\b"
    r"(?:\s+[A-Za-z][A-Za-z/-]*){0,5}\s+"
    r"(?:adjust(?:ment|ed|ing)?|increase|decrease|raise|lower|change|modify|"
    r"optimi[sz](?:e|ation)|tune|higher|split)\b|"
    r"\b(?:use|give|deliver|administer|inject)\b"
    r"(?:\s+[A-Za-z][A-Za-z/-]*){0,5}\s+"
    r"(?:basal|bolus|insulin|glucagon|dose|dosing|correction)\b|"
    r"\b(?:bolus|dose)\s+(?:earlier|later|more|less)\b",
    re.IGNORECASE,
)


def audit_ai_numeric_claims(answer: str, deterministic_context: str) -> tuple[str, ...]:
    """Flag numbers not present in the deterministic context.

    This is deliberately a lexical guard, not a scientific fact checker. It catches
    a common and high-impact failure mode: the model inventing statistics or
    converting units even though the desktop bridge did not compute those values.
    """

    allowed = set(_NUMERIC_CLAIM_PATTERN.findall(deterministic_context))
    without_ordinals = re.sub(r"(?m)^\s*\d+[.)]\s+", "", answer)
    claimed = set(_NUMERIC_CLAIM_PATTERN.findall(without_ordinals))
    unsupported = sorted(claimed - allowed, key=lambda value: (len(value), value))
    if not unsupported:
        return ()
    preview = ", ".join(unsupported[:12])
    suffix = "" if len(unsupported) <= 12 else f" (+{len(unsupported) - 12} more)"
    return (
        "Numeric-claim audit found values that were not present in the deterministic "
        f"summary: {preview}{suffix}. Treat the narrative as unverified and use the metrics panel as authoritative.",
    )


def suppress_unsupported_numeric_lines(
    answer: str,
    deterministic_context: str,
) -> tuple[str, int]:
    """Hide model lines that introduce quantities the SDK did not compute."""

    allowed = set(_NUMERIC_CLAIM_PATTERN.findall(deterministic_context))
    kept: list[str] = []
    suppressed = 0
    for line in answer.splitlines():
        audit_line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        unsupported = set(_NUMERIC_CLAIM_PATTERN.findall(audit_line)) - allowed
        if unsupported:
            suppressed += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if suppressed:
        notice = (
            f"{suppressed} AI-generated line(s) were hidden because they introduced "
            "quantities that were not present in the deterministic SDK summary."
        )
        cleaned = f"{notice}\n\n{cleaned}" if cleaned else notice
    return cleaned, suppressed


def suppress_treatment_advice_lines(answer: str) -> tuple[str, int]:
    """Remove generated treatment-adjustment suggestions from desktop reviews.

    Prompt instructions are not a safety boundary: local models can ignore them.
    Aggregate result review therefore applies a deterministic line-level filter
    before any generated text is shown in the desktop application.
    """

    kept: list[str] = []
    suppressed = 0
    for line in answer.splitlines():
        if _TREATMENT_ADJUSTMENT_PATTERN.search(line):
            suppressed += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if suppressed:
        notice = (
            "AI-generated treatment-adjustment language was hidden. The desktop "
            "assistant is limited to non-treatment research review."
        )
        cleaned = f"{notice}\n\n{cleaned}" if cleaned else notice
    return cleaned, suppressed


def renumber_ordered_lines(answer: str) -> str:
    """Close numbering gaps left by deterministic post-generation filters."""

    next_number = 1
    lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.lower().rstrip(":") in {
            "deterministic facts",
            "interpretation",
            "limitations",
            "next checks",
        }:
            next_number = 1
            lines.append(line)
            continue
        match = re.match(r"^(\s*)\d+[.)]\s+(.+)$", line)
        if match:
            lines.append(f"{match.group(1)}{next_number}. {match.group(2)}")
            next_number += 1
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def restrict_ai_to_review_sections(answer: str) -> str:
    """Keep only limitations and follow-up checks for aggregate CSV context.

    The desktop model receives aggregate metrics and column names, not enough
    evidence for patient-level causal or physiological interpretation. The SDK
    therefore owns the facts while the model is limited to proposing checks.
    """

    headings = {
        "deterministic facts",
        "interpretation",
        "limitations",
        "next checks",
    }
    allowed = {"limitations", "next checks"}
    current: str | None = None
    kept: list[str] = []
    for line in answer.splitlines():
        normalized = line.strip().lower().rstrip(":")
        if normalized in headings:
            current = normalized
            if current in allowed:
                kept.append(line.strip().rstrip(":"))
            continue
        if current in allowed and line.strip():
            kept.append(line)

    notice = (
        "AI scope is limited to limitations and follow-up checks because the local "
        "model received aggregate SDK metrics rather than row-level causal evidence."
    )
    if not kept:
        return (
            f"{notice}\n\nNo review notes were retained. Use the deterministic metrics "
            "and inspect the CSV, safety reasons, and generated report directly."
        )
    return f"{notice}\n\n" + "\n".join(kept).strip()


def ask_local_ai(
    *,
    question: str,
    model: str = DEFAULT_MINISTRAL_MODEL,
    host: str | None = None,
    result_csv: str | Path | None = None,
) -> LocalAIAnswer:
    if not question.strip():
        raise ValueError("Question is empty.")

    backend = OllamaBackend(
        model_name=model,
        base_url=host,
        timeout_seconds=900.0,
        temperature=0.1,
        top_p=0.8,
        num_predict=1000,
        num_ctx=2048,
    )
    context = build_ai_result_context(result_csv) if result_csv else "No result CSV is currently loaded."
    deterministic_metrics = (
        load_results_preview(result_csv, max_rows=1).metrics if result_csv is not None else None
    )
    user_prompt = (
        f"Result context:\n{context}\n\n"
        f"User question:\n{question.strip()}\n\n"
        "Summarize conservatively. Do not add quantitative claims beyond the authoritative facts. "
        "Distinguish observations, hypotheses, uncertainty, and limitations."
    )
    answer = backend.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    guarded = guard_ai_output(answer, source="desktop_local_ai")
    resolved = backend.resolved_model_name or model
    formatted = format_ai_answer(guarded.text)
    numeric_warnings = audit_ai_numeric_claims(formatted, context)
    filtered, suppressed_line_count = suppress_unsupported_numeric_lines(formatted, context)
    interpretation_restricted = result_csv is not None
    if interpretation_restricted:
        filtered = restrict_ai_to_review_sections(filtered)
    filtered, suppressed_advice_line_count = suppress_treatment_advice_lines(filtered)
    filtered = renumber_ordered_lines(filtered)
    policy_warnings = guarded.warnings
    policy_action = guarded.action
    if suppressed_advice_line_count:
        policy_warnings = tuple(
            dict.fromkeys(
                [*policy_warnings, "AI-generated treatment-adjustment language was removed"]
            )
        )
        if policy_action == "allow":
            policy_action = "warn"
    return LocalAIAnswer(
        answer=filtered,
        model=resolved,
        context_used=result_csv is not None,
        policy_violations=guarded.violations,
        policy_warnings=policy_warnings,
        policy_action=policy_action,
        numeric_claim_warnings=numeric_warnings,
        deterministic_metrics=deterministic_metrics,
        suppressed_line_count=suppressed_line_count,
        suppressed_advice_line_count=suppressed_advice_line_count,
        interpretation_restricted=interpretation_restricted,
    )
