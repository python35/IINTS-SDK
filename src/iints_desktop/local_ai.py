from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from iints.ai.backends.ollama import DEFAULT_MINISTRAL_MODEL, OllamaBackend

from iints_desktop.results import build_ai_result_context


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


SYSTEM_PROMPT = """You are the local IINTS-AF desktop research assistant.

Rules:
- Research and education only.
- Not a medical device.
- Do not provide diagnosis, insulin dosing, treatment instructions, or real-time patient care advice.
- Be critical about simulation limitations, data quality, uncertainty, and physiology.
- If the user asks for dosing/treatment decisions, refuse briefly and redirect to safe research interpretation.
- Prefer clear, plain-language explanations over hype.
- If result-summary context is provided, explain what it suggests without pretending it is clinically validated.
"""


def check_local_ai(*, model: str = DEFAULT_MINISTRAL_MODEL, host: str | None = None) -> LocalAIStatus:
    backend = OllamaBackend(model_name=model, base_url=host, timeout_seconds=20.0, num_predict=128)
    if not backend.available():
        return LocalAIStatus(
            available=False,
            message=f"Ollama is not reachable at {backend.base_url}. Start Ollama first.",
        )
    try:
        resolved = backend.ensure_model_ready()
    except Exception as exc:
        return LocalAIStatus(available=False, message=str(exc))
    return LocalAIStatus(available=True, message=f"Ready: {resolved}", resolved_model=resolved)


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
        timeout_seconds=180.0,
        temperature=0.1,
        top_p=0.8,
        num_predict=1000,
        num_ctx=8192,
    )
    context = build_ai_result_context(result_csv) if result_csv else "No result CSV is currently loaded."
    user_prompt = (
        f"Result context:\n{context}\n\n"
        f"User question:\n{question.strip()}\n\n"
        "Answer as a critical SDK research assistant. Mention limitations and avoid treatment advice."
    )
    answer = backend.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    resolved = backend.resolved_model_name or model
    return LocalAIAnswer(answer=answer, model=resolved, context_used=result_csv is not None)
