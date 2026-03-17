from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backends import DEFAULT_MINISTRAL_MODEL, CompletionBackend, MistralAPIBackend, OllamaBackend
from .mdmp_guard import GuardResult, MDMPGuard
from .prompts import TaskName, build_prompt


@dataclass(frozen=True)
class AIResponse:
    task: str
    text: str
    backend: str
    model: str
    certification: GuardResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "text": self.text,
            "backend": self.backend,
            "model": self.model,
            "certification": self.certification.to_dict(),
        }


class IINTSAssistant:
    """Research-only LLM assistant gated by MDMP certification."""

    def __init__(
        self,
        mdmp_cert: str | Path,
        *,
        mode: str = "auto",
        model: str = DEFAULT_MINISTRAL_MODEL,
        minimum_grade: str = "research_grade",
        public_key_path: str | Path | None = None,
        trust_store_path: str | Path | None = None,
        ollama_host: str | None = None,
        timeout_seconds: float = 60.0,
        backend: CompletionBackend | None = None,
        guard: MDMPGuard | None = None,
    ) -> None:
        self.guard = guard or MDMPGuard(
            mdmp_cert,
            minimum_grade=minimum_grade,
            public_key_path=public_key_path,
            trust_store_path=trust_store_path,
        )
        self.backend = backend or self._detect_backend(
            mode=mode,
            model=model,
            ollama_host=ollama_host,
            timeout_seconds=timeout_seconds,
        )

    def _detect_backend(
        self,
        *,
        mode: str,
        model: str,
        ollama_host: str | None,
        timeout_seconds: float,
    ) -> CompletionBackend:
        requested = mode.strip().lower()
        if requested in {"auto", "local", "ollama"}:
            local_backend: CompletionBackend = OllamaBackend(
                model_name=model,
                base_url=ollama_host,
                timeout_seconds=timeout_seconds,
            )
            if local_backend.available():
                return local_backend
            raise RuntimeError(
                "No local Ollama backend is available. Start Ollama and pull the Ministral model with:\n"
                f"  ollama pull {model}"
            )
        if requested == "api":
            api_backend: CompletionBackend = MistralAPIBackend()
            if api_backend.available():
                return api_backend
            raise RuntimeError(
                "Cloud API fallback is not enabled in this SDK build yet. "
                "Use mode='local' with Ollama."
            )
        raise ValueError(f"Unsupported AI mode: {mode}")

    def _run_task(self, task: TaskName, payload: Any) -> AIResponse:
        certification = self.guard.check()
        system_prompt, user_prompt = build_prompt(task, payload)
        text = self.guard.wrap(
            self.backend.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        )
        return AIResponse(
            task=task,
            text=text,
            backend=getattr(self.backend, "backend_name", type(self.backend).__name__),
            model=getattr(self.backend, "model_name", DEFAULT_MINISTRAL_MODEL),
            certification=certification,
        )

    def explain_decision(self, step: dict[str, Any]) -> AIResponse:
        return self._run_task("explain_decision", step)

    def analyze_trends(self, glucose_payload: list[Any] | dict[str, Any]) -> AIResponse:
        return self._run_task("analyze_trends", glucose_payload)

    def detect_anomalies(self, results: dict[str, Any]) -> AIResponse:
        return self._run_task("detect_anomalies", results)

    def generate_report(self, run: dict[str, Any]) -> AIResponse:
        return self._run_task("generate_report", run)
