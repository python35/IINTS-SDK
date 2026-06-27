from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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


@dataclass(frozen=True)
class LocalAIStartResult:
    available: bool
    message: str
    resolved_model: str | None = None
    started_process: bool = False
    pulled_model: bool = False


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
