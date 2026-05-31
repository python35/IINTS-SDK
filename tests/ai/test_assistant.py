from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints.ai import IINTSAssistant, MDMPGuard
from iints.ai.assistant import AIResponse
from iints.ai.backends.ollama import DEFAULT_MINISTRAL_MODEL, OllamaBackend
from iints.ai.mdmp_guard import GuardResult


class _FakeBackend:
    backend_name = "fake"
    model_name = DEFAULT_MINISTRAL_MODEL
    resolved_model_name = "ministral-3:8b"

    def available(self) -> bool:
        return True

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert "research use only" in system_prompt.lower()
        assert "glucose" in user_prompt.lower() or "simulation" in user_prompt.lower()
        return "Research-only explanation."


class _FakeGuard:
    def __init__(self) -> None:
        self.calls = 0

    def check(self) -> GuardResult:
        self.calls += 1
        return GuardResult(
            cert_path="cert.json",
            grade="research_grade",
            issued_by="MDMP-Authority-v1",
            verification_mode="bundled_root",
            key_id="mdmp_pub_v1",
            raw_result={"valid": True, "grade": "research_grade"},
        )

    def wrap(self, response: str) -> str:
        return response + "\n\nWARNING: For research use only. Not medical advice."


def test_assistant_runs_with_injected_backend_and_guard() -> None:
    guard = _FakeGuard()
    assistant = IINTSAssistant(
        "cert.json",
        backend=_FakeBackend(),
        guard=guard,  # type: ignore[arg-type]
    )

    response = assistant.explain_decision({"glucose": 145, "decision": {"insulin": 0.3}})

    assert isinstance(response, AIResponse)
    assert response.text.endswith("WARNING: For research use only. Not medical advice.")
    assert response.backend == "fake"
    assert response.model == "ministral-3:8b"
    assert response.certification.grade == "research_grade"
    assert guard.calls == 1


def test_assistant_auto_detects_ollama_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OllamaBackend, "available", lambda self: True)
    monkeypatch.setattr(
        OllamaBackend,
        "ensure_model_ready",
        lambda self: "ministral-3:8b",
    )
    assistant = IINTSAssistant(
        "cert.json",
        guard=_FakeGuard(),  # type: ignore[arg-type]
        mode="auto",
    )

    assert isinstance(assistant.backend, OllamaBackend)
    assert assistant.backend.model_name == DEFAULT_MINISTRAL_MODEL


def test_assistant_local_mode_fails_if_model_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OllamaBackend, "available", lambda self: True)

    def _raise_missing(self) -> str:
        raise RuntimeError("requested Ministral model is not installed locally")

    monkeypatch.setattr(OllamaBackend, "ensure_model_ready", _raise_missing)

    with pytest.raises(RuntimeError, match="not installed locally"):
        IINTSAssistant(
            "cert.json",
            guard=_FakeGuard(),  # type: ignore[arg-type]
            mode="local",
        )


def test_assistant_api_mode_reports_current_mistral_replacement() -> None:
    with pytest.raises(RuntimeError, match="mistral-small-latest"):
        IINTSAssistant(
            "cert.json",
            guard=_FakeGuard(),  # type: ignore[arg-type]
            mode="api",
            model="devstral-small-latest",
        )


def test_guard_rejects_invalid_certificate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cert_path = tmp_path / "report.signed.mdmp"
    cert_path.write_text(json.dumps({"signature": "abc"}), encoding="utf-8")

    class _Verifier:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def verify(self, payload: dict[str, object]) -> dict[str, object]:
            return {"valid": False, "error": "signature_verification_failed"}

    monkeypatch.setattr("iints.ai.mdmp_guard._load_mdmp_verifier", lambda: _Verifier)

    guard = MDMPGuard(cert_path)
    with pytest.raises(PermissionError):
        guard.check()


def test_guard_enforces_minimum_grade(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cert_path = tmp_path / "report.signed.mdmp"
    cert_path.write_text(json.dumps({"signature": "abc", "grade": "draft"}), encoding="utf-8")

    class _Verifier:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def verify(self, payload: dict[str, object]) -> dict[str, object]:
            return {"valid": True, "grade": "draft", "issued_by": "MDMP-Authority-v1"}

    monkeypatch.setattr("iints.ai.mdmp_guard._load_mdmp_verifier", lambda: _Verifier)

    guard = MDMPGuard(cert_path, minimum_grade="research_grade")
    with pytest.raises(PermissionError):
        guard.check()


def test_guard_wrap_appends_disclaimer() -> None:
    guard = MDMPGuard("cert.json")
    wrapped = guard.wrap("Generated report body.")
    assert wrapped.endswith("WARNING: For research use only. Not medical advice.")


def test_assistant_review_realism_uses_same_guarded_flow() -> None:
    guard = _FakeGuard()
    assistant = IINTSAssistant(
        "cert.json",
        backend=_FakeBackend(),
        guard=guard,  # type: ignore[arg-type]
    )

    response = assistant.review_realism({"summary": {"mean_glucose_mgdl": 145}})

    assert response.task == "review_realism"
    assert "research use only" in response.text.lower()
    assert guard.calls == 1
