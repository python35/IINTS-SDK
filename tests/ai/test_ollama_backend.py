from __future__ import annotations

import pytest

from iints.ai.backends.ollama import OllamaBackend


def test_ollama_backend_complete_returns_response(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral-3:8b")

    def _fake_request_json(path, payload=None, *, method="POST"):
        if path == "/api/version":
            assert method == "GET"
            return {"version": "0.13.1"}
        if path == "/api/tags":
            assert method == "GET"
            return {"models": [{"name": "ministral-3:8b"}]}
        assert path == "/api/generate"
        assert payload["model"] == "ministral-3:8b"
        assert payload["stream"] is False
        assert method == "POST"
        return {"response": "Local explanation from Ollama."}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)

    text = backend.complete(system_prompt="system", user_prompt="prompt")
    assert text == "Local explanation from Ollama."


def test_ollama_backend_available_uses_tags_endpoint(monkeypatch) -> None:
    backend = OllamaBackend()

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": []}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    assert backend.available() is True


def test_ollama_backend_resolves_ministral_alias(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": [{"name": "ministral-3:8b"}]}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    monkeypatch.setattr(backend, "version_supported", lambda: (True, "0.13.1"))
    assert backend.ensure_model_ready() == "ministral-3:8b"


def test_ollama_backend_accepts_legacy_ministral_model(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": [{"name": "mistral/ministral-8b-instruct"}]}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    monkeypatch.setattr(backend, "version_supported", lambda: (True, "0.13.1"))
    assert backend.ensure_model_ready() == "mistral/ministral-8b-instruct"


def test_ollama_backend_healthcheck_reports_missing_model(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        if path == "/api/version":
            assert method == "GET"
            return {"version": "0.13.1"}
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    status = backend.healthcheck()

    assert status["available"] is True
    assert status["ready"] is False
    assert status["resolved_model"] is None
    assert status["pull_command"] == "ollama pull ministral"
    assert status["server_version"] == "0.13.1"
    assert status["version_ok"] is True


def test_ollama_backend_complete_fails_when_model_missing(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": []}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    monkeypatch.setattr(backend, "version_supported", lambda: (True, "0.13.1"))

    with pytest.raises(RuntimeError, match="not installed locally"):
        backend.complete(system_prompt="system", user_prompt="prompt")


def test_ollama_backend_rejects_old_ollama_for_ministral_3(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral-3:8b")
    monkeypatch.setattr(backend, "version_supported", lambda: (False, "0.12.9"))

    with pytest.raises(RuntimeError, match="requires a newer Ollama runtime"):
        backend.ensure_model_ready()
