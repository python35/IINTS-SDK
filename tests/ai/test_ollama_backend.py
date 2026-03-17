from __future__ import annotations

import pytest

from iints.ai.backends.ollama import OllamaBackend


def test_ollama_backend_complete_returns_response(monkeypatch) -> None:
    backend = OllamaBackend(model_name="mistral/ministral-8b-instruct")

    def _fake_request_json(path, payload=None, *, method="POST"):
        if path == "/api/tags":
            assert method == "GET"
            return {"models": [{"name": "mistral/ministral-8b-instruct"}]}
        assert path == "/api/generate"
        assert payload["model"] == "mistral/ministral-8b-instruct"
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
        return {"models": [{"name": "mistral/ministral-8b-instruct"}]}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    assert backend.ensure_model_ready() == "mistral/ministral-8b-instruct"


def test_ollama_backend_healthcheck_reports_missing_model(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": [{"name": "llama3.2:latest"}]}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)
    status = backend.healthcheck()

    assert status["available"] is True
    assert status["ready"] is False
    assert status["resolved_model"] is None
    assert status["pull_command"] == "ollama pull ministral"


def test_ollama_backend_complete_fails_when_model_missing(monkeypatch) -> None:
    backend = OllamaBackend(model_name="ministral")

    def _fake_request_json(path, payload=None, *, method="POST"):
        assert path == "/api/tags"
        assert method == "GET"
        return {"models": []}

    monkeypatch.setattr(backend, "_request_json", _fake_request_json)

    with pytest.raises(RuntimeError, match="not installed locally"):
        backend.complete(system_prompt="system", user_prompt="prompt")
