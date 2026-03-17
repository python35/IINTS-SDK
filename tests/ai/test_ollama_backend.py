from __future__ import annotations

from iints.ai.backends.ollama import OllamaBackend


def test_ollama_backend_complete_returns_response(monkeypatch) -> None:
    backend = OllamaBackend(model_name="mistral/ministral-8b-instruct")

    def _fake_request_json(path, payload=None, *, method="POST"):
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
