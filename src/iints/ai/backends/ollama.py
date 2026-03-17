from __future__ import annotations

import json
import os
from urllib import error, request


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MINISTRAL_MODEL = "mistral/ministral-8b-instruct"


class OllamaBackend:
    backend_name = "ollama"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MINISTRAL_MODEL,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        method: str = "POST",
    ) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if exc.code == 404 and path == "/api/generate":
                raise RuntimeError(
                    f"Ollama model '{self.model_name}' is not available locally. "
                    f"Run: ollama pull {self.model_name}"
                ) from exc
            raise RuntimeError(f"Ollama request failed ({exc.code}): {detail or exc.reason}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start Ollama or set OLLAMA_HOST to the correct endpoint."
            ) from exc

        try:
            payload_json = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc
        if not isinstance(payload_json, dict):
            raise RuntimeError("Ollama returned an unexpected response shape.")
        return payload_json

    def available(self) -> bool:
        try:
            self._request_json("/api/tags", method="GET")
        except Exception:
            return False
        return True

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }
        response = self._request_json("/api/generate", payload)
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty completion.")
        return text.strip()
