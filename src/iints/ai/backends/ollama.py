from __future__ import annotations

import json
import os
from urllib import error, request


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MINISTRAL_MODEL = "mistral/ministral-8b-instruct"
MINISTRAL_MODEL_ALIASES = (
    DEFAULT_MINISTRAL_MODEL,
    "ministral",
    "ministral-8b",
    "ministral-8b-instruct",
)


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

    def _pull_hint(self) -> str:
        return f"ollama pull {self.model_name}"

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

    def list_models(self) -> list[str]:
        response = self._request_json("/api/tags", method="GET")
        raw_models = response.get("models", [])
        if not isinstance(raw_models, list):
            raise RuntimeError("Ollama returned an unexpected model list.")

        discovered: list[str] = []
        for entry in raw_models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name.strip():
                discovered.append(name.strip())
        return discovered

    def resolve_model_name(self) -> str | None:
        installed = self.list_models()
        installed_lookup = {name.lower(): name for name in installed}

        if self.model_name.lower() in installed_lookup:
            return installed_lookup[self.model_name.lower()]

        requested = self.model_name.strip().lower()
        if requested in {"ministral", "ministral-8b", "ministral-8b-instruct"}:
            for alias in MINISTRAL_MODEL_ALIASES:
                resolved = installed_lookup.get(alias.lower())
                if resolved is not None:
                    return resolved

        for installed_name in installed:
            lowered = installed_name.lower()
            if "ministral" in lowered and "8b" in lowered:
                return installed_name

        return None

    def ensure_model_ready(self) -> str:
        try:
            resolved = self.resolve_model_name()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect local Ollama models: {exc}") from exc

        if resolved is None:
            installed = self.list_models()
            installed_hint = ", ".join(installed) if installed else "none"
            raise RuntimeError(
                "Ollama is running, but the requested Ministral model is not installed locally.\n"
                f"Requested: {self.model_name}\n"
                f"Installed: {installed_hint}\n"
                f"Run: {self._pull_hint()}"
            )
        return resolved

    def healthcheck(self) -> dict[str, object]:
        installed = self.list_models()
        resolved = self.resolve_model_name() if installed else None
        return {
            "available": True,
            "base_url": self.base_url,
            "requested_model": self.model_name,
            "resolved_model": resolved,
            "installed_models": installed,
            "ready": resolved is not None,
            "pull_command": None if resolved is not None else self._pull_hint(),
        }

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        resolved_model = self.ensure_model_ready()
        payload = {
            "model": resolved_model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }
        response = self._request_json("/api/generate", payload)
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty completion.")
        return text.strip()
