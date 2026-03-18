from __future__ import annotations

import json
import os
from urllib import error, request


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MINISTRAL_MODEL = "ministral-3:8b"
LEGACY_MINISTRAL_MODEL = "mistral/ministral-8b-instruct"
MIN_OLLAMA_VERSION_FOR_MINISTRAL_3 = (0, 13, 1)
MINISTRAL_MODEL_ALIASES = (
    DEFAULT_MINISTRAL_MODEL,
    "ministral-3",
    "ministral-3:latest",
    "ministral-3:8b",
    "ministral-3:8b-instruct",
    "ministral",
    "ministral-8b",
    "ministral-8b-instruct",
    LEGACY_MINISTRAL_MODEL,
)


class OllamaBackend:
    backend_name = "ollama"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MINISTRAL_MODEL,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.resolved_model_name: str | None = None

    def _pull_hint(self) -> str:
        return f"ollama pull {self.model_name}"

    def _requires_ministral_3_runtime(self) -> bool:
        requested = self.model_name.strip().lower()
        return requested.startswith("ministral-3") or requested == "ministral"

    @staticmethod
    def _parse_version(raw_version: str) -> tuple[int, ...] | None:
        value = raw_version.strip().lower().lstrip("v")
        numeric_parts: list[int] = []
        for part in value.split("."):
            digits = ""
            for char in part:
                if char.isdigit():
                    digits += char
                else:
                    break
            if not digits:
                break
            numeric_parts.append(int(digits))
        if not numeric_parts:
            return None
        return tuple(numeric_parts)

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

    def server_version(self) -> str | None:
        try:
            response = self._request_json("/api/version", method="GET")
        except Exception:
            return None
        raw_version = response.get("version")
        if isinstance(raw_version, str) and raw_version.strip():
            return raw_version.strip()
        return None

    def version_supported(self) -> tuple[bool | None, str | None]:
        version = self.server_version()
        if version is None:
            return None, None
        if not self._requires_ministral_3_runtime():
            return True, version
        parsed = self._parse_version(version)
        if parsed is None:
            return None, version
        return parsed >= MIN_OLLAMA_VERSION_FOR_MINISTRAL_3, version

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
        if requested in {
            "ministral",
            "ministral-3",
            "ministral-3:latest",
            "ministral-3:8b",
            "ministral-3:8b-instruct",
            "ministral-8b",
            "ministral-8b-instruct",
        }:
            for alias in MINISTRAL_MODEL_ALIASES:
                resolved = installed_lookup.get(alias.lower())
                if resolved is not None:
                    return resolved

        for installed_name in installed:
            lowered = installed_name.lower()
            if "ministral-3" in lowered and "8b" in lowered:
                return installed_name
        for installed_name in installed:
            lowered = installed_name.lower()
            if "ministral" in lowered and "8b" in lowered:
                return installed_name

        return None

    def ensure_model_ready(self) -> str:
        version_ok, version = self.version_supported()
        if version_ok is False:
            required_version = ".".join(str(part) for part in MIN_OLLAMA_VERSION_FOR_MINISTRAL_3)
            raise RuntimeError(
                "The open Ministral 3 local model requires a newer Ollama runtime.\n"
                f"Detected Ollama: {version}\n"
                f"Required Ollama: >= {required_version}"
            )
        try:
            resolved = self.resolve_model_name()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect local Ollama models: {exc}") from exc

        if resolved is None:
            self.resolved_model_name = None
            installed = self.list_models()
            installed_hint = ", ".join(installed) if installed else "none"
            raise RuntimeError(
                "Ollama is running, but the requested Ministral model is not installed locally.\n"
                f"Requested: {self.model_name}\n"
                f"Installed: {installed_hint}\n"
                f"Run: {self._pull_hint()}"
            )
        self.resolved_model_name = resolved
        return resolved

    def healthcheck(self) -> dict[str, object]:
        version_ok, version = self.version_supported()
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
            "timeout_seconds": self.timeout_seconds,
            "server_version": version,
            "version_ok": version_ok,
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
