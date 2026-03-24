from __future__ import annotations

import json
import os
from ipaddress import ip_address
from http.client import IncompleteRead, RemoteDisconnected
from time import sleep
from urllib import error, request
from urllib.parse import urlparse


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
        raw_base_url = base_url or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        self.base_url = self._validate_base_url(raw_base_url)
        self.timeout_seconds = timeout_seconds
        self.resolved_model_name: str | None = None

    @staticmethod
    def _is_loopback_host(hostname: str) -> bool:
        if hostname == "localhost":
            return True
        try:
            return ip_address(hostname).is_loopback
        except ValueError:
            return False

    @classmethod
    def _validate_base_url(cls, raw_base_url: str) -> str:
        parsed = urlparse(raw_base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                "OLLAMA_HOST/base_url must use http or https. Other URL schemes are blocked."
            )
        if not parsed.hostname:
            raise ValueError("OLLAMA_HOST/base_url must include a hostname.")
        if parsed.path not in {"", "/"}:
            raise ValueError("OLLAMA_HOST/base_url must not include a path component.")
        if not cls._is_loopback_host(parsed.hostname) and os.getenv("IINTS_ALLOW_REMOTE_OLLAMA") != "1":
            raise ValueError(
                "Remote Ollama endpoints are disabled by default. "
                "Use localhost/127.0.0.1 or set IINTS_ALLOW_REMOTE_OLLAMA=1 explicitly."
            )
        return raw_base_url.rstrip("/")

    def _pull_hint(self) -> str:
        return f"ollama pull {self.model_name}"

    def _generation_failure_hint(self) -> str:
        resolved = self.resolved_model_name or self.model_name
        return (
            "Ollama closed the generation connection before returning a response.\n"
            f"Endpoint: {self.base_url}\n"
            f"Model: {resolved}\n"
            "This usually means the model crashed while loading, the daemon restarted, "
            "or the machine ran out of memory.\n"
            "Try one of these:\n"
            f"  1. Run `ollama run {resolved} \"Reply with OK.\"` to confirm direct inference works.\n"
            "  2. Run `iints ai local-check --smoke-test` to validate a real generation path.\n"
            "  3. Switch to a smaller local model such as `ministral-3:3b` if memory is tight."
        )

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
        return self._request_json_once(path, payload, method=method)

    def _request_json_once(
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
            with request.urlopen(req, timeout=self.timeout_seconds) as response:  # nosec B310 - base_url is scheme/host validated
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
        except (RemoteDisconnected, ConnectionResetError, IncompleteRead) as exc:
            if path == "/api/generate":
                raise RuntimeError(self._generation_failure_hint()) from exc
            raise RuntimeError(
                f"Ollama connection closed unexpectedly while calling {path} at {self.base_url}."
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

    def smoke_test(self) -> dict[str, object]:
        resolved_model = self.ensure_model_ready()
        payload = {
            "model": resolved_model,
            "system": "You are a health check. Reply with exactly: OK",
            "prompt": "Reply with exactly: OK",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 8,
            },
        }

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._request_json_once("/api/generate", payload)
                text = response.get("response")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Ollama returned an empty smoke-test completion.")
                return {
                    "ok": True,
                    "response": text.strip(),
                    "attempts": attempt + 1,
                }
            except (RuntimeError, RemoteDisconnected, ConnectionResetError, IncompleteRead) as exc:
                if not isinstance(exc, RuntimeError):
                    exc = RuntimeError(self._generation_failure_hint())
                last_error = exc
                if attempt == 0:
                    sleep(1.0)
                    continue
                break

        assert last_error is not None
        raise last_error

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        resolved_model = self.ensure_model_ready()
        payload = {
            "model": resolved_model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._request_json_once("/api/generate", payload)
                break
            except (RuntimeError, RemoteDisconnected, ConnectionResetError, IncompleteRead) as exc:
                if not isinstance(exc, RuntimeError):
                    exc = RuntimeError(self._generation_failure_hint())
                last_error = exc
                if attempt == 0:
                    sleep(1.0)
                    continue
                raise exc
        else:
            assert last_error is not None
            raise last_error
        text = response.get("response")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Ollama returned an empty completion.")
        return text.strip()
