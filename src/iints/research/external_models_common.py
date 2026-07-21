"""Shared safety and provenance helpers for optional external research engines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for evidence manifests."""

    return datetime.now(timezone.utc).isoformat()


def timestamp_token() -> str:
    """Return a collision-resistant, filename-safe UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def namespace(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1 : tag.index("}")]
    return ""


def normalised_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def read_local_file(
    path: Path,
    *,
    label: str,
    suffixes: Iterable[str],
    max_bytes: int,
    reject_xml_entities: bool = False,
) -> tuple[Path, bytes]:
    """Read a bounded local artifact without resolving remote or embedded paths."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    allowed = {suffix.lower() for suffix in suffixes}
    if resolved.suffix.lower() not in allowed:
        readable = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must use one of these extensions: {readable}.")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty.")
    if size > max_bytes:
        limit_mib = max_bytes / (1024 * 1024)
        raise ValueError(f"{label} exceeds the {limit_mib:g} MiB safety limit.")
    payload = resolved.read_bytes()
    if reject_xml_entities:
        upper = payload.upper()
        if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
            raise ValueError(f"DTD and entity declarations are not accepted in {label}.")
    return resolved, payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a stable JSON evidence artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_stem(value: str, *, fallback: str = "model") -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return cleaned.strip("._") or fallback


def find_executable(
    *,
    environment_variable: str,
    names: Iterable[str],
    common_paths: Iterable[Path] = (),
) -> Path | None:
    """Resolve one optional executable without invoking a shell."""

    configured = os.environ.get(environment_variable, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
    for name in names:
        if match := shutil.which(name):
            return Path(match).resolve()
    for candidate in common_paths:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return None


def run_external_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 30,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a fixed argv command with bounded output and no shell expansion."""

    if not command or any("\x00" in value for value in command):
        raise ValueError("External command arguments must be non-empty and contain no NUL bytes.")
    if timeout_seconds < 1 or timeout_seconds > 24 * 60 * 60:
        raise ValueError("timeout_seconds must be between 1 and 86,400.")
    max_output = 5 * 1024 * 1024
    with tempfile.TemporaryFile(mode="w+b") as stdout_file, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
        )
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if (
                os.fstat(stdout_file.fileno()).st_size > max_output
                or os.fstat(stderr_file.fileno()).st_size > max_output
            ):
                process.kill()
                process.wait()
                raise RuntimeError("External engine output exceeded the 5 MiB safety limit.")
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            time.sleep(0.02)

        if (
            os.fstat(stdout_file.fileno()).st_size > max_output
            or os.fstat(stderr_file.fileno()).st_size > max_output
        ):
            raise RuntimeError("External engine output exceeded the 5 MiB safety limit.")
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr = stderr_file.read().decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(command, int(process.returncode), stdout, stderr)
