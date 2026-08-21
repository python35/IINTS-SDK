"""Shared, network-bounded version inspection for the CLI and desktop app."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version


SDK_DISTRIBUTION = "iints-sdk-python35"
SDK_PYPI_JSON_URL = f"https://pypi.org/pypi/{SDK_DISTRIBUTION}/json"
SDK_RELEASE_URL = f"https://pypi.org/project/{SDK_DISTRIBUTION}/"
APP_RELEASES_API_URL = "https://api.github.com/repos/python35/IINTS-SDK/releases?per_page=30"
APP_RELEASE_URL = "https://github.com/python35/IINTS-SDK/releases/tag/tauri-beta-latest"
APP_TAG_PREFIX = "tauri-beta-v"
VERSION_CHECK_SCHEMA = 1
DEFAULT_CACHE_TTL = timedelta(hours=6)
DEFAULT_NETWORK_TIMEOUT_SECONDS = 4.0

JsonFetcher = Callable[[str, float], Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_version(value: str | None) -> Version | None:
    if not value:
        return None
    candidate = value.strip()
    if candidate.lower() in {"unknown", "editable", "development"}:
        return None
    if candidate.startswith("v"):
        candidate = candidate[1:]
    try:
        return Version(candidate)
    except InvalidVersion:
        return None


def _status_for_versions(installed: str, latest: str | None) -> tuple[str, bool | None]:
    installed_version = _normalized_version(installed)
    latest_version = _normalized_version(latest)
    if installed_version is None:
        return "development", None
    if latest_version is None:
        return "unknown", None
    if installed_version < latest_version:
        return "update_available", True
    if installed_version > latest_version:
        return "ahead", False
    return "current", False


def version_is_newer(candidate: str, reference: str) -> bool:
    """Return whether *candidate* is a valid version newer than *reference*."""

    candidate_version = _normalized_version(candidate)
    reference_version = _normalized_version(reference)
    return bool(
        candidate_version is not None
        and reference_version is not None
        and candidate_version > reference_version
    )


def _default_cache_path() -> Path:
    override = os.getenv("IINTS_VERSION_CACHE")
    if override:
        return Path(override).expanduser()
    cache_root = os.getenv("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root).expanduser() / "iints-af" / "version-check.json"
    return Path.home() / ".cache" / "iints-af" / "version-check.json"


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != VERSION_CHECK_SCHEMA:
        return {}
    return payload


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Version checks must remain usable in read-only or restricted environments.
        return


def clear_version_cache(path: Path | None = None) -> None:
    try:
        (path or _default_cache_path()).unlink(missing_ok=True)
    except OSError:
        return


def _fetch_json(url: str, timeout: float) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "IINTS-AF-Version-Check/1",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoints
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"version endpoint unavailable: {exc}") from exc


def installed_sdk_version() -> str:
    try:
        return metadata.version(SDK_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        try:
            import iints

            return str(getattr(iints, "__version__", "unknown"))
        except Exception:
            return "unknown"


def installed_sdk_environment() -> dict[str, Any]:
    package_location: str | None = None
    try:
        package_location = str(
            Path(str(metadata.distribution(SDK_DISTRIBUTION).locate_file(""))).resolve()
        )
    except metadata.PackageNotFoundError:
        pass

    try:
        import iints

        module_location = str(Path(iints.__file__).resolve()) if iints.__file__ else None
        code_version = str(getattr(iints, "__version__", "unknown"))
    except Exception:
        module_location = None
        code_version = "unknown"

    distribution_version = installed_sdk_version()

    invoked_cli = Path(sys.argv[0]).expanduser()
    cli_path: str | None
    if invoked_cli.name.lower() in {"iints", "iints.exe"} and invoked_cli.is_file():
        cli_path = str(invoked_cli.absolute())
    else:
        cli_path = shutil.which("iints")
    cli_shebang: str | None = None
    cli_matches_python: bool | None = None
    if cli_path and os.name != "nt":
        try:
            first_line = Path(cli_path).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
            if first_line.startswith("#!"):
                cli_shebang = first_line[2:].strip()
                if cli_shebang.startswith("/"):
                    # Do not resolve virtual-environment symlinks: two environments
                    # may share the same base interpreter but have different packages.
                    cli_matches_python = Path(cli_shebang).absolute() == Path(sys.executable).absolute()
        except (OSError, IndexError):
            pass

    return {
        "python_executable": sys.executable,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "distribution_version": distribution_version,
        "code_version": code_version,
        "version_metadata_matches_code": (
            distribution_version == code_version
            if "unknown" not in {distribution_version, code_version}
            else None
        ),
        "cli_path": cli_path,
        "cli_shebang": cli_shebang,
        "cli_matches_python": cli_matches_python,
        "package_location": package_location,
        "module_location": module_location,
    }


@dataclass(frozen=True)
class ComponentVersionStatus:
    component: str
    installed_version: str
    latest_version: str | None
    status: str
    update_available: bool | None
    source: str
    checked_at: str | None
    release_url: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cached_latest(
    cache: dict[str, Any],
    component: str,
    *,
    now: datetime,
    ttl: timedelta,
) -> tuple[str | None, str | None, bool]:
    entry = cache.get(component)
    if not isinstance(entry, dict):
        return None, None, False
    latest = entry.get("latest_version")
    checked_at = entry.get("checked_at")
    parsed = _parse_iso_utc(checked_at)
    fresh = parsed is not None and now - parsed <= ttl
    return (str(latest) if latest else None, str(checked_at) if checked_at else None, fresh)


def _component_status(
    *,
    component: str,
    installed: str,
    latest: str | None,
    source: str,
    checked_at: str | None,
    release_url: str,
    error: str | None = None,
) -> ComponentVersionStatus:
    status, update_available = _status_for_versions(installed, latest)
    if error and latest is None:
        status = "unknown"
        update_available = None
    return ComponentVersionStatus(
        component=component,
        installed_version=installed,
        latest_version=latest,
        status=status,
        update_available=update_available,
        source=source,
        checked_at=checked_at,
        release_url=release_url,
        error=error,
    )


def check_sdk_version(
    *,
    installed: str | None = None,
    refresh: bool = False,
    offline: bool = False,
    cache_path: Path | None = None,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    timeout: float = DEFAULT_NETWORK_TIMEOUT_SECONDS,
    fetch_json: JsonFetcher = _fetch_json,
) -> ComponentVersionStatus:
    current = installed or installed_sdk_version()
    path = cache_path or _default_cache_path()
    cache = _read_cache(path)
    now = _utc_now()
    cached_latest, cached_at, cache_fresh = _cached_latest(cache, "sdk", now=now, ttl=cache_ttl)

    if not refresh and cache_fresh:
        return _component_status(
            component="sdk",
            installed=current,
            latest=cached_latest,
            source="cache",
            checked_at=cached_at,
            release_url=SDK_RELEASE_URL,
        )
    if offline:
        return _component_status(
            component="sdk",
            installed=current,
            latest=cached_latest,
            source="cache-stale" if cached_latest else "offline",
            checked_at=cached_at,
            release_url=SDK_RELEASE_URL,
            error=None if cached_latest else "No cached SDK release information is available.",
        )

    try:
        payload = fetch_json(SDK_PYPI_JSON_URL, timeout)
        latest = str(payload["info"]["version"])
        if _normalized_version(latest) is None:
            raise ValueError(f"invalid PyPI version: {latest}")
        checked_at = _iso_utc(now)
        cache = {**cache, "schema": VERSION_CHECK_SCHEMA}
        cache["sdk"] = {"latest_version": latest, "checked_at": checked_at, "source": "pypi"}
        _write_cache(path, cache)
        return _component_status(
            component="sdk",
            installed=current,
            latest=latest,
            source="pypi",
            checked_at=checked_at,
            release_url=SDK_RELEASE_URL,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        return _component_status(
            component="sdk",
            installed=current,
            latest=cached_latest,
            source="cache-stale" if cached_latest else "unavailable",
            checked_at=cached_at,
            release_url=SDK_RELEASE_URL,
            error=str(exc),
        )


def _latest_app_release(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, list):
        raise ValueError("GitHub releases response is not a list")
    candidates: list[tuple[Version, str, str]] = []
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("draft"):
            continue
        tag = str(entry.get("tag_name", ""))
        if not tag.startswith(APP_TAG_PREFIX):
            continue
        raw_version = tag.removeprefix(APP_TAG_PREFIX)
        parsed = _normalized_version(raw_version)
        if parsed is None:
            continue
        url = str(entry.get("html_url") or APP_RELEASE_URL)
        candidates.append((parsed, raw_version, url))
    if not candidates:
        raise ValueError("No versioned Tauri beta release was found")
    _, version_text, release_url = max(candidates, key=lambda item: item[0])
    return version_text, release_url


def check_app_version(
    installed: str,
    *,
    refresh: bool = False,
    offline: bool = False,
    cache_path: Path | None = None,
    cache_ttl: timedelta = DEFAULT_CACHE_TTL,
    timeout: float = DEFAULT_NETWORK_TIMEOUT_SECONDS,
    fetch_json: JsonFetcher = _fetch_json,
) -> ComponentVersionStatus:
    path = cache_path or _default_cache_path()
    cache = _read_cache(path)
    now = _utc_now()
    cached_latest, cached_at, cache_fresh = _cached_latest(cache, "app", now=now, ttl=cache_ttl)
    raw_cached_entry = cache.get("app")
    cached_entry: dict[str, Any] = raw_cached_entry if isinstance(raw_cached_entry, dict) else {}
    cached_url = str(cached_entry.get("release_url") or APP_RELEASE_URL)

    if not refresh and cache_fresh:
        return _component_status(
            component="app",
            installed=installed,
            latest=cached_latest,
            source="cache",
            checked_at=cached_at,
            release_url=cached_url,
        )
    if offline:
        return _component_status(
            component="app",
            installed=installed,
            latest=cached_latest,
            source="cache-stale" if cached_latest else "offline",
            checked_at=cached_at,
            release_url=cached_url,
            error=None if cached_latest else "No cached desktop release information is available.",
        )

    try:
        payload = fetch_json(APP_RELEASES_API_URL, timeout)
        latest, release_url = _latest_app_release(payload)
        checked_at = _iso_utc(now)
        cache = {**cache, "schema": VERSION_CHECK_SCHEMA}
        cache["app"] = {
            "latest_version": latest,
            "checked_at": checked_at,
            "source": "github",
            "release_url": release_url,
        }
        _write_cache(path, cache)
        return _component_status(
            component="app",
            installed=installed,
            latest=latest,
            source="github",
            checked_at=checked_at,
            release_url=release_url,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return _component_status(
            component="app",
            installed=installed,
            latest=cached_latest,
            source="cache-stale" if cached_latest else "unavailable",
            checked_at=cached_at,
            release_url=cached_url,
            error=str(exc),
        )


def version_report(
    *,
    app_version: str | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": VERSION_CHECK_SCHEMA,
        "sdk": check_sdk_version(refresh=refresh, offline=offline).to_dict(),
        "environment": installed_sdk_environment(),
    }
    if app_version is not None:
        report["app"] = check_app_version(
            app_version,
            refresh=refresh,
            offline=offline,
        ).to_dict()
    return report


__all__ = [
    "APP_RELEASE_URL",
    "APP_RELEASES_API_URL",
    "ComponentVersionStatus",
    "SDK_DISTRIBUTION",
    "SDK_PYPI_JSON_URL",
    "SDK_RELEASE_URL",
    "check_app_version",
    "check_sdk_version",
    "clear_version_cache",
    "installed_sdk_environment",
    "installed_sdk_version",
    "version_is_newer",
    "version_report",
]
