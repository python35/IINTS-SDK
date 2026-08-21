from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

from iints.versioning import (
    APP_RELEASES_API_URL,
    SDK_PYPI_JSON_URL,
    check_app_version,
    check_sdk_version,
    installed_sdk_environment,
    version_is_newer,
)


def test_version_comparison_uses_pep_440() -> None:
    assert version_is_newer("1.6.0", "1.5.34") is True
    assert version_is_newer("1.5.34", "1.5.34") is False
    assert version_is_newer("not-a-version", "1.5.34") is False


def test_sdk_check_reports_pypi_update_and_writes_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "version-cache.json"

    def fetch(url: str, _timeout: float):
        assert url == SDK_PYPI_JSON_URL
        return {"info": {"version": "1.6.0"}}

    result = check_sdk_version(
        installed="1.5.34",
        refresh=True,
        cache_path=cache_path,
        fetch_json=fetch,
    )

    assert result.status == "update_available"
    assert result.update_available is True
    assert result.latest_version == "1.6.0"
    assert result.source == "pypi"
    assert json.loads(cache_path.read_text(encoding="utf-8"))["sdk"]["latest_version"] == "1.6.0"


def test_sdk_check_falls_back_to_stale_cache_on_network_failure(tmp_path: Path) -> None:
    cache_path = tmp_path / "version-cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "sdk": {
                    "latest_version": "1.5.35",
                    "checked_at": "2020-01-01T00:00:00Z",
                    "source": "pypi",
                },
            }
        ),
        encoding="utf-8",
    )

    def fail(_url: str, _timeout: float):
        raise RuntimeError("offline")

    result = check_sdk_version(
        installed="1.5.34",
        refresh=True,
        cache_path=cache_path,
        cache_ttl=timedelta(seconds=0),
        fetch_json=fail,
    )

    assert result.latest_version == "1.5.35"
    assert result.update_available is True
    assert result.source == "cache-stale"
    assert result.error == "offline"


def test_sdk_offline_without_cache_is_explicitly_unknown(tmp_path: Path) -> None:
    result = check_sdk_version(
        installed="1.5.34",
        offline=True,
        cache_path=tmp_path / "missing.json",
    )

    assert result.status == "unknown"
    assert result.update_available is None
    assert result.latest_version is None
    assert result.source == "offline"


def test_app_check_ignores_stable_alias_and_selects_highest_version(tmp_path: Path) -> None:
    def fetch(url: str, _timeout: float):
        assert url == APP_RELEASES_API_URL
        return [
            {"tag_name": "tauri-beta-latest", "draft": False, "html_url": "https://example.invalid/latest"},
            {"tag_name": "tauri-beta-v0.2.8", "draft": False, "html_url": "https://example.invalid/028"},
            {"tag_name": "tauri-beta-v0.2.7", "draft": False, "html_url": "https://example.invalid/027"},
            {"tag_name": "tauri-beta-v9.9.9", "draft": True, "html_url": "https://example.invalid/draft"},
        ]

    result = check_app_version(
        "0.2.7",
        refresh=True,
        cache_path=tmp_path / "cache.json",
        fetch_json=fetch,
    )

    assert result.latest_version == "0.2.8"
    assert result.update_available is True
    assert result.release_url == "https://example.invalid/028"


def test_environment_uses_invoked_cli_and_preserves_venv_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    global_python = tmp_path / "base" / "bin" / "python3"
    invoked_cli = tmp_path / "venv" / "bin" / "iints"
    invoked_cli.parent.mkdir(parents=True)
    invoked_cli.write_text(f"#!{global_python}\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [str(invoked_cli), "version"])
    monkeypatch.setattr("sys.executable", str(venv_python))

    environment = installed_sdk_environment()

    assert environment["cli_path"] == str(invoked_cli)
    assert environment["cli_shebang"] == str(global_python)
    assert environment["cli_matches_python"] is False
