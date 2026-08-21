from __future__ import annotations

import shlex
import platform
import subprocess
import sys
from dataclasses import dataclass

from iints.versioning import (
    APP_RELEASE_URL,
    check_app_version,
    check_sdk_version,
    installed_sdk_environment,
)


DESKTOP_RELEASE_URL = APP_RELEASE_URL
UPDATE_DOCS_URL = "https://python35.github.io/IINTS-SDK/APP_INSTALL/"
PYTHON_SDK_UPDATE_PACKAGE = "iints-sdk-python35[desktop-all]"


@dataclass(frozen=True)
class DesktopUpdateInfo:
    current_version: str
    active_code_version: str
    version_metadata_matches_code: bool | None
    latest_version: str | None
    sdk_status: str
    sdk_update_available: bool | None
    sdk_check_source: str
    sdk_checked_at: str | None
    sdk_check_error: str | None
    app_current_version: str | None
    app_latest_version: str | None
    app_status: str | None
    app_update_available: bool | None
    app_check_source: str | None
    app_checked_at: str | None
    app_check_error: str | None
    python_executable: str
    package_spec: str
    pip_command_args: tuple[str, ...]
    pip_command: str
    app_download_url: str
    update_docs_url: str
    research_only: bool = True
    medical_device: bool = False


def current_sdk_version() -> str:
    return check_sdk_version(offline=True).installed_version


def build_python_sdk_update_args(version: str | None = None) -> tuple[str, ...]:
    package_spec = PYTHON_SDK_UPDATE_PACKAGE
    if version:
        package_spec = f"{package_spec}=={version}"
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        package_spec,
    )


def build_python_sdk_update_command(version: str | None = None) -> str:
    return format_shell_command(build_python_sdk_update_args(version))


def format_shell_command(
    args: tuple[str, ...] | list[str],
    *,
    platform_name: str | None = None,
) -> str:
    """Quote argv for the shell used by the current desktop platform."""

    system = (platform_name or platform.system()).lower()
    values = [str(value) for value in args]
    if system == "windows":
        return subprocess.list2cmdline(values)
    return shlex.join(values)


def get_desktop_update_info(
    *,
    app_version: str | None = None,
    refresh: bool = False,
    offline: bool = False,
) -> DesktopUpdateInfo:
    sdk = check_sdk_version(refresh=refresh, offline=offline)
    environment = installed_sdk_environment()
    app = (
        check_app_version(app_version, refresh=refresh, offline=offline)
        if app_version is not None
        else None
    )
    args = build_python_sdk_update_args(sdk.latest_version)
    return DesktopUpdateInfo(
        current_version=sdk.installed_version,
        active_code_version=str(environment.get("code_version", "unknown")),
        version_metadata_matches_code=environment.get("version_metadata_matches_code"),
        latest_version=sdk.latest_version,
        sdk_status=sdk.status,
        sdk_update_available=sdk.update_available,
        sdk_check_source=sdk.source,
        sdk_checked_at=sdk.checked_at,
        sdk_check_error=sdk.error,
        app_current_version=app.installed_version if app else app_version,
        app_latest_version=app.latest_version if app else None,
        app_status=app.status if app else None,
        app_update_available=app.update_available if app else None,
        app_check_source=app.source if app else None,
        app_checked_at=app.checked_at if app else None,
        app_check_error=app.error if app else None,
        python_executable=sys.executable,
        package_spec=PYTHON_SDK_UPDATE_PACKAGE,
        pip_command_args=args,
        pip_command=format_shell_command(list(args)),
        app_download_url=app.release_url if app else DESKTOP_RELEASE_URL,
        update_docs_url=UPDATE_DOCS_URL,
    )
