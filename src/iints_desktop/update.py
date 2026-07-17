from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from importlib import metadata


DESKTOP_RELEASE_URL = "https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-latest"
UPDATE_DOCS_URL = "https://python35.github.io/IINTS-SDK/APP_INSTALL/"
PYTHON_SDK_UPDATE_PACKAGE = "iints-sdk-python35[full,desktop,mdmp,research,edge]"


@dataclass(frozen=True)
class DesktopUpdateInfo:
    current_version: str
    python_executable: str
    package_spec: str
    pip_command_args: tuple[str, ...]
    pip_command: str
    app_download_url: str
    update_docs_url: str
    research_only: bool = True
    medical_device: bool = False


def current_sdk_version() -> str:
    try:
        return metadata.version("iints-sdk-python35")
    except metadata.PackageNotFoundError:
        try:
            import iints

            return str(getattr(iints, "__version__", "unknown"))
        except Exception:
            return "unknown"


def build_python_sdk_update_args() -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        PYTHON_SDK_UPDATE_PACKAGE,
    )


def build_python_sdk_update_command() -> str:
    return shlex.join(build_python_sdk_update_args())


def get_desktop_update_info() -> DesktopUpdateInfo:
    args = build_python_sdk_update_args()
    return DesktopUpdateInfo(
        current_version=current_sdk_version(),
        python_executable=sys.executable,
        package_spec=PYTHON_SDK_UPDATE_PACKAGE,
        pip_command_args=args,
        pip_command=shlex.join(args),
        app_download_url=DESKTOP_RELEASE_URL,
        update_docs_url=UPDATE_DOCS_URL,
    )
