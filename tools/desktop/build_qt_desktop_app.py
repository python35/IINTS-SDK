#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = {
    "cocoa": REPO_ROOT / "src" / "iints_desktop" / "cocoa_app.py",
    "qt": REPO_ROOT / "src" / "iints_desktop" / "qt_app.py",
    "tk": REPO_ROOT / "src" / "iints_desktop" / "app.py",
}
ICON_ASSETS = {
    "darwin": REPO_ROOT / "src" / "iints_desktop" / "assets" / "app_icon.icns",
    "win32": REPO_ROOT / "src" / "iints_desktop" / "assets" / "app_icon.ico",
    "default": REPO_ROOT / "src" / "iints_desktop" / "assets" / "app_icon.png",
}


def require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(f"Missing {module_name}. Install it with:\n  {install_hint}")


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    pyinstaller_cache = REPO_ROOT / ".pyinstaller-cache"
    matplotlib_cache = REPO_ROOT / ".mplt_desktop_build"
    pyinstaller_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(pyinstaller_cache))
    env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    return env


def build_command(*, backend: str, onefile: bool, windowed: bool, name: str) -> list[str]:
    entrypoint = ENTRYPOINTS[backend]
    icon_path = ICON_ASSETS["darwin"] if sys.platform == "darwin" else ICON_ASSETS["win32"] if sys.platform.startswith("win") else ICON_ASSETS["default"]
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        name,
        "--clean",
        "--noconfirm",
        "--collect-data",
        "iints",
        "--collect-data",
        "iints_desktop",
        "--collect-submodules",
        "iints_desktop",
        "--hidden-import",
        "iints.core.algorithms.clinical_baseline",
    ]
    if icon_path.exists():
        command.extend(["--icon", str(icon_path)])
    if backend == "qt":
        command.extend(
            [
                "--collect-all",
                "PySide6",
                "--collect-all",
                "shiboken6",
                "--hidden-import",
                "PySide6.QtSvg",
                "--hidden-import",
                "PySide6.QtXml",
            ]
        )
        if sys.platform != "darwin":
            command.extend(
                [
                    "--hidden-import",
                    "PySide6.QtWebEngineCore",
                    "--hidden-import",
                    "PySide6.QtWebEngineWidgets",
                ]
            )
    if backend == "cocoa":
        command.extend(
            [
                "--hidden-import",
                "Cocoa",
                "--hidden-import",
                "objc",
            ]
        )
    if onefile:
        command.append("--onefile")
    if windowed:
        command.append("--windowed")
    if sys.platform == "darwin":
        command.extend(["--osx-bundle-identifier", "org.iints.desktop"])
    command.append(str(entrypoint))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the IINTS-AF desktop app with PyInstaller.")
    parser.add_argument(
        "--backend",
        choices=sorted(ENTRYPOINTS),
        default="qt",
        help=(
            "Desktop UI backend to package. Use 'cocoa' for the macOS beta "
            "when Qt packaging is unstable."
        ),
    )
    parser.add_argument("--name", default="IINTS-AF-Desktop-Qt", help="Executable/app name.")
    parser.add_argument("--onedir", action="store_true", help="Build an onedir bundle instead of a single-file app.")
    parser.add_argument("--console", action="store_true", help="Keep a console window for debugging.")
    args = parser.parse_args()

    entrypoint = ENTRYPOINTS[args.backend]
    if not entrypoint.exists():
        raise SystemExit(f"Desktop entrypoint not found for backend {args.backend!r}: {entrypoint}")

    install_extra = "desktop"
    if args.backend == "cocoa":
        install_extra = "desktop-macos"
    require_module("PyInstaller", f'python -m pip install -U -e ".[{install_extra}]"')
    if args.backend == "qt":
        require_module("PySide6", 'python -m pip install -U -e ".[desktop]"')
    if args.backend == "cocoa":
        require_module("Cocoa", 'python -m pip install -U -e ".[desktop-macos]"')

    command = build_command(
        backend=args.backend,
        onefile=not args.onedir,
        windowed=not args.console,
        name=args.name,
    )
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=REPO_ROOT, env=build_environment())


if __name__ == "__main__":
    raise SystemExit(main())
