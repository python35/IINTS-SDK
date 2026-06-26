#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "src" / "iints_desktop" / "app.py"


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


def build_command(*, onefile: bool, windowed: bool, name: str) -> list[str]:
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
        "--hidden-import",
        "iints.core.algorithms.clinical_baseline",
    ]
    if onefile:
        command.append("--onefile")
    if windowed:
        command.append("--windowed")
    command.append(str(ENTRYPOINT))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the IINTS-AF desktop companion app with PyInstaller.")
    parser.add_argument("--name", default="IINTS-AF-Desktop", help="Executable/app name.")
    parser.add_argument("--onedir", action="store_true", help="Build an onedir bundle instead of a single-file app.")
    parser.add_argument("--console", action="store_true", help="Keep a console window for debugging.")
    args = parser.parse_args()

    if not ENTRYPOINT.exists():
        raise SystemExit(f"Desktop entrypoint not found: {ENTRYPOINT}")

    require_module("PyInstaller", 'python -m pip install -U -e ".[desktop]"')

    command = build_command(onefile=not args.onedir, windowed=not args.console, name=args.name)
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=REPO_ROOT, env=build_environment())


if __name__ == "__main__":
    raise SystemExit(main())
