#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
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
OPTIONAL_BUNDLED_MODULES = ("plotly.graph_objects", "roadrunner", "fmpy")
BINARY_BUNDLED_MODULES = ("fmpy", "roadrunner")


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


def add_fmpy_sundials_binaries(command: list[str]) -> None:
    """Add the active-platform SUNDIALS binaries at FMPy's expected paths."""

    spec = importlib.util.find_spec("fmpy")
    if spec is None or not spec.submodule_search_locations:
        return
    fmpy_module = importlib.import_module("fmpy")
    platform_tuple = str(getattr(fmpy_module, "platform_tuple", ""))
    extension = str(getattr(fmpy_module, "sharedLibraryExtension", ""))
    if not platform_tuple or not extension:
        raise RuntimeError("FMPy does not expose its platform tuple or library extension.")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    source_dir = package_dir / "sundials" / platform_tuple
    binaries = sorted(source_dir.glob(f"*{extension}"))
    if not binaries:
        raise RuntimeError(f"No FMPy SUNDIALS binaries found in {source_dir}.")

    destination = f"fmpy/sundials/{platform_tuple}"
    for binary in binaries:
        command.extend(["--add-binary", f"{binary}{os.pathsep}{destination}"])


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
        "--hidden-import",
        "iints.research.academic_bundle",
        "--hidden-import",
        "iints.research.genomics_engine",
        "--hidden-import",
        "iints.research.mechanistic_models",
        "--hidden-import",
        "iints.research.structure",
        "--hidden-import",
        "iints.research.tissue_stressor",
        # Pandas exposes optional test helpers that can pull pytest and the
        # removed pkg_resources API into an otherwise production-only bundle.
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "pkg_resources",
    ]
    # These engines are imported lazily by the research workbench. Hidden
    # imports trigger their normal PyInstaller hooks without collecting test,
    # example, and unrelated GUI modules from the whole distributions.
    for module_name in OPTIONAL_BUNDLED_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            command.extend(["--hidden-import", module_name])
    # FMPy and libRoadRunner load nested native libraries at runtime. Their
    # Linux shared objects are not always discovered from Python imports, so
    # preserve the package-relative binary paths explicitly. This is also
    # harmless on Windows and macOS and keeps all platform bundles symmetric.
    for module_name in BINARY_BUNDLED_MODULES:
        if importlib.util.find_spec(module_name) is not None:
            command.extend(["--collect-binaries", module_name])
    add_fmpy_sundials_binaries(command)
    if icon_path.exists():
        command.extend(["--icon", str(icon_path)])
    if backend == "qt":
        command.extend(
            [
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

    install_extra = "desktop-all"
    if args.backend == "cocoa":
        install_extra = "desktop-macos"
    require_module("PyInstaller", f'python -m pip install -U -e ".[{install_extra}]"')
    if args.backend == "qt":
        require_module("PySide6", 'python -m pip install -U -e ".[desktop-all]"')
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
