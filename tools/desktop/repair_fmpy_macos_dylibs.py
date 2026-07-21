#!/usr/bin/env python3
"""Repair non-relocatable SUNDIALS links in the installed FMPy macOS wheel.

FMPy ships its SUNDIALS libraries beside the Python package. Some wheel builds
retain absolute CI build paths in Mach-O dependency records. A packaged app
cannot resolve those paths, so this release-only helper rewrites them to
``@loader_path`` before PyInstaller analyzes and copies the libraries.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import re
import subprocess


SUNDIALS_DEPENDENCY = re.compile(r"/libsundials_([A-Za-z0-9_]+)(?:\.[0-9]+)*\.dylib$")


def local_name_for_dependency(dependency: str) -> str | None:
    match = SUNDIALS_DEPENDENCY.search(dependency)
    if match is None:
        return None
    return f"sundials_{match.group(1)}.dylib"


def linked_libraries(path: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["otool", "-L", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        value = stripped.split(" (compatibility version", 1)[0]
        if value:
            rows.append(value)
    return tuple(rows)


def repair_fmpy_macos_dylibs(package_dir: Path) -> tuple[Path, ...]:
    sundials_root = package_dir / "sundials"
    candidates = sorted(sundials_root.glob("*-darwin/*.dylib"))
    if not candidates:
        raise RuntimeError(f"No FMPy macOS SUNDIALS libraries found under {sundials_root}.")

    repaired: list[Path] = []
    for library in candidates:
        subprocess.run(
            ["install_name_tool", "-id", f"@loader_path/{library.name}", str(library)],
            check=True,
        )
        for dependency in linked_libraries(library):
            local_name = local_name_for_dependency(dependency)
            if local_name is None or not dependency.startswith("/"):
                continue
            local_target = library.parent / local_name
            if not local_target.is_file():
                raise RuntimeError(
                    f"Cannot make FMPy dependency relocatable: {dependency} has no {local_target}."
                )
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    dependency,
                    f"@loader_path/{local_name}",
                    str(library),
                ],
                check=True,
            )
        unresolved = [
            dependency
            for dependency in linked_libraries(library)
            if dependency.startswith("/") and "libSystem.B.dylib" not in dependency
        ]
        if unresolved:
            raise RuntimeError(
                f"FMPy library still contains non-system absolute dependencies: {library}: {unresolved}"
            )
        repaired.append(library)
    return tuple(repaired)


def installed_fmpy_dir() -> Path:
    spec = importlib.util.find_spec("fmpy")
    if spec is None or spec.origin is None:
        raise RuntimeError("FMPy is not installed. Install the desktop-all extra first.")
    return Path(spec.origin).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-platform",
        choices=("Darwin",),
        help="Test hook; normally the helper runs only on macOS.",
    )
    args = parser.parse_args()
    system = args.force_platform or platform.system()
    if system != "Darwin":
        print("FMPy Mach-O repair skipped: not macOS.")
        return 0
    repaired = repair_fmpy_macos_dylibs(installed_fmpy_dir())
    print(f"Repaired {len(repaired)} FMPy SUNDIALS libraries for relocatable app packaging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
