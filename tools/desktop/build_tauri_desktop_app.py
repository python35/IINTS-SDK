#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TAURI_APP_DIR = REPO_ROOT / "apps" / "iints-tauri"


def require_tool(name: str, install_hint: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing {name}. {install_hint}")


def run(command: list[str], *, cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the experimental IINTS-AF Tauri desktop shell.")
    parser.add_argument("--skip-install", action="store_true", help="Do not run npm install before building.")
    parser.add_argument("--dev", action="store_true", help="Run `npm run tauri dev` instead of a production build.")
    args = parser.parse_args()

    if not TAURI_APP_DIR.exists():
        raise SystemExit(f"Tauri app directory not found: {TAURI_APP_DIR}")

    require_tool("npm", "Install Node.js and npm first.")
    require_tool("cargo", "Install Rust from https://rustup.rs first.")

    if not args.skip_install:
        run(["npm", "install"], cwd=TAURI_APP_DIR)
    run(["npm", "run", "check"], cwd=TAURI_APP_DIR)
    run(["npm", "run", "tauri", "dev" if args.dev else "build"], cwd=TAURI_APP_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
