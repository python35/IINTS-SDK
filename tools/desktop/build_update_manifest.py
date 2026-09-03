"""Build the Tauri updater's latest.json manifest from packaged release assets.

Reads the per-platform `IINTS-AF-Research-Workbench-<platform>-update.*` /
`.sig` pairs written by `package_tauri_bundle.py` and combines them with the
released version into the manifest shape the Tauri updater plugin expects.
Every configured platform must have an artifact; a build that silently
dropped one would otherwise leave that platform unable to auto-update without
any visible failure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# Maps this project's existing CI matrix labels to the target identifiers the
# Tauri updater plugin looks up in latest.json. macOS CI runs on Apple
# Silicon (`macos-latest`), so this is darwin-aarch64, not darwin-x86_64 --
# revisit this mapping if the runner architecture or matrix ever changes.
PLATFORM_TARGETS = {
    "windows-x64": "windows-x86_64",
    "macos": "darwin-aarch64",
    "linux-x64": "linux-x86_64",
}


def _find_update_artifact(assets_dir: Path, platform_label: str) -> Path:
    candidates = sorted(
        path
        for path in assets_dir.glob(f"IINTS-AF-Research-Workbench-{platform_label}-update.*")
        if path.suffix != ".sig"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No updater artifact for platform {platform_label!r} in {assets_dir}. "
            "Check that createUpdaterArtifacts and the TAURI_SIGNING_PRIVATE_KEY "
            "secret were both active for that build."
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"Expected one updater artifact for {platform_label!r}, found: {names}")
    return candidates[0]


def build_manifest(
    assets_dir: Path,
    version: str,
    notes: str,
    download_base_url: str,
) -> dict:
    platforms: dict[str, dict[str, str]] = {}
    for platform_label, target in PLATFORM_TARGETS.items():
        artifact = _find_update_artifact(assets_dir, platform_label)
        signature = artifact.with_name(f"{artifact.name}.sig").read_text(encoding="utf-8").strip()
        platforms[target] = {
            "signature": signature,
            "url": f"{download_base_url.rstrip('/')}/{artifact.name}",
        }

    return {
        "version": version,
        "notes": notes,
        "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": platforms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--version", required=True, help="App version, without a leading v")
    parser.add_argument("--notes-file", type=Path, required=True)
    parser.add_argument(
        "--download-base-url",
        default="https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        assets_dir=args.assets_dir,
        version=args.version,
        notes=args.notes_file.read_text(encoding="utf-8"),
        download_base_url=args.download_base_url,
    )
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
