"""Copy a platform-native Tauri installer to a stable release filename."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


RELEASE_NAMES = {
    "windows-x64": "IINTS-AF-Research-Workbench-windows-x64-setup.exe",
    "macos": "IINTS-AF-Research-Workbench-macos.dmg",
    "linux-x64": "IINTS-AF-Research-Workbench-linux-x64.AppImage",
}

SEARCH_PATTERNS = {
    "windows-x64": ("nsis", "*.exe"),
    "macos": ("dmg", "*.dmg"),
    "linux-x64": ("appimage", "*.AppImage"),
}


def _find_bundle(bundle_root: Path, platform_label: str) -> Path:
    subdirectory, pattern = SEARCH_PATTERNS[platform_label]
    candidates = sorted((bundle_root / subdirectory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No Tauri bundle matching {subdirectory}/{pattern} under {bundle_root}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"Expected one Tauri bundle, found: {names}")
    return candidates[0]


def package_bundle(bundle_root: Path, output_dir: Path, platform_label: str) -> Path:
    """Copy one installer and write a matching SHA-256 checksum file."""

    if platform_label not in RELEASE_NAMES:
        choices = ", ".join(sorted(RELEASE_NAMES))
        raise ValueError(f"Unsupported platform label {platform_label!r}; use one of: {choices}")

    source = _find_bundle(bundle_root, platform_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / RELEASE_NAMES[platform_label]
    shutil.copy2(source, destination)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    checksum = destination.with_name(f"{destination.name}.sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-label", choices=sorted(RELEASE_NAMES), required=True)
    args = parser.parse_args()

    packaged = package_bundle(args.bundle_root, args.output_dir, args.platform_label)
    print(packaged)


if __name__ == "__main__":
    main()
