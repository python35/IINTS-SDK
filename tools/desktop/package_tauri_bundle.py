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


def package_updater_artifact(bundle_root: Path, output_dir: Path, platform_label: str) -> Path | None:
    """Copy the signed updater artifact (if `createUpdaterArtifacts` produced one).

    `tauri build` writes exactly one `<artifact>.sig` file per platform next to
    the artifact it signs -- an `.app.tar.gz` on macOS, the installer itself on
    Windows, or the AppImage itself on Linux. Which of those it is varies by
    platform and by Tauri version, so this discovers the pairing from the
    `.sig` file rather than hard-coding an extension. Returns None (skipping
    quietly) when no signing key was configured for this build, matching the
    optional-signing fallback used for code-signing certificates elsewhere in
    this pipeline.
    """

    if platform_label not in RELEASE_NAMES:
        choices = ", ".join(sorted(RELEASE_NAMES))
        raise ValueError(f"Unsupported platform label {platform_label!r}; use one of: {choices}")

    signatures = sorted(bundle_root.rglob("*.sig"))
    if not signatures:
        return None
    if len(signatures) > 1:
        names = ", ".join(str(path.relative_to(bundle_root)) for path in signatures)
        raise RuntimeError(f"Expected at most one updater signature, found: {names}")

    signature_path = signatures[0]
    artifact_path = signature_path.with_suffix("")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Updater signature {signature_path} has no matching artifact {artifact_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Path.suffixes would also pick up dots from an embedded version number
    # (e.g. "App_0.2.10_x64-setup.exe"), so this only special-cases the one
    # real compound extension Tauri produces (macOS's .app.tar.gz) and
    # otherwise keeps just the final extension.
    if artifact_path.name.endswith(".tar.gz"):
        stable_suffix = ".tar.gz"
    elif artifact_path.suffix:
        stable_suffix = artifact_path.suffix
    else:
        raise RuntimeError(f"Updater artifact {artifact_path} has no recognizable file extension")
    destination = output_dir / f"IINTS-AF-Research-Workbench-{platform_label}-update{stable_suffix}"
    shutil.copy2(artifact_path, destination)
    shutil.copy2(signature_path, destination.with_name(f"{destination.name}.sig"))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform-label", choices=sorted(RELEASE_NAMES), required=True)
    args = parser.parse_args()

    packaged = package_bundle(args.bundle_root, args.output_dir, args.platform_label)
    print(packaged)

    updater_artifact = package_updater_artifact(args.bundle_root, args.output_dir, args.platform_label)
    if updater_artifact is not None:
        print(updater_artifact)


if __name__ == "__main__":
    main()
