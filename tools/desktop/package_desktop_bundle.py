#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def _candidate_bundle_paths(dist_dir: Path, app_name: str) -> list[Path]:
    return [
        dist_dir / f"{app_name}.app",
        dist_dir / app_name,
        dist_dir / f"{app_name}.exe",
        dist_dir / f"{app_name}.bin",
    ]


def find_bundle(dist_dir: Path, app_name: str) -> Path:
    for candidate in _candidate_bundle_paths(dist_dir, app_name):
        if candidate.exists():
            return candidate
    found = "\n".join(str(path) for path in sorted(dist_dir.glob("*"))) or "<empty>"
    raise SystemExit(
        f"Could not find a PyInstaller bundle for {app_name!r} in {dist_dir}.\nFound:\n{found}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_executable(source: Path, target: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Expected executable file, got: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    mode = target.stat().st_mode
    target.chmod(mode | 0o111)


def _bundle_executable(bundle: Path, app_name: str, suffix: str = "") -> Path:
    if bundle.is_file():
        return bundle
    candidate = bundle / f"{app_name}{suffix}"
    if candidate.exists():
        return candidate
    found = "\n".join(str(path) for path in sorted(bundle.glob("*"))) or "<empty>"
    raise SystemExit(f"Could not find executable for {app_name!r} in {bundle}.\nFound:\n{found}")


def _create_dmg(bundle: Path, output_path: Path, app_name: str) -> None:
    if bundle.suffix != ".app":
        raise SystemExit(
            f"macOS packaging expects a .app bundle so users can download a .dmg. Got: {bundle}"
        )
    if shutil.which("hdiutil") is None:
        raise SystemExit("hdiutil is required to package the macOS desktop app as a .dmg.")
    if output_path.exists():
        output_path.unlink()
    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            app_name,
            "-srcfolder",
            str(bundle),
            "-ov",
            "-format",
            "UDZO",
            str(output_path),
        ],
        check=True,
    )


def package_release_asset(bundle: Path, app_name: str, platform_label: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if platform_label.startswith("windows"):
        source = _bundle_executable(bundle, app_name, ".exe")
        target = output_dir / f"{app_name}-{platform_label}.exe"
        _copy_executable(source, target)
        return target

    if platform_label == "macos":
        target = output_dir / f"{app_name}-{platform_label}.dmg"
        _create_dmg(bundle, target, app_name)
        return target

    if platform_label.startswith("linux"):
        source = _bundle_executable(bundle, app_name)
        target = output_dir / f"{app_name}-{platform_label}"
        _copy_executable(source, target)
        return target

    raise SystemExit(f"Unsupported platform label for direct desktop asset packaging: {platform_label}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a PyInstaller desktop bundle as a direct release asset.")
    parser.add_argument("--dist-dir", default="dist", help="PyInstaller dist directory.")
    parser.add_argument("--app-name", default="IINTS-AF-Desktop-Beta", help="PyInstaller app/executable name.")
    parser.add_argument("--platform-label", required=True, help="Release platform label, e.g. windows-x64.")
    parser.add_argument("--output-dir", default="desktop-dist", help="Directory for release archives.")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    bundle = find_bundle(dist_dir, args.app_name)
    asset_path = package_release_asset(bundle, args.app_name, args.platform_label, output_dir)

    checksum = sha256_file(asset_path)
    checksum_path = asset_path.with_name(asset_path.name + ".sha256")
    checksum_path.write_text(f"{checksum}  {asset_path.name}\n", encoding="utf-8")

    print(f"Packaged {bundle} -> {asset_path}")
    print(f"SHA256 {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
