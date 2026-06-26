#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
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


def add_to_zip(zip_file: zipfile.ZipFile, source: Path, archive_root: str) -> None:
    if source.is_file():
        zip_file.write(source, archive_root)
        return

    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            zip_file.write(path, str(Path(archive_root) / relative))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a PyInstaller desktop bundle as a release zip.")
    parser.add_argument("--dist-dir", default="dist", help="PyInstaller dist directory.")
    parser.add_argument("--app-name", default="IINTS-AF-Desktop-Beta", help="PyInstaller app/executable name.")
    parser.add_argument("--platform-label", required=True, help="Release platform label, e.g. windows-x64.")
    parser.add_argument("--output-dir", default="desktop-dist", help="Directory for release archives.")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = find_bundle(dist_dir, args.app_name)
    archive_name = f"{args.app_name}-{args.platform_label}.zip"
    archive_path = output_dir / archive_name
    archive_root = bundle.name

    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(archive_path, "w", compression=compression, compresslevel=9) as zip_file:
        add_to_zip(zip_file, bundle, archive_root)

    checksum = sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive_name}\n", encoding="utf-8")

    print(f"Packaged {bundle} -> {archive_path}")
    print(f"SHA256 {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
