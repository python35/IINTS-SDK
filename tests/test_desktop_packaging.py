from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module(path: str):
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_repair_module():
    return _load_module("tools/desktop/repair_fmpy_macos_dylibs.py")


def test_fmpy_dependency_name_mapping_is_version_independent() -> None:
    module = _load_repair_module()

    assert (
        module.local_name_for_dependency(
            "/build/install/lib/libsundials_core.7.dylib"
        )
        == "sundials_core.dylib"
    )
    assert (
        module.local_name_for_dependency(
            "/build/install/lib/libsundials_sunmatrixdense.5.2.1.dylib"
        )
        == "sundials_sunmatrixdense.dylib"
    )
    assert module.local_name_for_dependency("/usr/lib/libSystem.B.dylib") is None


def _load_package_tauri_bundle_module():
    return _load_module("tools/desktop/package_tauri_bundle.py")


def _load_build_update_manifest_module():
    return _load_module("tools/desktop/build_update_manifest.py")


def test_package_updater_artifact_pairs_the_sig_with_its_own_artifact(tmp_path: Path) -> None:
    module = _load_package_tauri_bundle_module()
    bundle_root = tmp_path / "bundle"
    macos_dir = bundle_root / "macos"
    macos_dir.mkdir(parents=True)
    artifact = macos_dir / "IINTS-AF Research Workbench.app.tar.gz"
    artifact.write_bytes(b"fake-tarball")
    (macos_dir / f"{artifact.name}.sig").write_text("fake-signature-block\n", encoding="utf-8")

    output_dir = tmp_path / "dist"
    destination = module.package_updater_artifact(bundle_root, output_dir, "macos")

    assert destination is not None
    assert destination.name == "IINTS-AF-Research-Workbench-macos-update.tar.gz"
    assert destination.read_bytes() == b"fake-tarball"
    assert destination.with_name(f"{destination.name}.sig").read_text(encoding="utf-8") == "fake-signature-block\n"


def test_package_updater_artifact_ignores_dots_in_an_embedded_version_number(tmp_path: Path) -> None:
    """A dotted version like "_0.2.10_" in the source filename must not leak into the suffix."""
    module = _load_package_tauri_bundle_module()
    bundle_root = tmp_path / "bundle"
    nsis_dir = bundle_root / "nsis"
    nsis_dir.mkdir(parents=True)
    artifact = nsis_dir / "IINTS-AF Research Workbench_0.2.10_x64-setup.exe"
    artifact.write_bytes(b"fake-installer")
    (nsis_dir / f"{artifact.name}.sig").write_text("fake-signature-block\n", encoding="utf-8")

    destination = module.package_updater_artifact(bundle_root, tmp_path / "dist", "windows-x64")

    assert destination is not None
    assert destination.name == "IINTS-AF-Research-Workbench-windows-x64-update.exe"


def test_package_updater_artifact_skips_quietly_without_a_signing_key(tmp_path: Path) -> None:
    module = _load_package_tauri_bundle_module()
    bundle_root = tmp_path / "bundle"
    (bundle_root / "macos").mkdir(parents=True)

    assert module.package_updater_artifact(bundle_root, tmp_path / "dist", "macos") is None


def test_package_updater_artifact_rejects_a_sig_with_no_matching_file(tmp_path: Path) -> None:
    module = _load_package_tauri_bundle_module()
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "orphan.tar.gz.sig").write_text("sig", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        module.package_updater_artifact(bundle_root, tmp_path / "dist", "macos")


def _write_update_artifact(assets_dir: Path, platform_label: str, suffix: str, signature: str) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    artifact = assets_dir / f"IINTS-AF-Research-Workbench-{platform_label}-update{suffix}"
    artifact.write_bytes(b"fake-artifact")
    artifact.with_name(f"{artifact.name}.sig").write_text(signature, encoding="utf-8")


def test_build_update_manifest_maps_every_platform_and_embeds_signatures(tmp_path: Path) -> None:
    module = _load_build_update_manifest_module()
    assets_dir = tmp_path / "release-assets"
    _write_update_artifact(assets_dir, "macos", ".tar.gz", "mac-signature")
    _write_update_artifact(assets_dir, "windows-x64", ".exe", "windows-signature")
    _write_update_artifact(assets_dir, "linux-x64", ".AppImage", "linux-signature")

    manifest = module.build_manifest(
        assets_dir=assets_dir,
        version="0.2.10",
        notes="Test release notes",
        download_base_url="https://example.invalid/releases/download/tauri-beta-latest",
    )

    assert manifest["version"] == "0.2.10"
    assert manifest["notes"] == "Test release notes"
    assert manifest["pub_date"].endswith("Z")
    assert manifest["platforms"] == {
        "darwin-aarch64": {
            "signature": "mac-signature",
            "url": "https://example.invalid/releases/download/tauri-beta-latest/"
            "IINTS-AF-Research-Workbench-macos-update.tar.gz",
        },
        "windows-x86_64": {
            "signature": "windows-signature",
            "url": "https://example.invalid/releases/download/tauri-beta-latest/"
            "IINTS-AF-Research-Workbench-windows-x64-update.exe",
        },
        "linux-x86_64": {
            "signature": "linux-signature",
            "url": "https://example.invalid/releases/download/tauri-beta-latest/"
            "IINTS-AF-Research-Workbench-linux-x64-update.AppImage",
        },
    }
    json.dumps(manifest)  # must be plain-JSON serializable, as it is written verbatim


def test_build_update_manifest_fails_loudly_when_a_platform_is_missing(tmp_path: Path) -> None:
    module = _load_build_update_manifest_module()
    assets_dir = tmp_path / "release-assets"
    _write_update_artifact(assets_dir, "macos", ".app.tar.gz", "mac-signature")
    # windows-x64 and linux-x64 are missing.

    with pytest.raises(FileNotFoundError):
        module.build_manifest(
            assets_dir=assets_dir,
            version="0.2.10",
            notes="Test release notes",
            download_base_url="https://example.invalid/releases/download/tauri-beta-latest",
        )
