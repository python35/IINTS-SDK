from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools" / "install" / "install_omarchy.sh"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_omarchy_installer_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_standard_dry_run_uses_omarchy_and_fixed_mise_python() -> None:
    result = _run("--dry-run", "--profile", "standard")

    assert result.returncode == 0, result.stderr
    assert "omarchy pkg add mise" in result.stdout
    assert "mise use --global python@3.11" in result.stdout
    assert "SDK:     iints-sdk-python35[full,mdmp]" in result.stdout
    assert "doctor --smoke-run --suggest" in result.stdout
    assert "Mode:    dry run" in result.stdout


def test_research_profile_and_version_pin_are_visible() -> None:
    result = _run(
        "--dry-run",
        "--profile",
        "research",
        "--version",
        "1.5.34",
    )

    assert result.returncode == 0, result.stderr
    assert "iints-sdk-python35[full,mdmp,research]==1.5.34" in result.stdout
    assert "Desktop: false" in result.stdout


def test_desktop_dry_run_verifies_the_stable_appimage() -> None:
    result = _run("--dry-run", "--desktop")

    assert result.returncode == 0, result.stderr
    assert "omarchy pkg add fuse2" in result.stdout
    assert "IINTS-AF-Research-Workbench-linux-x64.AppImage" in result.stdout
    assert "sha256sum --check" in result.stdout
    assert "org.iints.research-workbench.desktop" in result.stdout


def test_desktop_profile_uses_complete_python_engine() -> None:
    result = _run("--dry-run", "--profile", "desktop")

    assert result.returncode == 0, result.stderr
    assert "SDK:     iints-sdk-python35[desktop-all]" in result.stdout
    assert "Desktop: true" in result.stdout


def test_unknown_profile_is_rejected() -> None:
    result = _run("--dry-run", "--profile", "unknown")

    assert result.returncode != 0
    assert "Unknown profile" in result.stderr
