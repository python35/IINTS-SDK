#!/usr/bin/env python3
from __future__ import annotations

import json
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "iints-cache"))

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_IO_PATH = REPO_ROOT / "src" / "iints" / "utils" / "run_io.py"
RUN_IO_SPEC = importlib.util.spec_from_file_location("_iints_run_io", RUN_IO_PATH)
assert RUN_IO_SPEC is not None
assert RUN_IO_SPEC.loader is not None
RUN_IO_MODULE = importlib.util.module_from_spec(RUN_IO_SPEC)
RUN_IO_SPEC.loader.exec_module(RUN_IO_MODULE)
build_run_manifest = RUN_IO_MODULE.build_run_manifest


def _check_license() -> List[str]:
    issues: List[str] = []
    license_path = REPO_ROOT / "LICENSE"
    if not license_path.exists():
        return ["Missing LICENSE file."]
    text = license_path.read_text(encoding="utf-8").strip()
    if not text:
        issues.append("LICENSE file is empty.")
    return issues


def _check_sbom() -> List[str]:
    issues: List[str] = []
    sbom_path = REPO_ROOT / "sbom.json"
    if not sbom_path.exists():
        return ["Missing sbom.json. Generate SBOM before governance checks."]
    try:
        payload = json.loads(sbom_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid sbom.json: {exc}"]
    if not isinstance(payload, dict):
        issues.append("sbom.json must be a JSON object.")
        return issues
    if payload.get("bomFormat") != "CycloneDX":
        issues.append("sbom.json bomFormat must be CycloneDX.")
    if "components" not in payload:
        issues.append("sbom.json missing components list.")
    return issues


def _check_dataset_licenses() -> List[str]:
    issues: List[str] = []
    registry_path = REPO_ROOT / "src/iints/data/datasets.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return ["src/iints/data/datasets.json must be a list."]

    for entry in payload:
        if not isinstance(entry, dict):
            issues.append("Dataset entry is not a JSON object.")
            continue
        dataset_id = str(entry.get("id", "<missing-id>"))
        if not str(entry.get("license", "")).strip():
            issues.append(f"{dataset_id}: missing license field.")
        if not str(entry.get("description", "")).strip():
            issues.append(f"{dataset_id}: missing description.")
        citation = entry.get("citation")
        if not isinstance(citation, dict) or not str(citation.get("text", "")).strip():
            issues.append(f"{dataset_id}: missing citation.text.")
    return issues


def _check_manifest_hashing() -> List[str]:
    issues: List[str] = []
    with tempfile.TemporaryDirectory(prefix="iints_governance_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        sample_file = tmp_root / "sample.txt"
        sample_file.write_text("iints-governance-check", encoding="utf-8")
        manifest = build_run_manifest(tmp_root, {"sample": sample_file})
        sample_entry = manifest.get("files", {}).get("sample", {})
        if "sha256" not in sample_entry:
            issues.append("Run manifest entry missing sha256.")
        if sample_entry.get("missing") is True:
            issues.append("Run manifest marked an existing file as missing.")
    return issues


def _read_text(relative_path: str) -> str:
    path = REPO_ROOT / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _require_text(relative_path: str, required: list[str]) -> List[str]:
    issues: List[str] = []
    text = _read_text(relative_path).lower()
    if not text:
        return [f"Missing required governance file: {relative_path}."]
    for phrase in required:
        if phrase.lower() not in text:
            issues.append(f"{relative_path}: missing required phrase '{phrase}'.")
    return issues


def _check_eu_research_software_dossier() -> List[str]:
    """Keep the EU research-only boundary visible and auditable."""

    issues: List[str] = []
    issues.extend(
        _require_text(
            "docs/governance/EU_RESEARCH_SOFTWARE_COMPLIANCE.md",
            [
                "Research only",
                "not a medical device",
                "EU AI Act",
                "Medical Devices Regulation",
                "GDPR",
                "Cyber Resilience Act",
                "human oversight",
                "logging and traceability",
                "Tauri",
            ],
        )
    )
    issues.extend(
        _require_text(
            "README.md",
            [
                "not a medical device",
                "treatment decisions",
                "docs",
            ],
        )
    )
    issues.extend(
        _require_text(
            "docs/governance/PRIVACY_POLICY.md",
            [
                "does not collect or transmit personal data",
                "processed locally",
                "third",
            ],
        )
    )
    return issues


def _check_tauri_security_boundary() -> List[str]:
    """Ensure the experimental Rust shell stays a narrow command boundary."""

    issues: List[str] = []
    capability_path = REPO_ROOT / "apps/iints-tauri/src-tauri/capabilities/main.json"
    if not capability_path.exists():
        return ["Missing Tauri capability file."]
    try:
        capability = json.loads(capability_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid Tauri capability JSON: {exc}"]

    forbidden_prefixes = ("shell:", "fs:", "http:", "updater:", "process:")
    permissions = capability.get("permissions", [])
    if not isinstance(permissions, list):
        issues.append("Tauri permissions must be a list.")
        permissions = []
    for permission in permissions:
        permission_text = str(permission)
        if permission_text.startswith(forbidden_prefixes):
            issues.append(f"Tauri capability grants broad or sensitive permission: {permission_text}")

    tauri_config = _read_text("apps/iints-tauri/src-tauri/tauri.conf.json")
    if "https://**" in tauri_config or "http://**" in tauri_config:
        issues.append("Tauri CSP appears to allow broad remote web content.")

    rust_main = _read_text("apps/iints-tauri/src-tauri/src/main.rs")
    if "iints_desktop.tauri_bridge" not in rust_main:
        issues.append("Tauri Rust shell must call the audited Python bridge module.")
    if "std::process::Command" not in rust_main:
        issues.append("Tauri Rust shell should keep Python process invocation explicit and reviewable.")

    frontend = _read_text("apps/iints-tauri/frontend/index.html")
    if "Not a medical device" not in frontend:
        issues.append("Tauri frontend must display the medical-device boundary.")
    return issues


def main() -> int:
    checks = [
        _check_license,
        _check_sbom,
        _check_dataset_licenses,
        _check_manifest_hashing,
        _check_eu_research_software_dossier,
        _check_tauri_security_boundary,
    ]
    issues: List[str] = []
    for check in checks:
        issues.extend(check())

    if issues:
        print("Governance checks failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Governance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
