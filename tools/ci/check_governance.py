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
                "control matrix",
            ],
        )
    )
    for path in [
        "docs/governance/INTENDED_USE_AND_CLAIMS.md",
        "docs/governance/RISK_REGISTER.md",
        "docs/governance/DPIA_LITE.md",
        "docs/governance/TAURI_THREAT_MODEL.md",
    ]:
        if not (REPO_ROOT / path).exists():
            issues.append(f"Missing deeper EU research governance document: {path}.")
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


def _check_security_policy() -> List[str]:
    return _require_text(
        "SECURITY.md",
        [
            "Reporting a Vulnerability",
            "private patient",
            "not a medical device",
            "treatment decisions",
        ],
    )


def _check_eu_control_matrix() -> List[str]:
    issues: List[str] = []
    matrix_path = REPO_ROOT / "docs/governance/EU_RESEARCH_CONTROL_MATRIX.json"
    if not matrix_path.exists():
        return ["Missing EU research control matrix JSON."]
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid EU research control matrix JSON: {exc}"]

    controls = payload.get("controls")
    if not isinstance(controls, list) or not controls:
        return ["EU research control matrix must contain a non-empty controls list."]

    required_fields = {"id", "framework", "theme", "control", "evidence_paths", "status"}
    seen_ids: set[str] = set()
    for index, control in enumerate(controls, start=1):
        if not isinstance(control, dict):
            issues.append(f"EU control #{index} is not an object.")
            continue
        missing = sorted(required_fields - set(control))
        control_id = str(control.get("id", f"<control-{index}>"))
        if missing:
            issues.append(f"{control_id}: missing fields {', '.join(missing)}.")
        if control_id in seen_ids:
            issues.append(f"Duplicate EU control id: {control_id}.")
        seen_ids.add(control_id)
        if str(control.get("status", "")).lower() not in {"implemented", "partial", "planned"}:
            issues.append(f"{control_id}: status must be implemented, partial, or planned.")
        evidence_paths = control.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            issues.append(f"{control_id}: evidence_paths must be a non-empty list.")
            continue
        for evidence in evidence_paths:
            evidence_path = REPO_ROOT / str(evidence)
            if not evidence_path.exists():
                issues.append(f"{control_id}: evidence path does not exist: {evidence}.")
    return issues


def _check_runtime_policy_hooks() -> List[str]:
    """Ensure generated AI text goes through the research-boundary policy guard."""

    issues: List[str] = []
    required_hooks = {
        "src/iints/analysis/run_quality.py": [
            "guard_ai_output",
            "blocked_policy",
            "policy_violations",
        ],
        "src/iints_desktop/local_ai.py": [
            "guard_ai_output",
            "policy_violations",
            "RESEARCH_ONLY_NOTICE",
        ],
        "src/iints_desktop/tauri_bridge.py": [
            "policy_violations",
        ],
        "src/iints/governance/research_policy.py": [
            "RESEARCH_ONLY_NOTICE",
            "scan_text_for_policy_violations",
            "guard_ai_output",
        ],
    }
    for relative_path, needles in required_hooks.items():
        text = _read_text(relative_path)
        if not text:
            issues.append(f"Missing runtime policy hook file: {relative_path}.")
            continue
        for needle in needles:
            if needle not in text:
                issues.append(f"{relative_path}: missing runtime policy hook '{needle}'.")
    return issues


def _check_public_claims() -> List[str]:
    """Block accidental positive clinical/regulatory claims in project files."""

    issues: List[str] = []
    excluded = {
        "docs/governance/INTENDED_USE_AND_CLAIMS.md",
        "tools/ci/check_governance.py",
        "tests/test_research_policy.py",
    }
    forbidden_claims = [
        "clinically validated insulin dosing",
        "safe for patient use",
        "approved medical ai",
        "ce marked",
        "mdr certified",
        "approved for clinical use",
        "use for treatment decisions",
        "use for insulin dosing",
    ]
    ignored_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".uv-python",
        ".venv",
        ".venv-ci310",
        ".venv_audit_tmp",
        ".venv_unoq",
        "__pycache__",
        "data_packs",
        "dist",
        "IINTS-SDK",
        "local",
        "models",
        "obsidian",
        "private_docs",
        "results",
        "scratch",
        "site",
    }
    negative_markers = (
        "not ",
        "not a ",
        "not an ",
        "no ",
        "never ",
        "must not ",
        "do not ",
        "does not ",
        "forbidden ",
        "excluded ",
        "without ",
    )
    suffixes = {".md", ".py", ".js", ".rs", ".toml", ".json", ".yml", ".yaml"}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if relative in excluded:
            continue
        if any(part in ignored_parts or part.startswith(".venv") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        scan_text = text.replace("*", "").replace("_", "")
        for claim in forbidden_claims:
            start = 0
            while True:
                index = scan_text.find(claim, start)
                if index < 0:
                    break
                context = scan_text[max(0, index - 48) : index]
                if not any(marker in context for marker in negative_markers):
                    issues.append(
                        f"{relative}: contains forbidden positive clinical/regulatory claim '{claim}'."
                    )
                    break
                start = index + len(claim)
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
        _check_security_policy,
        _check_eu_control_matrix,
        _check_runtime_policy_hooks,
        _check_public_claims,
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
