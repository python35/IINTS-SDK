#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iints-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "iints-cache"))

from iints.utils.run_io import build_run_manifest


def _check_license() -> List[str]:
    issues: List[str] = []
    license_path = Path("LICENSE")
    if not license_path.exists():
        return ["Missing LICENSE file."]
    text = license_path.read_text(encoding="utf-8").strip()
    if not text:
        issues.append("LICENSE file is empty.")
    return issues


def _check_sbom() -> List[str]:
    issues: List[str] = []
    sbom_path = Path("sbom.json")
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
    registry_path = Path("src/iints/data/datasets.json")
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


def main() -> int:
    checks = [
        _check_license,
        _check_sbom,
        _check_dataset_licenses,
        _check_manifest_hashing,
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
