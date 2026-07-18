from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_governance_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_governance.py"
    spec = importlib.util.spec_from_file_location("check_governance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_architecture_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_architecture_boundaries.py"
    spec = importlib.util.spec_from_file_location("check_architecture_boundaries", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_governance_uses_repo_root_for_license(monkeypatch, tmp_path) -> None:
    module = _load_governance_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "LICENSE").write_text("Apache License 2.0", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    monkeypatch.setattr(module, "REPO_ROOT", repo_root)
    monkeypatch.chdir(outside_dir)

    assert module._check_license() == []


def test_architecture_boundaries_have_no_current_violations() -> None:
    module = _load_architecture_module()

    assert module.find_violations() == []


def test_eu_research_software_dossier_is_present() -> None:
    module = _load_governance_module()

    assert module._check_eu_research_software_dossier() == []


def test_tauri_security_boundary_has_no_current_violations() -> None:
    module = _load_governance_module()

    assert module._check_tauri_security_boundary() == []


def test_security_policy_is_present() -> None:
    module = _load_governance_module()

    assert module._check_security_policy() == []


def test_eu_control_matrix_evidence_paths_exist() -> None:
    module = _load_governance_module()

    assert module._check_eu_control_matrix() == []


def test_sdk_check_validates_the_current_checkout() -> None:
    source = Path("tools/dev/sdk_check.sh").read_text(encoding="utf-8")

    assert 'export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "MPLCONFIGDIR" in source
    assert "XDG_CACHE_HOME" in source


def test_safety_type_gate_does_not_strict_check_unrelated_imports() -> None:
    workflow = Path(".github/workflows/python-package.yml").read_text(encoding="utf-8")
    gate = workflow.split("Type check safety core (strict gate)", 1)[1].split(
        "Run tests with PyTest",
        1,
    )[0]

    assert gate.count("--follow-imports=skip") == 2
    assert "mypy --strict" in gate
    assert "mypy --check-untyped-defs" in gate


def test_release_workflow_installs_declared_build_backend() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert '"setuptools>=83,<84"' in workflow
    assert "python -m build --no-isolation" in workflow
