from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_governance_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "ci" / "check_governance.py"
    spec = importlib.util.spec_from_file_location("check_governance", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_governance_uses_repo_root_for_license(monkeypatch, tmp_path) -> None:
    module = _load_governance_module()

    monkeypatch.chdir(tmp_path)

    assert (module.REPO_ROOT / "LICENSE").is_file()
    assert module._check_license() == []
