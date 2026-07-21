from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_repair_module():
    path = Path("tools/desktop/repair_fmpy_macos_dylibs.py")
    spec = importlib.util.spec_from_file_location("repair_fmpy_macos_dylibs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
