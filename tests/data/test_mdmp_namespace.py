from __future__ import annotations

import iints.mdmp as mdmp


def test_mdmp_namespace_exports_protocol_surface() -> None:
    assert hasattr(mdmp, "ContractRunner")
    assert hasattr(mdmp, "mdmp_gate")
    assert hasattr(mdmp, "generate_synthetic_mirror")
    assert hasattr(mdmp, "build_mdmp_dashboard_html")
