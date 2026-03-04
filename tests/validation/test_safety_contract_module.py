from __future__ import annotations

from iints.validation.safety_contract import (
    load_contract_spec,
    verify_safety_contract,
)


def test_default_contract_spec_loads() -> None:
    spec = load_contract_spec()
    assert spec.contract_enabled is True
    assert spec.contract_glucose_threshold > 0


def test_contract_verification_has_no_violations_on_small_grid() -> None:
    spec = load_contract_spec()
    report = verify_safety_contract(
        spec,
        glucose_values=[70.0, 85.0, 95.0, 120.0],
        trend_values=[-2.0, -1.0, 0.0, 1.0],
        proposed_doses=[0.0, 0.5, 1.0],
    )
    assert report.total_cases > 0
    assert report.passed is True
