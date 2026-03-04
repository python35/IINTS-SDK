from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies

from iints.core.supervisor import IndependentSupervisor


@settings(max_examples=250, deadline=None)
@given(
    glucose=st.floats(min_value=20.0, max_value=600.0, allow_nan=False, allow_infinity=False),
    proposed=st.floats(min_value=-5.0, max_value=25.0, allow_nan=False, allow_infinity=False),
    iob=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_supervisor_always_returns_bounded_non_negative_dose(glucose: float, proposed: float, iob: float) -> None:
    supervisor = IndependentSupervisor(max_insulin_per_bolus=5.0, max_60min=100.0)
    result = supervisor.evaluate_safety(
        current_glucose=glucose,
        proposed_insulin=proposed,
        current_time=0.0,
        current_iob=iob,
    )
    approved = float(result["approved_insulin"])
    assert approved >= 0.0
    assert approved <= 5.0
    # Supervisor must never increase insulin above what was requested.
    assert approved <= max(proposed, 0.0)


@settings(max_examples=200, deadline=None)
@given(
    severe_glucose=st.floats(min_value=20.0, max_value=54.0, allow_nan=False, allow_infinity=False),
    proposed=st.floats(min_value=0.0, max_value=15.0, allow_nan=False, allow_infinity=False),
)
def test_supervisor_hard_stops_dosing_in_severe_hypoglycemia(severe_glucose: float, proposed: float) -> None:
    supervisor = IndependentSupervisor(severe_hypoglycemia_threshold=54.0)
    result = supervisor.evaluate_safety(
        current_glucose=severe_glucose,
        proposed_insulin=proposed,
        current_time=0.0,
        current_iob=0.0,
    )
    assert float(result["approved_insulin"]) == 0.0


@settings(max_examples=200, deadline=None)
@given(
    glucose=st.floats(min_value=40.0, max_value=89.9, allow_nan=False, allow_infinity=False),
    trend=st.floats(min_value=-4.0, max_value=-1.0, allow_nan=False, allow_infinity=False),
    proposed=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_safety_contract_always_blocks_low_and_falling(glucose: float, trend: float, proposed: float) -> None:
    supervisor = IndependentSupervisor(
        contract_enabled=True,
        contract_glucose_threshold=90.0,
        contract_trend_threshold_mgdl_min=-1.0,
        max_60min=100.0,
    )
    # Build 10-minute history so computed trend equals requested trend.
    g0 = glucose - trend * 10.0
    supervisor.glucose_history = [(0.0, g0), (5.0, g0 + trend * 5.0)]
    result = supervisor.evaluate_safety(
        current_glucose=glucose,
        proposed_insulin=proposed,
        current_time=10.0,
        current_iob=0.0,
    )
    assert float(result["approved_insulin"]) == 0.0
    assert any("SAFETY_CONTRACT" in action for action in result["actions_taken"])
