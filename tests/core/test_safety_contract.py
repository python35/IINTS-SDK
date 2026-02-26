import pytest

from iints.core.supervisor import IndependentSupervisor
from iints.core.safety.config import SafetyConfig


def _evaluate_with_trend(supervisor: IndependentSupervisor, current_glucose: float, trend_mgdl_min: float):
    # Build a 10-minute history so the supervisor computes the desired trend.
    g0 = current_glucose - (trend_mgdl_min * 10.0)
    g5 = current_glucose - (trend_mgdl_min * 5.0)
    supervisor.glucose_history = [(0.0, g0), (5.0, g5)]
    return supervisor.evaluate_safety(
        current_glucose=current_glucose,
        proposed_insulin=1.0,
        current_time=10.0,
        current_iob=0.0,
    )


@pytest.mark.parametrize("glucose", [80.0, 95.0, 110.0])
@pytest.mark.parametrize("trend", [-2.0, -1.0, 0.0, 1.0])
def test_formal_safety_contract_blocks_when_low_and_falling(glucose, trend):
    config = SafetyConfig(
        hypoglycemia_threshold=50.0,
        severe_hypoglycemia_threshold=40.0,
        hypo_cutoff=50.0,
        trend_stop=-10.0,
        max_insulin_per_bolus=10.0,
        max_insulin_per_hour=50.0,
        max_iob=50.0,
        contract_enabled=True,
        contract_glucose_threshold=90.0,
        contract_trend_threshold_mgdl_min=-1.0,
    )
    supervisor = IndependentSupervisor(safety_config=config)
    result = _evaluate_with_trend(supervisor, glucose, trend)

    should_block = glucose < 90.0 and trend <= -1.0
    if should_block:
        assert result["approved_insulin"] == 0.0
        assert any("SAFETY_CONTRACT" in action for action in result["actions_taken"])
    else:
        assert result["approved_insulin"] >= 0.0


def test_formal_safety_contract_can_be_disabled():
    config = SafetyConfig(
        hypoglycemia_threshold=50.0,
        severe_hypoglycemia_threshold=40.0,
        hypo_cutoff=50.0,
        trend_stop=-10.0,
        max_insulin_per_bolus=10.0,
        max_insulin_per_hour=50.0,
        max_iob=50.0,
        contract_enabled=False,
    )
    supervisor = IndependentSupervisor(safety_config=config)
    result = _evaluate_with_trend(supervisor, current_glucose=80.0, trend_mgdl_min=-2.0)
    assert all("SAFETY_CONTRACT" not in action for action in result["actions_taken"])
