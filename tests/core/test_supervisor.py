import pytest

from iints.core.supervisor import IndependentSupervisor, SafetyLevel


def test_supervisor_caps_max_bolus():
    supervisor = IndependentSupervisor(max_insulin_per_bolus=5.0)

    result = supervisor.evaluate_safety(
        current_glucose=180.0,
        proposed_insulin=12.0,
        current_time=0.0,
        current_iob=0.0,
    )

    # 60-min cap (3.0U) should be enforced before max bolus
    assert result["approved_insulin"] == pytest.approx(3.0)
    assert result["insulin_reduction"] == pytest.approx(9.0)
    assert result["safety_level"] in (SafetyLevel.WARNING, SafetyLevel.CRITICAL)


def test_supervisor_blocks_severe_hypoglycemia():
    supervisor = IndependentSupervisor(severe_hypoglycemia_threshold=54)

    result = supervisor.evaluate_safety(
        current_glucose=50.0,
        proposed_insulin=2.0,
        current_time=0.0,
        current_iob=0.0,
    )

    assert result["approved_insulin"] == 0
    assert result["safety_level"] == SafetyLevel.EMERGENCY
    assert result["emergency_mode"] is True


def test_supervisor_reduces_dose_on_high_iob():
    supervisor = IndependentSupervisor(max_insulin_per_bolus=5.0)

    result = supervisor.evaluate_safety(
        current_glucose=120.0,
        proposed_insulin=4.0,
        current_time=0.0,
        current_iob=6.0,
    )

    assert result["approved_insulin"] < 4.0
    assert any("High IOB" in action for action in result["actions_taken"])


def test_supervisor_records_violation_when_actions_taken():
    supervisor = IndependentSupervisor(max_insulin_per_bolus=3.0)

    supervisor.evaluate_safety(
        current_glucose=200.0,
        proposed_insulin=10.0,
        current_time=0.0,
        current_iob=0.0,
    )

    report = supervisor.get_safety_report()
    assert report["total_violations"] == 1
    assert report["violation_breakdown"]["warning"] == 1


def test_supervisor_blocks_predicted_hypoglycemia():
    supervisor = IndependentSupervisor(predicted_hypoglycemia_threshold=60.0)

    result = supervisor.evaluate_safety(
        current_glucose=110.0,
        proposed_insulin=1.0,
        current_time=0.0,
        current_iob=0.5,
        predicted_glucose_30min=55.0,
    )

    assert result["approved_insulin"] == 0
    assert any("PREDICTED_HYPO" in action for action in result["actions_taken"])


def test_supervisor_caps_basal_limit():
    supervisor = IndependentSupervisor()

    result = supervisor.evaluate_safety(
        current_glucose=140.0,
        proposed_insulin=1.0,
        current_time=0.0,
        current_iob=0.0,
        basal_insulin_units=1.0,
        basal_limit_units=0.4,
    )

    assert result["approved_insulin"] == pytest.approx(0.4)
    assert any("BASAL_LIMIT" in action for action in result["actions_taken"])
