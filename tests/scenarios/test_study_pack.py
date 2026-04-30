from __future__ import annotations

from iints.scenarios.study_pack import build_eucys_arm_scenario, build_eucys_study_pack
from iints.validation import validate_scenario_dict


def test_official_scenarios_cover_a_full_day_rhythm() -> None:
    pack = build_eucys_study_pack(seeds=[1])

    for entry in pack["scenarios"]:
        events = entry["scenario"]["stress_events"]
        meal_times = sorted(event["start_time"] for event in events if event["event_type"] == "meal")

        assert len(events) >= 4
        assert any(start < 600 for start in meal_times)
        assert any(600 <= start < 900 for start in meal_times)
        assert any(start >= 900 for start in meal_times)


def test_build_eucys_arm_scenario_keeps_payload_schema_valid() -> None:
    pack = build_eucys_study_pack(seeds=[1])
    base_scenario = dict(pack["scenarios"][0]["scenario"])

    for arm_id in ("clean_certified", "corrupted_uncertified", "supervisor_off_ablation"):
        scenario, metadata = build_eucys_arm_scenario(base_scenario, arm_id=arm_id)

        validate_scenario_dict(scenario)

        assert "study_arm" not in scenario
        assert "condition_group" not in scenario
        assert "corruption_modes" not in scenario
        assert metadata["arm_id"] == arm_id


def test_build_eucys_arm_scenario_corrupted_arm_records_operations() -> None:
    pack = build_eucys_study_pack(seeds=[1])
    base_scenario = dict(pack["scenarios"][0]["scenario"])

    scenario, metadata = build_eucys_arm_scenario(base_scenario, arm_id="corrupted_uncertified")

    assert scenario["scenario_name"].endswith("[Corrupted Uncertified]")
    operation_modes = {item["mode"] for item in metadata["operations"]}
    assert {"timestamp_shift", "meal_annotation_mismatch", "sensor_error_injection"} <= operation_modes
