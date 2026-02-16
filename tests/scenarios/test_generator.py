from iints.scenarios import ScenarioGeneratorConfig, generate_random_scenario


def test_generate_random_scenario_counts():
    config = ScenarioGeneratorConfig(
        name="Test Scenario",
        duration_minutes=120,
        seed=123,
        meal_count=2,
        exercise_count=1,
        sensor_error_count=1,
    )
    scenario = generate_random_scenario(config)

    assert scenario["scenario_name"] == "Test Scenario"
    assert scenario["scenario_version"] == "1.0"
    assert "stress_events" in scenario
    assert len(scenario["stress_events"]) == 4

    event_types = {event["event_type"] for event in scenario["stress_events"]}
    assert "meal" in event_types
    assert "exercise" in event_types
    assert "sensor_error" in event_types
