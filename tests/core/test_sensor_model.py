from __future__ import annotations

from iints.core.devices.models import SENSOR_PROFILES, SensorModel, create_sensor_model


def test_create_sensor_model_profile_exposes_free_living_artifacts() -> None:
    sensor = create_sensor_model(profile="free_living_cgm", seed=7)
    state = sensor.get_state()

    assert "free_living_cgm" in SENSOR_PROFILES
    assert state["lag_minutes"] == 10
    assert state["noise_std"] == 8.0
    assert state["drift_std_per_hour"] > 0.0
    assert state["compression_low_prob"] > 0.0
    assert state["dropout_duration_steps"] == (2, 6)


def test_sensor_model_can_emit_compression_low_episode() -> None:
    sensor = SensorModel(
        noise_std=0.0,
        bias=0.0,
        lag_minutes=0,
        dropout_prob=0.0,
        compression_low_prob=1.0,
        compression_low_max_glucose=200.0,
        compression_low_mgdl_range=(20.0, 20.0),
        compression_low_duration_steps=(2, 2),
        seed=1,
    )

    first = sensor.read(112.0, 0.0)
    second = sensor.read(118.0, 5.0)

    assert first.status == "compression_low"
    assert second.status == "compression_low"
    assert first.value == 92.0
    assert second.value == 98.0


def test_sensor_model_can_hold_dropouts_across_multiple_steps() -> None:
    sensor = SensorModel(
        noise_std=0.0,
        bias=0.0,
        lag_minutes=0,
        dropout_prob=1.0,
        dropout_duration_steps=(2, 2),
        seed=3,
    )

    first = sensor.read(120.0, 0.0)
    second = sensor.read(146.0, 5.0)

    assert first.status == "dropout_hold"
    assert second.status == "dropout_hold"
    assert first.value == 120.0
    assert second.value == first.value
