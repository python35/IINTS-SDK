from __future__ import annotations

from fastapi.testclient import TestClient

from iints.live_patient.api import create_patient_app
from iints.live_patient.runtime import PatientRuntimeStore


def test_live_patient_api_exposes_status_history_and_commands(tmp_path) -> None:
    workspace = tmp_path / "patient"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(
        daemon_status="running",
        paused=0,
        simulated_clock="Day 1 07:30",
        last_glucose_mgdl=132.0,
        last_event_summary="Breakfast",
        workspace=str(workspace),
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
    )
    store.append_reading(
        {
            "time_minutes": 450,
            "simulated_clock": "Day 1 07:30",
            "glucose_actual_mgdl": 132.0,
            "delivered_insulin_units": 0.8,
            "algo_recommended_insulin_units": 0.8,
            "carb_intake_grams": 48.0,
            "safety_triggered": False,
            "safety_reason": "",
        },
        event_summary="Breakfast",
    )

    client = TestClient(create_patient_app(workspace))

    status = client.get("/status")
    assert status.status_code == 200
    assert status.json()["daemon_status"] == "running"

    latest = client.get("/glucose/latest")
    assert latest.status_code == 200
    assert latest.json()["simulated_clock"] == "Day 1 07:30"

    history = client.get("/glucose/history?limit=8")
    assert history.status_code == 200
    assert history.json()["records"][0]["glucose_mgdl"] == 132.0

    queued = client.post("/events/meal", json={"carbs": 60})
    assert queued.status_code == 200
    pending = store.fetch_pending_commands()
    assert pending[0]["command"] == "inject_meal"
