from __future__ import annotations

import re

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

    queued = client.post("/events/meal", json={"carbs": 60}, headers={"X-IINTS-Control": "1"})
    assert queued.status_code == 200
    pending = store.fetch_pending_commands()
    assert pending[0]["command"] == "inject_meal"


def test_live_patient_api_rejects_mutations_without_control_header(tmp_path) -> None:
    workspace = tmp_path / "patient"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(daemon_status="running", paused=0, workspace=str(workspace))

    client = TestClient(create_patient_app(workspace))
    response = client.post("/control/pause")

    assert response.status_code == 403
    assert "control header" in response.json()["detail"]


def test_live_patient_api_requires_bearer_token_when_configured(tmp_path) -> None:
    workspace = tmp_path / "patient"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(daemon_status="running", paused=0, workspace=str(workspace))

    client = TestClient(create_patient_app(workspace, api_token="secret-token"))

    missing = client.post("/control/pause", headers={"X-IINTS-Control": "1"})
    assert missing.status_code == 401

    wrong = client.post(
        "/control/pause",
        headers={"X-IINTS-Control": "1", "Authorization": "Bearer nope"},
    )
    assert wrong.status_code == 401

    okay = client.post(
        "/control/pause",
        headers={"X-IINTS-Control": "1", "Authorization": "Bearer secret-token"},
    )
    assert okay.status_code == 200

    hidden_status = client.get("/status")
    assert hidden_status.status_code == 401

    visible_status = client.get("/status", headers={"Authorization": "Bearer secret-token"})
    assert visible_status.status_code == 200


def test_live_patient_api_sets_security_headers(tmp_path) -> None:
    workspace = tmp_path / "patient"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(daemon_status="running", paused=0, workspace=str(workspace))

    client = TestClient(create_patient_app(workspace))
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-iints-version"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "'unsafe-inline'" not in response.headers["content-security-policy"]
    assert "nonce-" in response.headers["content-security-policy"]
    assert re.search(r'<script nonce="[^"]+">', response.text)
    assert re.search(r'<style nonce="[^"]+">', response.text)
    assert 'onclick=' not in response.text
