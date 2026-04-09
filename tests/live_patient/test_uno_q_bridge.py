from __future__ import annotations

from pathlib import Path

from iints.live_patient.uno_q import bridge_state_from_runtime_status, run_uno_q_bridge_forwarder


def test_bridge_state_from_runtime_status_maps_expected_states() -> None:
    assert bridge_state_from_runtime_status({}) == "OVERRIDE"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_glucose_mgdl": 120.0}) == "OK"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_safety_reason": "Supervisor override"}) == "OVERRIDE"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_glucose_mgdl": 65.0}) == "CRITICAL"


def test_run_uno_q_bridge_forwarder_once_sends_current_state(monkeypatch, tmp_path) -> None:
    sent: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        "iints.live_patient.runtime.load_runtime_status",
        lambda workspace: {"daemon_status": "running", "last_glucose_mgdl": 64.0},
    )
    monkeypatch.setattr("iints.live_patient.uno_q.resolve_uno_q_port", lambda port: "/dev/ttyTEST0")

    def _fake_send(port, state, *, baudrate, timeout_seconds, expect_response):
        sent.append((port, state, baudrate))
        return {"port": port, "state": state, "baudrate": baudrate, "response": None}

    monkeypatch.setattr("iints.live_patient.uno_q.send_uno_q_bridge_state", _fake_send)

    payload = run_uno_q_bridge_forwarder(
        tmp_path / "patient_runtime",
        "/dev/ttyTEST0",
        once=True,
    )

    assert payload["state"] == "CRITICAL"
    assert sent == [("/dev/ttyTEST0", "CRITICAL", 115200)]
