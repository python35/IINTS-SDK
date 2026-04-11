from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from iints.live_patient.uno_q import (
    bridge_state_from_runtime_status,
    run_uno_q_bridge_forwarder,
    run_uno_q_bridge_test,
    send_uno_q_bridge_state,
)


def test_bridge_state_from_runtime_status_maps_expected_states() -> None:
    assert bridge_state_from_runtime_status({}) == "OVERRIDE"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_glucose_mgdl": 120.0}) == "OK"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_safety_reason": "Supervisor override"}) == "OVERRIDE"
    assert bridge_state_from_runtime_status({"daemon_status": "running", "last_glucose_mgdl": 65.0}) == "CRITICAL"


def test_send_uno_q_bridge_state_waits_for_banner_and_reads_ack(monkeypatch) -> None:
    class _FakeConnection:
        def __init__(self):
            self.pending = [b"IINTS UNO Q supervisor bridge ready\n"]
            self.written: list[str] = []

        def reset_input_buffer(self):
            return None

        def reset_output_buffer(self):
            return None

        def write(self, payload: bytes):
            state = payload.decode("utf-8").strip()
            self.written.append(state)
            self.pending.append(f"STATE={state}\n".encode("utf-8"))

        def flush(self):
            return None

        def readline(self):
            if self.pending:
                return self.pending.pop(0)
            return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeSerialModule:
        def __init__(self):
            self.connection = _FakeConnection()

        def Serial(self, *args, **kwargs):
            return self.connection

    monkeypatch.setattr("iints.live_patient.uno_q._require_pyserial", lambda: (_FakeSerialModule(), object()))
    monkeypatch.setattr("iints.live_patient.uno_q.resolve_uno_q_port", lambda port: "/dev/ttyTEST0")
    monkeypatch.setattr("iints.live_patient.uno_q.time.sleep", lambda _seconds: None)

    payload = send_uno_q_bridge_state("/dev/ttyTEST0", "OK")

    assert payload["response"] == "STATE=OK"
    assert payload["startup_lines"] == ["IINTS UNO Q supervisor bridge ready"]
    assert payload["response_lines"] == ["STATE=OK"]


def test_run_uno_q_bridge_test_reuses_single_serial_session(monkeypatch) -> None:
    class _FakeConnection:
        def __init__(self):
            self.written: list[str] = []

    connection = _FakeConnection()
    open_count = {"value": 0}

    @contextmanager
    def _fake_session(port: str, *, baudrate: int, timeout_seconds: float, boot_delay_seconds: float = 1.2):
        open_count["value"] += 1
        yield connection, ["IINTS UNO Q supervisor bridge ready"]

    def _fake_send(connection_obj, state, *, timeout_seconds, expect_response):
        connection_obj.written.append(state)
        return {
            "state": state,
            "response": f"STATE={state}",
            "response_lines": [f"STATE={state}"],
        }

    monkeypatch.setattr("iints.live_patient.uno_q.resolve_uno_q_port", lambda port: "/dev/ttyTEST0")
    monkeypatch.setattr("iints.live_patient.uno_q._uno_q_serial_connection", _fake_session)
    monkeypatch.setattr("iints.live_patient.uno_q._send_state_over_connection", _fake_send)
    monkeypatch.setattr("iints.live_patient.uno_q.time.sleep", lambda _seconds: None)

    payload = run_uno_q_bridge_test("/dev/ttyTEST0", delay_seconds=0.0)

    assert open_count["value"] == 1
    assert [item["state"] for item in payload] == ["OK", "OVERRIDE", "CRITICAL"]
    assert payload[0]["startup_lines"] == ["IINTS UNO Q supervisor bridge ready"]
    assert payload[1]["startup_lines"] == []
    assert connection.written == ["OK", "OVERRIDE", "CRITICAL"]


def test_run_uno_q_bridge_forwarder_once_sends_current_state(monkeypatch, tmp_path) -> None:
    sent: list[tuple[str, str, int]] = []

    monkeypatch.setattr(
        "iints.live_patient.runtime.load_runtime_status",
        lambda workspace: {"daemon_status": "running", "last_glucose_mgdl": 64.0},
    )
    monkeypatch.setattr("iints.live_patient.uno_q.resolve_uno_q_port", lambda port: "/dev/ttyTEST0")

    @contextmanager
    def _fake_session(port: str, *, baudrate: int, timeout_seconds: float, boot_delay_seconds: float = 1.2):
        yield object(), ["IINTS UNO Q supervisor bridge ready"]

    def _fake_send(connection_obj, state, *, timeout_seconds, expect_response):
        sent.append(("/dev/ttyTEST0", state, 115200))
        return {"state": state, "response": None, "response_lines": []}

    monkeypatch.setattr("iints.live_patient.uno_q._uno_q_serial_connection", _fake_session)
    monkeypatch.setattr("iints.live_patient.uno_q._send_state_over_connection", _fake_send)

    payload = run_uno_q_bridge_forwarder(
        tmp_path / "patient_runtime",
        "/dev/ttyTEST0",
        once=True,
    )

    assert payload["state"] == "CRITICAL"
    assert sent == [("/dev/ttyTEST0", "CRITICAL", 115200)]
