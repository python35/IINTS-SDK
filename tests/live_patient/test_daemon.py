from __future__ import annotations

import pytest

from iints.core.simulator import SimulationLimitError
from iints.live_patient.daemon import _resolve_api_token
from iints.live_patient.runtime import LivePatientDaemon, PatientRuntimeConfig, PatientRuntimeStore


def test_resolve_api_token_rejects_missing_or_blank_env(monkeypatch, tmp_path) -> None:
    config = PatientRuntimeConfig(
        workspace=str(tmp_path / "workspace"),
        algo_path=str(tmp_path / "algo.py"),
        api_token_env="IINTS_TEST_TOKEN",
    )

    monkeypatch.delenv("IINTS_TEST_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        _resolve_api_token(config)

    monkeypatch.setenv("IINTS_TEST_TOKEN", "   ")
    with pytest.raises(RuntimeError):
        _resolve_api_token(config)

    monkeypatch.setenv("IINTS_TEST_TOKEN", " secret-token ")
    assert _resolve_api_token(config) == "secret-token"


def test_prime_profile_start_aborts_if_generator_stalls(monkeypatch, tmp_path) -> None:
    algo_path = tmp_path / "algo.py"
    algo_path.write_text("class Dummy: pass\n", encoding="utf-8")
    config = PatientRuntimeConfig(
        workspace=str(tmp_path / "workspace"),
        algo_path=str(algo_path),
        scenario_profile="expo_hot_start",
    )
    daemon = LivePatientDaemon(config)
    daemon.generator = iter({"time_minutes": 0, "glucose_actual_mgdl": 120.0, "safety_triggered": False} for _ in range(64))

    monkeypatch.setattr(daemon, "_record_to_csv", lambda record: None)
    monkeypatch.setattr(daemon, "_save_snapshot", lambda record: None)
    monkeypatch.setattr(daemon, "_update_live_status_from_record", lambda *args, **kwargs: None)

    with pytest.raises(SimulationLimitError):
        daemon._prime_profile_start()


def test_runtime_store_claims_commands_before_completion(tmp_path) -> None:
    store = PatientRuntimeStore(tmp_path / "workspace" / "patient_state.db")

    command_id = store.enqueue_command("pause", {"requested_by": "test"})
    claimed = store.fetch_pending_commands()

    assert len(claimed) == 1
    assert claimed[0]["id"] == command_id
    assert claimed[0]["status"] == "processing"
    assert store.fetch_pending_commands() == []

    store.complete_command(command_id, status="done", result={"paused": True})
    resolved = store.await_command(command_id, timeout_seconds=0.2)

    assert resolved is not None
    assert resolved["status"] == "done"
    assert resolved["result"]["paused"] is True
