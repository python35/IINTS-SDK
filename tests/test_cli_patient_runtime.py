from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.live_patient.runtime import PatientRuntimeStore


runner = CliRunner()


ALGO_TEMPLATE = '''from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata


class DemoPID(InsulinAlgorithm):
    def __init__(self):
        super().__init__()
        self.set_algorithm_metadata(AlgorithmMetadata(name="DemoPID", version="1.0.0"))

    def predict_insulin(self, data: AlgorithmInput):
        self.why_log = []
        dose = 0.6 if data.current_glucose > 110 else 0.0
        return {
            "total_insulin_delivered": dose,
            "basal_insulin": dose,
            "bolus_insulin": 0.0,
            "correction_bolus": 0.0,
            "meal_bolus": 0.0,
        }
'''


class _FakeServer:
    should_exit = False


class _FakeThread:
    def join(self, timeout: float | None = None) -> None:
        return None


def test_patient_start_foreground_writes_runtime_bundle(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")

    monkeypatch.setattr("iints.cli.patient_cli._start_api_server", lambda cfg: (_FakeServer(), _FakeThread()))

    result = runner.invoke(
        app,
        [
            "patient",
            "start",
            "--algo",
            str(algo),
            "--workspace",
            str(workspace),
            "--scenario-profile",
            "sport_day",
            "--foreground",
            "--max-steps",
            "2",
            "--reset",
        ],
    )

    assert result.exit_code == 0
    assert (workspace / "live_bundle" / "results.csv").is_file()
    store = PatientRuntimeStore(workspace / "patient_state.db")
    assert len(store.get_recent_readings(limit=8)) == 2

    status = runner.invoke(app, ["patient", "status", "--workspace", str(workspace)])
    assert status.exit_code == 0
    assert "Digital Patient Status" in status.stdout
    assert "sport_day" in status.stdout
    assert "2202" in status.stdout


def test_patient_scenarios_and_service_export(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")

    scenarios = runner.invoke(app, ["patient", "scenarios"])
    assert scenarios.exit_code == 0
    assert "normal_day" in scenarios.stdout
    assert "expo_hot_start" in scenarios.stdout

    monkeypatch.setattr("iints.cli.patient_cli._start_api_server", lambda cfg: (_FakeServer(), _FakeThread()))
    start = runner.invoke(
        app,
        [
            "patient",
            "start",
            "--algo",
            str(algo),
            "--workspace",
            str(workspace),
            "--scenario-profile",
            "night_hypo_risk",
            "--foreground",
            "--max-steps",
            "1",
            "--reset",
        ],
    )
    assert start.exit_code == 0

    service_path = tmp_path / "iints-digital-patient.service"
    exported = runner.invoke(
        app,
        [
            "patient",
            "export-service",
            "--workspace",
            str(workspace),
            "--output",
            str(service_path),
            "--service-name",
            "iints-digital-patient",
            "--user-name",
            "pi",
        ],
    )
    assert exported.exit_code == 0
    contents = service_path.read_text(encoding="utf-8")
    assert "ExecStart=" in contents
    assert "iints.live_patient.daemon" in contents
    assert str(workspace / "patient_runtime_config.json") in contents

    bridge_dir = tmp_path / "uno_q_bridge"
    bridge = runner.invoke(app, ["patient", "export-uno-bridge", "--output-dir", str(bridge_dir)])
    assert bridge.exit_code == 0
    assert (bridge_dir / "iints_supervisor_bridge.ino").is_file()
    assert (bridge_dir / "bridge_protocol.txt").is_file()


def test_patient_start_rejects_remote_api_without_opt_in(tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "patient",
            "start",
            "--algo",
            str(algo),
            "--workspace",
            str(workspace),
            "--api-host",
            "0.0.0.0",
        ],
    )

    assert result.exit_code == 1
    assert "Remote API exposure is blocked by default" in result.stdout


def test_patient_start_rejects_remote_api_without_token(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")
    monkeypatch.setattr("iints.cli.patient_cli._start_api_server", lambda cfg: (_FakeServer(), _FakeThread()))

    result = runner.invoke(
        app,
        [
            "patient",
            "start",
            "--algo",
            str(algo),
            "--workspace",
            str(workspace),
            "--api-host",
            "0.0.0.0",
            "--allow-remote-api",
            "--foreground",
            "--max-steps",
            "1",
            "--reset",
        ],
    )

    assert result.exit_code == 1
    assert "requires a control token" in result.stdout


def test_patient_start_allows_remote_api_with_env_token(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    algo = tmp_path / "algo.py"
    algo.write_text(ALGO_TEMPLATE, encoding="utf-8")

    monkeypatch.setenv("IINTS_REMOTE_TOKEN", "demo-token")
    monkeypatch.setattr("iints.cli.patient_cli._start_api_server", lambda cfg: (_FakeServer(), _FakeThread()))

    result = runner.invoke(
        app,
        [
            "patient",
            "start",
            "--algo",
            str(algo),
            "--workspace",
            str(workspace),
            "--api-host",
            "0.0.0.0",
            "--allow-remote-api",
            "--api-token-env",
            "IINTS_REMOTE_TOKEN",
            "--foreground",
            "--max-steps",
            "1",
            "--reset",
        ],
    )

    assert result.exit_code == 0
    config = json.loads((workspace / "patient_runtime_config.json").read_text(encoding="utf-8"))
    assert config["allow_remote_api"] is True
    assert config["api_token_env"] == "IINTS_REMOTE_TOKEN"
