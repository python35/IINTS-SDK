from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.live_patient.runtime import PatientRuntimeConfig, PatientRuntimeStore


runner = CliRunner()


def test_edge_setup_and_status_commands(tmp_path) -> None:
    setup_dir = tmp_path / "edge_demo"
    result = runner.invoke(app, ["edge", "setup", "--output-dir", str(setup_dir), "--board", "raspberry_pi"])
    assert result.exit_code == 0
    assert (setup_dir / "run_edge_patient.sh").is_file()
    assert (setup_dir / "patient_runtime" / "iints-digital-patient.service").is_file()

    workspace = tmp_path / "patient_runtime"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(
        daemon_status="running",
        paused=0,
        scenario_profile="normal_day",
        active_seed=1101,
        algorithm_name="DemoPID",
        workspace=str(workspace),
        api_host="127.0.0.1",
        api_port=8765,
    )
    (workspace / "live_bundle").mkdir(parents=True, exist_ok=True)
    (workspace / "live_bundle" / "certification.json").write_text(
        json.dumps({"mdmp_grade": "research_grade", "certified_for_medical_research": True}),
        encoding="utf-8",
    )

    status = runner.invoke(app, ["edge", "status", "--workspace", str(workspace)])
    assert status.exit_code == 0
    assert "IINTS Edge Runtime Status" in status.stdout
    assert "research_grade" in status.stdout


def test_edge_up_uses_generated_project_config(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "edge_demo"
    workspace = project_dir / "patient_runtime"
    workspace.mkdir(parents=True, exist_ok=True)
    algo = project_dir / "algorithms" / "example_algorithm.py"
    algo.parent.mkdir(parents=True, exist_ok=True)
    algo.write_text("class Placeholder: pass\n", encoding="utf-8")

    cfg = PatientRuntimeConfig(
        workspace=str(workspace),
        algo_path=str(algo),
        patient_config="default_patient",
        patient_model_type="auto",
        scenario_profile="normal_day",
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
        seed=1101,
    )
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_start(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("iints.cli.cli.patient_cli_module.start", _fake_start)

    result = runner.invoke(app, ["edge", "up", "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert captured["workspace"] == workspace
    assert captured["algo"] == algo
    assert captured["scenario_profile"] == "normal_day"
    assert captured["speed"] == "60x"


def test_makerfaire_up_starts_pi_runtime_and_prints_kiosk(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "pi_demo"
    workspace = project_dir / "patient_runtime"
    workspace.mkdir(parents=True, exist_ok=True)
    algo = project_dir / "algorithms" / "example_algorithm.py"
    algo.parent.mkdir(parents=True, exist_ok=True)
    algo.write_text("class Placeholder: pass\n", encoding="utf-8")
    (project_dir / "start_makerfaire_patient.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_START.md").write_text("# guide\n", encoding="utf-8")
    (workspace / "iints-digital-patient.service").write_text("[Unit]\n", encoding="utf-8")

    cfg = PatientRuntimeConfig(
        workspace=str(workspace),
        algo_path=str(algo),
        patient_config="default_patient",
        patient_model_type="auto",
        scenario_profile="normal_day",
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
        seed=1101,
    )
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    start_calls: list[dict[str, object]] = []
    kiosk_calls: list[Path] = []
    summary_calls = {"count": 0}

    def _fake_start(**kwargs):
        start_calls.append(kwargs)

    def _fake_kiosk(*, workspace):
        kiosk_calls.append(workspace)

    def _fake_summary(workspace_path):
        summary_calls["count"] += 1
        if summary_calls["count"] == 1:
            return {}
        return {
            "pid_alive": True,
            "daemon_status": "running",
            "scenario_profile": "expo_hot_start",
            "active_seed": 1101,
            "algorithm_name": "EdgeDemoAlgorithm",
            "dashboard_url": "http://127.0.0.1:8765/dashboard",
            "kiosk_url": "http://127.0.0.1:8765/kiosk",
        }

    monkeypatch.setattr("iints.cli.cli.patient_cli_module.start", _fake_start)
    monkeypatch.setattr("iints.cli.cli.patient_cli_module.kiosk", _fake_kiosk)
    monkeypatch.setattr("iints.cli.cli.summarize_edge_workspace", _fake_summary)

    result = runner.invoke(app, ["makerfaire", "up", "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert len(start_calls) == 1
    assert start_calls[0]["workspace"] == workspace
    assert start_calls[0]["scenario_profile"] == "expo_hot_start"
    assert start_calls[0]["speed"] == "60x"
    assert kiosk_calls == [workspace]
    assert "IINTS Maker Faire Pi" in result.stdout
    assert "expo_hot_start" in result.stdout
    assert "start_makerfaire_patient.sh" in result.stdout


def test_makerfaire_up_resets_existing_runtime_without_restart(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "pi_demo"
    workspace = project_dir / "patient_runtime"
    workspace.mkdir(parents=True, exist_ok=True)
    algo = project_dir / "algorithms" / "example_algorithm.py"
    algo.parent.mkdir(parents=True, exist_ok=True)
    algo.write_text("class Placeholder: pass\n", encoding="utf-8")

    cfg = PatientRuntimeConfig(
        workspace=str(workspace),
        algo_path=str(algo),
        patient_config="default_patient",
        patient_model_type="auto",
        scenario_profile="normal_day",
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
        seed=1101,
    )
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    reset_calls: list[dict[str, object]] = []
    kiosk_calls: list[Path] = []

    monkeypatch.setattr(
        "iints.cli.cli.summarize_edge_workspace",
        lambda workspace_path: {
            "pid_alive": True,
            "daemon_status": "running",
            "scenario_profile": "normal_day",
            "active_seed": 1101,
            "algorithm_name": "EdgeDemoAlgorithm",
            "dashboard_url": "http://127.0.0.1:8765/dashboard",
            "kiosk_url": "http://127.0.0.1:8765/kiosk",
        },
    )
    monkeypatch.setattr("iints.cli.cli.patient_cli_module.start", lambda **kwargs: (_ for _ in ()).throw(AssertionError("start should not run")))
    monkeypatch.setattr(
        "iints.cli.cli.patient_cli_module.expo_reset",
        lambda *, scenario_profile, seed, workspace: reset_calls.append(
            {"scenario_profile": scenario_profile, "seed": seed, "workspace": workspace}
        ),
    )
    monkeypatch.setattr("iints.cli.cli.patient_cli_module.kiosk", lambda *, workspace: kiosk_calls.append(workspace))

    result = runner.invoke(
        app,
        ["makerfaire", "up", "--project-dir", str(project_dir), "--scenario-profile", "bad_carb_count", "--seed", "777"],
    )

    assert result.exit_code == 0
    assert reset_calls == [{"scenario_profile": "bad_carb_count", "seed": 777, "workspace": workspace}]
    assert kiosk_calls == [workspace]
    assert "bad_carb_count" in result.stdout


def test_edge_bridge_commands_and_doctor(monkeypatch, tmp_path) -> None:
    project_dir = tmp_path / "uno_q_demo"
    workspace = project_dir / "patient_runtime"
    workspace.mkdir(parents=True, exist_ok=True)
    (project_dir / "uno_q_bridge").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "iints.cli.cli.run_uno_q_bridge_test",
        lambda port, *, baudrate, delay_seconds: [
            {"state": "OK", "port": "/dev/ttyACM0", "response": "STATE=OK"},
            {"state": "OVERRIDE", "port": "/dev/ttyACM0", "response": "STATE=OVERRIDE"},
            {"state": "CRITICAL", "port": "/dev/ttyACM0", "response": "STATE=CRITICAL"},
        ],
    )
    bridge_test = runner.invoke(app, ["edge", "bridge-test", "--port", "/dev/ttyACM0"])
    assert bridge_test.exit_code == 0
    assert "UNO Q Bridge Test" in bridge_test.stdout
    assert "STATE=CRITICAL" in bridge_test.stdout

    monkeypatch.setattr(
        "iints.cli.cli.run_uno_q_bridge_forwarder",
        lambda workspace, port, *, baudrate, poll_interval, once, max_cycles: {
            "workspace": str(workspace),
            "port": "/dev/ttyACM0",
            "state": "OK",
        },
    )
    bridge_run = runner.invoke(
        app,
        ["edge", "bridge-run", "--project-dir", str(project_dir), "--port", "/dev/ttyACM0", "--once"],
    )
    assert bridge_run.exit_code == 0
    assert "UNO Q bridge state sent" in bridge_run.stdout

    monkeypatch.setattr(
        "iints.cli.cli.flash_uno_q_bridge",
        lambda sketch_dir, *, port, fqbn, arduino_cli: {
            "sketch_dir": str(sketch_dir),
            "port": port,
            "fqbn": fqbn,
            "arduino_cli": arduino_cli,
        },
    )
    bridge_flash = runner.invoke(
        app,
        [
            "edge",
            "bridge-flash",
            "--project-dir",
            str(project_dir),
            "--port",
            "/dev/ttyACM0",
            "--fqbn",
            "vendor:arch:board",
        ],
    )
    assert bridge_flash.exit_code == 0
    assert "UNO Q Bridge Flashed" in bridge_flash.stdout

    monkeypatch.setattr(
        "iints.cli.cli.uno_q_bridge_environment_report",
        lambda: {
            "pyserial_available": True,
            "pyserial_error": None,
            "serial_ports": ["/dev/ttyACM0"],
            "arduino_cli_path": "/usr/local/bin/arduino-cli",
        },
    )
    doctor = runner.invoke(app, ["edge", "doctor", "--board", "uno_q", "--project-dir", str(project_dir)])
    assert doctor.exit_code == 0
    assert "Arduino UNO Q Edge Check" in doctor.stdout
    assert "Do This Next" in doctor.stdout
    assert "/usr/local/bin/arduino-cli" in doctor.stdout
    assert "Create the project" in doctor.stdout


def test_edge_doctor_explains_missing_uno_q_requirements(monkeypatch) -> None:
    monkeypatch.setattr(
        "iints.cli.cli.uno_q_bridge_environment_report",
        lambda: {
            "pyserial_available": False,
            "pyserial_error": "No module named 'serial'",
            "serial_ports": [],
            "arduino_cli_path": None,
        },
    )

    doctor = runner.invoke(app, ["edge", "doctor", "--board", "uno_q"])

    assert doctor.exit_code == 1
    assert "Not Ready Yet" in doctor.stdout
    assert "Install serial support with the edge extras" in doctor.stdout
    assert "USB data cable" in doctor.stdout
    assert "You can still run the Linux-side demo now" in doctor.stdout
