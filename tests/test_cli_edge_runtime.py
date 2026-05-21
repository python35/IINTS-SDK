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
    assert (setup_dir / "start_edge_easy.sh").is_file()
    assert (setup_dir / "EDGE_EASY_START.md").is_file()
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


def test_edge_quickstart_starts_uno_q_easy_path(monkeypatch, tmp_path) -> None:
    start_calls: list[dict[str, object]] = []

    def _fake_start(**kwargs):
        start_calls.append(kwargs)

    monkeypatch.setattr("iints.cli.cli.patient_cli_module.start", _fake_start)

    setup_dir = tmp_path / "uno_q_demo"
    result = runner.invoke(
        app,
        [
            "edge",
            "quickstart",
            "--board",
            "uno_q",
            "--output-dir",
            str(setup_dir),
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert len(start_calls) == 1
    assert start_calls[0]["scenario_profile"] == "expo_hot_start"
    assert start_calls[0]["reset"] is True
    assert (setup_dir / "start_edge_easy.sh").is_file()
    assert (setup_dir / "test_uno_q_bridge.sh").is_file()
    assert (setup_dir / "run_uno_q_bridge.sh").is_file()
    assert (setup_dir / "EDGE_EASY_START.md").is_file()
    assert "UNO Q Simple Path" in result.stdout
    assert "uno_q_bridge/iints_supervisor_bridge.ino" in result.stdout


def test_edge_deploy_invokes_remote_deployer(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_deploy(**kwargs):
        captured.update(kwargs)
        project_root = tmp_path / "edge_demo"
        project_root.mkdir(parents=True, exist_ok=True)
        guide = project_root / "EDGE_REMOTE_ACCESS.md"
        guide.write_text("# Remote\n", encoding="utf-8")
        setup = project_root / "EDGE_SETUP.md"
        setup.write_text("# Setup\n", encoding="utf-8")
        return {
            "destination": "pi@pi.local",
            "remote_dir": "~/booth_demo",
            "local_output_dir": str(project_root),
            "board": "uno_q",
            "scenario_profile": "expo_hot_start",
            "dry_run": False,
            "deploy_command": "ssh pi@pi.local ...",
            "deploy_stdout": "",
            "remote_commands": {
                "status": "ssh pi@pi.local status",
                "reset": "ssh pi@pi.local reset",
                "stop": "ssh pi@pi.local stop",
            },
            "artifacts": {
                "remote_access_guide": str(guide),
                "setup_guide": str(setup),
                "uno_bridge_service": str(project_root / "iints-uno-q-bridge.service"),
            },
        }

    monkeypatch.setattr("iints.cli.cli.deploy_edge_project", _fake_deploy)

    result = runner.invoke(
        app,
        [
            "edge",
            "deploy",
            "--host",
            "pi.local",
            "--user",
            "pi",
            "--board",
            "uno_q",
            "--local-output-dir",
            str(tmp_path / "edge_demo"),
            "--remote-dir",
            "~/booth_demo",
            "--uno-bridge-port",
            "/dev/ttyACM0",
        ],
    )

    assert result.exit_code == 0
    assert captured["host"] == "pi.local"
    assert captured["user_name"] == "pi"
    assert captured["board"] == "uno_q"
    assert captured["uno_bridge_port"] == "/dev/ttyACM0"
    assert "Raspberry Pi Connect recommended" in result.stdout
    assert "EDGE_REMOTE_ACCESS.md" in result.stdout
    assert "ssh pi@pi.local status" in result.stdout


def test_edge_offline_bundle_invokes_builder(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_bundle(output, **kwargs):
        captured["output"] = output
        captured.update(kwargs)
        return {
            "archive": str(tmp_path / "iints_offline.tar.gz"),
            "install_script": "iints_offline/install_offline_edge.sh",
            "install_guide": "iints_offline/OFFLINE_INSTALL.md",
            "package_spec": "iints-sdk-python35[edge,mdmp]==1.5.3",
        }

    monkeypatch.setattr("iints.cli.cli.build_edge_offline_bundle", _fake_bundle)

    result = runner.invoke(
        app,
        [
            "edge",
            "offline-bundle",
            "--output",
            str(tmp_path / "iints_offline.tar.gz"),
            "--board",
            "uno_q",
        ],
    )

    assert result.exit_code == 0
    assert captured["board"] == "uno_q"
    assert "install_offline_edge.sh" in result.stdout
    assert "iints-sdk-python35[edge,mdmp]==1.5.3" in result.stdout


def test_edge_study_invokes_study_bundle(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_study_bundle(**kwargs):
        captured.update(kwargs)
        target_root = tmp_path / "pi_study"
        target_root.mkdir(parents=True, exist_ok=True)
        protocol = target_root / "protocol" / "STUDY_PROTOCOL.md"
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text("# Protocol\n", encoding="utf-8")
        matrix = target_root / "protocol" / "study_matrix.csv"
        matrix.write_text("seed\n1\n", encoding="utf-8")
        summary = target_root / "study_summary.json"
        summary.write_text("{}", encoding="utf-8")
        return {
            "target_root": target_root,
            "protocol_outputs": {
                "protocol_markdown": str(protocol),
                "study_matrix_csv": str(matrix),
            },
            "root_summary_json": summary,
        }

    monkeypatch.setattr("iints.cli.cli._run_study_bundle", _fake_run_study_bundle)

    algo = tmp_path / "algo.py"
    algo.write_text("class Placeholder: pass\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "edge",
            "study",
            "--algo",
            str(algo),
            "--output-dir",
            str(tmp_path / "pi_study"),
            "--seeds",
            "1,2,3",
        ],
    )

    assert result.exit_code == 0
    assert captured["seeds"] == [1, 2, 3]
    assert captured["prepare_ai"] is False
    assert "Edge Study Complete" in result.stdout
    assert (tmp_path / "pi_study" / "edge_study_metadata.json").is_file()


def test_edge_long_study_invokes_runner(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_long_study(**kwargs):
        captured.update(kwargs)
        study_root = tmp_path / "long_study"
        study_root.mkdir(parents=True, exist_ok=True)
        return {
            "project_dir": str(tmp_path / "pi_demo"),
            "output_dir": str(study_root),
            "scratch_dir": str(tmp_path / "scratch" / "long_study"),
            "index_csv": str(study_root / "long_study_index.csv"),
            "summary_json": str(study_root / "study_summary.json"),
            "progress_json": str(study_root / "study_progress.json"),
            "export_readme": str(study_root / "EXPORT_README.md"),
            "completed_runs": 20,
            "skipped_runs": 0,
            "resume_start_day": 2,
            "resume": True,
            "elapsed_seconds": 12.34,
            "snapshots": [{"archive": str(study_root / "snapshots" / "snapshot.tar.gz")}],
        }

    monkeypatch.setattr("iints.cli.cli.run_edge_long_study", _fake_long_study)

    result = runner.invoke(
        app,
        [
            "edge",
            "long-study",
            "--config",
            str(tmp_path / "edge_long_study.yaml"),
            "--project-dir",
            str(tmp_path / "pi_demo"),
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert captured["project_dir"] == tmp_path / "pi_demo"
    assert captured["resume"] is True
    assert "Edge Long Study Complete" in result.stdout
    assert "study_summary.json" in result.stdout
    assert "Scratch dir" in result.stdout


def test_edge_study_snapshot_invokes_snapshotter(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _Snapshot:
        input_dir = str(tmp_path / "pi_demo" / "results" / "long_study")
        archive = str(tmp_path / "pi_demo" / "snapshots" / "study_snapshot.tar.gz")
        generated_at_utc = "2026-04-25T10:00:00Z"

    def _fake_snapshot(input_dir, *, output):
        captured["input_dir"] = input_dir
        captured["output"] = output
        return _Snapshot()

    monkeypatch.setattr("iints.cli.cli.create_edge_study_snapshot", _fake_snapshot)

    result = runner.invoke(
        app,
        [
            "edge",
            "study-snapshot",
            "--project-dir",
            str(tmp_path / "pi_demo"),
        ],
    )

    assert result.exit_code == 0
    assert str(captured["input_dir"]).endswith("results/long_study")
    assert "study_snapshot.tar.gz" in result.stdout


def test_edge_study_export_invokes_exporter(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_export(input_dir, *, output):
        captured["input_dir"] = input_dir
        captured["output"] = output
        return {
            "input_dir": str(input_dir),
            "archive": str(output),
            "manifest": str(Path(output).with_name("long_study_export_manifest.json")),
        }

    monkeypatch.setattr("iints.cli.cli.export_edge_study_archive", _fake_export)

    result = runner.invoke(
        app,
        [
            "edge",
            "study-export",
            "--project-dir",
            str(tmp_path / "pi_demo"),
        ],
    )

    assert result.exit_code == 0
    assert str(captured["output"]).endswith("results/long_study_export.zip")
    assert "long_study_export.zip" in result.stdout


def test_edge_remote_status_uses_ssh_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_remote(**kwargs):
        captured.update(kwargs)
        return {"stdout": "IINTS Edge Runtime Status\n"}

    monkeypatch.setattr("iints.cli.cli.run_remote_edge_command", _fake_remote)

    result = runner.invoke(
        app,
        [
            "edge",
            "remote-status",
            "--host",
            "pi.local",
            "--user",
            "pi",
            "--remote-dir",
            "~/booth_demo",
        ],
    )

    assert result.exit_code == 0
    assert captured["host"] == "pi.local"
    assert captured["user_name"] == "pi"
    assert captured["action"] == "status"
    assert "IINTS Edge Runtime Status" in result.stdout


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
    (project_dir / "install_makerfaire_autostart.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_AUTOSTART.md").write_text("# autostart\n", encoding="utf-8")
    (project_dir / "run_makerfaire_watchdog.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_CHECKLIST.md").write_text("# checklist\n", encoding="utf-8")
    (project_dir / "iints-digital-patient-watchdog.timer").write_text("[Timer]\n", encoding="utf-8")
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
    assert "install_makerfaire_autostart.sh" in result.stdout
    assert "run_makerfaire_watchdog.sh" in result.stdout
    assert "MAKERFAIRE_CHECKLIST.md" in result.stdout


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


def test_makerfaire_autostart_prints_generated_paths(tmp_path) -> None:
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
        scenario_profile="expo_hot_start",
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
        seed=1101,
    )
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    (project_dir / "start_makerfaire_patient.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_START.md").write_text("# guide\n", encoding="utf-8")
    (project_dir / "open_makerfaire_kiosk.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "iints-makerfaire-kiosk.desktop").write_text("[Desktop Entry]\n", encoding="utf-8")
    (project_dir / "install_makerfaire_autostart.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_AUTOSTART.md").write_text("# autostart\n", encoding="utf-8")
    (project_dir / "run_makerfaire_watchdog.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (project_dir / "MAKERFAIRE_CHECKLIST.md").write_text("# checklist\n", encoding="utf-8")
    (project_dir / "iints-digital-patient-watchdog.service").write_text("[Unit]\n", encoding="utf-8")
    (project_dir / "iints-digital-patient-watchdog.timer").write_text("[Timer]\n", encoding="utf-8")
    (workspace / "iints-digital-patient.service").write_text("[Unit]\n", encoding="utf-8")
    (workspace / "iints-digital-patient.INSTALL.txt").write_text("sudo systemctl enable\n", encoding="utf-8")

    result = runner.invoke(app, ["makerfaire", "autostart", "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert "IINTS Maker Faire Autostart" in result.stdout
    assert "install_makerfaire_autostart.sh" in result.stdout
    assert "Desktop Autologin" in result.stdout
    assert "run_makerfaire_watchdog.sh" in result.stdout


def test_makerfaire_watchdog_restarts_missing_runtime(monkeypatch, tmp_path) -> None:
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
        scenario_profile="expo_hot_start",
        mode="demo-time",
        speed=60.0,
        api_host="127.0.0.1",
        api_port=8765,
        seed=1101,
    )
    cfg.config_path.write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")

    start_calls: list[dict[str, object]] = []
    summary_calls = {"count": 0}

    def _fake_start(**kwargs):
        start_calls.append(kwargs)

    def _fake_summary(workspace_path):
        summary_calls["count"] += 1
        if summary_calls["count"] == 1:
            return {"daemon_status": "stopped", "pid_alive": False}
        return {
            "pid_alive": True,
            "daemon_status": "running",
            "scenario_profile": "expo_hot_start",
            "active_seed": 1101,
            "dashboard_url": "http://127.0.0.1:8765/dashboard",
            "kiosk_url": "http://127.0.0.1:8765/kiosk",
        }

    monkeypatch.setattr("iints.cli.cli.patient_cli_module.start", _fake_start)
    monkeypatch.setattr("iints.cli.cli.summarize_edge_workspace", _fake_summary)

    result = runner.invoke(app, ["makerfaire", "watchdog", "--project-dir", str(project_dir)])

    assert result.exit_code == 0
    assert len(start_calls) == 1
    assert start_calls[0]["scenario_profile"] == "expo_hot_start"
    assert start_calls[0]["reset"] is True
    assert "IINTS Maker Faire Watchdog" in result.stdout
    assert "restarted" in result.stdout


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


def test_edge_pump_cli_creates_packages_and_uploads_dry_run(tmp_path) -> None:
    lab_dir = tmp_path / "pico_lab"
    init_result = runner.invoke(app, ["edge", "pump", "init", "--output-dir", str(lab_dir)])
    assert init_result.exit_code == 0
    assert "IINTS Pico Pump Lab" in init_result.stdout
    assert (lab_dir / "algorithms" / "pico_bench_algorithm.py").is_file()

    bundle_dir = tmp_path / "pico_bundle"
    package_result = runner.invoke(
        app,
        [
            "edge",
            "pump",
            "package",
            "--algorithm",
            str(lab_dir / "algorithms" / "pico_bench_algorithm.py"),
            "--output-dir",
            str(bundle_dir),
            "--safety-contract",
            str(lab_dir / "safety_contract.json"),
        ],
    )
    assert package_result.exit_code == 0
    assert "Pico Pump Bench Bundle" in package_result.stdout
    assert (bundle_dir / "manifest.json").is_file()

    mount_dir = tmp_path / "CIRCUITPY"
    mount_dir.mkdir()
    upload_result = runner.invoke(
        app,
        [
            "edge",
            "pump",
            "upload",
            "--bundle-dir",
            str(bundle_dir),
            "--mount-dir",
            str(mount_dir),
            "--bench-only-confirm",
            "I understand this is bench-only and not for human use",
        ],
    )
    assert upload_result.exit_code == 0
    assert "Dry run only" in upload_result.stdout
    assert "code.py" in upload_result.stdout
