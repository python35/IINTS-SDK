from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

from iints.live_patient.edge_ops import (
    build_edge_offline_bundle,
    create_edge_bundle,
    deploy_edge_project,
    export_edge_setup,
    summarize_edge_workspace,
)
from iints.live_patient.runtime import PatientRuntimeStore


def test_summarize_edge_workspace_includes_certification_and_review(tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(
        daemon_status="running",
        paused=0,
        simulated_clock="Day 1 12:00",
        last_glucose_mgdl=148.0,
        scenario_profile="sport_day",
        active_seed=2202,
        algorithm_name="DemoPID",
        workspace=str(workspace),
        api_host="127.0.0.1",
        api_port=8765,
    )
    bundle_dir = workspace / "live_bundle"
    (bundle_dir / "ai").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "certification.json").write_text(
        json.dumps(
            {
                "mdmp_grade": "research_grade",
                "certified_for_medical_research": True,
                "compliance_score": 98.0,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "ai" / "realism_review.md").write_text(
        "# Review\n\nLooks realistic overall.\n",
        encoding="utf-8",
    )

    summary = summarize_edge_workspace(workspace)

    assert summary["kiosk_url"] == "http://127.0.0.1:8765/kiosk"
    assert summary["certification"]["grade"] == "research_grade"
    assert summary["review"]["exists"] is True
    assert summary["scenario_profile"] == "sport_day"


def test_create_edge_bundle_writes_zip_with_summary(tmp_path) -> None:
    workspace = tmp_path / "patient_runtime"
    store = PatientRuntimeStore(workspace / "patient_state.db")
    store.update_status(
        daemon_status="running",
        paused=0,
        scenario_profile="normal_day",
        active_seed=1101,
        workspace=str(workspace),
        api_host="127.0.0.1",
        api_port=8765,
    )
    (workspace / "live_bundle").mkdir(parents=True, exist_ok=True)
    (workspace / "live_bundle" / "results.csv").write_text("time_minutes,glucose_actual_mgdl\n0,110\n", encoding="utf-8")

    archive_path = tmp_path / "edge_bundle.zip"
    payload = create_edge_bundle(workspace, output_path=archive_path)

    assert Path(payload["archive"]).is_file()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "patient_runtime/EDGE_BUNDLE_SUMMARY.json" in names
        assert "patient_runtime/live_bundle/results.csv" in names


def test_export_edge_setup_scaffolds_project(tmp_path) -> None:
    output_dir = tmp_path / "edge_demo"
    outputs = export_edge_setup(output_dir, board="uno_q", include_uno_bridge=True, uno_bridge_port="/dev/ttyACM0")

    assert Path(outputs["algorithm"]).is_file()
    assert Path(outputs["run_script"]).is_file()
    assert Path(outputs["kiosk_script"]).is_file()
    assert Path(outputs["makerfaire_script"]).is_file()
    assert Path(outputs["makerfaire_guide"]).is_file()
    assert Path(outputs["makerfaire_kiosk_script"]).is_file()
    assert Path(outputs["makerfaire_desktop_entry"]).is_file()
    assert Path(outputs["makerfaire_autostart_script"]).is_file()
    assert Path(outputs["makerfaire_autostart_guide"]).is_file()
    assert Path(outputs["makerfaire_watchdog_script"]).is_file()
    assert Path(outputs["makerfaire_watchdog_service"]).is_file()
    assert Path(outputs["makerfaire_watchdog_timer"]).is_file()
    assert Path(outputs["makerfaire_checklist"]).is_file()
    assert Path(outputs["remote_access_guide"]).is_file()
    assert Path(outputs["long_study_template"]).is_file()
    assert Path(outputs["service_file"]).is_file()
    assert Path(outputs["setup_guide"]).is_file()
    assert Path(outputs["uno_q_bridge"]).is_dir()
    assert Path(outputs["uno_bridge_service"]).is_file()
    assert Path(outputs["uno_bridge_service_notes"]).is_file()
    guide_text = Path(outputs["setup_guide"]).read_text(encoding="utf-8")
    assert "iints edge up --project-dir ." in guide_text
    assert "iints edge bridge-run --project-dir . --port /dev/ttyACM0" in guide_text
    makerfaire_text = Path(outputs["makerfaire_guide"]).read_text(encoding="utf-8")
    assert "iints makerfaire up --project-dir ." in makerfaire_text
    autostart_text = Path(outputs["makerfaire_autostart_guide"]).read_text(encoding="utf-8")
    assert "./install_makerfaire_autostart.sh" in autostart_text
    assert "watchdog timer" in autostart_text.lower()
    kiosk_script_text = Path(outputs["makerfaire_kiosk_script"]).read_text(encoding="utf-8")
    assert "--kiosk" in kiosk_script_text
    assert "chromium-browser" in kiosk_script_text
    assert "xset -dpms" in kiosk_script_text
    assert "--disable-background-networking" in kiosk_script_text
    assert "--disable-session-crashed-bubble" in kiosk_script_text
    desktop_entry_text = Path(outputs["makerfaire_desktop_entry"]).read_text(encoding="utf-8")
    assert "open_makerfaire_kiosk.sh" in desktop_entry_text
    assert "X-GNOME-Autostart-Delay=15" in desktop_entry_text
    watchdog_script_text = Path(outputs["makerfaire_watchdog_script"]).read_text(encoding="utf-8")
    assert "makerfaire watchdog" in watchdog_script_text
    checklist_text = Path(outputs["makerfaire_checklist"]).read_text(encoding="utf-8")
    assert "Day Before The Event" in checklist_text
    remote_text = Path(outputs["remote_access_guide"]).read_text(encoding="utf-8")
    assert "Raspberry Pi Connect" in remote_text
    long_study_text = Path(outputs["long_study_template"]).read_text(encoding="utf-8")
    assert "duration_days: 14" in long_study_text
    assert "relaxed_day" in long_study_text
    assert "output_dir: /media/pi/usb_ssd/results/long_study" in long_study_text
    assert "scratch_dir: /tmp/iints_edge_long_study" in long_study_text
    bridge_service_text = Path(outputs["uno_bridge_service"]).read_text(encoding="utf-8")
    assert "edge bridge-run" in bridge_service_text
    assert "/dev/ttyACM0" in bridge_service_text


def test_deploy_edge_project_runs_expected_remote_steps(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_process(command: list[str], *, step: str, timeout_seconds: float = 300.0, retries: int = 0) -> str:
        calls.append({"command": command, "step": step, "timeout_seconds": timeout_seconds, "retries": retries})
        return "ok"

    monkeypatch.setattr("iints.live_patient.edge_ops._run_process", _fake_run_process)

    payload = deploy_edge_project(
        host="pi.local",
        user_name="pi",
        ssh_port=2222,
        remote_dir="~/booth_demo",
        local_output_dir=tmp_path / "edge_demo",
        board="uno_q",
        scenario_profile="expo_hot_start",
        uno_bridge_port="/dev/ttyACM0",
        install_autostart=True,
        start_runtime=True,
    )

    assert payload["destination"] == "pi@pi.local"
    assert payload["remote_dir"] == "~/booth_demo"
    assert "uno_bridge_service" in payload["artifacts"]
    joined_commands = [" ".join(str(part) for part in call["command"]) for call in calls]
    assert any(command.startswith("ssh -p 2222 pi@pi.local") for command in joined_commands)
    assert any("iints edge setup --output-dir \"$REMOTE_DIR\" --board uno_q" in command for command in joined_commands)
    assert any("./install_makerfaire_autostart.sh" in command for command in joined_commands)
    assert any("iints makerfaire up --project-dir . --scenario-profile expo_hot_start" in command for command in joined_commands)
    assert any(call["retries"] == 1 for call in calls)


def test_build_edge_offline_bundle_writes_tarball(monkeypatch, tmp_path) -> None:
    def _fake_run_process(command: list[str], *, step: str, timeout_seconds: float = 300.0, retries: int = 0, cwd: Path | None = None) -> str:
        wheel_dir = Path(command[command.index("--wheel-dir") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (wheel_dir / "iints_sdk_python35-1.5.3-py3-none-any.whl").write_text("wheel", encoding="utf-8")
        return "ok"

    monkeypatch.setattr("iints.live_patient.edge_ops._run_process", _fake_run_process)

    bundle_path = tmp_path / "iints_offline.tar.gz"
    outputs = build_edge_offline_bundle(bundle_path, board="raspberry_pi")

    assert Path(outputs["archive"]).is_file()
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = set(archive.getnames())
        assert "iints_offline/install_offline_edge.sh" in names
        assert "iints_offline/OFFLINE_INSTALL.md" in names
        assert "iints_offline/edge_project/EDGE_SETUP.md" in names
