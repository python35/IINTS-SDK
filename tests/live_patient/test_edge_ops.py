from __future__ import annotations

import json
import zipfile
from pathlib import Path

from iints.live_patient.edge_ops import create_edge_bundle, export_edge_setup, summarize_edge_workspace
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
    outputs = export_edge_setup(output_dir, board="uno_q", include_uno_bridge=True)

    assert Path(outputs["algorithm"]).is_file()
    assert Path(outputs["run_script"]).is_file()
    assert Path(outputs["kiosk_script"]).is_file()
    assert Path(outputs["makerfaire_script"]).is_file()
    assert Path(outputs["makerfaire_guide"]).is_file()
    assert Path(outputs["makerfaire_kiosk_script"]).is_file()
    assert Path(outputs["makerfaire_desktop_entry"]).is_file()
    assert Path(outputs["makerfaire_autostart_script"]).is_file()
    assert Path(outputs["makerfaire_autostart_guide"]).is_file()
    assert Path(outputs["service_file"]).is_file()
    assert Path(outputs["setup_guide"]).is_file()
    assert Path(outputs["uno_q_bridge"]).is_dir()
    guide_text = Path(outputs["setup_guide"]).read_text(encoding="utf-8")
    assert "iints edge up --project-dir ." in guide_text
    assert "iints edge bridge-run --project-dir . --port /dev/ttyACM0" in guide_text
    makerfaire_text = Path(outputs["makerfaire_guide"]).read_text(encoding="utf-8")
    assert "iints makerfaire up --project-dir ." in makerfaire_text
    autostart_text = Path(outputs["makerfaire_autostart_guide"]).read_text(encoding="utf-8")
    assert "./install_makerfaire_autostart.sh" in autostart_text
    desktop_entry_text = Path(outputs["makerfaire_desktop_entry"]).read_text(encoding="utf-8")
    assert "open_makerfaire_kiosk.sh" in desktop_entry_text
