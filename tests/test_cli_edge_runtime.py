from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.live_patient.runtime import PatientRuntimeStore


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
