from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _bridge(*args: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "iints_desktop.tauri_bridge", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def test_tauri_bridge_status_reports_sdk_context() -> None:
    payload = _bridge("status")

    assert payload["ok"] is True
    data = payload["data"]
    assert isinstance(data, dict)
    assert data["bridge"] == "iints_desktop.tauri_bridge"
    assert data["medical_device"] is False


def test_tauri_bridge_lists_curated_workflows() -> None:
    payload = _bridge("workflows")

    assert payload["ok"] is True
    workflows = payload["data"]["workflows"]  # type: ignore[index]
    assert isinstance(workflows, list)
    assert any(workflow["key"] == "doctor-safety" for workflow in workflows)


def test_tauri_bridge_previews_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units\n"
        "0,110,0,0\n"
        "5,140,10,0.2\n",
        encoding="utf-8",
    )

    payload = _bridge("preview", "--csv", str(csv_path), "--max-rows", "2")

    assert payload["ok"] is True
    data = payload["data"]
    assert data["row_count"] == 2  # type: ignore[index]
    assert "Mean glucose" in data["metrics"]  # type: ignore[index]
