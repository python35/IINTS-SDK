from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from iints_desktop.engine import RUN_HISTORY_FILENAME


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


def test_tauri_bridge_reports_desktop_diagnostics() -> None:
    payload = _bridge("diagnostics")

    assert payload["ok"] is True
    data = payload["data"]
    assert data["medical_device"] is False  # type: ignore[index]
    assert "python_version" in data  # type: ignore[operator]
    assert "optional_modules" in data  # type: ignore[operator]
    assert "pandas" in data["optional_modules"]  # type: ignore[index]
    assert isinstance(data["recommended_checks"], list)  # type: ignore[index]


def test_tauri_bridge_lists_molecule_assets() -> None:
    payload = _bridge("molecules")

    assert payload["ok"] is True
    molecules = payload["data"]["molecules"]  # type: ignore[index]
    assert isinstance(molecules, list)
    assert len(molecules) >= 5
    first = molecules[0]
    assert "image_data_url" in first
    assert str(first["image_data_url"]).startswith("data:image/png;base64,")
    assert first["structure_path"]
    assert first["sdk_link"].startswith("Connects to:")


def test_tauri_bridge_lists_evidence_connectors() -> None:
    payload = _bridge("evidence-connectors")

    assert payload["ok"] is True
    connectors = payload["data"]["connectors"]  # type: ignore[index]
    assert isinstance(connectors, list)
    keys = {connector["key"] for connector in connectors}
    assert {
        "alphafold-db",
        "ensembl-vep-alphamissense",
        "open-targets",
        "reactome",
        "rcsb-pdb",
        "uniprot",
        "human-protein-atlas",
        "gtex",
        "chembl",
        "clinpgx-pharmgkb",
        "biomodels",
        "string-db",
        "clinvar",
    }.issubset(keys)
    assert all(str(connector["primary_url"]).startswith("https://") for connector in connectors)


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


def test_tauri_bridge_reads_run_history(tmp_path: Path) -> None:
    history_path = tmp_path / RUN_HISTORY_FILENAME
    history_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-07-16T12:00:00+00:00",
                "workflow_title": "Doctor safety discussion",
                "preset_name": "hypo_prone_night",
                "seed": 42,
                "run_id": "run-test",
                "output_dir": str(tmp_path / "run-test"),
                "results_csv": str(tmp_path / "run-test" / "results.csv"),
                "report_pdf": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = _bridge("history", "--output-dir", str(tmp_path), "--limit", "5")

    assert payload["ok"] is True
    entries = payload["data"]["history"]  # type: ignore[index]
    assert entries[0]["run_id"] == "run-test"
    assert entries[0]["preset_name"] == "hypo_prone_night"


def test_tauri_bridge_mdmp_certify_smoke(tmp_path: Path) -> None:
    pytest.importorskip("mdmp_core.crypto")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units\n"
        "0,110,0,0\n"
        "5,140,10,0.2\n",
        encoding="utf-8",
    )

    payload = _bridge("mdmp-certify", "--csv", str(csv_path), "--quick-rows", "50")

    assert payload["ok"] is True
    data = payload["data"]
    assert Path(data["certificate_path"]).is_file()  # type: ignore[index,arg-type]
    assert data["row_count"] >= 1  # type: ignore[index]
