from __future__ import annotations

import json

from typer.testing import CliRunner

from iints.analysis.study_protocol import build_study_protocol_payload, write_study_protocol_bundle
from iints.cli.cli import app


runner = CliRunner()


def test_build_study_protocol_payload_includes_hypotheses_and_metrics() -> None:
    payload = build_study_protocol_payload()
    assert payload["hypotheses"][0]["id"] == "H1"
    assert "tir_70_180" in payload["metrics"]
    assert "external_validation" in payload
    assert payload["profile_set"] == "clinic_safe_core"
    assert payload["algorithms"][0]["role"] == "candidate"


def test_write_study_protocol_bundle_writes_files(tmp_path) -> None:
    outputs = write_study_protocol_bundle(tmp_path / "protocol", seeds=[1, 2], algorithms=["AlgoA"])
    assert (tmp_path / "protocol" / "STUDY_PROTOCOL.md").is_file()
    design = json.loads((tmp_path / "protocol" / "study_design.json").read_text(encoding="utf-8"))
    registry = json.loads((tmp_path / "protocol" / "algorithms.json").read_text(encoding="utf-8"))
    assert design["seed_policy"]["seeds"] == [1, 2]
    assert registry[0]["display_name"] == "AlgoA"
    assert registry[0]["role"] == "candidate"
    assert {entry["display_name"] for entry in registry[1:]} >= {"PID Controller", "Standard Pump", "Correction Bolus"}
    assert outputs["study_matrix_csv"].endswith("study_matrix.csv")
    assert outputs["algorithms_json"].endswith("algorithms.json")


def test_write_study_protocol_bundle_supports_eucys_preset(tmp_path) -> None:
    write_study_protocol_bundle(tmp_path / "protocol", preset="eucys")
    design = json.loads((tmp_path / "protocol" / "study_design.json").read_text(encoding="utf-8"))
    assert design["preset"] == "eucys"
    assert len(design["seed_policy"]["seeds"]) == 10


def test_cli_study_protocol_writes_bundle(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "study-protocol",
            "--output-dir",
            str(tmp_path / "protocol"),
            "--seeds",
            "1,2,3",
            "--algorithms",
            "AlgoA,Standard PID",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "protocol" / "STUDY_PROTOCOL.md").is_file()
    assert (tmp_path / "protocol" / "study_design.json").is_file()
    assert (tmp_path / "protocol" / "algorithms.json").is_file()


def test_cli_study_protocol_supports_eucys_preset(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "study-protocol",
            "--output-dir",
            str(tmp_path / "protocol"),
            "--preset",
            "eucys",
        ],
    )

    assert result.exit_code == 0
    design = json.loads((tmp_path / "protocol" / "study_design.json").read_text(encoding="utf-8"))
    assert design["preset"] == "eucys"
