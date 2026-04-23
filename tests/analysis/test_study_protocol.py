from __future__ import annotations

import json

from typer.testing import CliRunner

from iints.analysis.study_experiment import load_study_experiment_config
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
    assert {entry["display_name"] for entry in registry[1:]} >= {"Clinical Baseline", "PID Controller", "Standard Pump", "Correction Bolus"}
    assert outputs["study_matrix_csv"].endswith("study_matrix.csv")
    assert outputs["algorithms_json"].endswith("algorithms.json")
    assert outputs["study_experiment_yaml"].endswith("study_experiment.yaml")
    assert (tmp_path / "protocol" / "study_experiment.yaml").is_file()


def test_write_study_protocol_bundle_supports_eucys_preset(tmp_path) -> None:
    write_study_protocol_bundle(tmp_path / "protocol", preset="eucys")
    design = json.loads((tmp_path / "protocol" / "study_design.json").read_text(encoding="utf-8"))
    assert design["preset"] == "eucys"
    assert len(design["seed_policy"]["seeds"]) == 10


def test_write_study_protocol_bundle_sanitizes_csv_formula_cells(tmp_path) -> None:
    write_study_protocol_bundle(
        tmp_path / "protocol",
        algorithms=["=CandidateAlgo"],
        extra_algorithms=["@AltAlgo"],
    )

    csv_text = (tmp_path / "protocol" / "study_matrix.csv").read_text(encoding="utf-8")

    assert "'=CandidateAlgo" in csv_text
    assert "'@AltAlgo" in csv_text


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


def test_load_study_experiment_config_resolves_relative_paths(tmp_path) -> None:
    experiment_path = tmp_path / "study_experiment.yaml"
    candidate_path = tmp_path / "algorithms" / "candidate.py"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("class Demo: pass\n", encoding="utf-8")
    experiment_path.write_text(
        """
experiment:
  name: meal_stress_test
  preset: default
  profile_set: clinic_safe_core
  seeds: [7, 9]
  time_step: 10
  include_default_baselines: true
study:
  scenarios:
    - baseline_day
    - meal_challenge
algorithm:
  candidate: algorithms/candidate.py
  extra_algorithms:
    - Standard Pump
paths:
  output_dir: results/custom_bundle
  carelink_metrics: refs/carelink_metrics.json
""".strip(),
        encoding="utf-8",
    )

    config = load_study_experiment_config(experiment_path)

    assert config.name == "meal_stress_test"
    assert config.seeds == [7, 9]
    assert config.time_step == 10
    assert config.scenarios == ["baseline_day", "meal_challenge"]
    assert config.candidate_algorithm == candidate_path.resolve()
    assert config.output_dir == (tmp_path / "results" / "custom_bundle").resolve()
    assert config.carelink_metrics == (tmp_path / "refs" / "carelink_metrics.json").resolve()
