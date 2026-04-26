from __future__ import annotations

import json

from typer.testing import CliRunner

from iints.api.registry import install_algorithm_plugin, list_algorithm_plugins, list_local_plugin_records
from iints.cli.cli import app
from iints.utils.run_io import RESULTS_CSV_FORMAT_VERSION, build_run_manifest, build_run_metadata

runner = CliRunner()


def _write_algorithm(path):
    path.write_text(
        """
import iints


class LocalDemoAlgorithm(iints.InsulinAlgorithm):
    def __init__(self):
        super().__init__()
        self.set_algorithm_metadata(iints.AlgorithmMetadata(name="Local Demo Algo", version="0.1.0"))

    def predict_insulin(self, data):
        return {"total_insulin_delivered": 0.0}
""".lstrip(),
        encoding="utf-8",
    )


def test_local_algorithm_plugin_install_and_discovery(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IINTS_PLUGIN_HOME", str(tmp_path / "plugins"))
    algo_path = tmp_path / "my_algo.py"
    _write_algorithm(algo_path)

    record = install_algorithm_plugin(algo_path)

    assert record.kind == "algorithm"
    assert record.name == "Local Demo Algo"
    assert (tmp_path / "plugins" / "registry.json").is_file()
    assert list_local_plugin_records("algorithm")[0].name == "Local Demo Algo"
    listings = list_algorithm_plugins()
    local = [entry for entry in listings if entry.name == "Local Demo Algo"]
    assert local
    assert local[0].source == "local"
    assert local[0].status == "available"


def test_plugin_cli_registers_algorithm_and_patient_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IINTS_PLUGIN_HOME", str(tmp_path / "plugins"))
    algo_path = tmp_path / "my_algo.py"
    patient_model_path = tmp_path / "my_model.py"
    _write_algorithm(algo_path)
    patient_model_path.write_text("class MyModel:\n    pass\n", encoding="utf-8")

    installed = runner.invoke(app, ["plugin", "install", str(algo_path)])
    assert installed.exit_code == 0
    assert "Local Demo Algo" in installed.stdout
    assert "iints algorithms list" in installed.stdout

    registered = runner.invoke(app, ["plugin", "register", "algo", str(algo_path), "--name", "Local Demo Alias"])
    assert registered.exit_code == 0
    assert "Local Demo Alias" in registered.stdout

    listed = runner.invoke(app, ["plugin", "list"])
    assert listed.exit_code == 0
    assert "Local Demo Algo" in listed.stdout
    assert "Local Demo Alias" in listed.stdout

    model = runner.invoke(app, ["plugin", "register", "patient-model", str(patient_model_path), "--name", "My Model"])
    assert model.exit_code == 0
    assert "My Model" in model.stdout

    patient_models = runner.invoke(app, ["patientmodel", "list"])
    assert patient_models.exit_code == 0
    assert "custom" in patient_models.stdout
    assert "My Model" in patient_models.stdout


def test_run_metadata_exposes_data_format_versions(tmp_path) -> None:
    metadata = build_run_metadata("run-1", 42, {"demo": True}, tmp_path)
    manifest = build_run_manifest(tmp_path, {})

    assert metadata["schema_version"] == "1.0"
    assert metadata["format_versions"]["results_csv"] == RESULTS_CSV_FORMAT_VERSION
    assert manifest["schema_version"] == "1.0"
    json.dumps(metadata)
    json.dumps(manifest)
