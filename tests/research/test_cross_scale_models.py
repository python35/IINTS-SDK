from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from iints.research.binding_evidence import query_bindingdb_uniprot
from iints.research.cellml_models import inspect_cellml_model, validate_cellml_model
from iints.research.copasi_models import inspect_copasi_model, run_copasi_model
from iints.research.fmi_models import inspect_fmu_model, run_fmu_model
from iints.research.external_models_common import run_external_command


COPASI_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<COPASI versionMajor="4" versionMinor="45" versionDevel="300">
  <Model key="Model_1" name="Insulin sensitivity example" />
  <ListOfTasks>
    <Task key="Task_1" name="Local sensitivities" type="sensitivities" scheduled="true" updateModel="false">
      <Report reference="Report_1" target="sensitivity.tsv" />
      <Method name="Sensitivities Method" type="SensitivitiesMethod" />
    </Task>
    <Task key="Task_2" name="Parameter fit" type="parameterFitting" scheduled="false" updateModel="true">
      <Problem><Parameter name="File Name" type="file" value="observations.tsv" /></Problem>
      <Method name="Levenberg - Marquardt" type="LevenbergMarquardt" />
    </Task>
  </ListOfTasks>
</COPASI>
"""


CELLML_MODEL = """<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" name="glucose_reference">
  <units name="millimolar" />
  <component name="glucose">
    <variable name="t" units="second" interface="public" />
    <variable name="G" units="millimolar" initial_value="5.5" />
    <math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math>
  </component>
</model>
"""


FMU_DESCRIPTION = """<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription fmiVersion="2.0" modelName="PumpFlow" guid="test-guid"
  generationTool="IINTS test fixture" generationDateAndTime="2026-07-18T00:00:00Z">
  <CoSimulation modelIdentifier="pump_flow" />
  <DefaultExperiment startTime="0" stopTime="2" stepSize="1" />
  <ModelVariables>
    <ScalarVariable name="flow_ml_min" valueReference="1" causality="output" variability="continuous">
      <Real unit="mL/min" start="0" />
    </ScalarVariable>
  </ModelVariables>
</fmiModelDescription>
"""


def _write_fmu(path: Path, *, unsafe_member: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("modelDescription.xml", FMU_DESCRIPTION)
        archive.writestr("binaries/x86_64-linux/pump_flow.so", b"not executable in tests")
        if unsafe_member is not None:
            archive.writestr(unsafe_member, b"unsafe")


def test_external_engine_runner_bounds_output_before_loading_it_into_memory() -> None:
    command = [sys.executable, "-c", "import sys; sys.stdout.write('x' * (6 * 1024 * 1024))"]

    with pytest.raises(RuntimeError, match="5 MiB"):
        run_external_command(command, timeout_seconds=10)


def test_copasi_inspection_finds_sensitivity_fitting_and_external_data(tmp_path: Path) -> None:
    model = tmp_path / "model.cps"
    model.write_text(COPASI_MODEL, encoding="utf-8")

    summary = inspect_copasi_model(model)

    assert summary.model_name == "Insulin sensitivity example"
    assert summary.scheduled_task_count == 1
    assert summary.sensitivity_task_count == 1
    assert summary.parameter_estimation_task_count == 1
    assert "observations.tsv" in summary.external_file_references
    assert "sensitivity.tsv" in summary.external_file_references


def test_copasi_execution_requires_review_and_writes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.cps"
    model.write_text(COPASI_MODEL, encoding="utf-8")
    engine = tmp_path / "CopasiSE"
    engine.write_text("fixture", encoding="utf-8")

    with pytest.raises(PermissionError, match="opt-in"):
        run_copasi_model(model, tmp_path / "blocked", executable=engine)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        save_path = Path(command[command.index("--save") + 1])
        report_path = Path(command[command.index("--report-file") + 1])
        save_path.write_text(COPASI_MODEL, encoding="utf-8")
        report_path.write_text("sensitivity result", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "COPASI complete\n", "")

    monkeypatch.setattr("iints.research.copasi_models.run_external_command", fake_run)
    result = run_copasi_model(
        model,
        tmp_path / "runs",
        allow_external_execution=True,
        executable=engine,
    )

    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["task"]["scheduled_task_count"] == 1
    assert result.report_txt.read_text(encoding="utf-8") == "sensitivity result"


def test_cellml_inspection_and_opencor_validation_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.cellml"
    model.write_text(CELLML_MODEL, encoding="utf-8")
    summary = inspect_cellml_model(model)
    assert summary.cellml_version == "2.0"
    assert summary.component_count == 1
    assert summary.variable_count == 2
    assert summary.opencor_validation_performed is False

    engine = tmp_path / "OpenCOR"
    engine.write_text("fixture", encoding="utf-8")

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1:3] == ["-c", "CellMLTools::validate"]
        return subprocess.CompletedProcess(command, 0, "CellML file is valid\n", "")

    monkeypatch.setattr("iints.research.cellml_models.run_external_command", fake_run)
    result = validate_cellml_model(model, tmp_path / "validated", executable=engine)
    assert result.valid is True
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["validation"]["command"] == "CellMLTools::validate"


def test_fmu_static_inspection_never_executes_and_rejects_archive_traversal(tmp_path: Path) -> None:
    fmu = tmp_path / "pump.fmu"
    _write_fmu(fmu)
    summary = inspect_fmu_model(fmu)
    assert summary.fmi_version == "2.0"
    assert summary.has_native_binaries is True
    assert summary.platforms == ("x86_64-linux",)
    assert summary.variable_count == 1
    assert any("native binaries" in warning for warning in summary.warnings)

    unsafe = tmp_path / "unsafe.fmu"
    _write_fmu(unsafe, unsafe_member="../escape.txt")
    with pytest.raises(ValueError, match="unsafe archive path"):
        inspect_fmu_model(unsafe)


def test_fmpy_execution_requires_native_code_consent_and_exports_rows(tmp_path: Path) -> None:
    fmu = tmp_path / "pump.fmu"
    _write_fmu(fmu)
    with pytest.raises(PermissionError, match="native code"):
        run_fmu_model(fmu, tmp_path / "blocked", start=0, end=2, output_interval=1)

    class FakeResult(list[dict[str, float]]):
        dtype = SimpleNamespace(names=("time", "flow_ml_min"))

    calls: list[dict[str, object]] = []

    def simulate_fmu(_path: str, **kwargs: object) -> FakeResult:
        calls.append(kwargs)
        return FakeResult(
            [
                {"time": 0.0, "flow_ml_min": 0.0},
                {"time": 1.0, "flow_ml_min": 0.2},
                {"time": 2.0, "flow_ml_min": 0.2},
            ]
        )

    engine = SimpleNamespace(__version__="0.3.test", simulate_fmu=simulate_fmu)
    result = run_fmu_model(
        fmu,
        tmp_path / "runs",
        start=0,
        end=2,
        output_interval=1,
        variables=["flow_ml_min"],
        allow_native_execution=True,
        _engine_module=engine,
    )
    assert result.row_count == 3
    assert calls[0]["validate"] is True
    rows = list(csv.DictReader(result.results_csv.open(encoding="utf-8")))
    assert rows[-1]["flow_ml_min"] == "0.2"
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["native_code_execution"] is True


def test_fmpy_can_select_a_variable_beyond_the_inspection_preview(tmp_path: Path) -> None:
    variables = "\n".join(
        (
            f'<ScalarVariable name="value_{index}" valueReference="{index}" causality="output">'
            f'<Real unit="1" start="0" /></ScalarVariable>'
        )
        for index in range(1, 1_003)
    )
    description = FMU_DESCRIPTION.replace(
        """    <ScalarVariable name="flow_ml_min" valueReference="1" causality="output" variability="continuous">
      <Real unit="mL/min" start="0" />
    </ScalarVariable>""",
        variables,
    )
    fmu = tmp_path / "large.fmu"
    with zipfile.ZipFile(fmu, "w") as archive:
        archive.writestr("modelDescription.xml", description)

    class FakeResult(list[dict[str, float]]):
        dtype = SimpleNamespace(names=("time", "value_1002"))

    requested_outputs: list[list[str] | None] = []

    def simulate_fmu(_path: str, **kwargs: object) -> FakeResult:
        requested_outputs.append(kwargs.get("output"))  # type: ignore[arg-type]
        return FakeResult([{"time": 0.0, "value_1002": 1.0}])

    result = run_fmu_model(
        fmu,
        tmp_path / "runs",
        start=0,
        end=1,
        output_interval=1,
        variables=["value_1002"],
        allow_native_execution=True,
        _engine_module=SimpleNamespace(__version__="test", simulate_fmu=simulate_fmu),
    )

    assert result.row_count == 1
    assert requested_outputs == [["value_1002"]]


def test_bindingdb_export_keeps_assay_types_and_censoring_distinct(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "getLindsByUniprotsResponse": {
                "affinities": [
                    {
                        "query": "=Untrusted spreadsheet formula",
                        "monomerid": "1",
                        "smile": "CCO",
                        "affinity_type": "Ki",
                        "affinity": "4101",
                        "pmid": "123",
                        "doi": "10.1000/test",
                    },
                    {
                        "query": "Insulin receptor",
                        "monomerid": "2",
                        "smile": "CCC",
                        "affinity_type": "IC50",
                        "affinity": ">10000",
                        "pmid": "456",
                        "doi": "",
                    },
                ]
            }
        }
    ).encode()

    result = query_bindingdb_uniprot(
        "P06213",
        tmp_path,
        cutoff_nm=20_000,
        _fetcher=lambda _url, _timeout: payload,
    )

    evidence = json.loads(result.evidence_json.read_text(encoding="utf-8"))
    assert evidence["summary"]["affinity_types"] == {"IC50": 1, "Ki": 1}
    assert evidence["records"][1]["affinity_relation"] == ">"
    assert evidence["records"][1]["affinity_value_nm"] == 10000.0
    assert evidence["records"][0]["target_name"] == "=Untrusted spreadsheet formula"
    csv_rows = list(csv.DictReader(result.records_csv.open(encoding="utf-8")))
    assert csv_rows[0]["target_name"] == "'=Untrusted spreadsheet formula"
    assert evidence["csv_formula_protection"] is True
    assert any("not interchangeable" in item for item in evidence["limitations"])
