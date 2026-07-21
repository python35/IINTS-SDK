from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints.research.mechanistic_models import (
    inspect_sbml_model,
    run_sbml_model,
    sbml_summary_payload,
)


SBML_DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="glucose_reference" name="Glucose reference" timeUnits="minute"
         substanceUnits="millimole" extentUnits="millimole" volumeUnits="litre">
    <listOfUnitDefinitions>
      <unitDefinition id="minute" />
    </listOfUnitDefinitions>
    <listOfCompartments>
      <compartment id="plasma" size="1" units="litre" constant="true" />
    </listOfCompartments>
    <listOfSpecies>
      <species id="G" compartment="plasma" initialConcentration="6"
               substanceUnits="millimole" boundaryCondition="false" constant="false" />
      <species id="I" compartment="plasma" initialConcentration="1"
               substanceUnits="millimole" boundaryCondition="false" constant="false" />
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.1" units="per_minute" constant="true" />
    </listOfParameters>
    <listOfReactions>
      <reaction id="clearance" reversible="false">
        <listOfReactants><speciesReference species="G" stoichiometry="1" constant="true" /></listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k</ci><ci>G</ci></apply></math>
          <listOfLocalParameters>
            <localParameter id="unused_local" value="1" />
          </listOfLocalParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _write_model(path: Path) -> Path:
    path.write_text(SBML_DOCUMENT, encoding="utf-8")
    return path


class _FakeNamedArray(list[list[float]]):
    def __init__(self, rows: list[list[float]], columns: list[str]) -> None:
        super().__init__(rows)
        self.colnames = columns


class _FakeRoadRunner:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.timeCourseSelections: list[str] = []

    def simulate(self, start: float, end: float, points: int) -> _FakeNamedArray:
        assert Path(self.model_path).is_file()
        step = (end - start) / (points - 1)
        rows: list[list[float]] = []
        for index in range(points):
            time = start + index * step
            values = [time]
            for selection in self.timeCourseSelections[1:]:
                values.append(6.0 - 0.1 * time if selection == "[G]" else 1.0)
            rows.append(values)
        return _FakeNamedArray(rows, list(self.timeCourseSelections))

    def getCurrentIntegratorName(self) -> str:
        return "fake-cvode"


class _FakeRoadRunnerModule:
    __version__ = "test-1.0"
    RoadRunner = _FakeRoadRunner


def test_inspect_sbml_model_records_structure_units_and_hash(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    summary = inspect_sbml_model(model)

    assert summary.readiness_status == "inspectable"
    assert summary.sbml_level == 3
    assert summary.sbml_version == 2
    assert summary.model_id == "glucose_reference"
    assert summary.counts["species"] == 2
    assert summary.counts["parameters"] == 1
    assert summary.counts["local_parameters"] == 1
    assert summary.counts["reactions"] == 1
    assert summary.model_units["timeUnits"] == "minute"
    assert {row["id"] for row in summary.species} == {"G", "I"}
    assert len(summary.sha256) == 64


def test_sbml_summary_can_hide_absolute_local_path(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.xml")

    payload = sbml_summary_payload(inspect_sbml_model(model), include_local_path=False)

    assert payload["model_path"] == "reference.xml"
    assert str(tmp_path) not in json.dumps(payload)


def test_inspect_rejects_dtd_before_xml_execution(tmp_path: Path) -> None:
    model = tmp_path / "unsafe.xml"
    model.write_text(
        '<!DOCTYPE sbml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><sbml>&xxe;</sbml>',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="DTD and entity"):
        inspect_sbml_model(model)


def test_run_sbml_model_writes_provenance_and_finite_results(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    result = run_sbml_model(
        model,
        tmp_path / "runs",
        start=0.0,
        end=10.0,
        points=3,
        variables=["G", "k"],
        source_url="https://example.org/reference.sbml",
        model_license="CC-BY-4.0",
        _engine_module=_FakeRoadRunnerModule,
    )

    assert result.row_count == 3
    assert result.selections == ("time", "[G]", "k")
    assert result.results_csv.read_text(encoding="utf-8").splitlines()[0] == "time,[G],k"
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    assert manifest["model"]["sha256"] == inspect_sbml_model(model).sha256
    assert manifest["model"]["license"] == "CC-BY-4.0"
    assert manifest["evidence_source_ids"] == ["sbml_2019_l3v2_core", "libroadrunner_2015"]
    assert manifest["engine"] == {
        "integrator": "fake-cvode",
        "integrator_settings": {},
        "name": "libRoadRunner",
        "version": "test-1.0",
    }
    assert manifest["simulation"]["time_units"] == "minute"
    assert manifest["simulation"]["selection_metadata"][1]["semantic"] == "species_concentration"
    assert manifest["simulation"]["selection_metadata"][2] == {
        "declared_units": "per_minute",
        "identifier": "k",
        "selection": "k",
        "semantic": "global_parameter",
    }
    assert str(tmp_path) not in result.manifest_json.read_text(encoding="utf-8")
    assert "not biological or clinical validation" in result.report_md.read_text(encoding="utf-8")


def test_run_sbml_model_rejects_unknown_variable(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    with pytest.raises(ValueError, match="Unknown SBML selection"):
        run_sbml_model(
            model,
            tmp_path / "runs",
            variables=["not_declared"],
            _engine_module=_FakeRoadRunnerModule,
        )


def test_run_sbml_model_preserves_explicit_amount_and_concentration_semantics(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    result = run_sbml_model(
        model,
        tmp_path / "runs",
        start=0,
        end=1,
        points=2,
        variables=["amount:G", "concentration:I"],
        _engine_module=_FakeRoadRunnerModule,
    )

    assert result.selections == ("time", "G", "[I]")
    manifest = json.loads(result.manifest_json.read_text(encoding="utf-8"))
    semantics = [row["semantic"] for row in manifest["simulation"]["selection_metadata"]]
    assert semantics == ["time", "species_amount", "species_concentration"]


def test_run_sbml_model_requires_explicit_https_source(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    with pytest.raises(ValueError, match="source_url must use HTTPS"):
        run_sbml_model(
            model,
            tmp_path / "runs",
            source_url="http://example.org/model.xml",
            _engine_module=_FakeRoadRunnerModule,
        )


def test_run_sbml_model_rejects_source_credentials_and_multiline_license(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "reference.sbml")

    with pytest.raises(ValueError, match="embedded credentials"):
        run_sbml_model(
            model,
            tmp_path / "runs",
            source_url="https://user:secret@example.org/model.xml",
            _engine_module=_FakeRoadRunnerModule,
        )
    with pytest.raises(ValueError, match="single-line"):
        run_sbml_model(
            model,
            tmp_path / "runs",
            model_license="CC-BY-4.0\ninjected",
            _engine_module=_FakeRoadRunnerModule,
        )
