from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest

from iints_desktop.engine import RUN_HISTORY_FILENAME


def _write_minimal_sbml(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="bridge_reference" timeUnits="minute" substanceUnits="mole">
    <listOfCompartments><compartment id="c" size="1" constant="true" /></listOfCompartments>
    <listOfSpecies>
      <species id="G" compartment="c" initialConcentration="1" boundaryCondition="false" constant="false" />
    </listOfSpecies>
    <listOfRules><rateRule variable="G"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math></rateRule></listOfRules>
  </model>
</sbml>
""",
        encoding="utf-8",
    )


def _write_cross_scale_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    copasi = tmp_path / "analysis.cps"
    copasi.write_text(
        """<COPASI versionMajor="4" versionMinor="45"><Model name="test" />
<ListOfTasks><Task name="Sensitivities" type="sensitivities" scheduled="true" updateModel="false">
<Method name="Sensitivities Method" type="SensitivitiesMethod" /></Task></ListOfTasks></COPASI>""",
        encoding="utf-8",
    )
    cellml = tmp_path / "reference.cellml"
    cellml.write_text(
        """<model xmlns="http://www.cellml.org/cellml/2.0#" name="reference">
<component name="c"><variable name="x" units="dimensionless" />
<math xmlns="http://www.w3.org/1998/Math/MathML"><cn>0</cn></math></component></model>""",
        encoding="utf-8",
    )
    fmu = tmp_path / "device.fmu"
    with zipfile.ZipFile(fmu, "w") as archive:
        archive.writestr(
            "modelDescription.xml",
            """<fmiModelDescription fmiVersion="2.0" modelName="device" guid="fixture">
<CoSimulation modelIdentifier="device" /><ModelVariables>
<ScalarVariable name="flow" valueReference="1" causality="output"><Real unit="mL/min" /></ScalarVariable>
</ModelVariables></fmiModelDescription>""",
        )
    return copasi, cellml, fmu


def _bridge(*args: str) -> dict[str, object]:
    source_root = str(Path.cwd() / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "iints_desktop.tauri_bridge", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": source_root + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
        },
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
    assert "roadrunner" in data["optional_modules"]  # type: ignore[index]
    assert "fmpy" in data["optional_modules"]  # type: ignore[index]
    assert isinstance(data["recommended_checks"], list)  # type: ignore[index]


def test_tauri_bridge_inspects_local_sbml_without_execution(tmp_path: Path) -> None:
    model = tmp_path / "reference.sbml"
    _write_minimal_sbml(model)

    payload = _bridge("mechanistic-inspect", "--model", str(model))

    assert payload["ok"] is True
    data = payload["data"]
    assert data["medical_device"] is False  # type: ignore[index]
    assert data["schema_validation_performed"] is False  # type: ignore[index]
    summary = data["summary"]  # type: ignore[index]
    assert summary["model_id"] == "bridge_reference"
    assert summary["counts"]["species"] == 1


def test_tauri_bridge_reports_optional_mechanistic_engine_status() -> None:
    payload = _bridge("mechanistic-status")

    assert payload["ok"] is True
    data = payload["data"]
    assert data["engine"] == "libRoadRunner"  # type: ignore[index]
    assert data["inspection_available"] is True  # type: ignore[index]
    assert isinstance(data["available"], bool)  # type: ignore[index]


def test_tauri_bridge_reports_cross_scale_engine_status() -> None:
    payload = _bridge("cross-scale-status")

    assert payload["ok"] is True
    data = payload["data"]
    assert data["static_inspection"] == {"copasi": True, "cellml": True, "fmu": True}  # type: ignore[index]
    assert data["bindingdb"]["verified_tls"] is True  # type: ignore[index]
    assert data["medical_device"] is False  # type: ignore[index]


def test_tauri_bridge_inspects_cross_scale_models_without_optional_engines(tmp_path: Path) -> None:
    copasi, cellml, fmu = _write_cross_scale_fixtures(tmp_path)

    copasi_payload = _bridge("copasi-inspect", "--model", str(copasi))
    cellml_payload = _bridge("cellml-inspect", "--model", str(cellml))
    fmi_payload = _bridge("fmi-inspect", "--model", str(fmu))

    assert copasi_payload["ok"] is True
    assert copasi_payload["data"]["summary"]["sensitivity_task_count"] == 1  # type: ignore[index]
    assert cellml_payload["ok"] is True
    assert cellml_payload["data"]["summary"]["cellml_version"] == "2.0"  # type: ignore[index]
    assert fmi_payload["ok"] is True
    assert fmi_payload["data"]["summary"]["fmi_version"] == "2.0"  # type: ignore[index]


def test_tauri_bridge_reports_update_info() -> None:
    payload = _bridge("update-info")

    assert payload["ok"] is True
    data = payload["data"]
    assert data["package_spec"] == "iints-sdk-python35[desktop-all]"  # type: ignore[index]
    assert "pip install -U" in str(data["pip_command"])  # type: ignore[index]
    assert data["app_download_url"] == "https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-latest"  # type: ignore[index]
    assert data["update_docs_url"] == "https://python35.github.io/IINTS-SDK/APP_INSTALL/"  # type: ignore[index]
    assert data["medical_device"] is False  # type: ignore[index]


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
        "libroadrunner",
        "copasi",
        "opencor-physiome",
        "fmi-fmpy",
        "bindingdb",
        "string-db",
        "clinvar",
        "ro-crate",
        "fair4rs",
        "sed-ml",
        "sbml",
        "pubmed",
        "clinicaltrials-gov",
        "zenodo",
    }.issubset(keys)
    assert all(str(connector["primary_url"]).startswith("https://") for connector in connectors)
    assert all(connector["integration_level"] in {"integrated", "partial", "planned", "portal"} for connector in connectors)
    ro_crate = next(connector for connector in connectors if connector["key"] == "ro-crate")
    assert ro_crate["writes_local_evidence"] is True
    sbml = next(connector for connector in connectors if connector["key"] == "sbml")
    assert sbml["integration_level"] == "integrated"
    assert sbml["writes_local_evidence"] is True


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


def test_tauri_bridge_exports_academic_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "time_minutes,glucose_actual_mgdl\n0,110\n5,120\n",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "git_sha": "0123456789abcdef",
                "python_version": "3.11.15",
                "dependencies": [{"name": "numpy", "version": "2.0.0"}],
                "config": {"patient_model_type": "hovorka"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"files": {"results": {"path": "results.csv"}}}),
        encoding="utf-8",
    )

    payload = _bridge(
        "academic-bundle",
        "--run-dir",
        str(run_dir),
        "--creator",
        "Researcher Example",
        "--license",
        "CC-BY-4.0",
        "--source-id",
        "hovorka_2004_nmpc_t1d",
    )

    assert payload["ok"] is True
    data = payload["data"]
    assert data["readiness_status"] == "ready"  # type: ignore[index]
    assert Path(data["ro_crate_metadata"]).is_file()  # type: ignore[index,arg-type]
    assert data["medical_device"] is False  # type: ignore[index]
