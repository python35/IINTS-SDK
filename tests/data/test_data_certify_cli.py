from __future__ import annotations

import json
import importlib.util

import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data import (
    MDMP_GRADE_DEFINITIONS,
    certify_csv,
    certify_dataset,
    standard_diabetes_contract_path,
)


runner = CliRunner()


def _write_contract(path) -> None:
    path.write_text(
        """
version: 1
streams:
  - name: PatientHealth
    source: sdk.iints_af.v1
    metadata:
      required_columns: [timestamp, glucose]
      column_types:
        glucose: float
      ranges:
        glucose:
          min: 20
          max: 250
processes:
  - name: GlucoseData
    input_stream: PatientHealth.glucose
    validations:
      - expression: glucose is not null and glucose > 20
""".strip()
    )


def test_main_help_exposes_first_class_mdmp_namespace() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "\n  mdmp " in result.stdout.lower()
    assert "data" in result.stdout.lower()


def test_data_certify_commands_work_end_to_end(tmp_path) -> None:
    contract_path = tmp_path / "data_contract.yaml"
    input_csv = tmp_path / "input.csv"
    output_json = tmp_path / "certification.json"
    output_html = tmp_path / "certification_dashboard.html"

    result_template = runner.invoke(
        app,
        ["data", "certify-template", "--output-path", str(contract_path)],
    )
    assert result_template.exit_code == 0
    assert contract_path.is_file()
    assert "IINTS Diabetes CGM + Insulin Research Contract" in contract_path.read_text(encoding="utf-8")

    _write_contract(contract_path)
    pd.DataFrame({"timestamp": [1, 2], "glucose": [110.0, 120.0]}).to_csv(input_csv, index=False)

    result_certify = runner.invoke(
        app,
        ["data", "certify", str(contract_path), str(input_csv), "--output-json", str(output_json)],
    )
    assert result_certify.exit_code == 0
    payload = json.loads(output_json.read_text())
    assert payload["is_compliant"] is True
    assert "mdmp_grade" in payload
    assert payload["scan_mode"] == "full"
    assert payload["grade_definitions"]["clinical_grade"]["allowed_use"] == MDMP_GRADE_DEFINITIONS["clinical_grade"]["allowed_use"]

    result_visualizer = runner.invoke(
        app,
        ["data", "certify-visualizer", str(output_json), "--output-html", str(output_html)],
    )
    assert result_visualizer.exit_code == 0
    assert output_html.is_file()


def test_data_certify_python_helpers_wrap_backend(tmp_path) -> None:
    contract_path = tmp_path / "data_contract.yaml"
    input_csv = tmp_path / "input.csv"
    _write_contract(contract_path)

    df = pd.DataFrame({"timestamp": [1, 2], "glucose": [115.0, 118.0]})
    df.to_csv(input_csv, index=False)

    csv_result = certify_csv(contract_path, input_csv)
    df_result = certify_dataset(contract_path, df)

    assert csv_result.is_compliant is True
    assert df_result.is_compliant is True
    assert csv_result.mdmp_grade == df_result.mdmp_grade


def test_standard_diabetes_contract_accepts_iints_result_aliases(tmp_path) -> None:
    input_csv = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [110.0, 118.0, 126.0],
            "carb_intake_grams": [0.0, 15.0, 0.0],
            "delivered_insulin_units": [0.01, 0.03, 0.02],
        }
    ).to_csv(input_csv, index=False)

    report = certify_csv(standard_diabetes_contract_path(), input_csv, quick=True, quick_rows=2)

    assert report.is_compliant is True
    assert report.row_count == 2
    assert report.mdmp_grade == "clinical_grade"


def test_data_certify_quick_mode_and_unsigned_certificate(tmp_path) -> None:
    contract_path = tmp_path / "data_contract.yaml"
    input_csv = tmp_path / "input.csv"
    report_json = tmp_path / "report.json"
    cert_json = tmp_path / "certificate.json"

    _write_contract(contract_path)
    pd.DataFrame({"timestamp": range(10), "glucose": [120.0] * 10}).to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "certify",
            str(contract_path),
            str(input_csv),
            "--quick",
            "--quick-rows",
            "3",
            "--output-json",
            str(report_json),
            "--certificate-output",
            str(cert_json),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    certificate = json.loads(cert_json.read_text(encoding="utf-8"))
    assert payload["scan_mode"] == "quick"
    assert payload["quick_rows_limit"] == 3
    assert payload["row_count"] == 3
    assert certificate["signature_status"] == "unsigned_sha256_only"
    assert "signature_sha256" in certificate


def test_data_certify_can_write_signed_certificate(tmp_path) -> None:
    if importlib.util.find_spec("cryptography") is None:
        pytest.skip("cryptography is not installed")

    from mdmp_core.crypto import MDMPVerifier, generate_keypair

    contract_path = tmp_path / "data_contract.yaml"
    input_csv = tmp_path / "input.csv"
    cert_json = tmp_path / "certificate.json"
    keys = generate_keypair(output_dir=tmp_path / "keys")
    _write_contract(contract_path)
    pd.DataFrame({"timestamp": [1, 2], "glucose": [110.0, 118.0]}).to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "certify",
            str(contract_path),
            str(input_csv),
            "--certificate-output",
            str(cert_json),
            "--signing-key",
            str(keys["private_key"]),
            "--signing-key-id",
            "test_key_v1",
        ],
    )

    assert result.exit_code == 0
    certificate = json.loads(cert_json.read_text(encoding="utf-8"))
    verification = MDMPVerifier(public_key_path=keys["public_key"]).verify(certificate)
    assert certificate["signature_status"] == "ed25519_signed"
    assert verification["valid"] is True
