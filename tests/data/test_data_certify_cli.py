from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data import certify_csv, certify_dataset


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


def test_main_help_hides_legacy_mdmp_namespace() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "\n  mdmp " not in result.stdout.lower()
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
