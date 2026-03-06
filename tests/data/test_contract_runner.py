from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.contracts import parse_contract
from iints.data.runner import ContractRunner


runner = CliRunner()


def _base_contract() -> dict:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "metadata": {
                    "required_columns": ["timestamp", "glucose"],
                    "column_types": {"glucose": "float"},
                    "ranges": {"glucose": {"min": 70, "max": 250}},
                },
            }
        ],
        "processes": [
            {
                "name": "GlucoseData",
                "input_stream": "PatientHealth.glucose",
                "validations": [
                    {"expression": "glucose is not null and glucose > 20"},
                ],
            }
        ],
    }


def test_contract_runner_passes_for_valid_dataframe() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "glucose": [110.0, 125.0, 140.0],
        }
    )
    runner_impl = ContractRunner(parse_contract(_base_contract()))
    result = runner_impl.run(df)

    assert result.is_compliant is True
    assert result.compliance_score == 100.0


def test_contract_runner_fails_on_missing_column() -> None:
    df = pd.DataFrame({"timestamp": [1, 2, 3]})
    runner_impl = ContractRunner(parse_contract(_base_contract()))
    result = runner_impl.run(df)

    assert result.is_compliant is False
    schema_check = next(check for check in result.checks if check.name == "schema_columns")
    assert schema_check.passed is False


def test_contract_runner_applies_builtin_unit_conversion() -> None:
    payload = _base_contract()
    payload["streams"][0]["metadata"]["unit_conversions"] = {  # type: ignore[index]
        "glucose": {"from": "mmol/L", "to": "mg/dL"}
    }
    df = pd.DataFrame(
        {
            "timestamp": [1, 2],
            "glucose": [6.0, 7.0],  # mmol/L -> ~108 and ~126 mg/dL
        }
    )
    runner_impl = ContractRunner(parse_contract(payload))
    result = runner_impl.run(df, apply_builtin_transforms=True)

    assert result.is_compliant is True


def test_cli_data_contract_run_writes_report(tmp_path) -> None:
    contract_path = tmp_path / "contract.yaml"
    input_csv = tmp_path / "input.csv"
    out_json = tmp_path / "contract_report.json"

    contract_path.write_text(
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
          min: 70
          max: 250
processes:
  - name: GlucoseData
    input_stream: PatientHealth.glucose
    validations:
      - expression: glucose is not null and glucose > 20
""".strip()
    )
    pd.DataFrame({"timestamp": [1, 2], "glucose": [105.0, 120.0]}).to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "contract-run",
            str(contract_path),
            str(input_csv),
            "--output-json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out_json.read_text())
    assert payload["is_compliant"] is True
    assert payload["compliance_score"] == 100.0
