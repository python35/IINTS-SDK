from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.synthetic_mirror import generate_synthetic_mirror


runner = CliRunner()


def _contract_payload() -> dict:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "metadata": {
                    "required_columns": ["timestamp", "glucose", "carbs"],
                    "column_types": {
                        "glucose": "float",
                        "carbs": "float",
                    },
                    "ranges": {
                        "glucose": {"min": 70, "max": 250},
                        "carbs": {"min": 0, "max": 120},
                    },
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


def test_generate_synthetic_mirror_is_deterministic_with_seed() -> None:
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:15:00Z",
            ],
            "glucose": [110.0, 118.0, 125.0, 119.0],
            "carbs": [0.0, 5.0, 0.0, 20.0],
        }
    )
    synth_a, artifact_a = generate_synthetic_mirror(source, _contract_payload(), rows=12, seed=7)
    synth_b, artifact_b = generate_synthetic_mirror(source, _contract_payload(), rows=12, seed=7)

    pd.testing.assert_frame_equal(synth_a, synth_b)
    assert artifact_a.validation.dataset_fingerprint_sha256 == artifact_b.validation.dataset_fingerprint_sha256


def test_generate_synthetic_mirror_respects_contract_ranges() -> None:
    source = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
            ],
            "glucose": [95.0, 180.0, 130.0],
            "carbs": [5.0, 15.0, 10.0],
        }
    )
    synth, artifact = generate_synthetic_mirror(source, _contract_payload(), rows=30, seed=42, noise_scale=0.15)

    assert len(synth) == 30
    assert synth["glucose"].min() >= 70.0
    assert synth["glucose"].max() <= 250.0
    assert artifact.validation.is_compliant is True


def test_cli_synthetic_mirror_writes_artifacts(tmp_path) -> None:
    input_csv = tmp_path / "input.csv"
    contract_yaml = tmp_path / "contract.yaml"
    output_csv = tmp_path / "synthetic.csv"
    output_json = tmp_path / "synthetic_report.json"

    pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:05:00Z",
                "2026-01-01T00:10:00Z",
                "2026-01-01T00:15:00Z",
            ],
            "glucose": [110.0, 120.0, 130.0, 140.0],
            "carbs": [0.0, 10.0, 0.0, 12.0],
        }
    ).to_csv(input_csv, index=False)

    contract_yaml.write_text(
        """
version: 1
streams:
  - name: PatientHealth
    source: sdk.iints_af.v1
    metadata:
      required_columns: [timestamp, glucose, carbs]
      column_types:
        glucose: float
        carbs: float
      ranges:
        glucose:
          min: 70
          max: 250
        carbs:
          min: 0
          max: 120
processes:
  - name: GlucoseData
    input_stream: PatientHealth.glucose
    validations:
      - expression: glucose is not null and glucose > 20
""".strip()
    )

    result = runner.invoke(
        app,
        [
            "data",
            "synthetic-mirror",
            str(input_csv),
            str(contract_yaml),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
            "--rows",
            "24",
            "--seed",
            "11",
        ],
    )

    assert result.exit_code == 0
    assert output_csv.is_file()
    assert output_json.is_file()

    payload = json.loads(output_json.read_text())
    assert payload["summary"]["synthetic_rows"] == 24
    assert payload["validation"]["is_compliant"] is True
    assert "mdmp_grade" in payload["validation"]
