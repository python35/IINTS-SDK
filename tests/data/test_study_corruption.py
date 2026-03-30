from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.study_corruption import apply_study_corruptions, write_corrupted_study_csv


runner = CliRunner()


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01 00:00:00",
                "2026-01-01 00:05:00",
                "2026-01-01 00:10:00",
                "2026-01-01 00:15:00",
            ],
            "glucose": [110.0, 115.0, 120.0, 125.0],
            "carbs": [0.0, 15.0, 0.0, 10.0],
        }
    )


def test_apply_study_corruptions_applies_multiple_modes() -> None:
    corrupted, manifest = apply_study_corruptions(
        _sample_dataframe(),
        modes=["timestamp_shift", "drop_meal_annotations", "glucose_spikes"],
        seed=7,
    )
    assert len(manifest["operations"]) == 3
    assert "timestamp" in corrupted.columns
    assert float(corrupted["carbs"].sum()) == 0.0


def test_write_corrupted_study_csv_writes_csv_and_manifest(tmp_path) -> None:
    input_csv = tmp_path / "source.csv"
    output_csv = tmp_path / "corrupted.csv"
    _sample_dataframe().to_csv(input_csv, index=False)

    outputs = write_corrupted_study_csv(
        input_csv,
        output_csv=output_csv,
        modes=["missing_block", "duplicate_rows"],
        seed=7,
    )

    assert output_csv.is_file()
    manifest = json.loads((tmp_path / "corrupted.manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_csv"].endswith("source.csv")
    assert outputs["corrupted_csv"].endswith("corrupted.csv")


def test_cli_data_corrupt_for_study(tmp_path) -> None:
    input_csv = tmp_path / "source.csv"
    output_csv = tmp_path / "corrupted.csv"
    manifest_json = tmp_path / "corruption_manifest.json"
    _sample_dataframe().to_csv(input_csv, index=False)

    result = runner.invoke(
        app,
        [
            "data",
            "corrupt-for-study",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--manifest-output",
            str(manifest_json),
            "--mode",
            "timestamp_shift",
            "--mode",
            "missing_block",
        ],
    )

    assert result.exit_code == 0
    assert output_csv.is_file()
    assert manifest_json.is_file()
