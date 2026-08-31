from __future__ import annotations

from pathlib import Path
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.cgmacros_downloader import (
    download_or_generate_cgmacros,
    fetch_and_import_cgmacros_pipeline,
)

runner = CliRunner()


def test_download_or_generate_cgmacros(tmp_path: Path):
    dest = tmp_path / "cgmacros_raw"
    res_path = download_or_generate_cgmacros(dest, participant_count=5, force_download=False)

    assert res_path.is_dir()
    assert (dest / "bio.csv").is_file()
    assert (dest / "CGMacros-01.csv").is_file()
    assert (dest / "CGMacros-05.csv").is_file()


def test_fetch_and_import_cgmacros_pipeline(tmp_path: Path):
    pytest.importorskip("pyarrow", reason="cgmacros_timeseries.parquet output requires pyarrow")
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"
    result = fetch_and_import_cgmacros_pipeline(raw_dir=raw_dir, processed_dir=proc_dir, participant_count=5)

    assert result.subject_count == 5
    assert result.meal_count > 0
    assert result.dexcom_measurements > 0
    assert result.libre_measurements > 0
    assert (proc_dir / "cgmacros_manifest.json").is_file()


def test_cli_download_cgmacros_command(tmp_path: Path):
    pytest.importorskip("pyarrow", reason="cgmacros_timeseries.parquet output requires pyarrow")
    out_dir = tmp_path / "cli_cgmacros"
    res = runner.invoke(app, ["data", "download-cgmacros", "--output-dir", str(out_dir), "--participants", "3"])
    assert res.exit_code == 0
    assert "CGMacros Dataset Acquired and Standardized Successfully" in res.output
    assert (out_dir / "standardized" / "cgmacros_manifest.json").is_file()
