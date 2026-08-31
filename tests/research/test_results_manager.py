from __future__ import annotations

import json
import sqlite3

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.research.results_manager import index_results, summarize_results_csv


runner = CliRunner()


def _write_run(run_dir, glucose_values):
    run_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "time_minutes": [idx * 5 for idx in range(len(glucose_values))],
            "glucose_actual_mgdl": glucose_values,
            "carb_intake_grams": [0, 35, 0, 0, 0, 0][: len(glucose_values)],
            "delivered_insulin_units": [0.04, 1.2, 0.04, 0.04, 0.04, 0.04][: len(glucose_values)],
            "safety_triggered": [False, False, True, False, False, False][: len(glucose_values)],
            "fallback_triggered": [False] * len(glucose_values),
        }
    )
    df.to_csv(run_dir / "results.csv", index=False)
    (run_dir / "realism_report.json").write_text(
        json.dumps({"verdict": "likely_realistic", "realism_score": 0.98})
    )
    (run_dir / "safety_report.json").write_text(
        json.dumps({"total_interventions": 1, "bolus_interventions_count": 1})
    )
    return run_dir / "results.csv"


def test_summarize_results_csv_extracts_research_metrics(tmp_path) -> None:
    results_csv = _write_run(tmp_path / "results" / "run_a", [110, 122, 134, 145, 132, 120])

    record = summarize_results_csv(results_csv, root=tmp_path / "results")

    assert record["quality_flag"] == "ok"
    assert record["rows"] == 6
    assert record["tir_70_180_pct"] == 100.0
    assert record["meal_event_count"] == 1
    assert record["insulin_event_count"] == 6
    assert record["safety_triggered_count"] == 1
    assert record["realism_verdict"] == "likely_realistic"


def test_index_results_writes_catalogue_and_optional_raw_table(tmp_path) -> None:
    root = tmp_path / "results"
    _write_run(root / "study" / "run_a", [110, 122, 134, 145, 132, 120])
    _write_run(root / "study" / "run_b", [95, 100, 112, 126, 121, 115])
    (root / "study" / "run_a" / "report.pdf").write_bytes(b"%PDF-1.4\n")

    bundle = index_results(root, include_raw=True)

    assert bundle.run_count == 2
    assert bundle.artifact_count >= 7
    assert bundle.run_index_csv.exists()
    assert bundle.artifact_inventory_csv.exists()
    assert bundle.report_md.exists()
    assert bundle.manifest_json.exists()
    assert bundle.catalog_sqlite.exists()
    assert bundle.raw_long_csv is not None and bundle.raw_long_csv.exists()

    run_index = pd.read_csv(bundle.run_index_csv)
    assert set(run_index["quality_flag"]) == {"ok"}
    assert "max_step_delta_mgdl" in run_index.columns

    raw = pd.read_csv(bundle.raw_long_csv)
    assert set(raw["run_id"]) == {"study__run_a", "study__run_b"}
    assert len(raw) == 12

    manifest = json.loads(bundle.manifest_json.read_text())
    assert manifest["run_count"] == 2
    assert manifest["artifacts"]["raw_long_csv"] == str(bundle.raw_long_csv)
    assert manifest["artifacts"]["catalog_sqlite"] == str(bundle.catalog_sqlite)


def test_results_catalog_reuses_unchanged_run_summaries(tmp_path, monkeypatch) -> None:
    from iints.research import results_manager

    root = tmp_path / "results"
    _write_run(root / "run_a", [110, 122, 134, 145, 132, 120])

    first = index_results(root)
    assert first.runs_updated == 1
    assert first.runs_reused == 0

    def fail_if_reparsed(*_args, **_kwargs):
        raise AssertionError("unchanged results.csv should be loaded from the SQLite catalogue")

    monkeypatch.setattr(results_manager, "summarize_results_csv", fail_if_reparsed)
    second = index_results(root)

    assert second.runs_updated == 0
    assert second.runs_reused == 1
    with sqlite3.connect(second.catalog_sqlite) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_results_cli_indexes_a_results_root(tmp_path) -> None:
    root = tmp_path / "results"
    output_dir = tmp_path / "index"
    _write_run(root / "run_a", [110, 122, 134, 145, 132, 120])

    result = runner.invoke(app, ["results", "--root", str(root), "--output-dir", str(output_dir)])

    assert result.exit_code == 0
    assert "Runs indexed" in result.stdout
    assert (output_dir / "run_index.csv").exists()
    assert (output_dir / "artifact_inventory.csv").exists()
