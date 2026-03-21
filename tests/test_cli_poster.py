from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def _write_run_bundle(run_dir: Path, *, glucose_base: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(36):
        rows.append(
            {
                "time_minutes": step * 5,
                "glucose_actual_mgdl": glucose_base + step,
                "algo_recommended_insulin_units": 0.3 if step == 10 else 0.15,
                "delivered_insulin_units": 0.1 if step == 10 else 0.15,
                "carb_intake_grams": 30.0 if step == 8 else 0.0,
                "safety_triggered": step == 10,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_dir.name}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"results.csv": "sha256:demo"}), encoding="utf-8")


def test_cli_poster_generates_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_run_bundle(run_a, glucose_base=110.0)
    _write_run_bundle(run_b, glucose_base=130.0)
    output_path = tmp_path / "poster.png"

    result = runner.invoke(
        app,
        [
            "poster",
            "--run-dir",
            str(run_a),
            "--run-dir",
            str(run_b),
            "--label",
            "Normal Run",
            "--label",
            "Stress Test",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.is_file()
    assert output_path.with_suffix(".json").is_file()
    assert "Poster PNG" in result.stdout
