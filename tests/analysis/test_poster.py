from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iints.analysis.poster import generate_results_poster


def _write_run_bundle(run_dir: Path, *, offset: float = 0.0, meal_index: int = 24, override_index: int = 30) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(48):
        glucose = 118.0 + offset + (step % 12) * 2.0
        rows.append(
            {
                "time_minutes": step * 5,
                "glucose_actual_mgdl": glucose,
                "algo_recommended_insulin_units": 0.35 if step == override_index else 0.2,
                "delivered_insulin_units": 0.05 if step == override_index else 0.2,
                "carb_intake_grams": 45.0 if step == meal_index else 0.0,
                "safety_triggered": step == override_index,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_dir.name}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"results.csv": "sha256:demo"}), encoding="utf-8")


def test_generate_results_poster_writes_png_and_summary(tmp_path: Path) -> None:
    run_a = tmp_path / "normal_run"
    run_b = tmp_path / "meal_stress"
    run_c = tmp_path / "supervisor_override"
    _write_run_bundle(run_a, offset=0.0, meal_index=999, override_index=999)
    _write_run_bundle(run_b, offset=16.0, meal_index=20, override_index=999)
    _write_run_bundle(run_c, offset=8.0, meal_index=18, override_index=22)

    outputs = generate_results_poster(
        [run_a, run_b, run_c],
        labels=["Normal Run", "Meal Stress Test", "Supervisor Override"],
        output_path=tmp_path / "poster.png",
    )

    poster_path = Path(outputs["poster_png"])
    summary_path = Path(outputs["summary_json"])

    assert poster_path.is_file()
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["poster_title"] == "288 Decisions. Every Day. We Test Them All."
    assert len(summary["scenarios"]) == 3
    assert summary["scenarios"][2]["supervisor_events"] == 1
    assert summary["scenarios"][1]["meal_events"] == 1


def test_generate_results_poster_auto_discovers_latest_runs(tmp_path: Path) -> None:
    for idx in range(4):
        run_dir = tmp_path / f"run_{idx}"
        _write_run_bundle(run_dir, offset=float(idx))

    outputs = generate_results_poster(
        output_path=tmp_path / "poster_auto.png",
        results_root=tmp_path,
        auto_limit=3,
    )

    summary = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    assert len(summary["scenarios"]) == 3
