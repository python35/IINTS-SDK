from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from iints.analysis.demo_cockpit import build_demo_cockpit


MINI_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+y2ioAAAAASUVORK5CYII="
)


def _write_run_bundle(run_dir: Path, *, glucose_base: float, overrides: int, meals: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(24):
        rows.append(
            {
                "time_minutes": step * 5,
                "glucose_actual_mgdl": glucose_base + step,
                "algo_recommended_insulin_units": 0.35 if step < overrides else 0.15,
                "delivered_insulin_units": 0.15,
                "carb_intake_grams": 25.0 if step < meals else 0.0,
                "safety_triggered": step < overrides,
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "clinical_report.pdf").write_text("pdf", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"results_csv": "sha256:demo"}), encoding="utf-8")


def test_build_demo_cockpit_generates_html_bundle(tmp_path: Path, monkeypatch) -> None:
    def _fake_build_booth_demo(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        _write_run_bundle(output_dir / "01_normal_run", glucose_base=110.0, overrides=1, meals=1)
        _write_run_bundle(output_dir / "02_meal_stress_test", glucose_base=125.0, overrides=3, meals=2)
        _write_run_bundle(output_dir / "03_supervisor_override", glucose_base=95.0, overrides=8, meals=1)

        poster_png = output_dir / "booth_demo_poster.png"
        poster_png.write_bytes(MINI_PNG)
        (output_dir / "booth_demo_poster.json").write_text(
            json.dumps(
                {
                    "poster_title": "Demo Poster",
                    "scenarios": [
                        {
                            "label": "Normal Run",
                            "run_dir": str(output_dir / "01_normal_run"),
                            "results_csv": str(output_dir / "01_normal_run" / "results.csv"),
                            "duration_hours": 2.0,
                            "total_steps": 24,
                            "tir_70_180": 100.0,
                            "tir_below_70": 0.0,
                            "tir_above_180": 0.0,
                            "mean_glucose": 120.0,
                            "max_glucose": 135.0,
                            "min_glucose": 110.0,
                            "supervisor_events": 1,
                            "meal_events": 1,
                        },
                        {
                            "label": "Meal Stress Test",
                            "run_dir": str(output_dir / "02_meal_stress_test"),
                            "results_csv": str(output_dir / "02_meal_stress_test" / "results.csv"),
                            "duration_hours": 2.0,
                            "total_steps": 24,
                            "tir_70_180": 96.0,
                            "tir_below_70": 0.0,
                            "tir_above_180": 4.0,
                            "mean_glucose": 132.0,
                            "max_glucose": 165.0,
                            "min_glucose": 120.0,
                            "supervisor_events": 3,
                            "meal_events": 2,
                        },
                        {
                            "label": "Supervisor Override",
                            "run_dir": str(output_dir / "03_supervisor_override"),
                            "results_csv": str(output_dir / "03_supervisor_override" / "results.csv"),
                            "duration_hours": 2.0,
                            "total_steps": 24,
                            "tir_70_180": 88.0,
                            "tir_below_70": 6.0,
                            "tir_above_180": 6.0,
                            "mean_glucose": 118.0,
                            "max_glucose": 185.0,
                            "min_glucose": 68.0,
                            "supervisor_events": 8,
                            "meal_events": 1,
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "demo_summary.json").write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "slug": "01_normal_run",
                            "headline": "Calm day control.",
                            "jury_takeaway": "A stable baseline case.",
                            "report_pdf": str(output_dir / "01_normal_run" / "clinical_report.pdf"),
                            "run_manifest_path": str(output_dir / "01_normal_run" / "run_manifest.json"),
                        },
                        {
                            "slug": "02_meal_stress_test",
                            "headline": "Meals and exercise.",
                            "jury_takeaway": "Stress without chaos.",
                            "report_pdf": str(output_dir / "02_meal_stress_test" / "clinical_report.pdf"),
                            "run_manifest_path": str(output_dir / "02_meal_stress_test" / "run_manifest.json"),
                        },
                        {
                            "slug": "03_supervisor_override",
                            "headline": "Unsafe AI gets blocked.",
                            "jury_takeaway": "Safety supervisor prevents dangerous dosing.",
                            "report_pdf": str(output_dir / "03_supervisor_override" / "clinical_report.pdf"),
                            "run_manifest_path": str(output_dir / "03_supervisor_override" / "run_manifest.json"),
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "run_commands.md").write_text("./scripts/run_live_stage_demo.sh", encoding="utf-8")
        (output_dir / "BEURS_LIVE_DEMO_SCRIPT.txt").write_text("WHAT CODE TO SHOW FIRST", encoding="utf-8")
        (output_dir / "JURY_TALK_TRACK.md").write_text("Supervisor Override", encoding="utf-8")
        return {
            "output_dir": str(output_dir),
            "poster_png": str(poster_png),
            "poster_summary_json": str(output_dir / "booth_demo_poster.json"),
            "demo_summary_json": str(output_dir / "demo_summary.json"),
            "jury_talk_track": str(output_dir / "JURY_TALK_TRACK.md"),
            "live_demo_script": str(output_dir / "BEURS_LIVE_DEMO_SCRIPT.txt"),
            "run_commands": str(output_dir / "run_commands.md"),
        }

    monkeypatch.setattr("iints.analysis.demo_cockpit.build_booth_demo", _fake_build_booth_demo)

    outputs = build_demo_cockpit(tmp_path / "demo_cockpit", patient_config="patient_559_config", prepare_ai=False)

    html_path = Path(outputs["html_path"])
    summary_path = Path(outputs["summary_json"])
    assert html_path.is_file()
    assert summary_path.is_file()

    html_text = html_path.read_text(encoding="utf-8")
    assert "Show Code. Run It. Make The Safety Story Visible." in html_text
    assert "patient_559_config" in html_text
    assert "Normal Run" in html_text
    assert "Unsafe AI gets blocked." in html_text
    assert "./scripts/run_live_stage_demo.sh" in html_text

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["patient_config"] == "patient_559_config"
    assert len(payload["scenarios"]) == 3
