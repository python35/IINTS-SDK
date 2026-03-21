from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from iints.analysis.booth_demo import build_booth_demo


def test_build_booth_demo_writes_bundle_files(tmp_path: Path, monkeypatch) -> None:
    def _fake_run_full(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "time_minutes": 0,
                    "glucose_actual_mgdl": 120.0,
                    "algo_recommended_insulin_units": 0.2,
                    "delivered_insulin_units": 0.2,
                    "carb_intake_grams": 0.0,
                    "safety_triggered": False,
                },
                {
                    "time_minutes": 5,
                    "glucose_actual_mgdl": 95.0,
                    "algo_recommended_insulin_units": 0.4,
                    "delivered_insulin_units": 0.0,
                    "carb_intake_grams": 40.0,
                    "safety_triggered": output_dir.name == "03_supervisor_override",
                },
            ]
        ).to_csv(output_dir / "results.csv", index=False)
        (output_dir / "run_metadata.json").write_text(json.dumps({"run_id": output_dir.name}), encoding="utf-8")
        (output_dir / "run_manifest.json").write_text(json.dumps({"results": "sha256:demo"}), encoding="utf-8")
        (output_dir / "clinical_report.pdf").write_text("pdf", encoding="utf-8")
        return {
            "results_csv": str(output_dir / "results.csv"),
            "report_pdf": str(output_dir / "clinical_report.pdf"),
            "run_manifest_path": str(output_dir / "run_manifest.json"),
        }

    def _fake_poster(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_text("png", encoding="utf-8")
        summary_path = Path(kwargs["summary_output_path"])
        summary_path.write_text(json.dumps({"scenarios": 3}), encoding="utf-8")
        return {"poster_png": str(output_path), "summary_json": str(summary_path)}

    def _fake_prepare(run_dir, **kwargs):
        run_dir_path = Path(run_dir)
        ai_dir = run_dir_path / "ai"
        ai_dir.mkdir(parents=True, exist_ok=True)
        cert = ai_dir / "report.signed.mdmp"
        cert.write_text("signed", encoding="utf-8")
        return {"mdmp_cert": str(cert)}

    monkeypatch.setattr("iints.analysis.booth_demo.run_full", _fake_run_full)
    monkeypatch.setattr("iints.analysis.booth_demo.generate_results_poster", _fake_poster)
    monkeypatch.setattr("iints.analysis.booth_demo.prepare_ai_ready_artifacts", _fake_prepare)

    outputs = build_booth_demo(tmp_path / "booth_demo")

    assert Path(outputs["poster_png"]).is_file()
    assert Path(outputs["poster_summary_json"]).is_file()
    assert Path(outputs["jury_talk_track"]).is_file()
    assert Path(outputs["live_demo_script"]).is_file()
    assert Path(outputs["run_commands"]).is_file()
    assert Path(outputs["demo_summary_json"]).is_file()
    assert Path(outputs["mdmp_cert"]).is_file()

    talk_track = Path(outputs["jury_talk_track"]).read_text(encoding="utf-8")
    assert "Supervisor Override" in talk_track
    assert "iints ai report" in talk_track
    live_script = Path(outputs["live_demo_script"]).read_text(encoding="utf-8")
    assert "WHAT CODE TO SHOW FIRST" in live_script
    assert "./scripts/run_booth_demo.sh" in live_script
