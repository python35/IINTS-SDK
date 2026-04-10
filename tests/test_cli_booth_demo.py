from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_cli_demo_booth_prints_artifacts(tmp_path: Path, monkeypatch) -> None:
    outputs = {
        "poster_png": str(tmp_path / "booth_demo_poster.png"),
        "poster_summary_json": str(tmp_path / "booth_demo_poster.json"),
        "demo_summary_json": str(tmp_path / "demo_summary.json"),
        "jury_talk_track": str(tmp_path / "JURY_TALK_TRACK.md"),
        "live_demo_script": str(tmp_path / "BEURS_LIVE_DEMO_SCRIPT.txt"),
        "run_commands": str(tmp_path / "run_commands.md"),
        "showcase_study_summary_json": str(tmp_path / "showcase_study" / "showcase_study_summary.json"),
        "showcase_study_poster_png": str(tmp_path / "showcase_study" / "showcase_study_poster.png"),
        "showcase_research_sync_md": str(tmp_path / "showcase_study" / "SHOWCASE_RESEARCH_SYNC.md"),
        "showcase_explanation_panel_md": str(tmp_path / "showcase_study" / "SHOWCASE_EXPLANATION_PANEL.md"),
        "01_normal_run_dir": str(tmp_path / "01_normal_run"),
        "02_meal_stress_test_dir": str(tmp_path / "02_meal_stress_test"),
        "03_supervisor_override_dir": str(tmp_path / "03_supervisor_override"),
    }

    monkeypatch.setattr("iints.cli.cli.build_booth_demo", lambda *args, **kwargs: outputs)

    result = runner.invoke(app, ["demo-booth", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "IINTS Booth Demo" in result.stdout
    assert "JURY_TALK_TRACK.md" in result.stdout
    assert "BEURS_LIVE_DEMO_SCRIPT.txt" in result.stdout
    assert "SHOWCASE_RESEARCH_SYNC" in result.stdout
    assert "SHOWCASE_EXPLANATION_PANEL" in result.stdout
