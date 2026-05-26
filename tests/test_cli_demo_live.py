from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_cli_demo_live_can_prepare_code_without_running(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo-live", "--output-dir", str(tmp_path), "--no-run"])

    assert result.exit_code == 0
    assert "IINTS Live Demo" in result.stdout
    assert "What To Say First" in result.stdout
    assert "Code to explain on the call" in result.stdout
    assert "run_full(" in result.stdout
    assert "Prepared only" in result.stdout
    assert (tmp_path / "showable_code" / "07_live_stage_demo.py").is_file()
    assert (tmp_path / "PRESENTER_GUIDE.md").is_file()
    assert (tmp_path / "DEMO_CUE_CARD.md").is_file()
    assert (tmp_path / "DEMO_ARTIFACTS.md").is_file()
    assert (tmp_path / "RUN_LIVE_DEMO.sh").is_file()
    assert "Cue card" in result.stdout


def test_cli_demo_live_supports_clinical_presenter_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["demo-live", "--output-dir", str(tmp_path), "--no-run", "--audience", "clinical"],
    )

    assert result.exit_code == 0
    assert "clinical audience" in result.stdout
    guide = (tmp_path / "PRESENTER_GUIDE.md").read_text(encoding="utf-8")
    assert "pre-clinical research tool" in guide
    assert "not treatment advice" in guide
    assert "EUCYS_05_PHYSIOLOGY_REFERENCE_BROCHURE.pdf" in guide
    assert "EUCYS_06_JURY_PHYSIOLOGY_BRIEF.pdf" in guide


def test_cli_demo_live_runs_exported_script_and_summarizes_outputs(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _fake_run(command: list[str], check: bool) -> SimpleNamespace:
        calls.append(command)
        results_dir = Path(command[command.index("--output-dir") + 1])
        results_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "poster_png": str(results_dir / "booth_demo_poster.png"),
            "scenarios": [
                {"label": "Normal Run", "output_dir": str(results_dir / "01_normal_run")},
                {"label": "Meal Stress Test", "output_dir": str(results_dir / "02_meal_stress_test")},
            ],
        }
        (results_dir / "demo_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("iints.cli.cli.subprocess.run", _fake_run)

    result = runner.invoke(app, ["demo-live", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert calls
    assert "--skip-ai" in calls[0]
    assert "IINTS Live Demo Results" in result.stdout
    assert "booth_demo_poster.png" in result.stdout
    assert "Normal Run" in result.stdout
    assert "Presenter guide updated" in result.stdout
    assert "Cue card updated" in result.stdout
    assert "Suggested call flow" in result.stdout
    assert (tmp_path / "DEMO_CUE_CARD.md").is_file()
    assert (tmp_path / "DEMO_ARTIFACTS.md").is_file()
    assert (tmp_path / "RUN_LIVE_DEMO.sh").is_file()
