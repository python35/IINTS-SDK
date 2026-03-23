from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_cli_demo_cockpit_prints_outputs(tmp_path: Path, monkeypatch) -> None:
    outputs = {
        "html_path": str(tmp_path / "demo_cockpit.html"),
        "summary_json": str(tmp_path / "demo_cockpit.json"),
        "poster_png": str(tmp_path / "booth_demo_poster.png"),
        "demo_summary_json": str(tmp_path / "demo_summary.json"),
        "jury_talk_track": str(tmp_path / "JURY_TALK_TRACK.md"),
        "live_demo_script": str(tmp_path / "BEURS_LIVE_DEMO_SCRIPT.txt"),
        "run_commands": str(tmp_path / "run_commands.md"),
    }

    monkeypatch.setattr("iints.cli.cli.build_demo_cockpit", lambda *args, **kwargs: outputs)

    result = runner.invoke(app, ["demo-cockpit", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "IINTS Demo Cockpit" in result.stdout
    assert "Open this in your browser" in result.stdout
    assert "demo_cockpit.html" in result.stdout
