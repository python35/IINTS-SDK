from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_cli_demo_export_writes_script_and_notes(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo-export", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "IINTS Demo Export" in result.stdout
    assert "07_live_stage_demo.py" in result.stdout

    script_path = tmp_path / "07_live_stage_demo.py"
    notes_path = tmp_path / "RUN_ME_FIRST.txt"

    assert script_path.is_file()
    assert notes_path.is_file()

    script_text = script_path.read_text(encoding="utf-8")
    assert "run_full(...)" in script_text
    assert "generate_results_poster(...)" in script_text
    assert "prepare_ai_ready_artifacts(...)" in script_text
