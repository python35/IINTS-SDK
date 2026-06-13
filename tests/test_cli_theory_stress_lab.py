from __future__ import annotations

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_jetson_theory_stress_cli_writes_report(tmp_path):
    output_dir = tmp_path / "theory_cli"

    result = runner.invoke(
        app,
        [
            "jetson",
            "theory-stress",
            "run",
            "--output-dir",
            str(output_dir),
            "--profile",
            "ci",
            "--seed",
            "9",
        ],
    )

    assert result.exit_code == 0
    assert "IINTS-AF Theory Stress Lab" in result.stdout
    assert (output_dir / "summary.md").is_file()
    assert (output_dir / "checks.json").is_file()
