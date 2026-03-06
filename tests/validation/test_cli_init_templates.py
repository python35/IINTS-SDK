from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_init_clinical_trial_template_scaffolds_mdmp_assets() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "init",
                "--project-name",
                "trial_project",
                "--template",
                "clinical-trial",
            ],
        )
        assert result.exit_code == 0

        root = Path("trial_project")
        assert (root / "contracts" / "clinical_mdmp_contract.yaml").is_file()
        assert (root / "data" / "demo" / "diabetes_cgm.csv").is_file()
        assert (root / "audit").is_dir()
        assert (root / "reports").is_dir()


def test_init_rejects_unknown_template() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "init",
                "--project-name",
                "bad_project",
                "--template",
                "unknown-template",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid --template" in result.stdout
