from __future__ import annotations

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_quickstart_scaffolds_certification_assets(tmp_path) -> None:
    project_dir = tmp_path / "quickstart_demo"

    result = runner.invoke(
        app,
        ["quickstart", "--project-name", str(project_dir)],
    )

    assert result.exit_code == 0
    assert (project_dir / "contracts" / "clinical_mdmp_contract.yaml").is_file()
    assert (project_dir / "data" / "demo" / "diabetes_cgm.csv").is_file()
    assert (project_dir / "audit").is_dir()
    assert (project_dir / "patients" / "stable_patient.yaml").is_file()
    assert "--patient-config-path patients/stable_patient.yaml" in (
        project_dir / "README.md"
    ).read_text(encoding="utf-8")
