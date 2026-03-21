from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_cli_carelink_workbench_reports_outputs(tmp_path, monkeypatch) -> None:
    sample = tmp_path / "carelink.csv"
    sample.write_text("demo", encoding="utf-8")
    output_dir = tmp_path / "workbench"

    fake_outputs = {
        "scenario": str(output_dir / "scenario.json"),
        "dashboard_png": str(output_dir / "carelink_dashboard.png"),
        "poster_png": str(output_dir / "carelink_poster.png"),
        "dashboard_html": str(output_dir / "carelink_dashboard.html"),
        "summary": str(output_dir / "carelink_summary.json"),
        "metrics": str(output_dir / "carelink_metrics.json"),
        "timeline": str(output_dir / "carelink_timeline.csv"),
        "report_payload": str(output_dir / "ai" / "report_payload.json"),
    }
    monkeypatch.setattr("iints.cli.cli.build_carelink_workbench", lambda *args, **kwargs: fake_outputs)

    result = runner.invoke(
        app,
        [
            "carelink-workbench",
            "--input-csv",
            str(sample),
            "--output-dir",
            str(output_dir),
            "--no-create-dev-mdmp-cert",
        ],
    )

    assert result.exit_code == 0
    assert "Personal CareLink workspace is ready" in result.stdout
    assert "carelink_dashboard.png" in result.stdout
    assert "carelink_poster.png" in result.stdout
    assert "iints run --algo" in result.stdout
