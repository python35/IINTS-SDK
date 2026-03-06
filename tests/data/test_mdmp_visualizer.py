from __future__ import annotations

import json

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.mdmp_visualizer import build_mdmp_dashboard_html


runner = CliRunner()


def _sample_report() -> dict:
    return {
        "is_compliant": True,
        "compliance_score": 100.0,
        "mdmp_grade": "clinical_grade",
        "mdmp_protocol_version": "1.0-draft",
        "certified_for_medical_research": True,
        "contract_fingerprint_sha256": "a" * 64,
        "dataset_fingerprint_sha256": "b" * 64,
        "row_count": 42,
        "output_columns": ["timestamp", "glucose"],
        "checks": [
            {"name": "schema_columns", "passed": True, "detail": "ok", "failed_rows": 0},
            {"name": "value_ranges", "passed": True, "detail": "ok", "failed_rows": 0},
        ],
    }


def test_build_mdmp_dashboard_html_contains_embedded_report() -> None:
    html = build_mdmp_dashboard_html(_sample_report(), title="MDMP Test")
    assert "<!doctype html>" in html.lower()
    assert "MDMP Test" in html
    assert "clinical_grade" in html
    assert "schema_columns" in html


def test_cli_mdmp_visualizer_writes_html(tmp_path) -> None:
    report_path = tmp_path / "contract_data_report.json"
    out_html = tmp_path / "mdmp_dashboard.html"
    report_path.write_text(json.dumps(_sample_report()))

    result = runner.invoke(
        app,
        [
            "data",
            "mdmp-visualizer",
            str(report_path),
            "--output-html",
            str(out_html),
            "--title",
            "MDMP Cert",
        ],
    )

    assert result.exit_code == 0
    assert out_html.is_file()
    content = out_html.read_text()
    assert "MDMP Cert" in content
    assert "clinical_grade" in content
