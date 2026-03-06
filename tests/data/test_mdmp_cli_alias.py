from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from iints.cli.cli import app


runner = CliRunner()


def test_mdmp_template_and_validate_alias(tmp_path) -> None:
    contract_path = tmp_path / "mdmp_contract.yaml"
    input_csv = tmp_path / "input.csv"
    out_json = tmp_path / "mdmp_report.json"

    result_template = runner.invoke(
        app,
        [
            "mdmp",
            "template",
            "--output-path",
            str(contract_path),
        ],
    )
    assert result_template.exit_code == 0
    assert contract_path.is_file()

    pd.DataFrame({"timestamp": [1, 2], "glucose": [110.0, 120.0]}).to_csv(input_csv, index=False)
    result_validate = runner.invoke(
        app,
        [
            "mdmp",
            "validate",
            str(contract_path),
            str(input_csv),
            "--output-json",
            str(out_json),
        ],
    )
    assert result_validate.exit_code == 0
    payload = json.loads(out_json.read_text())
    assert payload["is_compliant"] is True
    assert "mdmp_grade" in payload


def test_mdmp_visualizer_alias(tmp_path) -> None:
    report_path = tmp_path / "mdmp_report.json"
    out_html = tmp_path / "mdmp_dashboard.html"
    report_path.write_text(
        json.dumps(
            {
                "is_compliant": True,
                "compliance_score": 100.0,
                "mdmp_grade": "clinical_grade",
                "mdmp_protocol_version": "1.0-draft",
                "certified_for_medical_research": True,
                "contract_fingerprint_sha256": "a" * 64,
                "dataset_fingerprint_sha256": "b" * 64,
                "row_count": 2,
                "checks": [{"name": "schema_columns", "passed": True, "detail": "ok", "failed_rows": 0}],
                "output_columns": ["timestamp", "glucose"],
            }
        )
    )

    result = runner.invoke(
        app,
        [
            "mdmp",
            "visualizer",
            str(report_path),
            "--output-html",
            str(out_html),
        ],
    )
    assert result.exit_code == 0
    assert out_html.is_file()
