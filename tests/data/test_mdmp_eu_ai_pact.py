from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.mdmp.eu_ai_pact import (
    CORE_AI_PACT_CONTROLS,
    HIGH_RISK_READINESS_CONTROLS,
    review_eu_ai_pact_readiness,
)


runner = CliRunner()


def test_eu_ai_pact_review_blocks_missing_governance_controls() -> None:
    result = review_eu_ai_pact_readiness(
        {
            "mdmp_grade": "draft",
            "compliance_score": 0.6,
            "row_count": 0,
        }
    )

    assert result.status == "blocked"
    assert result.passed is False
    assert any("Missing governance control" in item for item in result.critical_failures)
    assert any("dataset fingerprint" in item.lower() for item in result.critical_failures)


def test_eu_ai_pact_review_passes_complete_research_payload() -> None:
    controls = {control: True for control in (*CORE_AI_PACT_CONTROLS, *HIGH_RISK_READINESS_CONTROLS)}
    result = review_eu_ai_pact_readiness(
        {
            "mdmp_grade": "research_grade",
            "compliance_score": 0.99,
            "dataset_fingerprint_sha256": "abc",
            "contract_fingerprint_sha256": "def",
            "row_count": 288,
            "governance": controls,
        }
    )

    assert result.status == "research_ready"
    assert result.passed is True
    assert result.critical_failures == []


def test_eu_ai_pact_cli_writes_review_json(tmp_path: Path) -> None:
    controls = {control: True for control in (*CORE_AI_PACT_CONTROLS, *HIGH_RISK_READINESS_CONTROLS)}
    report_path = tmp_path / "mdmp_report.json"
    output_path = tmp_path / "eu_ai_pact.json"
    report_path.write_text(
        json.dumps(
            {
                "mdmp_grade": "research_grade",
                "compliance_score": 0.99,
                "dataset_fingerprint_sha256": "abc",
                "contract_fingerprint_sha256": "def",
                "row_count": 288,
                "governance": controls,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "data",
            "eu-ai-pact-review",
            str(report_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "research_ready"
