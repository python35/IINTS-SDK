from __future__ import annotations

import pandas as pd

from iints.tools.ai_realism_auditor import AIRealismAuditor


def test_auditor_detects_corrupt_advanced_model_row(tmp_path):
    csv_path = tmp_path / "advanced.csv"
    pd.DataFrame(
        [
            {"time_min": 0, "glucose": 110, "insulin_delivered": 0.8, "ffa": 0.4, "ketones": 0.1, "event": ""},
            {"time_min": 5, "glucose": -50, "insulin_delivered": 0.8, "ffa": 0.4, "ketones": 20.0, "event": "injected_corrupt_row"},
        ]
    ).to_csv(csv_path, index=False)

    auditor = AIRealismAuditor(csv_path, enable_ai=False)
    anomalies = auditor.find_anomalies()

    assert len(anomalies) == 1
    assert anomalies[0].kind == "Negative glucose"


def test_auditor_accepts_official_jetson_endurance_columns(tmp_path):
    csv_path = tmp_path / "steps.csv"
    pd.DataFrame(
        [
            {"time_minutes": 0, "glucose_actual_mgdl": 120, "delivered_insulin_units": 0.0},
            {"time_minutes": 5, "glucose_actual_mgdl": 118, "delivered_insulin_units": 0.0},
            {"time_minutes": 10, "glucose_actual_mgdl": 116, "delivered_insulin_units": 0.0},
        ]
    ).to_csv(csv_path, index=False)

    auditor = AIRealismAuditor(csv_path, enable_ai=False)

    assert auditor.df["time_min"].tolist() == [0, 5, 10]
    assert auditor.df["glucose"].tolist() == [120, 118, 116]
    assert auditor.find_anomalies() == []


def test_auditor_writes_report_without_ollama(tmp_path):
    csv_path = tmp_path / "advanced.csv"
    report_path = tmp_path / "report.md"
    pd.DataFrame(
        [{"time_min": 0, "glucose": 900, "insulin_delivered": 0.0, "ketones": 0.2}]
    ).to_csv(csv_path, index=False)

    result = AIRealismAuditor(csv_path, enable_ai=False).run_audit(report_path)

    assert result["anomalies"] == 1
    text = report_path.read_text(encoding="utf-8")
    assert "Extreme hyperglycemia" in text
    assert "**Local AI Verdicts:** **offline / disabled**" in text
