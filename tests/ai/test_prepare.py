from __future__ import annotations

import json

import pandas as pd

from iints.ai.prepare import prepare_ai_ready_artifacts


def _write_run_bundle(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "time_minutes": 0,
                "glucose_actual_mgdl": 118.0,
                "glucose_to_algo_mgdl": 120.0,
                "glucose_trend_mgdl_min": 0.0,
                "predicted_glucose_30min": 128.0,
                "algo_recommended_insulin_units": 0.2,
                "delivered_insulin_units": 0.2,
                "safety_triggered": False,
                "safety_reason": "",
            },
            {
                "time_minutes": 5,
                "glucose_actual_mgdl": 62.0,
                "glucose_to_algo_mgdl": 64.0,
                "glucose_trend_mgdl_min": -1.4,
                "predicted_glucose_30min": 58.0,
                "algo_recommended_insulin_units": 0.0,
                "delivered_insulin_units": 0.0,
                "safety_triggered": True,
                "safety_reason": "hypo_guard",
            },
        ]
    )
    results.to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "audit").mkdir()
    (tmp_path / "baseline").mkdir()
    (tmp_path / "audit" / "audit_summary.json").write_text(
        json.dumps({"total_overrides": 1, "top_reasons": {"hypo_guard": 1}}),
        encoding="utf-8",
    )
    (tmp_path / "baseline" / "baseline_comparison.json").write_text(
        json.dumps({"primary_label": "QuickstartAlgorithm"}),
        encoding="utf-8",
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "demo-run",
                "sdk_version": "1.1.1",
                "output_dir": str(tmp_path),
                "config": {
                    "duration_minutes": 10,
                    "algorithm": {"metadata": {"name": "QuickstartAlgorithm"}},
                    "scenario": {"scenario_name": "demo"},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"files": {"results_csv": {"path": str(tmp_path / "results.csv")}}}),
        encoding="utf-8",
    )


def test_prepare_ai_ready_artifacts_creates_payloads_and_local_cert(tmp_path, monkeypatch) -> None:
    _write_run_bundle(tmp_path)

    class _Signer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def sign_card(self, payload, *, expires_days):
            assert payload["grade"] == "research_grade"
            assert expires_days == 30
            return {**payload, "signature": "demo", "signed_by": "IINTS-Local-AI", "key_id": "iints_local_ai_v1"}

    def _fake_keygen(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mdmp_private_v1.pem").write_text("private", encoding="utf-8")
        (output_dir / "mdmp_pub_v1.pem").write_text("public", encoding="utf-8")
        return {}

    monkeypatch.setattr("iints.ai.prepare._load_mdmp_signer_tools", lambda: (_Signer, _fake_keygen))

    outputs = prepare_ai_ready_artifacts(tmp_path)

    assert "report_payload" in outputs
    assert "review_payload" in outputs
    assert "anomalies_payload" in outputs
    assert "trends_payload" in outputs
    assert "step_riskiest" in outputs
    assert "mdmp_cert" in outputs
    cert_payload = json.loads((tmp_path / "ai" / "report.signed.mdmp").read_text(encoding="utf-8"))
    assert cert_payload["signature"] == "demo"
