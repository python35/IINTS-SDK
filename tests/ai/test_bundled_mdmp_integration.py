from __future__ import annotations

import json

import pandas as pd
import pytest

from iints.ai.mdmp_guard import MDMPGuard
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
                "glucose_actual_mgdl": 92.0,
                "glucose_to_algo_mgdl": 94.0,
                "glucose_trend_mgdl_min": -0.4,
                "predicted_glucose_30min": 90.0,
                "algo_recommended_insulin_units": 0.1,
                "delivered_insulin_units": 0.1,
                "safety_triggered": False,
                "safety_reason": "",
            },
        ]
    )
    results.to_csv(tmp_path / "results.csv", index=False)
    (tmp_path / "audit").mkdir()
    (tmp_path / "baseline").mkdir()
    (tmp_path / "audit" / "audit_summary.json").write_text(
        json.dumps({"total_overrides": 0, "top_reasons": {}}),
        encoding="utf-8",
    )
    (tmp_path / "baseline" / "baseline_comparison.json").write_text(
        json.dumps({"primary_label": "QuickstartAlgorithm"}),
        encoding="utf-8",
    )
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "bundled-mdmp-demo",
                "sdk_version": "1.3.2",
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


def test_bundled_mdmp_prepare_and_guard_roundtrip(tmp_path) -> None:
    pytest.importorskip("cryptography", reason="bundled MDMP crypto needs cryptography")
    pytest.importorskip("mdmp_core")
    _write_run_bundle(tmp_path)

    outputs = prepare_ai_ready_artifacts(tmp_path)

    assert "mdmp_cert" in outputs
    guard = MDMPGuard(outputs["mdmp_cert"], public_key_path=outputs["mdmp_public_key"])
    result = guard.check()

    assert result.grade == "research_grade"
    assert result.key_id == "iints_local_ai_v1"
