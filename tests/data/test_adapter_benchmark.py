from __future__ import annotations

import pandas as pd

from iints.data.adapter import DataAdapter


def _adapter_with_baseline(monkeypatch) -> DataAdapter:
    adapter = DataAdapter()
    baseline = pd.DataFrame({"glucose_mg_dl": [100.0, 120.0, 200.0, 220.0]})
    baseline.attrs["clinical_benchmarks"] = {
        "original_tir": 50.0,
        "original_gmi": 7.0,
        "original_cv": 30.0,
    }
    monkeypatch.setattr(adapter, "load_ohio_dataset", lambda patient_id: baseline)
    return adapter


def test_benchmark_never_invents_missing_algorithm_results(monkeypatch) -> None:
    adapter = _adapter_with_baseline(monkeypatch)

    result = adapter.clinical_benchmark_comparison("559", ["lstm", "hybrid"])

    assert result["synthetic_improvements_used"] is False
    assert result["algorithm_results"]["lstm"]["status"] == "not_evaluated"
    assert result["algorithm_results"]["lstm"]["tir_70_180"] is None
    assert result["algorithm_results"]["hybrid"]["improvement_percent"] is None


def test_benchmark_uses_only_measured_trace_values(monkeypatch) -> None:
    adapter = _adapter_with_baseline(monkeypatch)
    evaluated = pd.DataFrame({"glucose_actual_mgdl": [90.0, 110.0, 130.0, 200.0]})

    result = adapter.clinical_benchmark_comparison(
        "559",
        ["lstm"],
        evaluated_outputs={"lstm": evaluated},
    )

    metrics = result["algorithm_results"]["lstm"]
    assert metrics["status"] == "measured"
    assert metrics["tir_70_180"] == 75.0
    assert metrics["improvement_percent"] == 25.0
    assert metrics["relative_improvement"] == 50.0
