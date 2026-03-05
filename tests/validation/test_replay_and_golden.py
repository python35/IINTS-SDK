from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from iints.api.base_algorithm import AlgorithmInput, InsulinAlgorithm
from iints.validation.golden import evaluate_expected_ranges, load_golden_benchmark_pack
from iints.validation.replay import run_deterministic_replay_check


class _DummyAlgorithm(InsulinAlgorithm):
    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        return {
            "total_insulin_delivered": 0.0,
            "basal_insulin": 0.0,
            "bolus_insulin": 0.0,
            "meal_bolus": 0.0,
            "correction_bolus": 0.0,
            "uncertainty": 0.0,
            "fallback_triggered": False,
        }


def test_load_default_golden_pack() -> None:
    pack = load_golden_benchmark_pack()
    assert pack.name
    assert pack.scenarios
    assert all(spec.preset for spec in pack.scenarios)


def test_evaluate_expected_ranges_handles_missing_metrics() -> None:
    checks = evaluate_expected_ranges(
        metrics={"tir_70_180": 60.0},
        expected={
            "tir_70_180": {"min": 55.0},
            "tir_below_54": {"max": 5.0},
        },
    )
    assert checks["tir_70_180"]["passed"] is True
    assert checks["tir_below_54"]["passed"] is False
    assert checks["tir_below_54"]["reason"] == "missing_metric"


def test_replay_check_passes_for_stable_runs(monkeypatch) -> None:
    def _fake_run_simulation(**kwargs: Any) -> Dict[str, Any]:
        results = pd.DataFrame(
            {
                "time_minutes": [0.0, 5.0, 10.0],
                "glucose_actual_mgdl": [110.0, 112.0, 111.0],
                "algorithm_latency_ms": [0.10, 0.11, 0.09],  # ignored in hash
            }
        )
        safety_report = {
            "terminated_early": False,
            "bolus_interventions_count": 0,
            "performance_report": {"runtime_ms": 12.3},  # ignored in hash
        }
        return {"results": results, "safety_report": safety_report}

    import iints.validation.replay as replay_mod

    monkeypatch.setattr(replay_mod.iints, "run_simulation", _fake_run_simulation)
    result = run_deterministic_replay_check(
        algorithm=_DummyAlgorithm(),
        scenario=None,
        patient_config="default_patient",
        duration_minutes=15,
        time_step=5,
        seed=42,
        repeats=3,
    )
    assert result.passed is True
    assert len(result.digests) == 3


def test_replay_check_fails_on_output_mismatch(monkeypatch) -> None:
    call_count = {"n": 0}

    def _fake_run_simulation(**kwargs: Any) -> Dict[str, Any]:
        call_count["n"] += 1
        glucose = [110.0, 112.0, 111.0]
        if call_count["n"] == 2:
            glucose = [110.0, 130.0, 111.0]
        results = pd.DataFrame(
            {
                "time_minutes": [0.0, 5.0, 10.0],
                "glucose_actual_mgdl": glucose,
            }
        )
        return {"results": results, "safety_report": {"terminated_early": False}}

    import iints.validation.replay as replay_mod

    monkeypatch.setattr(replay_mod.iints, "run_simulation", _fake_run_simulation)
    result = run_deterministic_replay_check(
        algorithm=_DummyAlgorithm(),
        scenario=None,
        patient_config="default_patient",
        duration_minutes=15,
        time_step=5,
        seed=42,
        repeats=2,
    )
    assert result.passed is False
