from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "research" / "calibrate_simulator_realism.py"
    spec = importlib.util.spec_from_file_location("calibrate_simulator_realism", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_grid_only_changes_requested_dimensions() -> None:
    module = _load_module()
    base = {
        "initial_glucose": 140.0,
        "dawn_phenomenon_strength": 0.0,
        "meal_mismatch_epsilon": 1.0,
        "glucose_decay_rate": 0.03,
        "insulin_sensitivity": 50.0,
    }

    candidates = module.build_candidate_grid(
        base,
        initial_glucose_values=[140.0, 150.0],
        dawn_strength_values=[0.0, 8.0],
        meal_mismatch_values=[1.0],
        glucose_decay_values=[0.03],
    )

    assert len(candidates) == 4
    assert all(candidate["insulin_sensitivity"] == 50.0 for candidate in candidates)
    assert {candidate["initial_glucose"] for candidate in candidates} == {140.0, 150.0}
    assert {candidate["dawn_phenomenon_strength"] for candidate in candidates} == {0.0, 8.0}


def test_candidate_ranking_prefers_robust_realism_before_center_distance() -> None:
    module = _load_module()
    more_robust = {
        "likely_realistic_runs": 5,
        "mean_verdict_rank": 2.0,
        "mean_realism_score": 0.88,
        "failed_checks": 0,
        "warning_checks": 1,
        "mean_normalized_reference_distance": 0.9,
    }
    closer_but_less_robust = {
        "likely_realistic_runs": 4,
        "mean_verdict_rank": 1.8,
        "mean_realism_score": 0.91,
        "failed_checks": 0,
        "warning_checks": 0,
        "mean_normalized_reference_distance": 0.3,
    }

    ranked = sorted(
        [closer_but_less_robust, more_robust],
        key=module.candidate_rank_key,
        reverse=True,
    )

    assert ranked[0] is more_robust
