from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "research" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scenario_rank_key_prefers_cross_reference_robustness() -> None:
    module = _load_module("search_realistic_scenarios")
    robust = {
        "min_likely_realistic_runs": 3,
        "total_likely_realistic_runs": 8,
        "mean_realism_score": 0.89,
        "failed_checks": 0,
        "warning_checks": 2,
    }
    brittle = {
        "min_likely_realistic_runs": 2,
        "total_likely_realistic_runs": 9,
        "mean_realism_score": 0.93,
        "failed_checks": 0,
        "warning_checks": 0,
    }

    ranked = sorted([brittle, robust], key=module.scenario_rank_key, reverse=True)

    assert ranked[0] is robust


def test_mutated_scenario_preserves_four_meals_and_one_exercise() -> None:
    module = _load_module("search_realistic_scenarios")
    events = [
        {"start_time": 450, "event_type": "meal", "value": 42, "reported_value": 40},
        {"start_time": 735, "event_type": "meal", "value": 56, "reported_value": 54},
        {"start_time": 960, "event_type": "exercise", "value": 0.2, "duration": 25},
        {"start_time": 1080, "event_type": "meal", "value": 68, "reported_value": 64},
        {"start_time": 1290, "event_type": "meal", "value": 18, "reported_value": 18},
    ]

    mutated = module.mutate_events(events, module.random.Random(7))

    assert sum(event["event_type"] == "meal" for event in mutated) == 4
    assert sum(event["event_type"] == "exercise" for event in mutated) == 1
