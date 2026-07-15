from __future__ import annotations

import pandas as pd

from iints.research.stem_cell_optimizer import StemCellOptimizer


def test_stem_cell_optimizer_accepts_simple_meal_schedule() -> None:
    optimizer = StemCellOptimizer(duration_minutes=30, time_step=5)

    result = optimizer.evaluate_graft_configuration(
        engraftment_percent=50.0,
        subq_fraction=0.0,
        immune_decay=0.0,
        meal_schedule=[{"time": 10, "carbs": 20.0}],
        seed=7,
    )

    assert result["engraftment_percent"] == 50.0
    assert result["subq_fraction"] == 0.0
    assert result["final_graft_mass_fraction"] == 0.5
    assert result["min_glucose"] <= result["max_glucose"]
    assert isinstance(result["df"], pd.DataFrame)
    assert not result["df"].empty


def test_stem_cell_optimizer_reports_rejection_decay() -> None:
    optimizer = StemCellOptimizer(duration_minutes=60, time_step=5)

    result = optimizer.evaluate_graft_configuration(
        engraftment_percent=80.0,
        subq_fraction=0.0,
        immune_decay=0.001,
        meal_schedule=[],
        seed=7,
    )

    assert 0.0 < result["final_graft_mass_fraction"] < 0.8


def test_stem_cell_optimizer_runs_multicompartment_transplant_mode() -> None:
    optimizer = StemCellOptimizer(duration_minutes=120, time_step=5)

    result = optimizer.evaluate_transplant_configuration(
        placement="encapsulated",
        initial_cell_mass=1.0,
        initial_maturation_fraction=0.8,
        immunosuppression_effect=0.5,
        meal_schedule=[{"time": 30, "carbs": 30.0}],
    )

    assert result["placement"] == "encapsulated"
    assert result["final_functional_mass"] > 0.0
    assert result["total_released_insulin_units"] > 0.0
    assert isinstance(result["df"], pd.DataFrame)
    assert "fibrosis" in result["df"].columns
