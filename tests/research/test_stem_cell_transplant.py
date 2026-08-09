from __future__ import annotations

import pytest

from iints.research.stem_cell_transplant import (
    StemCellTransplantModel,
    StemCellTransplantParameters,
    run_stem_cell_transplant_simulation,
)


def _run_model(model: StemCellTransplantModel, *, glucose: float = 180.0, steps: int = 288) -> None:
    for _ in range(steps):
        model.step(glucose, 5.0)


def test_transplant_model_matures_and_revascularizes() -> None:
    model = StemCellTransplantModel(
        StemCellTransplantParameters(
            placement="subcutaneous",
            initial_cell_mass=1.0,
            initial_maturation_fraction=0.20,
            immunosuppression_effect=0.8,
        )
    )
    initial = model.state

    _run_model(model, steps=7 * 288)

    assert model.state.vascularization > initial.vascularization
    assert model.state.functional_mass > initial.functional_mass
    assert model.state.oxygenation > initial.oxygenation


def test_immunosuppression_preserves_more_functional_mass() -> None:
    unprotected = StemCellTransplantModel(
        StemCellTransplantParameters(
            placement="portal",
            initial_cell_mass=1.0,
            initial_maturation_fraction=0.7,
            immunosuppression_effect=0.0,
        )
    )
    protected = StemCellTransplantModel(
        StemCellTransplantParameters(
            placement="portal",
            initial_cell_mass=1.0,
            initial_maturation_fraction=0.7,
            immunosuppression_effect=0.9,
        )
    )

    _run_model(unprotected, steps=30 * 288)
    _run_model(protected, steps=30 * 288)

    assert protected.state.functional_mass > unprotected.state.functional_mass
    assert protected.state.adaptive_immunity < unprotected.state.adaptive_immunity


def test_encapsulated_graft_has_delayed_release_pool() -> None:
    model = StemCellTransplantModel(
        StemCellTransplantParameters(
            placement="encapsulated",
            initial_cell_mass=1.0,
            initial_maturation_fraction=0.9,
        )
    )

    step = model.step(220.0, 5.0)

    assert step.secreted_insulin_units > step.released_insulin_units
    assert step.subcutaneous_units == step.released_insulin_units
    assert model.state.insulin_delay_pool_units > 0.0


def test_transplant_runner_couples_graft_to_patient_glucose() -> None:
    no_graft = run_stem_cell_transplant_simulation(
        duration_minutes=6 * 60,
        parameters=StemCellTransplantParameters(initial_cell_mass=0.0),
        meal_schedule=[{"time": 60, "carbs": 60.0}],
    )
    graft = run_stem_cell_transplant_simulation(
        duration_minutes=6 * 60,
        parameters=StemCellTransplantParameters(
            placement="portal",
            initial_cell_mass=1.5,
            initial_maturation_fraction=0.9,
            immunosuppression_effect=0.8,
        ),
        meal_schedule=[{"time": 60, "carbs": 60.0}],
    )

    assert set(
        [
            "functional_mass",
            "vascularization",
            "oxygenation",
            "innate_inflammation",
            "adaptive_immunity",
            "fibrosis",
            "released_insulin_units",
            "glucose_mgdl",
        ]
    ) <= set(graft.columns)
    assert graft["released_insulin_units"].sum() > no_graft["released_insulin_units"].sum()
    assert graft["glucose_mgdl"].max() < no_graft["glucose_mgdl"].max()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_cell_mass": -1.0},
        {"initial_maturation_fraction": 1.1},
        {"immunosuppression_effect": -0.1},
        {"oxygen_time_constant_minutes": 0.0},
        {"max_secretion_units_per_min": float("nan")},
    ],
)
def test_transplant_parameters_fail_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        StemCellTransplantParameters(**kwargs)


def test_transplant_step_rejects_invalid_observation_or_step() -> None:
    model = StemCellTransplantModel()
    with pytest.raises(ValueError):
        model.step(float("nan"), 5.0)
    with pytest.raises(ValueError):
        model.step(120.0, 0.0)
    with pytest.raises(ValueError):
        model.step(120.0, 120.0)
