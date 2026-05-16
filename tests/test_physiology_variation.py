from __future__ import annotations

from iints.core.physiology_variation import EmpiricalResidualModel, EmpiricalResidualProfile


def test_empirical_residual_model_interpolates_between_samples() -> None:
    profile = EmpiricalResidualProfile(
        id="tiny",
        label="Tiny profile",
        source_dataset_ids=["test"],
        sample_interval_minutes=5,
        templates=[[0.0, 10.0, 20.0]],
    )
    model = EmpiricalResidualModel(profile, seed=0, scale=0.5)

    assert model.offset_at(0.0) == 0.0
    assert model.offset_at(2.5) == 2.5
    assert model.offset_at(5.0) == 5.0


def test_empirical_residual_model_rotates_templates_by_seed_and_day() -> None:
    profile = EmpiricalResidualProfile(
        id="tiny",
        label="Tiny profile",
        source_dataset_ids=["test"],
        sample_interval_minutes=5,
        templates=[[1.0], [2.0], [3.0]],
    )
    model = EmpiricalResidualModel(profile, seed=1)

    assert model.offset_at(0.0) == 2.0
    assert model.offset_at(1440.0) == 3.0
