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
    model = EmpiricalResidualModel(
        profile,
        seed=0,
        scale=0.5,
        max_residual_rate_mgdl_per_min=None,
    )

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


def test_empirical_residual_model_limits_unphysiological_template_jumps() -> None:
    profile = EmpiricalResidualProfile(
        id="jumpy",
        label="Jumpy profile",
        source_dataset_ids=["test"],
        sample_interval_minutes=5,
        templates=[[0.0, 50.0, -50.0, 50.0]],
    )
    model = EmpiricalResidualModel(
        profile,
        seed=0,
        scale=1.0,
        max_residual_rate_mgdl_per_min=0.75,
    )

    offsets = [model.offset_at(minute) for minute in (0.0, 5.0, 10.0, 15.0)]
    max_step = max(abs(next_value - value) for value, next_value in zip(offsets, offsets[1:]))

    assert max_step <= 3.75
