from __future__ import annotations

import pandas as pd
import pytest

from iints.research import tissue_stressor


def test_simulation_arm_passes_explicit_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_seeds: list[int | None] = []

    class FakeSimulator:
        def __init__(self, *, patient_model, algorithm, seed=None) -> None:
            observed_seeds.append(seed)

        def add_stress_event(self, _event) -> None:
            return None

        def run(self, *, duration_minutes: int):
            return (
                pd.DataFrame(
                    {
                        "time_minutes": [0, duration_minutes],
                        "glucose_actual_mgdl": [110.0, 115.0],
                    }
                ),
                {},
            )

    monkeypatch.setattr(tissue_stressor, "Simulator", FakeSimulator)

    tissue_stressor._simulate_arm(
        muscle_scalar=1.0,
        liver_scalar=1.0,
        basal_rate=0.8,
        duration_minutes=60,
        seed=123,
    )

    assert observed_seeds == [123]


@pytest.mark.parametrize(
    ("muscle_scalar", "liver_scalar", "message"),
    [(-0.1, 1.0, "muscle_scalar"), (1.0, 1.1, "liver_scalar")],
)
def test_tissue_scalars_are_bounded(
    muscle_scalar: float,
    liver_scalar: float,
    message: str,
    tmp_path,
) -> None:
    pytest.importorskip("plotly")

    with pytest.raises(ValueError, match=message):
        tissue_stressor.TissueStressor.run_stress_test(
            muscle_scalar,
            liver_scalar,
            tmp_path,
        )
