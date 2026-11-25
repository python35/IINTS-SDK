"""Tests for the integrated MiniMed v2 nonlinear propagator, backtest, and what-if engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iints.analysis import minimed_backtest as bt
from iints.analysis import minimed_predictor_v2 as v2
from iints.analysis import nonlinear_propagator as nlp
from iints.core.patient import equilibrium as pf


def _synthetic_timeline(days: int = 8) -> pd.DataFrame:
    n = days * 288
    start = pd.Timestamp("2026-10-01 00:00:00")
    times = start + pd.to_timedelta(np.arange(n) * 5.0, unit="m")
    t_min = np.arange(n, dtype=float) * 5.0
    minute_of_day = (times.hour * 60 + times.minute).to_numpy(dtype=float)

    rng = np.random.default_rng(42)
    glucose = 140.0 + 25.0 * np.sin(2.0 * np.pi * minute_of_day / 1440.0) + rng.normal(0.0, 4.0, size=n)
    carbs = np.zeros(n)
    bolus = np.zeros(n)
    carbs[minute_of_day == 480] = 45.0
    bolus[minute_of_day == 480] = 4.5
    auto_basal = np.full(n, 2.0 * 5.0 / 60.0)
    return pd.DataFrame(
        {
            "timestamp_dt": times,
            "glucose": np.clip(glucose, 45.0, 380.0),
            "isig": np.nan,
            "sensor_exception": "",
            "is_filled": False,
            "t_min": t_min,
            "day": times.normalize(),
            "minute_of_day": minute_of_day,
            "bolus_u": bolus,
            "carbs_g": carbs,
            "prog_basal_u_per_h": 2.0,
            "auto_basal_u": auto_basal,
            "insulin_u": auto_basal + bolus,
            "insulin_rate_u_per_h": (auto_basal + bolus) * 12.0,
        }
    )


@pytest.mark.parametrize("meal_model", ("hovorka_2comp", "biphasic", "glucose_gated"))
def test_nonlinear_propagator_meal_models_behave_physiologically(meal_model: str) -> None:
    basal = 2.0
    sigma = pf.basal_scale(basal, 140.0)
    patient = pf.make_equilibrium_patient(basal, sigma, 140.0)
    params = nlp.from_patient(
        patient,
        meal_tau_min=64.0,
        bioavailability=0.85,
        meal_model=meal_model,
    )
    ode_state = np.asarray(patient.get_state()["ode_state"], dtype=float)[:8]
    state0 = np.zeros((3, 10), dtype=float)
    state0[:, :8] = ode_state[None, :]

    steps = 24  # 120 minutes
    rate = np.full((3, steps), basal)
    rate[2, 0] += 3.0 * 12.0  # 3 U bolus on trajectory 2
    carbs = np.zeros((3, steps))
    carbs[1, 0] = 50.0  # 50 g meal on trajectory 1

    out = nlp.propagate(state0, rate, carbs, params, step_minutes=5.0, substeps=2)
    assert out.shape == (3, steps)
    assert np.all(np.isfinite(out))
    # Fasting trajectory stays at equilibrium (~140 mg/dL)
    assert out[0, -1] == pytest.approx(140.0, abs=1.0)
    # Meal raises glucose substantially
    assert np.max(out[1]) > 190.0
    # Bolus lowers glucose substantially
    assert np.min(out[2]) < 120.0


def test_walk_forward_ar2_nested_and_what_if_consistency() -> None:
    timeline = _synthetic_timeline(days=8)
    cfg = bt.PhysicsConfig(
        propagation="nonlinear",
        sigma=0.12,
        meal_tau_min=64.0,
        bioavailability=0.85,
        feedback_gain=0.075,
        ar2_nested_graybox=True,
        substeps=2,
    )
    table = bt.walk_forward(
        timeline,
        cfg,
        train_days=4.0,
        test_days=2.0,
        initial_train_days=4.0,
    )
    assert not table.empty
    assert set(table["horizon"].unique()) == {30, 60, 120}
    scores = bt.score(table)
    assert not scores.empty

    with pytest.warns(RuntimeWarning, match="extrapolated beyond 120 min"):
        scenario = v2.what_if(
            timeline,
            anchor_index=400,
            config=cfg,
            horizon_minutes=180,
            carbs_g=40.0,
            bolus_u=4.0,
        )
    assert scenario["min_p05_mgdl"] >= 20.0
    assert scenario["min_p05_mgdl"] == pytest.approx(min(scenario["p05_mgdl"]), abs=1e-6)
    assert scenario["interval_validated_to_min"] == 120
