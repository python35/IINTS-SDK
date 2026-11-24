"""Unit tests for the Hybrid Gray-Box MiniMed Predictor and In-Silico BoxQP."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from iints.analysis.minimed_predictor import MiniMedGrayBoxPredictor, PredictorConfig
from iints.core.patient.hovorka_model import HovorkaPatientModel


def _synthetic_minimed_timeline(n_steps: int = 220) -> pd.DataFrame:
    patient = HovorkaPatientModel(initial_glucose=120.0, basal_insulin_rate=1.2)
    dt = 5.0
    start = pd.Timestamp("2026-11-01 06:00:00")
    rows = []
    for k in range(n_steps):
        t = k * dt
        ts = start + pd.Timedelta(minutes=t)
        carbs = 35.0 if k in (20, 90, 160) else 0.0
        bolus = 3.0 if k in (20, 90, 160) else 0.0
        basal_u = 1.2 * dt / 60.0
        rows.append(
            {
                "timestamp_dt": ts,
                "timestamp": t,
                "glucose": patient.get_current_glucose(),
                "carbs": carbs,
                "insulin": bolus + basal_u,
                "bolus_units": bolus,
                "basal_units": basal_u,
                "basal_rate_u_per_hr": 1.2,
                "source": "synthetic_minimed",
            }
        )
        patient.update(
            time_step=dt,
            delivered_insulin=bolus + basal_u,
            carb_intake=carbs,
            current_time=t,
        )
    return pd.DataFrame(rows)


def test_minimed_graybox_predictor_end_to_end_benchmark() -> None:
    timeline = _synthetic_minimed_timeline(n_steps=220)
    cfg = PredictorConfig(horizons_minutes=(30, 60), train_fraction=0.65)
    predictor = MiniMedGrayBoxPredictor(config=cfg)

    report = predictor.evaluate_timeline(timeline)
    assert report["train_rows"] > 100
    assert report["test_rows"] > 50
    assert len(report["horizon_evaluations"]) == 2

    for ev in report["horizon_evaluations"]:
        assert ev["n_samples"] > 10
        assert math.isfinite(ev["graybox_rmse_mgdl"])
        assert ev["graybox_clarke_ab_pct"] >= 80.0
        assert 0.0 <= ev["graybox_coverage_90_pct"] <= 100.0

    qp = report["latest_in_silico_boxqp"]
    assert 0.0 <= qp["recommended_rate_u_per_h"] <= cfg.max_insulin_rate_u_per_h
    assert qp["kkt_residual"] < 1e-4


def test_minimed_boxqp_suspends_insulin_when_p05_crosses_hypo_floor() -> None:
    timeline = _synthetic_minimed_timeline(n_steps=160)
    cfg = PredictorConfig(horizons_minutes=(30, 60), safety_floor_mgdl=70.0)
    predictor = MiniMedGrayBoxPredictor(config=cfg)
    predictor.fit(timeline)

    obs = predictor._new_observer()
    # Set the observer's latent glucose estimate directly to 65 mg/dL so the
    # 90% lower confidence band crosses the 70 mg/dL safety floor.
    assert predictor.model is not None
    obs.x[0] = (65.0 - predictor.model.glucose_mgdl) / predictor.model.C[0]

    decision = predictor.recommend_in_silico_insulin(obs)
    assert decision["safety_floor_suspended"] is True
    assert decision["recommended_rate_u_per_h"] == 0.0
    assert decision["recommended_step_units"] == 0.0
