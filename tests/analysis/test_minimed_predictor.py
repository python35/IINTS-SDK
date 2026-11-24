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


def test_minimed_what_if_forecast_causal_monotonicity() -> None:
    timeline = _synthetic_minimed_timeline(n_steps=160)
    cfg = PredictorConfig(horizons_minutes=(30, 60))
    predictor = MiniMedGrayBoxPredictor(config=cfg)
    predictor.fit(timeline)

    obs = predictor.reconstruct_observer_at(timeline, anchor_idx=120)

    base = predictor.what_if_forecast(obs, horizon_minutes=120, carbs_g=0.0, bolus_u=0.0, basal_pct=100.0)
    with_carbs = predictor.what_if_forecast(obs, horizon_minutes=120, carbs_g=40.0, bolus_u=0.0, basal_pct=100.0)
    with_bolus = predictor.what_if_forecast(obs, horizon_minutes=120, carbs_g=0.0, bolus_u=4.0, basal_pct=100.0)
    basal_off = predictor.what_if_forecast(obs, horizon_minutes=120, carbs_g=0.0, bolus_u=0.0, basal_pct=0.0)

    # Index 12 = +60 min, index 24 = +120 min
    assert with_carbs["median_mgdl"][12] > base["median_mgdl"][12]
    assert with_bolus["median_mgdl"][12] < base["median_mgdl"][12]
    assert basal_off["median_mgdl"][24] > base["median_mgdl"][24]
    assert all(lo <= med <= hi for lo, med, hi in zip(with_carbs["p05_mgdl"], with_carbs["median_mgdl"], with_carbs["p95_mgdl"]))


def test_carelink_three_section_conservation_and_no_daily_spike(tmp_path) -> None:
    from iints.data.importer import import_carelink_timeline

    csv_text = "\n".join(
        [
            "Last Name;First Name;Patient ID;System ID;Start Date;End Date;Device;MiniMed 780G",
            "User;Test;;;2026-11-21;2026-11-21;Serial Number;NG000001",
            "",
            "-------;MiniMed 780G MMT-1886;Pump;NG000001;-------",
            "Index;Date;Time;Basal Rate (U/h);Bolus Volume Delivered (U);Bolus Source;BWZ Carb Input (grams);Sensor Glucose (mg/dL);ISIG Value",
            "0;2026/11/21;10:00:00;1,5;;;;;",
            "1;2026/11/21;10:20:00;;5,0;BOLUS_WIZARD;25,0;;",
            "",
            "-------;MiniMed 780G MMT-1886;Pump;NG000001;Aggregated Auto Insulin Data",
            "Index;Date;Time;Basal Rate (U/h);Bolus Volume Delivered (U);Bolus Source;BWZ Carb Input (grams);Sensor Glucose (mg/dL);ISIG Value",
            "2;2026/11/21;00:00:00;;60,0;CLOSED_LOOP_AUTO_INSULIN;;;",
            "",
            "-------;MiniMed 780G MMT-1886;Sensor;NG000001;-------",
            "Index;Date;Time;Basal Rate (U/h);Bolus Volume Delivered (U);Bolus Source;BWZ Carb Input (grams);Sensor Glucose (mg/dL);ISIG Value",
            "3;2026/11/21;10:00:00;;;;;130;22,5",
            "4;2026/11/21;10:05:00;;;;;135;23,1",
            # 30-min CGM gap from 10:05 to 10:35 — the 5.0 U bolus & 25g meal at 10:20 fall INSIDE this gap!
            "5;2026/11/21;10:35:00;;;;;160;26,4",
        ]
    )
    sample = tmp_path / "three_section.csv"
    sample.write_text(csv_text, encoding="utf-8")

    tl = import_carelink_timeline(sample, fill_gaps=True, grid_step_minutes=5.0)
    # Exact conservation of bolus & carbs delivered inside the CGM gap
    assert math.isclose(float(tl["bolus_units"].sum()), 5.0, abs_tol=1e-9)
    assert math.isclose(float(tl["carbs"].sum()), 25.0, abs_tol=1e-9)
    # Daily aggregated 60.0 U must be distributed as 2.5 U/h (0.2083 U / 5 min), never a 60 U spike
    assert float(tl["basal_units"].max()) < 0.3
    assert math.isclose(float(tl["basal_rate_u_per_hr"].iloc[0]), 60.0 / 24.0, rel_tol=1e-9)
    # Sensor gap rows carry NaN glucose while observed rows carry ISIG
    assert tl["glucose"].isna().sum() >= 5
    assert tl["isig"].notna().sum() == 3


def test_equilibrium_hovorka_has_machine_zero_drift() -> None:
    import pytest
    from iints.analysis.minimed_predictor import build_equilibrium_hovorka, solve_equilibrium_glucose

    basal = 2.0
    sigma = 0.175
    patient = build_equilibrium_hovorka(basal, sigma)
    y0 = patient._default_ode_state(glucose_mgdl=patient.initial_glucose)
    dy = patient._ode(0.0, y0, basal * 1000.0 / 60.0, 0.0, 0.0)
    drift_mgdl_per_hour = abs(float(dy[0] / patient._glucose_volume_dl())) * 60.0
    assert drift_mgdl_per_hour < 1e-8

    # Inadmissible sensitivity (far too high -> equilibrium below 82 mg/dL knee) must raise ValueError
    with pytest.raises(ValueError):
        solve_equilibrium_glucose(basal_rate_u_per_h=2.0, sensitivity_scale=5.0)


def test_smartguard_closed_loop_feedback_speeds_up_slowest_pole() -> None:
    from iints.analysis.minimed_predictor import _apply_smartguard_feedback

    timeline = _synthetic_minimed_timeline(n_steps=120)
    predictor = MiniMedGrayBoxPredictor(PredictorConfig(horizons_minutes=(30, 60)))
    predictor._init_physics_model(timeline)
    assert predictor.open_loop_model is not None

    ol_eigs = np.linalg.eigvals(predictor.open_loop_model.A)
    cl_model = _apply_smartguard_feedback(predictor.open_loop_model, k_sg_u_per_h_per_mgdl=0.02)
    cl_eigs = np.linalg.eigvals(cl_model.A)

    assert float(np.max(cl_eigs.real)) < 0.0
    # Slowest glucose pole must become strictly faster (more negative real part) under SmartGuard feedback
    assert float(np.max(cl_eigs.real)) < float(np.max(ol_eigs.real))
