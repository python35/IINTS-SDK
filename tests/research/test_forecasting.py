from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from iints.research.forecasting import (
    ForecastConfig,
    PhysiologyAwareBaseline,
    assess_forecast_risk,
    attach_forecasts_to_frame,
    write_forecast_bundle,
)
from iints.research.predictor import evaluate_baselines


def test_physiology_aware_baseline_uses_cob_and_iob() -> None:
    feature_columns = [
        "glucose_actual_mgdl",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "glucose_trend_mgdl_min",
    ]
    X = np.zeros((2, 4, len(feature_columns)), dtype=np.float32)
    X[:, :, 0] = 120.0
    X[:, :, 3] = 50.0
    X[:, :, 4] = 10.0
    X[0, -1, 2] = 45.0
    X[1, -1, 1] = 3.0

    predictor = PhysiologyAwareBaseline(6, feature_columns=feature_columns)
    preds = predictor.predict(X)

    assert preds.shape == (2, 6)
    assert preds[0, -1] > 120.0
    assert preds[1, -1] < 120.0


def test_physiology_aware_baseline_can_model_insulin_antibody_delay() -> None:
    feature_columns = [
        "glucose_actual_mgdl",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
        "insulin_antibody_binding_fraction",
        "insulin_antibody_release_fraction",
    ]
    X = np.zeros((2, 8, len(feature_columns)), dtype=np.float32)
    X[:, :, 0] = 180.0
    X[:, :, 1] = 4.0
    X[:, :, 3] = 50.0
    X[:, :, 4] = 10.0
    X[1, :, 5] = 0.8
    X[1, :, 6] = 0.01

    predictor = PhysiologyAwareBaseline(6, feature_columns=feature_columns)
    preds = predictor.predict(X)

    assert preds[1, -1] > preds[0, -1]


def test_assess_forecast_risk_flags_hypoglycemia_and_uncertainty() -> None:
    risk = assess_forecast_risk([92.0, 78.0, 66.0], current_glucose=110.0)
    assert risk["risk_level"] == "hypo_risk"
    assert risk["guardrail_action"] == "block_extra_insulin"

    uncertain = assess_forecast_risk(
        [120.0, 122.0, 124.0],
        current_glucose=118.0,
        predicted_std=[8.0, 40.0, 42.0],
    )
    assert uncertain["risk_level"] == "uncertain"
    assert uncertain["guardrail_action"] == "fallback_or_human_review"


def test_attach_forecasts_to_frame_writes_predictions_and_scoring_columns() -> None:
    n = 80
    df = pd.DataFrame(
        {
            "time_minutes": np.arange(n) * 5,
            "glucose_actual_mgdl": 130.0 + np.sin(np.arange(n) / 8.0) * 15.0,
            "patient_iob_units": np.linspace(1.0, 0.1, n),
            "patient_cob_grams": np.linspace(20.0, 0.0, n),
            "effective_isf": 50.0,
            "effective_icr": 10.0,
        }
    )
    config = ForecastConfig(history_minutes=30, horizon_minutes=30)

    out = attach_forecasts_to_frame(df, config=config)

    assert "predicted_glucose_physiology_30min" in out.columns
    assert "predicted_glucose_ai_30min" in out.columns
    assert "observed_glucose_30min" in out.columns
    scored = out["observed_glucose_30min"].notna() & out["predicted_glucose_ai_30min"].notna()
    assert int(scored.sum()) > 20
    assert set(out.loc[out["forecast_risk_level"] != "", "forecast_guardrail_action"]) != set()


def test_write_forecast_bundle_creates_research_artifacts(tmp_path) -> None:
    n = 72
    source = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "time_minutes": np.arange(n) * 5,
            "glucose_actual_mgdl": np.linspace(105.0, 175.0, n),
            "patient_iob_units": 0.5,
            "patient_cob_grams": 12.0,
            "effective_isf": 45.0,
            "effective_icr": 11.0,
        }
    ).to_csv(source, index=False)

    bundle = write_forecast_bundle(
        source,
        tmp_path / "forecast",
        config=ForecastConfig(history_minutes=30, horizon_minutes=30),
    )

    artifacts = bundle["artifacts"]
    for path in artifacts.values():
        artifact_path = Path(path)
        assert tmp_path in artifact_path.parents
        assert artifact_path.is_file()
    report = json.loads(Path(artifacts["report_json"]).read_text())
    assert "ai" in report["models"]
    assert "physiology" in report["models"]


def test_write_forecast_bundle_accepts_hidden_biology_feature_overrides(tmp_path) -> None:
    n = 72
    source = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "time_minutes": np.arange(n) * 5,
            "glucose_actual_mgdl": np.linspace(150.0, 190.0, n),
            "patient_iob_units": 2.0,
            "patient_cob_grams": 0.0,
            "effective_isf": 50.0,
            "effective_icr": 10.0,
        }
    ).to_csv(source, index=False)

    bundle = write_forecast_bundle(
        source,
        tmp_path / "forecast",
        config=ForecastConfig(history_minutes=30, horizon_minutes=30),
        feature_overrides={
            "insulin_antibody_binding_fraction": 0.6,
            "insulin_antibody_release_fraction": 0.02,
        },
    )

    predictions = pd.read_csv(bundle["artifacts"]["predictions_csv"])
    assert "insulin_antibody_binding_fraction" in predictions.columns
    assert float(predictions["insulin_antibody_binding_fraction"].max()) == 0.6


def test_evaluate_baselines_includes_stronger_physiology_baseline() -> None:
    feature_columns = [
        "glucose_actual_mgdl",
        "patient_iob_units",
        "patient_cob_grams",
        "effective_isf",
        "effective_icr",
    ]
    X = np.zeros((4, 8, len(feature_columns)), dtype=np.float32)
    X[:, :, 0] = 120.0
    X[:, :, 3] = 50.0
    X[:, :, 4] = 10.0
    y = np.ones((4, 6), dtype=np.float32) * 120.0

    results = evaluate_baselines(X, y, horizon_steps=6, feature_columns=feature_columns)

    assert "PhysiologyAware" in results
    assert results["PhysiologyAware"]["mae"] >= 0.0
