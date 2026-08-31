from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.research.dual_stream import (
    decompose_dual_stream,
    extract_dual_stream_pre_meal_features,
)
from iints.research.ppgr import (
    CarbOnlyLinearPPGR,
    DualStreamGlucoFMPPGR,
    MultiMacroLinearPPGR,
    compute_trajectory_metrics,
    run_ppgr_benchmark,
)

runner = CliRunner()


def test_decompose_dual_stream():
    # 24 steps (2 hours) of glucose values
    time_series = np.array([100.0 + 5.0 * np.sin(i / 3.0) + (15.0 if i > 12 else 0.0) for i in range(24)])
    decomp = decompose_dual_stream(time_series, sampling_interval_minutes=5.0, filter_window_minutes=60.0)

    assert decomp.length == 24
    assert len(decomp.baseline_stream) == 24
    assert len(decomp.event_stream) == 24
    # Check that baseline + event = raw glucose exactly
    reconstructed = decomp.baseline_stream + decomp.event_stream
    np.testing.assert_allclose(reconstructed, time_series, atol=1e-5)


def test_extract_dual_stream_pre_meal_features():
    pre_meal = [100.0, 102.0, 105.0, 108.0, 110.0, 112.0]
    feats = extract_dual_stream_pre_meal_features(pre_meal, sampling_interval_minutes=5.0, time_of_day_minutes=720.0)

    assert "baseline_last" in feats
    assert "baseline_slope_60min" in feats
    assert "event_auc_60min" in feats
    assert "tod_sin" in feats
    assert "tod_cos" in feats


def test_ppgr_models_and_metrics():
    np.random.seed(42)
    N = 20
    # Features: carbs, protein, fat, fiber, cal, g0, ...
    carbs = np.random.uniform(20, 80, size=(N, 1))
    protein = np.random.uniform(5, 30, size=(N, 1))
    fat = np.random.uniform(5, 25, size=(N, 1))
    fiber = np.random.uniform(2, 10, size=(N, 1))
    cal = carbs * 4 + protein * 4 + fat * 9
    g0 = np.random.uniform(80, 120, size=(N, 1))
    extras = np.zeros((N, 7))

    X = np.hstack([carbs, protein, fat, fiber, cal, g0, extras])
    # Target Delta G: 24 steps with delayed peak based on fat/protein
    t_steps = np.arange(1, 25)
    y = np.array([
        (c * 0.8 * np.sin(t_steps / 8.0) + (p + f) * 0.2 * np.sin(t_steps / 12.0))
        for c, p, f in zip(carbs.squeeze(), protein.squeeze(), fat.squeeze())
    ])

    m_carb = CarbOnlyLinearPPGR()
    m_carb.fit(X, y)
    pred_carb = m_carb.predict(X)
    metrics_carb = compute_trajectory_metrics(y, pred_carb)

    m_macro = MultiMacroLinearPPGR()
    m_macro.fit(X, y)
    pred_macro = m_macro.predict(X)
    metrics_macro = compute_trajectory_metrics(y, pred_macro)

    m_gluco = DualStreamGlucoFMPPGR()
    m_gluco.fit(X, y)
    pred_gluco = m_gluco.predict(X)
    metrics_gluco = compute_trajectory_metrics(y, pred_gluco)

    assert metrics_carb.mae_mgdl > 0.0
    assert metrics_macro.mae_mgdl <= metrics_carb.mae_mgdl + 1.0
    assert metrics_gluco.mae_mgdl < metrics_carb.mae_mgdl


def test_run_ppgr_benchmark(tmp_path: Path):
    meals_file = tmp_path / "mock_meals.csv"
    rows = []
    np.random.seed(42)
    for i in range(16):
        c = float(np.random.uniform(20, 80))
        p = float(np.random.uniform(5, 30))
        f = float(np.random.uniform(5, 25))
        g0 = float(np.random.uniform(85, 115))
        row = {
            "meal_id": f"m_{i}",
            "subject_id": f"CGMacros-{i%4 + 1:02d}",
            "carbs_g": c,
            "protein_g": p,
            "fat_g": f,
            "fiber_g": 5.0,
            "calories_kcal": c*4 + p*4 + f*9,
            "pre_meal_glucose_dexcom": g0,
            "pre_meal_glucose_libre": g0 - 2.0,
            "time_minutes": 720.0,
        }
        for t in range(5, 125, 5):
            row[f"dexcom_t{t}"] = g0 + c * 0.5 * np.sin(t / 40.0)
            row[f"libre_t{t}"] = g0 - 2.0 + c * 0.48 * np.sin(t / 40.0)
        rows.append(row)

    pd.DataFrame(rows).to_csv(meals_file, index=False)
    out_dir = tmp_path / "ppgr_results"

    res = run_ppgr_benchmark(
        meals_path=meals_file,
        output_dir=out_dir,
        sensor="dexcom",
        test_split=0.25,
        seed=42,
    )

    assert res.sample_count == 16
    assert "Carb-Only Baseline" in res.models
    assert "Dual-Stream GlucoFM" in res.models
    assert res.report_json.is_file()
    assert res.report_md.is_file()


def test_cli_ppgr_benchmark(tmp_path: Path):
    meals_file = tmp_path / "mock_meals.csv"
    rows = []
    np.random.seed(42)
    for i in range(10):
        row = {
            "meal_id": f"m_{i}",
            "subject_id": f"CGMacros-{i%2 + 1:02d}",
            "carbs_g": 50.0,
            "protein_g": 20.0,
            "fat_g": 15.0,
            "fiber_g": 5.0,
            "calories_kcal": 415.0,
            "pre_meal_glucose_dexcom": 100.0,
            "pre_meal_glucose_libre": 98.0,
            "time_minutes": 600.0,
        }
        for t in range(5, 125, 5):
            row[f"dexcom_t{t}"] = 100.0 + 30.0 * np.sin(t / 50.0)
            row[f"libre_t{t}"] = 98.0 + 28.0 * np.sin(t / 50.0)
        rows.append(row)

    pd.DataFrame(rows).to_csv(meals_file, index=False)
    out_dir = tmp_path / "cli_ppgr_results"

    result = runner.invoke(
        app,
        [
            "research",
            "ppgr-benchmark",
            "--meals-file",
            str(meals_file),
            "--output-dir",
            str(out_dir),
            "--sensor",
            "dexcom",
        ],
    )
    assert result.exit_code == 0
    assert "PPGR 2-Hour Trajectory Benchmark" in result.output
    assert (out_dir / "ppgr_benchmark_report.md").is_file()
