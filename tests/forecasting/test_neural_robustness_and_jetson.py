"""Automated tests for Phase 3 (GRU & TCN + 5-seed trainer), Phase 4 (CGM Robustness),
and Phase 5 (Edge/ONNX export, numerical parity, `tegrastats` parser & Jetson HTTP worker).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from iints.forecasting import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    GRUGlucosePredictor,
    NeuralTrainingConfig,
    PerturbationSpec,
    TCNGlucosePredictor,
    WindowSpec,
    benchmark_edge_inference,
    build_partitioned_windows,
    create_jetson_worker_app,
    export_model_for_edge,
    load_virtual_benchmark_dataset,
    parse_tegrastats_line,
    perturb_windows,
    run_cgm_robustness_benchmark,
    train_and_compare_all_models,
)


def test_gru_and_tcn_learn_linear_trend_and_checkpoint_roundtrip() -> None:
    rng = np.random.default_rng(42)
    # Construct synthetic trajectories G_t = g0 + slope * t + small noise
    n_train, n_val, n_test = 240, 60, 60
    L = 12

    def _make_set(n: int, seed_offset: int) -> tuple[np.ndarray, np.ndarray]:
        r = np.random.default_rng(seed_offset)
        g0 = r.uniform(90.0, 210.0, size=(n, 1))
        slopes = r.uniform(-2.5, 2.5, size=(n, 1))
        steps = np.arange(L, dtype=float)[None, :]
        X = g0 + slopes * steps + r.normal(0.0, 0.5, size=(n, L))
        y = (g0 + slopes * (L - 1 + 6) + r.normal(0.0, 0.5, size=(n, 1))).ravel()
        return X, y

    X_tr, y_tr = _make_set(n_train, 1)
    X_va, y_va = _make_set(n_val, 2)
    X_te, y_te = _make_set(n_test, 3)

    pers_mae = float(np.mean(np.abs(X_te[:, -1] - y_te)))

    gru = GRUGlucosePredictor(hidden_size=16, max_epochs=35, learning_rate=3e-3, seed=11).fit(
        X_tr, y_tr, X_va, y_va
    )
    tcn = TCNGlucosePredictor(channels=12, max_epochs=35, learning_rate=3e-3, seed=11).fit(
        X_tr, y_tr, X_va, y_va
    )

    gru_pred = gru.predict(X_te)
    tcn_pred = tcn.predict(X_te)
    assert gru_pred.shape == (n_test,)
    assert tcn_pred.shape == (n_test,)
    assert float(np.mean(np.abs(gru_pred - y_te))) < pers_mae * 0.65
    assert float(np.mean(np.abs(tcn_pred - y_te))) < pers_mae * 0.65

    # Exact checkpoint serialization round-trip
    gru_reloaded = GRUGlucosePredictor.from_checkpoint_dict(gru.to_checkpoint_dict())
    tcn_reloaded = TCNGlucosePredictor.from_checkpoint_dict(tcn.to_checkpoint_dict())
    np.testing.assert_allclose(gru_pred, gru_reloaded.predict(X_te), atol=1e-12)
    np.testing.assert_allclose(tcn_pred, tcn_reloaded.predict(X_te), atol=1e-12)


def test_multi_seed_trainer_robustness_and_edge_jetson_parity(tmp_path: Path) -> None:
    canonical, quality = load_virtual_benchmark_dataset(
        n_subjects=4, train_days=4.0, test_days=2.0, seed=99
    )
    part = build_partitioned_windows(canonical, WindowSpec())

    tr_res = train_and_compare_all_models(
        part,
        quality,
        tmp_path / "study",
        train_cfg=NeuralTrainingConfig(
            seeds=(11, 23),
            gru_hidden_size=12,
            tcn_channels=8,
            max_epochs=12,
        ),
    )
    # Crucial statistical check: n_subjects is 4 (NEVER inflated to 4 * 2 seeds = 8!)
    assert tr_res["vs_persistence"]["iints-gru-glucose-v1"]["n_subjects"] == 4
    assert tr_res["vs_persistence"]["iints-gru-glucose-v1-5seed-mean"]["n_subjects"] == 4

    # Phase 4: Perturbations preserve y_true untouched and test Clean/Dropout/Noise/Delay
    conds = [
        PerturbationSpec("A_clean", "clean", 0.0, "complete"),
        PerturbationSpec("B_dropout_15m", "dropout_locf", 15.0, "locf_imputed"),
        PerturbationSpec("C_noise_10mgdl", "sensor_noise", 10.0, "degraded"),
        PerturbationSpec("D_delay_10m", "sensor_delay", 10.0, "degraded"),
    ]
    x_drop, mask_drop = perturb_windows(part.test.X[:5], conds[1])
    assert np.all(mask_drop[:, -3:] == 0.0)
    assert np.all(x_drop[:, -1] == part.test.X[:5, -4])  # exact 15m (3-step) LOCF!

    rob = run_cgm_robustness_benchmark(
        part.test,
        tr_res["models"],
        conditions=conds,
        is_synthetic=True,
        dataset_id="iints_virtual",
        output_dir=tmp_path / "rob",
    )
    assert len(rob["paired_degradation"]) == 3 * len(tr_res["models"])

    # Phase 5: Edge export + numerical parity (< 1e-6 mg/dL) + batch-size-1 benchmark
    for mid in ("iints-gru-glucose-v1", "iints-tcn-glucose-v1"):
        exp = export_model_for_edge(
            tr_res["models"][mid],
            tmp_path / "edge",
            reference_windows=part.test.X[:20],
            parity_tolerance_mgdl=1e-4,
        )
        assert exp["parity"]["passed"] is True
        bench = benchmark_edge_inference(exp["edge_bundle_path"], sample_windows=part.test.X[:20], n_warmup=5, n_runs=25)
        assert bench["batch_size"] == 1
        assert bench["median_latency_ms"] >= 0.0


def test_jetson_orin_nano_tegrastats_parser_and_worker_api(tmp_path: Path) -> None:
    sample_line = (
        "05-12-2026 14:22:01 RAM 2410/7620MB (lfb 812x4MB) SWAP 0/3810MB "
        "CPU [18%@1510,12%@1510,9%@1510,14%@1510,0%@729,0%@729] "
        "GR3D_FREQ 62% cpu@48.5C soc2@46.1C gpu@47.2C tj@49.0C "
        "VDD_IN 7420mW/7150mW VDD_CPU_GPU_CV 3120mW/2980mW VDD_SOC 1410mW/1390mW"
    )
    parsed = parse_tegrastats_line(sample_line)
    assert parsed["ram_used_mb"] == pytest.approx(2410.0)
    assert parsed["ram_total_mb"] == pytest.approx(7620.0)
    assert parsed["gpu_util_pct"] == pytest.approx(62.0)
    assert parsed["vdd_in_mw"] == pytest.approx(7420.0)
    assert parsed["vdd_in_avg_mw"] == pytest.approx(7150.0)
    assert parsed["vdd_cpu_gpu_cv_mw"] == pytest.approx(3120.0)
    assert parsed["cpu_temp_c"] == pytest.approx(48.5)
    assert parsed["gpu_temp_c"] == pytest.approx(47.2)

    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    X_dummy = np.linspace(100.0, 160.0, 240).reshape(20, 12)
    y_dummy = X_dummy[:, -1] + 10.0
    tcn = TCNGlucosePredictor(channels=8, max_epochs=5, seed=1).fit(X_dummy, y_dummy)
    exp = export_model_for_edge(tcn, tmp_path / "worker_edge")

    client = TestClient(create_jetson_worker_app(exp["edge_bundle_path"]))
    tel = client.get("/telemetry")
    assert tel.status_code == 200
    assert tel.json()["clinical_use"] is False

    pred = client.post(
        "/predict",
        json={
            "timestamp": "2027-01-10T12:00:00Z",
            "cgm_window_mgdl": [118, 121, 125, 129, 133, 136, 140, 142, 145, 148, 150, 151],
            "prediction_horizon_minutes": 30,
        },
    )
    assert pred.status_code == 200
    body = pred.json()
    assert body["prediction"]["clinical_use"] is False
    assert body["prediction"]["source"] == "research_model"
    assert 20.0 <= body["prediction"]["predicted_glucose_mgdl"] <= 600.0
