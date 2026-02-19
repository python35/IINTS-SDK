"""
Unit tests for iints.research.predictor

Covers:
- LSTMPredictor: forward pass shapes, MC Dropout uncertainty
- LastValueBaseline / LinearTrendBaseline: correctness and shapes
- evaluate_baselines: smoke test
- load_predictor / PredictorService: checkpoint round-trip (with scaler)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from iints.research.dataset import FeatureScaler
from iints.research.predictor import (
    LastValueBaseline,
    LinearTrendBaseline,
    LSTMPredictor,
    PredictorService,
    evaluate_baselines,
    load_predictor,
)


# ---------------------------------------------------------------------------
# LSTMPredictor
# ---------------------------------------------------------------------------

class TestLSTMPredictor:
    def _model(self, input_size: int = 4, horizon: int = 6) -> LSTMPredictor:
        return LSTMPredictor(input_size=input_size, hidden_size=16, num_layers=2,
                             dropout=0.1, horizon_steps=horizon)

    def test_forward_shape(self):
        model = self._model()
        x = torch.randn(8, 10, 4)  # [batch, time, features]
        out = model(x)
        assert out.shape == (8, 6)

    def test_forward_single_sample(self):
        model = self._model()
        x = torch.randn(1, 10, 4)
        out = model(x)
        assert out.shape == (1, 6)

    def test_mc_dropout_returns_mean_std(self):
        model = self._model()
        x = torch.randn(4, 10, 4)
        mean, std = model.predict_with_uncertainty(x, n_samples=20)
        assert mean.shape == (4, 6)
        assert std.shape == (4, 6)
        # Std should be non-negative
        assert (std >= 0).all()

    def test_mc_dropout_variance_nonzero(self):
        """With dropout > 0 and multiple passes, std should be > 0."""
        model = self._model()
        x = torch.randn(4, 10, 4)
        _, std = model.predict_with_uncertainty(x, n_samples=50)
        # At least some uncertainty should exist
        assert std.mean().item() > 0

    def test_eval_mode_deterministic(self):
        """In eval mode (no MC dropout), forward should be deterministic."""
        model = self._model()
        model.eval()
        x = torch.randn(4, 10, 4)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Checkpoint / PredictorService round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_save_load_predictor(self):
        model = LSTMPredictor(input_size=3, hidden_size=8, num_layers=1, dropout=0.0, horizon_steps=4)
        scaler = FeatureScaler("zscore")
        X_dummy = np.random.randn(50, 10, 3).astype(np.float32)
        scaler.fit(X_dummy)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            payload = {
                "state_dict": model.state_dict(),
                "config": {
                    "input_size": 3,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "horizon_steps": 4,
                    "history_steps": 10,
                    "time_step_minutes": 5,
                    "feature_columns": ["a", "b", "c"],
                    "target_column": "a",
                    "scaler": scaler.to_dict(),
                },
            }
            torch.save(payload, path)

            loaded_model, config = load_predictor(path)
            service = PredictorService(loaded_model, config)

        # Predict on raw (unscaled) input — PredictorService applies scaler internally
        X_new = np.random.randn(5, 10, 3).astype(np.float32)
        preds = service.predict(X_new)
        assert preds.shape == (5, 4)

    def test_predictor_service_uncertainty(self):
        model = LSTMPredictor(input_size=2, hidden_size=8, num_layers=1, dropout=0.2, horizon_steps=3)
        config = {
            "input_size": 2, "hidden_size": 8, "num_layers": 1, "dropout": 0.2,
            "horizon_steps": 3, "history_steps": 5, "time_step_minutes": 5,
            "feature_columns": ["g", "i"], "target_column": "g",
        }
        service = PredictorService(model, config)
        X = np.random.randn(4, 5, 2).astype(np.float32)
        mean, std = service.predict_with_uncertainty(X, n_samples=30)
        assert mean.shape == (4, 3)
        assert std.shape == (4, 3)
        assert (std >= 0).all()


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------

class TestLastValueBaseline:
    def test_output_shape(self):
        X = np.random.randn(10, 8, 3).astype(np.float32)
        bl = LastValueBaseline(horizon_steps=4)
        preds = bl.predict(X)
        assert preds.shape == (10, 4)

    def test_all_steps_equal_last_glucose(self):
        """Every horizon step must equal the last glucose in the window."""
        X = np.random.randn(5, 6, 3).astype(np.float32)
        bl = LastValueBaseline(horizon_steps=3)
        preds = bl.predict(X)
        last_glucose = X[:, -1, 0]
        for k in range(3):
            np.testing.assert_allclose(preds[:, k], last_glucose, rtol=1e-5)

    def test_dtype(self):
        X = np.random.randn(4, 6, 2).astype(np.float32)
        preds = LastValueBaseline(6).predict(X)
        assert preds.dtype == np.float32


class TestLinearTrendBaseline:
    def test_output_shape(self):
        X = np.random.randn(10, 8, 3).astype(np.float32)
        bl = LinearTrendBaseline(horizon_steps=4)
        preds = bl.predict(X)
        assert preds.shape == (10, 4)

    def test_constant_series_is_flat(self):
        """For a flat glucose series, linear trend should predict the same value."""
        N, T, F = 5, 10, 2
        X = np.ones((N, T, F), dtype=np.float32)
        X[:, :, 0] = 120.0  # constant glucose
        bl = LinearTrendBaseline(horizon_steps=3)
        preds = bl.predict(X)
        np.testing.assert_allclose(preds, 120.0, atol=1e-4)

    def test_rising_trend_extrapolates_upward(self):
        """A linearly rising glucose series should extrapolate upward."""
        T = 12
        X = np.zeros((1, T, 1), dtype=np.float32)
        X[0, :, 0] = np.arange(T, dtype=np.float32)  # glucose 0..11
        bl = LinearTrendBaseline(horizon_steps=3)
        preds = bl.predict(X)[0]
        assert preds[0] > X[0, -1, 0]
        assert preds[1] > preds[0]
        assert preds[2] > preds[1]

    def test_dtype(self):
        X = np.random.randn(4, 6, 2).astype(np.float32)
        preds = LinearTrendBaseline(4).predict(X)
        assert preds.dtype == np.float32


class TestEvaluateBaselines:
    def test_returns_both_keys(self):
        X = np.random.randn(20, 8, 3).astype(np.float32)
        y = np.random.randn(20, 4).astype(np.float32)
        result = evaluate_baselines(X, y, horizon_steps=4)
        assert "LastValue" in result
        assert "LinearTrend" in result

    def test_metrics_present(self):
        X = np.random.randn(20, 8, 3).astype(np.float32)
        y = np.random.randn(20, 4).astype(np.float32)
        result = evaluate_baselines(X, y, horizon_steps=4)
        for bname in result:
            assert "mae" in result[bname]
            assert "rmse" in result[bname]
            assert result[bname]["mae"] >= 0
            assert result[bname]["rmse"] >= 0

    def test_perfect_prediction_gives_zero_error(self):
        """Baseline should have zero MAE/RMSE when prediction equals target."""
        N, T, F = 10, 8, 1
        glucose = np.ones((N, T, F), dtype=np.float32) * 120.0
        y = np.ones((N, 4), dtype=np.float32) * 120.0  # target == last value
        result = evaluate_baselines(glucose, y, horizon_steps=4)
        np.testing.assert_allclose(result["LastValue"]["mae"], 0.0, atol=1e-5)
        np.testing.assert_allclose(result["LastValue"]["rmse"], 0.0, atol=1e-5)
