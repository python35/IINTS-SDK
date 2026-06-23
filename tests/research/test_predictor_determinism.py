from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from iints.research.predictor import LSTMPredictor, PredictorService


def test_mc_dropout_uncertainty_is_repeatable_for_fixed_seed() -> None:
    torch.manual_seed(7)
    model = LSTMPredictor(
        input_size=3,
        hidden_size=8,
        num_layers=1,
        dropout=0.4,
        horizon_steps=2,
    )
    service = PredictorService(
        model,
        {
            "feature_columns": ["glucose", "iob", "cob"],
            "history_steps": 4,
            "horizon_steps": 2,
            "uncertainty_seed": 1234,
        },
    )
    features = np.ones((1, 4, 3), dtype=np.float32)

    first_mean, first_std = service.predict_with_uncertainty(features, n_samples=12)
    second_mean, second_std = service.predict_with_uncertainty(features, n_samples=12)

    np.testing.assert_array_equal(first_mean, second_mean)
    np.testing.assert_array_equal(first_std, second_std)
