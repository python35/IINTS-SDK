from __future__ import annotations

import random

from iints.research.jetson_hf_trainer import _trial_config, model_score


def test_model_score_penalizes_physiology_and_hypo_risk() -> None:
    clean = {
        "mae": 20.0,
        "missed_hypo_rate_pct": 1.0,
        "any_physiology_violation_pct": 1.0,
    }
    risky = {
        "mae": 19.0,
        "missed_hypo_rate_pct": 8.0,
        "any_physiology_violation_pct": 25.0,
    }

    assert model_score(clean) < model_score(risky)


def test_trial_config_preserves_warm_start_architecture() -> None:
    base = {
        "predictor": {
            "history_minutes": 360,
            "horizon_minutes": 120,
            "time_step_minutes": 5,
            "feature_columns": ["glucose_actual_mgdl"],
            "target_column": "glucose_actual_mgdl",
        },
        "training": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "loss": "pinn",
        },
    }

    cfg = _trial_config(
        base,
        rng=random.Random(42),
        trial_index=1,
        seed=42,
        epochs=3,
        batch_size=32,
        min_lr=1e-5,
        max_lr=1e-4,
        min_pinn_lambda=0.1,
        max_pinn_lambda=0.2,
        weight_decay_choices=[0.0],
    )

    assert cfg["training"]["hidden_size"] == 128
    assert cfg["training"]["num_layers"] == 2
    assert cfg["training"]["epochs"] == 3
    assert cfg["training"]["batch_size"] == 32
    assert 1e-5 <= cfg["training"]["learning_rate"] <= 1e-4
    assert 0.1 <= cfg["training"]["pinn_lambda"] <= 0.2
