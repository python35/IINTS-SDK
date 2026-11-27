"""Phase 3 — Multi-Seed Neural Training & Subject-Level Comparison Engine.

Trains `Persistence`, `Ridge`, `GRU` (across 5 initialization seeds), and `TCN`
(across 5 initialization seeds) under identical embargoed windows:
- Hyperparameters & early stopping use `val` ONLY; `test` is strictly held out.
- Multi-seed repetitions (`[11, 23, 42, 67, 89]`) quantify initialization sensitivity
  (`seed_std_mae`) and are **averaged within each subject first** before cross-subject
  Wilcoxon / bootstrap testing so `N` is never inflated by seed count!
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from iints.forecasting.data.windows import PartitionedWindowDataset
from iints.forecasting.models.base import WindowForecaster
from iints.forecasting.models.gru import GRUGlucosePredictor
from iints.forecasting.models.persistence import PersistencePredictor
from iints.forecasting.models.ridge import RidgeCGMPredictor
from iints.forecasting.models.tcn import TCNGlucosePredictor
from iints.forecasting.schemas import DataQualityReport
from iints.forecasting.training.registry import ForecastingModelRegistry
from iints.sdk2.reproducibility import sha256_json

__all__ = ["NeuralTrainingConfig", "train_and_compare_all_models"]


@dataclass(frozen=True)
class NeuralTrainingConfig:
    seeds: Tuple[int, ...] | Sequence[int] = (11, 23, 42, 67, 89)
    gru_hidden_size: int = 32
    tcn_channels: int = 16
    tcn_kernel_size: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 50
    early_stopping_patience: int = 8
    backend: Literal["auto", "torch", "numpy"] = "auto"
    device: str = "auto"


def train_and_compare_all_models(
    partitioned: PartitionedWindowDataset,
    quality: DataQualityReport,
    output_dir: Union[str, Path],
    *,
    train_cfg: NeuralTrainingConfig | None = None,
) -> Dict[str, Any]:
    """Train Persistence, Ridge, 5-seed GRU, and 5-seed TCN; register & compare cleanly."""
    from iints.forecasting.evaluation.metrics import compare_models_by_subject, evaluate_predictions

    cfg = train_cfg or NeuralTrainingConfig()
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    registry = ForecastingModelRegistry(out_root / "model_registry")
    ds_hash = sha256_json(quality.model_dump(mode="json"))
    wspec = partitioned.spec
    splits = partitioned.split_counts()

    tr_X, tr_y = partitioned.train_fit.X, partitioned.train_fit.y
    va_X = partitioned.val.X if len(partitioned.val) > 0 else None
    va_y = partitioned.val.y if len(partitioned.val) > 0 else None
    te_split = partitioned.test

    # 1. Deterministic Baselines
    pers = PersistencePredictor().fit(tr_X, tr_y)
    ridge = RidgeCGMPredictor().fit(tr_X, tr_y, va_X, va_y)

    evaluations: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, WindowForecaster] = {
        pers.model_id: pers,
        ridge.model_id: ridge,
    }
    seed_sensitivity_rows: List[Dict[str, Any]] = []

    for base_m in (pers, ridge):
        v_m = (
            evaluate_predictions(partitioned.val, base_m.predict(va_X), model_id=base_m.model_id)["pooled"]
            if va_X is not None
            else {}
        )
        t_ev = evaluate_predictions(te_split, base_m.predict(te_split.X), model_id=base_m.model_id)
        evaluations[base_m.model_id] = t_ev
        registry.register(
            base_m,
            dataset_id=quality.dataset_id,
            is_synthetic=quality.is_synthetic,
            benchmark_tier=quality.benchmark_tier,
            dataset_sha256=ds_hash,
            input_length=wspec.input_length,
            elapsed_input_span_minutes=wspec.elapsed_span_minutes,
            horizon_minutes=wspec.prediction_horizon_minutes,
            training_config={"type": "baseline"},
            split_counts=splits,
            validation_metrics=v_m,
            test_metrics_available=True,
        )

    # 2. Multi-seed GRU & TCN
    for arch in ("gru", "tcn"):
        seed_test_preds: List[np.ndarray] = []
        seed_val_maes: List[float] = []
        seed_test_maes: List[float] = []
        best_val_mae = float("inf")
        best_seed_model: WindowForecaster | None = None

        for s in cfg.seeds:
            model_id_seed = f"iints-{arch}-glucose-v1-seed{int(s)}"
            if arch == "gru":
                m: WindowForecaster = GRUGlucosePredictor(
                    hidden_size=cfg.gru_hidden_size,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size,
                    max_epochs=cfg.max_epochs,
                    early_stopping_patience=cfg.early_stopping_patience,
                    seed=int(s),
                    backend=cfg.backend,
                    device=cfg.device,
                    model_id=model_id_seed,
                )
            else:
                m = TCNGlucosePredictor(
                    channels=cfg.tcn_channels,
                    kernel_size=cfg.tcn_kernel_size,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size,
                    max_epochs=cfg.max_epochs,
                    early_stopping_patience=cfg.early_stopping_patience,
                    seed=int(s),
                    backend=cfg.backend,
                    device=cfg.device,
                    model_id=model_id_seed,
                )

            m.fit(tr_X, tr_y, va_X, va_y)
            v_mae = (
                float(np.mean(np.abs(m.predict(va_X) - va_y)))
                if va_X is not None and va_y is not None
                else float("nan")
            )
            t_pred = m.predict(te_split.X)
            t_mae = float(np.mean(np.abs(t_pred - te_split.y)))
            seed_test_preds.append(t_pred)
            seed_val_maes.append(v_mae)
            seed_test_maes.append(t_mae)

            seed_sensitivity_rows.append(
                {
                    "architecture": arch,
                    "seed": int(s),
                    "val_mae": v_mae,
                    "test_mae": t_mae,
                    "backend": getattr(m, "active_backend", "unknown"),
                }
            )
            if np.isnan(v_mae) or v_mae < best_val_mae:
                best_val_mae = v_mae
                best_seed_model = m

        assert best_seed_model is not None
        # Register canonical best-validation-seed model for single-model Jetson/ONNX export!
        canonical_id = f"iints-{arch}-glucose-v1"
        best_seed_model.model_id = canonical_id
        trained_models[canonical_id] = best_seed_model
        registry.register(
            best_seed_model,
            dataset_id=quality.dataset_id,
            is_synthetic=quality.is_synthetic,
            benchmark_tier=quality.benchmark_tier,
            dataset_sha256=ds_hash,
            input_length=wspec.input_length,
            elapsed_input_span_minutes=wspec.elapsed_span_minutes,
            horizon_minutes=wspec.prediction_horizon_minutes,
            training_config={
                "seeds_evaluated": list(cfg.seeds),
                "selected_by": "lowest_validation_mae",
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
            },
            split_counts=splits,
            validation_metrics={"mae": best_val_mae} if np.isfinite(best_val_mae) else {},
            test_metrics_available=True,
        )

        # Evaluate BOTH the validation-selected single checkpoint AND the 5-seed mean prediction
        evaluations[canonical_id] = evaluate_predictions(
            te_split,
            best_seed_model.predict(te_split.X),
            model_id=canonical_id,
        )
        mean_pred_across_seeds = np.mean(np.vstack(seed_test_preds), axis=0)
        evaluations[f"{canonical_id}-5seed-mean"] = evaluate_predictions(
            te_split,
            mean_pred_across_seeds,
            model_id=f"{canonical_id}-5seed-mean",
        )

    seed_df = pd.DataFrame(seed_sensitivity_rows)
    seed_df.to_csv(out_root / "neural_seed_sensitivity.csv", index=False)

    comparisons_vs_persistence = {
        mid: compare_models_by_subject(evaluations[pers.model_id], ev)
        for mid, ev in evaluations.items()
        if mid != pers.model_id
    }
    comparisons_vs_ridge = {
        mid: compare_models_by_subject(evaluations[ridge.model_id], ev)
        for mid, ev in evaluations.items()
        if mid not in (pers.model_id, ridge.model_id)
    }

    summary_rows = [
        {
            "dataset_id": quality.dataset_id,
            "is_synthetic": quality.is_synthetic,
            "model_id": mid,
            **ev["pooled"],
            **ev["subject_distribution"],
        }
        for mid, ev in evaluations.items()
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("mae").reset_index(drop=True)
    summary_df.to_csv(out_root / "all_models_test_summary.csv", index=False)

    (out_root / "training_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": quality.dataset_id,
                "is_synthetic": quality.is_synthetic,
                "seeds": list(cfg.seeds),
                "models_registered": [e.model_id for e in registry.list_entries()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "models": trained_models,
        "evaluations": evaluations,
        "summary_table": summary_df,
        "seed_sensitivity": seed_df,
        "vs_persistence": comparisons_vs_persistence,
        "vs_ridge": comparisons_vs_ridge,
        "registry": registry,
    }
