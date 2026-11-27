"""Phase 4 — Controlled CGM Robustness Benchmark (`iints.forecasting.evaluation.robustness`).

Evaluates trained forecasting models (`Persistence`, `Ridge`, `GRU`, `TCN`) across
four pre-registered sensor degradation families injected onto clean held-out test
windows while always scoring against the **unperturbed true target `y_true`**:

- **Condition A — Clean (`clean`)**: Original valid 12-sample CGM windows.
- **Condition B — Dropout (`dropout_5m`, `dropout_15m`, `dropout_30m`)**: Drops the
  final 1, 3, or 6 samples up to `t` and applies explicit Last Observation Carried
  Forward (`LOCF`), tagging `input_quality='locf_imputed'`.
- **Condition C — Sensor Noise (`noise_5mgdl`, `noise_10mgdl`, `noise_15mgdl`)**:
  Adds deterministic seeded zero-mean AR(1) colored noise (`phi=0.70`).
- **Condition D — Sensor Delay (`delay_5m`, `delay_10m`, `delay_15m`)**: Shifts input
  observations backward by 1, 2, or 3 steps (5, 10, 15 minutes).

Pre-existing missingness in the raw dataset is already excluded from `test` windows,
cleanly isolating controlled synthetic degradation from native dataset gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from iints.forecasting.data.windows import SplitWindows
from iints.forecasting.evaluation.metrics import compare_models_by_subject, evaluate_predictions
from iints.forecasting.models.base import WindowForecaster
from iints.research.statistics import holm_bonferroni
from iints.sdk2.reproducibility import derive_seed

__all__ = [
    "PerturbationSpec",
    "DEFAULT_ROBUSTNESS_CONDITIONS",
    "perturb_windows",
    "run_cgm_robustness_benchmark",
]


@dataclass(frozen=True)
class PerturbationSpec:
    condition_id: str
    family: Literal["clean", "dropout_locf", "sensor_noise", "sensor_delay"]
    severity_value: float = 0.0  # minutes for dropout/delay; mg/dL SD for noise
    input_quality: Literal["complete", "locf_imputed", "degraded"] = "complete"


DEFAULT_ROBUSTNESS_CONDITIONS: tuple[PerturbationSpec, ...] = (
    PerturbationSpec("A_clean", "clean", 0.0, "complete"),
    PerturbationSpec("B_dropout_5m", "dropout_locf", 5.0, "locf_imputed"),
    PerturbationSpec("B_dropout_15m", "dropout_locf", 15.0, "locf_imputed"),
    PerturbationSpec("B_dropout_30m", "dropout_locf", 30.0, "locf_imputed"),
    PerturbationSpec("C_noise_5mgdl", "sensor_noise", 5.0, "degraded"),
    PerturbationSpec("C_noise_10mgdl", "sensor_noise", 10.0, "degraded"),
    PerturbationSpec("C_noise_15mgdl", "sensor_noise", 15.0, "degraded"),
    PerturbationSpec("D_delay_5m", "sensor_delay", 5.0, "degraded"),
    PerturbationSpec("D_delay_10m", "sensor_delay", 10.0, "degraded"),
    PerturbationSpec("D_delay_15m", "sensor_delay", 15.0, "degraded"),
)


def perturb_windows(
    X_clean: np.ndarray,
    spec: PerturbationSpec,
    *,
    sample_interval_minutes: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(X_perturbed, observed_mask)` of shape `(N, L)`.

    `observed_mask` is `1.0` where original reading is retained and `0.0` where
    a reading was dropped and LOCF-imputed.
    """
    arr = np.asarray(X_clean, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError(f"perturb_windows expects 2-D [N, L], got {arr.shape}")
    n, L = arr.shape
    mask = np.ones((n, L), dtype=np.float64)

    if spec.family == "clean" or spec.severity_value <= 0.0:
        return arr, mask

    if spec.family == "dropout_locf":
        drop_steps = min(L - 1, max(1, int(round(spec.severity_value / sample_interval_minutes))))
        anchor_col = L - 1 - drop_steps
        arr[:, anchor_col + 1 :] = arr[:, anchor_col : anchor_col + 1]
        mask[:, anchor_col + 1 :] = 0.0
        return arr, mask

    if spec.family == "sensor_noise":
        rng = np.random.default_rng(derive_seed(seed, "cgm_robustness_noise", spec.condition_id))
        phi = 0.70
        sd = float(spec.severity_value)
        innov_sd = sd * np.sqrt(1.0 - phi * phi)
        noise = np.empty((n, L), dtype=np.float64)
        noise[:, 0] = rng.normal(0.0, sd, size=n)
        for k in range(1, L):
            noise[:, k] = phi * noise[:, k - 1] + rng.normal(0.0, innov_sd, size=n)
        return np.clip(arr + noise, 20.0, 600.0), mask

    if spec.family == "sensor_delay":
        shift_steps = min(L - 1, max(1, int(round(spec.severity_value / sample_interval_minutes))))
        delayed = np.empty_like(arr)
        delayed[:, :shift_steps] = arr[:, 0:1]
        delayed[:, shift_steps:] = arr[:, : L - shift_steps]
        return delayed, mask

    raise ValueError(f"Unsupported perturbation family '{spec.family}'")


def run_cgm_robustness_benchmark(
    test_split: SplitWindows,
    models: Mapping[str, WindowForecaster],
    *,
    conditions: Sequence[PerturbationSpec] = DEFAULT_ROBUSTNESS_CONDITIONS,
    sample_interval_minutes: float = 5.0,
    seed: int = 42,
    is_synthetic: bool = False,
    dataset_id: str = "unknown",
    output_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Evaluate all `models` across `conditions` against unperturbed `test_split.y`."""
    clean_candidates = [c for c in conditions if c.family == "clean"]
    if not clean_candidates:
        raise ValueError("conditions must include a 'clean' reference PerturbationSpec.")
    clean_id = clean_candidates[0].condition_id

    cond_evals: Dict[tuple[str, str], Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for cond in conditions:
        x_pert, _ = perturb_windows(
            test_split.X,
            cond,
            sample_interval_minutes=sample_interval_minutes,
            seed=seed,
        )
        pert_split = SplitWindows(
            X=x_pert,
            y=test_split.y,  # ALWAYS evaluate against original unperturbed reference truth!
            subject_ids=test_split.subject_ids,
            window_start_ts=test_split.window_start_ts,
            origin_ts=test_split.origin_ts,
            target_ts=test_split.target_ts,
        )
        for mid, model in models.items():
            preds = model.predict(x_pert)
            ev = evaluate_predictions(
                pert_split,
                preds,
                model_id=mid,
                sample_interval_minutes=sample_interval_minutes,
            )
            cond_evals[(cond.condition_id, mid)] = ev
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "is_synthetic": is_synthetic,
                    "condition_id": cond.condition_id,
                    "family": cond.family,
                    "severity_value": cond.severity_value,
                    "model_id": mid,
                    **ev["pooled"],
                    **ev["subject_distribution"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    # Within-subject paired performance loss vs Clean reference condition
    degradation_rows: List[Dict[str, Any]] = []
    for cond in conditions:
        if cond.condition_id == clean_id:
            continue
        for mid in models:
            comp = compare_models_by_subject(
                cond_evals[(clean_id, mid)],
                cond_evals[(cond.condition_id, mid)],
                seed=seed,
            )
            degradation_rows.append(
                {
                    "condition_id": cond.condition_id,
                    "family": cond.family,
                    "severity_value": cond.severity_value,
                    "model_id": mid,
                    "n_subjects": comp["n_subjects"],
                    "clean_mean_subject_mae": cond_evals[(clean_id, mid)]["subject_distribution"]["mean_subject_mae"],
                    "perturbed_mean_subject_mae": cond_evals[(cond.condition_id, mid)]["subject_distribution"][
                        "mean_subject_mae"
                    ],
                    "mean_delta_mae": comp["mean_delta_mae"],
                    "ci95_delta_low": comp["ci95_delta_low"],
                    "ci95_delta_high": comp["ci95_delta_high"],
                    "median_delta_mae": comp["median_delta_mae"],
                    "wilcoxon_p_raw": comp["wilcoxon_p_value"],
                }
            )

    degradation_df = pd.DataFrame(degradation_rows)
    if not degradation_df.empty:
        degradation_df["p_holm_bonferroni"] = holm_bonferroni(degradation_df["wilcoxon_p_raw"].tolist())

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_p / "robustness_condition_summary.csv", index=False)
        degradation_df.to_csv(out_p / "robustness_paired_degradation.csv", index=False)

    return {
        "clean_condition_id": clean_id,
        "condition_summary": summary_df,
        "paired_degradation": degradation_df,
    }
