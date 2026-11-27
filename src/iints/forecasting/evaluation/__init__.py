"""Evaluation & CGM robustness benchmarks for `iints.forecasting`."""

from iints.forecasting.evaluation.metrics import (
    compare_models_by_subject,
    evaluate_predictions,
    run_phase2_baseline_benchmark,
)
from iints.forecasting.evaluation.robustness import (
    DEFAULT_ROBUSTNESS_CONDITIONS,
    PerturbationSpec,
    perturb_windows,
    run_cgm_robustness_benchmark,
)

__all__ = [
    "evaluate_predictions",
    "compare_models_by_subject",
    "run_phase2_baseline_benchmark",
    "PerturbationSpec",
    "DEFAULT_ROBUSTNESS_CONDITIONS",
    "perturb_windows",
    "run_cgm_robustness_benchmark",
]
