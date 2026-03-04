from __future__ import annotations

import numpy as np

from iints.research.metrics import (
    band_regression_metrics,
    interval_coverage_metrics,
    regression_metrics,
)


def test_regression_metrics_basic() -> None:
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0], [5.0, 4.0]], dtype=np.float32)
    m = regression_metrics(y_true, y_pred)
    assert abs(m["mae"] - 0.75) < 1e-6
    assert abs(m["rmse"] - np.sqrt(1.25)) < 1e-6
    assert abs(m["bias"] - 0.25) < 1e-6


def test_band_regression_metrics_counts() -> None:
    y_true = np.array([[60.0, 100.0, 220.0]], dtype=np.float32)
    y_pred = np.array([[65.0, 95.0, 210.0]], dtype=np.float32)
    bands = band_regression_metrics(y_true, y_pred)
    assert bands["hypo_lt_70"]["count"] == 1
    assert bands["in_range_70_180"]["count"] == 1
    assert bands["hyper_gt_180"]["count"] == 1
    assert abs(bands["hypo_lt_70"]["mae"] - 5.0) < 1e-6


def test_interval_coverage_metrics_reports_global_and_band() -> None:
    y_true = np.array([[80.0, 120.0, 220.0]], dtype=np.float32)
    mean = np.array([[80.0, 120.0, 220.0]], dtype=np.float32)
    std = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    cov = interval_coverage_metrics(y_true, mean, std, confidence=0.95)
    assert abs(cov["overall_coverage"] - 1.0) < 1e-6
    assert cov["bands"]["hypo_lt_70"]["count"] == 0
    assert cov["bands"]["in_range_70_180"]["count"] == 2
    assert cov["bands"]["hyper_gt_180"]["count"] == 1
