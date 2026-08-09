from __future__ import annotations

import numpy as np
import pytest

from iints.research.evaluation import (
    feature_drift_report,
    forecast_error_report,
    hypoglycemia_detection_report,
    subgroup_error_report,
    uncertainty_reliability_report,
)


def test_forecast_error_report_basic_metrics() -> None:
    observed = np.array([100.0, 110.0, 140.0, 190.0], dtype=float)
    predicted = np.array([102.0, 108.0, 150.0, 170.0], dtype=float)
    std = np.array([5.0, 5.0, 8.0, 10.0], dtype=float)

    report = forecast_error_report(observed, predicted, std)
    assert report["n"] == 4
    assert report["mae"] > 0
    assert "band_metrics" in report
    assert "interval_95_coverage_pct" in report


def test_forecast_error_report_false_hypo_alarm() -> None:
    observed = np.array([120.0, 130.0, 65.0], dtype=float)
    predicted = np.array([60.0, 80.0, 75.0], dtype=float)
    report = forecast_error_report(observed, predicted)
    assert report["false_hypo_alarm_rate_pct"] > 0
    assert report["missed_hypo_rate_pct"] > 0


def test_hypoglycemia_detection_report_counts_and_sensitivity() -> None:
    observed = np.array([60.0, 65.0, 110.0, 125.0], dtype=float)
    predicted = np.array([62.0, 80.0, 68.0, 130.0], dtype=float)

    report = hypoglycemia_detection_report(observed, predicted)

    assert report["counts"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 1,
    }
    assert report["sensitivity_pct"] == 50.0
    assert report["specificity_pct"] == 50.0


def test_uncertainty_reliability_report_bins_predictions() -> None:
    observed = np.array([100.0, 105.0, 110.0, 115.0], dtype=float)
    predicted = np.array([100.0, 104.0, 111.0, 130.0], dtype=float)
    std = np.array([2.0, 3.0, 5.0, 10.0], dtype=float)

    report = uncertainty_reliability_report(observed, predicted, std, bins=2)

    assert report["target_coverage_pct"] == 95.0
    assert len(report["bins"]) == 2
    assert sum(row["count"] for row in report["bins"]) == 4


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_forecast_reports_reject_nonfinite_values(bad_value: float) -> None:
    observed = np.array([100.0, bad_value], dtype=float)
    predicted = np.array([101.0, 102.0], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        forecast_error_report(observed, predicted)
    with pytest.raises(ValueError, match="finite"):
        hypoglycemia_detection_report(observed, predicted)


def test_uncertainty_reports_reject_negative_or_nonfinite_standard_deviation() -> None:
    observed = np.array([100.0, 105.0], dtype=float)
    predicted = np.array([101.0, 106.0], dtype=float)

    with pytest.raises(ValueError, match="non-negative"):
        forecast_error_report(observed, predicted, np.array([2.0, -1.0]))
    with pytest.raises(ValueError, match="finite"):
        uncertainty_reliability_report(observed, predicted, np.array([2.0, np.nan]))


def test_subgroup_error_report_splits_labels() -> None:
    observed = np.array([90.0, 100.0, 190.0, 200.0], dtype=float)
    predicted = np.array([95.0, 98.0, 175.0, 205.0], dtype=float)
    groups = np.array(["adult", "adult", "child", "child"], dtype=object)

    report = subgroup_error_report(observed, predicted, groups)

    assert set(report) == {"adult", "child"}
    assert report["adult"]["n"] == 2
    assert "hypoglycemia_detection" in report["child"]


def test_feature_drift_report_flags_shift_score() -> None:
    reference = np.array([[100.0, 0.0], [110.0, 1.0], [120.0, 2.0]], dtype=float)
    candidate = np.array([[130.0, 0.0], [140.0, 1.0], [150.0, 2.0]], dtype=float)

    report = feature_drift_report(reference, candidate, feature_names=["glucose", "carbs"])

    assert report["feature_count"] == 2
    assert report["max_robust_shift_score"] is not None
    assert report["features"][0]["feature"] == "glucose"
