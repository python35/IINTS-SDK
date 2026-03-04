from __future__ import annotations

import numpy as np

from iints.research.evaluation import forecast_error_report


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
