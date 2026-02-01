import pytest
import pandas as pd
import numpy as np
from src.data.quality_checker import DataQualityChecker, DataAnomaly, QualityReport

def test_physiological_rapid_change_detection():
    """
    Test that DataQualityChecker detects physiologically impossible rapid glucose changes.
    """
    checker = DataQualityChecker(expected_interval=1) # Set interval to 1 minute for this test

    # Create a DataFrame with a rapid glucose increase (20 mg/dL in 1 minute)
    data = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 120, 121, 122] # 20 mg/dL in 1 minute from 0 to 1
    }
    df = pd.DataFrame(data)

    # Run the validity check
    validity_score, anomalies = checker.check_validity(df)

    # Assert that a rapid_change anomaly is detected
    assert any(
        a.anomaly_type == 'rapid_change' and
        "Impossible glucose rise of 20.0 mg/dL/min" in a.description
        for a in anomalies
    ), "Physiologically impossible rapid glucose rise was not detected."

    # Test a rapid glucose decrease (20 mg/dL in 1 minute)
    data_decrease = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [120, 100, 99, 98] # 20 mg/dL in 1 minute from 0 to 1
    }
    df_decrease = pd.DataFrame(data_decrease)

    validity_score_dec, anomalies_dec = checker.check_validity(df_decrease)

    assert any(
        a.anomaly_type == 'rapid_change' and
        "Impossible glucose drop of -20.0 mg/dL/min" in a.description
        for a in anomalies_dec
    ), "Physiologically impossible rapid glucose drop was not detected."

    # Test with a change below the threshold (should not trigger anomaly)
    data_safe = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 110, 111, 112] # 10 mg/dL in 1 minute
    }
    df_safe = pd.DataFrame(data_safe)

    validity_score_safe, anomalies_safe = checker.check_validity(df_safe)

    assert not any(a.anomaly_type == 'rapid_change' for a in anomalies_safe), \
        "Rapid glucose change anomaly detected for a safe change."

def test_overall_report_with_rapid_change_anomaly():
    """
    Test that the overall quality report includes warnings for rapid glucose changes
    and that the overall score is affected.
    """
    checker = DataQualityChecker(expected_interval=1)

    data = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 120, 121, 122] # Rapid change here
    }
    df = pd.DataFrame(data)

    report = checker.check(df)

    assert report.overall_score < 1.0, "Overall score should be affected by rapid change anomaly."
    assert any("CRITICAL ANOMALY: Impossible glucose rise of 20.0 mg/dL/min" in w for w in report.warnings), \
        "Warning for rapid change not found in report."
