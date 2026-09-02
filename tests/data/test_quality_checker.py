import pytest
import pandas as pd
import numpy as np
from iints.data.quality_checker import DataQualityChecker, DataAnomaly, QualityReport

def test_physiological_rapid_change_detection():
    """
    Test that DataQualityChecker flags implausibly fast glucose changes.

    The threshold is the measured one (see core/safety/config.py): 11 mg/dL/min,
    set above the 99.9th percentile of real 5-minute steps in AZT1D, HUPA-UCM and
    OhioT1DM. This test used to assert that 5 mg/dL/min was "physiologically
    impossible", which the cohort measurements contradict — rates that high occur
    in roughly 1% of genuine 5-minute steps.
    """
    checker = DataQualityChecker(expected_interval=1) # Set interval to 1 minute for this test

    # A 15 mg/dL/min rise is above the measured ceiling and should be flagged.
    data = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 115, 116, 117]
    }
    df = pd.DataFrame(data)

    # Run the validity check
    validity_score, anomalies = checker.check_validity(df)

    # Assert that a rapid_change anomaly is detected
    assert any(
        a.anomaly_type == 'rapid_change' and
        "Implausible glucose rise of 15.0 mg/dL/min" in a.description
        for a in anomalies
    ), "Implausibly fast glucose rise was not detected."

    # The same in the falling direction.
    data_decrease = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [130, 115, 114, 113]
    }
    df_decrease = pd.DataFrame(data_decrease)

    validity_score_dec, anomalies_dec = checker.check_validity(df_decrease)

    assert any(
        a.anomaly_type == 'rapid_change' and
        "Implausible glucose drop of -15.0 mg/dL/min" in a.description
        for a in anomalies_dec
    ), "Implausibly fast glucose drop was not detected."

    # A 7 mg/dL/min step is fast but occurs in the real cohorts, so it must pass.
    # This is the case the old 4 mg/dL/min ceiling wrongly flagged.
    data_safe = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 107, 111, 115]
    }
    df_safe = pd.DataFrame(data_safe)

    validity_score_safe, anomalies_safe = checker.check_validity(df_safe)

    assert not any(a.anomaly_type == 'rapid_change' for a in anomalies_safe), \
        "A step that occurs in the real cohorts was flagged as implausible."

def test_overall_report_with_rapid_change_anomaly():
    """
    Test that the overall quality report includes warnings for rapid glucose changes
    and that the overall score is affected.
    """
    checker = DataQualityChecker(expected_interval=1)

    data = {
        'timestamp': [0, 1, 2, 3],
        'glucose': [100, 115, 116, 117] # 15 mg/dL/min, above the measured ceiling
    }
    df = pd.DataFrame(data)

    report = checker.check(df)

    assert report.overall_score < 1.0, "Overall score should be affected by rapid change anomaly."
    assert any("CRITICAL ANOMALY: Implausible glucose rise of 15.0 mg/dL/min" in w for w in report.warnings), \
        "Warning for rapid change not found in report."
