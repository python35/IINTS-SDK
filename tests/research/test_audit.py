from __future__ import annotations

import pandas as pd

from iints.research.audit import audit_subject_split_and_leakage


def _make_subject_df() -> pd.DataFrame:
    rows = []
    for subject_idx, subject_id in enumerate(["S1", "S2", "S3", "S4", "S5", "S6"]):
        for t in range(40):
            rows.append(
                {
                    "subject_id": subject_id,
                    "segment_id": f"{subject_id}_seg",
                    "time_minutes": t * 5,
                    "glucose_actual_mgdl": 100 + subject_idx + (t * 0.1),
                    "patient_iob_units": 0.5,
                    "patient_cob_grams": 20.0,
                    "effective_isf": 50.0,
                    "effective_icr": 10.0,
                    "effective_basal_rate_u_per_hr": 0.8,
                    "glucose_trend_mgdl_min": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_audit_subject_split_and_leakage_reports_no_overlap() -> None:
    df = _make_subject_df()
    report = audit_subject_split_and_leakage(
        df,
        history_steps=12,
        horizon_steps=3,
        feature_columns=[
            "glucose_actual_mgdl",
            "patient_iob_units",
            "patient_cob_grams",
            "effective_isf",
            "effective_icr",
            "effective_basal_rate_u_per_hr",
            "glucose_trend_mgdl_min",
        ],
        target_column="glucose_actual_mgdl",
    )
    assert report["leakage_free"] is True
    assert report["sequence_overlap_counts"]["train_val"] == 0
