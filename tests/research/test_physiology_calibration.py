from __future__ import annotations

import pandas as pd

from iints.research.physiology_calibration import (
    physiology_calibration_report,
    resolve_calibration_columns,
    standardize_calibration_dataframe,
)


def _synthetic_real_dataset() -> pd.DataFrame:
    rows = []
    for subject_id in ("p1", "p2"):
        for minute in range(0, 24 * 60, 5):
            hour = (minute % 1440) / 60.0
            meal = 0.0
            if minute in {8 * 60, 13 * 60, 19 * 60}:
                meal = 45.0 if minute != 19 * 60 else 65.0
            exercise = 1.0 if 16 * 60 <= minute < 17 * 60 else 0.0
            dawn = 18.0 if 4 <= hour < 8 else 0.0
            meal_wave = 0.0
            for meal_min, grams in ((8 * 60, 45.0), (13 * 60, 45.0), (19 * 60, 65.0)):
                dt = minute - meal_min
                if 0 <= dt <= 240:
                    meal_wave += (grams / 10.0) * 8.0 * (dt / 70.0) * (2.718281828 ** (-dt / 90.0))
            exercise_drop = -18.0 if 16 * 60 <= minute < 18 * 60 else 0.0
            glucose = 118.0 + dawn + meal_wave + exercise_drop
            rows.append(
                {
                    "subject_id": subject_id,
                    "time_minutes": minute + (0 if subject_id == "p1" else 1440),
                    "glucose_actual_mgdl": glucose,
                    "carb_grams": meal,
                    "insulin_units": meal / 10.0 if meal else 0.0,
                    "exercise_flag": exercise,
                }
            )
    return pd.DataFrame(rows)


def test_physiology_calibration_report_generates_conservative_hints() -> None:
    raw = _synthetic_real_dataset()
    report = physiology_calibration_report(raw)

    glucose = report["real_dataset"]["glucose_summary"]
    meal = report["real_dataset"]["meal_response_summary"]
    hints = report["calibration"]["patient_profile_hints"]

    assert glucose["rows"] == len(raw)
    assert glucose["subjects"] == 2
    assert meal["eligible_meal_count"] >= 4
    assert 80.0 <= hints["initial_glucose"] <= 220.0
    assert 120.0 <= hints["carb_absorption_duration_minutes"] <= 420.0
    assert 1.0 <= hints["max_glucose_rate_mgdl_per_min"] <= 4.0
    assert report["purpose"].startswith("research physiology calibration")


def test_calibration_standardizer_resolves_ohio_style_columns() -> None:
    raw = _synthetic_real_dataset().rename(columns={"glucose_actual_mgdl": "glucose"})
    columns = resolve_calibration_columns(raw)
    standardized = standardize_calibration_dataframe(raw, columns)

    assert columns.glucose == "glucose"
    assert set(["time_minutes", "glucose_mgdl", "carbs_g", "insulin_u", "exercise_flag"]).issubset(
        standardized.columns
    )
    assert standardized["glucose_mgdl"].between(20, 600).all()
