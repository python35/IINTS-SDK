from __future__ import annotations

from iints.data.tidepool import MMOL_L_TO_MG_DL, _events_to_dataframe


def test_tidepool_events_convert_units_and_attach_nearby_events() -> None:
    dataframe = _events_to_dataframe(
        [
            {
                "type": "cbg",
                "time": "2026-05-17T08:00:00Z",
                "value": 6.0,
                "units": "mmol/L",
            },
            {
                "type": "wizard",
                "time": "2026-05-17T08:03:00Z",
                "carbInput": 42,
            },
            {
                "type": "bolus",
                "time": "2026-05-17T08:04:00Z",
                "normal": 3.5,
            },
        ],
        event_tolerance_minutes=7.5,
    )

    assert len(dataframe) == 1
    assert abs(float(dataframe.loc[0, "glucose"]) - 6.0 * MMOL_L_TO_MG_DL) < 1e-6
    assert float(dataframe.loc[0, "carbs"]) == 42.0
    assert float(dataframe.loc[0, "insulin"]) == 3.5


def test_tidepool_events_ignore_faraway_non_cgm_events() -> None:
    dataframe = _events_to_dataframe(
        [
            {
                "type": "cbg",
                "time": "2026-05-17T08:00:00Z",
                "value": 110,
                "units": "mg/dL",
            },
            {
                "type": "bolus",
                "time": "2026-05-17T08:30:00Z",
                "normal": 2.0,
            },
        ],
        event_tolerance_minutes=7.5,
    )

    assert float(dataframe.loc[0, "insulin"]) == 0.0
