from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.medtronic_live import (
    MedtronicLiveConfig,
    medtronic_live_timeline_to_standard,
    normalize_medtronic_live_payload,
)
from iints.data.tidepool import MMOL_L_TO_MG_DL


def test_medtronic_live_payload_normalizes_glucose_meal_and_bolus() -> None:
    payload = {
        "data": [
            {
                "type": "sensor_glucose",
                "dateTime": "2026-05-25T10:00:00Z",
                "value": 6.2,
                "units": "mmol/L",
            },
            {
                "eventType": "meal",
                "eventTime": "2026-05-25T10:03:00Z",
                "carbInput": 35,
            },
            {
                "eventType": "bolus",
                "eventTime": "2026-05-25T10:04:00Z",
                "bolusAmount": 3.2,
            },
        ]
    }

    timeline = normalize_medtronic_live_payload(payload)

    assert len(timeline) == 1
    assert abs(float(timeline.loc[0, "glucose"]) - 6.2 * MMOL_L_TO_MG_DL) < 1e-6
    assert float(timeline.loc[0, "carbs"]) == 35.0
    assert float(timeline.loc[0, "insulin"]) == 3.2
    assert timeline.loc[0, "source"] == "medtronic_carelink_live"


def test_medtronic_live_standard_conversion_uses_relative_minutes() -> None:
    timeline = pd.DataFrame(
        [
            {
                "timestamp_dt": pd.Timestamp("2026-05-25T10:00:00Z"),
                "glucose": 120.0,
                "carbs": 0.0,
                "insulin": 0.0,
                "source": "medtronic_carelink_live",
            },
            {
                "timestamp_dt": pd.Timestamp("2026-05-25T10:05:00Z"),
                "glucose": 125.0,
                "carbs": 15.0,
                "insulin": 1.5,
                "source": "medtronic_carelink_live",
            },
        ]
    )

    standard = medtronic_live_timeline_to_standard(timeline)

    assert list(standard.columns) == ["timestamp", "glucose", "carbs", "insulin", "source"]
    assert standard["timestamp"].tolist() == [0.0, 5.0]
    assert standard["source"].unique().tolist() == ["medtronic_carelink_live"]


def test_medtronic_live_config_rejects_unsafe_urls_and_endpoint_paths() -> None:
    with pytest.raises(ValueError):
        MedtronicLiveConfig(base_url="file:///etc/passwd")

    with pytest.raises(ValueError):
        MedtronicLiveConfig(base_url="http://carelink.example")

    with pytest.raises(ValueError):
        MedtronicLiveConfig(
            base_url="https://relay.example",
            endpoint_path="https://other.example/carelink/live",
        )


def test_cli_medtronic_live_writes_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = pd.DataFrame(
        [
            {
                "timestamp_dt": pd.Timestamp("2026-05-25T10:00:00Z"),
                "glucose": 118.0,
                "carbs": 0.0,
                "insulin": 0.0,
                "source": "medtronic_carelink_live",
            }
        ]
    )

    def _fake_poll(*args, **kwargs):
        return [timeline]

    monkeypatch.setattr("iints.cli.cli.poll_medtronic_live_timeline", _fake_poll)
    output_dir = tmp_path / "live"
    result = CliRunner().invoke(
        app,
        [
            "medtronic-live",
            "--base-url",
            "http://localhost:9000",
            "--endpoint-path",
            "/carelink/live",
            "--output-dir",
            str(output_dir),
            "--samples",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "live_timeline.csv").is_file()
    assert (output_dir / "cgm_standard.csv").is_file()
    latest = json.loads((output_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["latest"]["glucose"] == 118.0
