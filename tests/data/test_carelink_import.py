from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.importer import (
    import_carelink_csv,
    import_carelink_timeline,
    load_carelink_event_log,
    summarize_carelink_csv,
)


CARELINK_HEADER = [
    "Index",
    "Date",
    "Time",
    "New Device Time",
    "BG Source",
    "BG Reading (mg/dL)",
    "Linked BG Meter ID",
    "Basal Rate (U/h)",
    "Temp Basal Amount",
    "Temp Basal Type",
    "Temp Basal Duration (h:mm:ss)",
    "Bolus Type",
    "Bolus Volume Selected (U)",
    "Bolus Volume Delivered (U)",
    "Bolus Duration (h:mm:ss)",
    "Prime Type",
    "Prime Volume Delivered (U)",
    "Estimated Reservoir Volume after Fill (U)",
    "Alert",
    "User Cleared Alerts",
    "Suspend",
    "Rewind",
    "BWZ Estimate (U)",
    "BWZ Target High BG (mg/dL)",
    "BWZ Target Low BG (mg/dL)",
    "BWZ Carb Ratio (g/U)",
    "BWZ Insulin Sensitivity (mg/dL/U)",
    "BWZ Carb Input (grams)",
    "BWZ BG/SG Input (mg/dL)",
    "BWZ Correction Estimate (U)",
    "BWZ Food Estimate (U)",
    "BWZ Active Insulin (U)",
    "BWZ Status",
    "Sensor Calibration BG (mg/dL)",
    "Sensor Glucose (mg/dL)",
    "ISIG Value",
    "Event Marker",
    "Bolus Number",
    "Bolus Cancellation Reason",
    "BWZ Unabsorbed Insulin Total (U)",
    "Final Bolus Estimate",
    "Scroll Step Size",
    "Insulin Action Curve Time",
    "Sensor Calibration Rejected Reason",
    "Preset Bolus",
    "Bolus Source",
    "BLE Network Device",
    "Device Update Event",
    "Network Device Associated Reason",
    "Network Device Disassociated Reason",
    "Network Device Disconnected Reason",
    "Sensor Exception",
    "Preset Temp Basal Name",
    "Sensor State",
]


def _write_carelink_sample(path: Path) -> None:
    preamble = [
        'Last Name;First Name;Patient ID;System ID;Start Date;End Date;Device;MiniMed 780G MMT-1886',
        '"Bobbaers";"Rune";"";"";"21/03/2026 00:00:00";"21/03/2026 23:59:00";"Serial Number";NG4052135H',
        'Patient DOB;;;;;;CGM;Simplera Sync™',
        "",
    ]
    rows = []

    def make_row(**values: str) -> list[str]:
        row = [""] * len(CARELINK_HEADER)
        for key, value in values.items():
            row[CARELINK_HEADER.index(key)] = value
        return row

    rows.append(
        make_row(
            Index="0,00000",
            Date="2026/03/21",
            Time="18:00:00",
            **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "140", "ISIG Value": "22,64"},
        )
    )
    rows.append(
        make_row(
            Index="1,00000",
            Date="2026/03/21",
            Time="18:05:00",
            **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "145", "ISIG Value": "23,10"},
        )
    )
    rows.append(
        make_row(
            Index="2,00000",
            Date="2026/03/21",
            Time="18:06:00",
            **{
                "Bolus Type": "Normal",
                "Bolus Volume Delivered (U)": "4,5",
                "BWZ Carb Input (grams)": "24,0",
                "BWZ BG/SG Input (mg/dL)": "145",
                "Bolus Source": "BOLUS_WIZARD",
            },
        )
    )
    rows.append(
        make_row(
            Index="3,00000",
            Date="2026/03/21",
            Time="18:10:00",
            **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "155", "ISIG Value": "24,00"},
        )
    )
    rows.append(
        make_row(
            Index="4,00000",
            Date="2026/03/21",
            Time="18:10:30",
            **{"Alert": "ALERT ON HIGH: tone and vibration"},
        )
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in preamble:
            handle.write(line + "\n")
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CARELINK_HEADER)
        writer.writerows(rows)


def test_import_carelink_csv_builds_standard_schema(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    _write_carelink_sample(sample)

    df = import_carelink_csv(sample)

    assert list(df.columns) == ["timestamp", "glucose", "carbs", "insulin", "source"]
    assert len(df) == 3
    assert df["source"].unique().tolist() == ["carelink_minimed"]
    assert abs(df["carbs"].sum() - 24.0) < 1e-6
    assert df["insulin"].max() > 4.5


def test_summarize_carelink_csv_reads_metadata(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    _write_carelink_sample(sample)

    summary = summarize_carelink_csv(sample)

    assert summary["patient_name"] == "Rune Bobbaers"
    assert summary["device"] == "MiniMed 780G MMT-1886"
    assert summary["cgm"] == "Simplera Sync™"
    assert summary["meal_rows"] == 1
    assert summary["bolus_rows"] == 1
    assert summary["alert_rows"] == 1


def test_load_carelink_event_log_returns_events_and_metadata(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    _write_carelink_sample(sample)

    raw_df, metadata = load_carelink_event_log(sample)

    assert len(raw_df) == 5
    assert "timestamp_dt" in raw_df.columns
    assert metadata["patient_name"] == "Rune Bobbaers"


def test_import_carelink_timeline_preserves_datetime_context(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    _write_carelink_sample(sample)

    timeline = import_carelink_timeline(sample)

    assert "timestamp_dt" in timeline.columns
    assert timeline["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S").iloc[0] == "2026-03-21T18:00:00"
    assert abs(timeline["carbs"].sum() - 24.0) < 1e-6
    assert timeline["insulin"].max() > 4.5


def test_cli_import_carelink_writes_outputs(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    _write_carelink_sample(sample)
    output_dir = tmp_path / "imported"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "import-carelink",
            "--input-csv",
            str(sample),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "cgm_standard.csv").is_file()
    assert (output_dir / "scenario.json").is_file()
    assert (output_dir / "carelink_summary.json").is_file()

    scenario = json.loads((output_dir / "scenario.json").read_text(encoding="utf-8"))
    assert len(scenario["stress_events"]) == 1
