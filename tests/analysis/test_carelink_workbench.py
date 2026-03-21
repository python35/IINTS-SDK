from __future__ import annotations

import csv
import json
from pathlib import Path

from iints.analysis.carelink_workbench import build_carelink_workbench


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

    rows.extend(
        [
            make_row(
                Index="0,00000",
                Date="2026/03/21",
                Time="18:00:00",
                **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "140"},
            ),
            make_row(
                Index="1,00000",
                Date="2026/03/21",
                Time="18:05:00",
                **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "145"},
            ),
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
            ),
            make_row(
                Index="3,00000",
                Date="2026/03/21",
                Time="18:10:00",
                **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "95"},
            ),
            make_row(
                Index="4,00000",
                Date="2026/03/21",
                Time="18:15:00",
                **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "62", "Alert": "ALERT ON LOW"},
            ),
            make_row(
                Index="5,00000",
                Date="2026/03/21",
                Time="18:20:00",
                **{"Basal Rate (U/h)": "1,2", "Sensor Glucose (mg/dL)": "88", "Sensor Exception": "SENSOR_ERROR"},
            ),
        ]
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in preamble:
            handle.write(line + "\n")
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CARELINK_HEADER)
        writer.writerows(rows)


def test_build_carelink_workbench_generates_dashboard_and_ai_payloads(tmp_path: Path) -> None:
    sample = tmp_path / "carelink.csv"
    workbench = tmp_path / "workbench"
    _write_carelink_sample(sample)

    outputs = build_carelink_workbench(
        sample,
        output_dir=workbench,
        create_dev_mdmp_cert=False,
    )

    assert Path(outputs["standard_csv"]).is_file()
    assert Path(outputs["scenario"]).is_file()
    assert Path(outputs["dashboard_png"]).is_file()
    assert Path(outputs["poster_png"]).is_file()
    assert Path(outputs["dashboard_html"]).is_file()
    assert Path(outputs["report_payload"]).is_file()
    assert Path(outputs["trends_payload"]).is_file()
    assert Path(outputs["anomalies_payload"]).is_file()
    assert Path(outputs["step_riskiest"]).is_file()

    metrics = json.loads((workbench / "carelink_metrics.json").read_text(encoding="utf-8"))
    assert metrics["reading_count"] >= 5
    assert metrics["time_below_70_pct"] > 0

    report_payload = json.loads((workbench / "ai" / "report_payload.json").read_text(encoding="utf-8"))
    assert report_payload["source"] == "carelink_personal_workbench"
    assert report_payload["summary"]["alert_count"] == 1
