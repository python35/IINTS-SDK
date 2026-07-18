from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

from iints.validation import validate_patient_config_dict, validate_scenario_dict
from iints_desktop.engine import DesktopRunHistoryEntry
from iints_desktop.qt_app import IINTSQtDesktopApp


@pytest.fixture
def qt_window() -> IINTSQtDesktopApp:
    _app = QApplication.instance() or QApplication([])
    window = IINTSQtDesktopApp()
    yield window
    window.close()


def test_scenario_builder_emits_sdk_valid_payload(qt_window: IINTSQtDesktopApp) -> None:
    qt_window._add_custom_meal()

    payload = qt_window._get_custom_preset_dict()

    validate_patient_config_dict(payload["patient_config"])
    scenario = validate_scenario_dict(payload["scenario"])
    assert scenario.stress_events[0].event_type == "meal"
    assert hasattr(qt_window, "_run_custom_payload")
    assert hasattr(qt_window, "_handle_success")
    assert hasattr(qt_window, "_handle_error")


def test_structural_outputs_follow_selected_workspace(
    qt_window: IINTSQtDesktopApp,
    tmp_path: Path,
) -> None:
    qt_window.output_dir.setText(str(tmp_path))

    assert qt_window._structural_output_dir() == (tmp_path / "structural").resolve()


def test_compare_selected_runs_accepts_current_sdk_columns(
    qt_window: IINTSQtDesktopApp,
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")
    qt_window.output_dir.setText(str(tmp_path))
    entries: list[DesktopRunHistoryEntry] = []
    for index in range(2):
        csv_path = tmp_path / f"run-{index}.csv"
        csv_path.write_text(
            "time_minutes,glucose_actual_mgdl\n0,110\n5,115\n",
            encoding="utf-8",
        )
        entries.append(
            DesktopRunHistoryEntry(
                timestamp_utc="2026-07-18T00:00:00+00:00",
                workflow_title="Runtime test",
                preset_name=f"run-{index}",
                seed=index,
                run_id=f"run-{index}",
                output_dir=str(tmp_path),
                results_csv=str(csv_path),
                report_pdf=None,
            )
        )

    qt_window.history_entries = entries
    qt_window.history_table.setRowCount(2)
    qt_window.history_table.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 6), True)
    qt_window.history_table.setRangeSelected(QTableWidgetSelectionRange(1, 0, 1, 6), True)

    qt_window._compare_selected_runs()

    assert (tmp_path / ".cache" / "comparison_graph.html").is_file()
    assert "Not enough valid" not in qt_window.status.text()
