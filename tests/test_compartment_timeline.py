"""Tests for the compartment timeline the desktop compartment view consumes.

The behaviour worth pinning down is what happens for runs that carry no
compartment columns -- runs made before the export existed, and runs whose
patient backend publishes no schema. Those must report why the view is
unavailable rather than present an empty diagram as if it meant something.
"""

import json

import pytest

from iints_desktop.results import load_compartment_timeline

SCHEMA = {
    "model_key": "toy",
    "model_label": "Toy two-compartment model",
    "compartments": [
        {
            "key": "A",
            "symbol": "A",
            "label": "Depot",
            "unit": "mg",
            "state_index": 0,
            "kind": "pool",
            "site": "subcutaneous",
            "provenance": "canonical",
            "description": "",
        },
        {
            "key": "B",
            "symbol": "B",
            "label": "Plasma",
            "unit": "mg",
            "state_index": 1,
            "kind": "pool",
            "site": "plasma",
            "provenance": "canonical",
            "description": "",
        },
    ],
    "fluxes": [
        {
            "key": "transfer",
            "source": "A",
            "target": "B",
            "label": "Transfer",
            "unit": "mg/min",
            "rate_expression": "k * A",
            "parameters": ["k"],
            "provenance": "canonical",
            "description": "",
            "recorded": True,
        }
    ],
}


def _write_run(tmp_path, *, steps=10, schema=True, state=True):
    header = ["time_minutes", "glucose_actual_mgdl"]
    if state:
        header += ["patient_state_A", "patient_state_B", "patient_flux_transfer"]
    lines = [",".join(header)]
    for index in range(steps):
        row = [str(index * 5), str(110 + index)]
        if state:
            row += [str(100 - index), str(index), str(-index)]
        lines.append(",".join(row))
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    if schema:
        (tmp_path / "compartment_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return csv_path


def test_timeline_reports_series_and_schema(tmp_path):
    timeline = load_compartment_timeline(_write_run(tmp_path))

    assert timeline.available
    assert timeline.schema["model_label"] == "Toy two-compartment model"
    assert set(timeline.compartments) == {"A", "B"}
    assert set(timeline.fluxes) == {"transfer"}
    assert timeline.times[0] == 0.0
    assert timeline.step_count == 10
    assert timeline.stride == 1
    # Aligned to `times`/`compartments` the same way, from the same CSV
    # column the rest of the desktop app already treats as canonical.
    assert timeline.plasma_glucose_mgdl == [110.0 + index for index in range(10)]


def test_missing_schema_is_reported_not_guessed(tmp_path):
    timeline = load_compartment_timeline(_write_run(tmp_path, schema=False))

    assert not timeline.available
    assert "compartment_schema.json" in timeline.reason
    assert timeline.compartments == {}


def test_run_without_state_columns_is_reported(tmp_path):
    timeline = load_compartment_timeline(_write_run(tmp_path, state=False))

    assert not timeline.available
    assert "predates" in timeline.reason
    # The schema is still handed back, so the UI can name the model it found.
    assert timeline.schema["model_label"] == "Toy two-compartment model"


def test_downsampling_keeps_real_samples_and_full_range(tmp_path):
    timeline = load_compartment_timeline(_write_run(tmp_path, steps=100), max_points=10)

    assert timeline.stride == 10
    assert len(timeline.times) == 10
    # Every retained value must be one the run actually recorded, not an average.
    assert timeline.compartments["A"] == [float(100 - index) for index in range(0, 100, 10)]
    # Extremes come from the full trace, so a scale built on them does not
    # depend on the stride.
    assert timeline.flux_extremes["transfer"] == (-99.0, 0.0)
    assert timeline.plasma_glucose_mgdl == [float(110 + index) for index in range(0, 100, 10)]


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_compartment_timeline(tmp_path / "absent.csv")


def test_plasma_glucose_is_empty_when_no_glucose_column_present(tmp_path):
    # A run whose CSV carries no recognizable glucose column (see
    # GLUCOSE_COLUMNS) must degrade gracefully -- the illustrated diagram's
    # hypo coloring has nothing to key off, not a crash.
    header = ["time_minutes", "patient_state_A", "patient_state_B", "patient_flux_transfer"]
    lines = [",".join(header)]
    for index in range(5):
        lines.append(",".join([str(index * 5), str(100 - index), str(index), str(-index)]))
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "compartment_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")

    timeline = load_compartment_timeline(csv_path)

    assert timeline.available
    assert timeline.plasma_glucose_mgdl == []
