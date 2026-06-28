from __future__ import annotations

import pandas as pd
import pytest

from iints.research.genomics_engine import GenomicsEngine, _extract_glucose_trace


def test_extract_glucose_trace_accepts_current_simulator_columns() -> None:
    results = pd.DataFrame(
        {
            "time_minutes": [0, 5, 10],
            "glucose_actual_mgdl": [100.0, 105.5, 112.0],
        }
    )

    time, glucose = _extract_glucose_trace(results)

    assert time == [0.0, 5.0, 10.0]
    assert glucose == [100.0, 105.5, 112.0]


def test_extract_glucose_trace_accepts_legacy_columns() -> None:
    results = pd.DataFrame({"time": [0, 5], "glucose": [90, 95]})

    time, glucose = _extract_glucose_trace(results)

    assert time == [0.0, 5.0]
    assert glucose == [90.0, 95.0]


def test_extract_glucose_trace_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        _extract_glucose_trace(pd.DataFrame({"timestamp": [0], "value": [100]}))


def test_genomics_engine_known_mutation_metadata_is_deterministic() -> None:
    data = GenomicsEngine.evaluate_mutation("INSR", "v938m")

    assert data["scalar"] == 0.1
    assert data["residue"] == 938
    assert "Donohue" in data["desc"]
