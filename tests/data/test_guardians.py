from __future__ import annotations

import pandas as pd
import pytest

from iints.data.guardians import MDMPGateError, mdmp_gate


def _contract_payload() -> dict:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "metadata": {
                    "required_columns": ["timestamp", "glucose"],
                    "column_types": {"glucose": "float"},
                    "ranges": {"glucose": {"min": 70, "max": 250}},
                },
            }
        ],
        "processes": [
            {
                "name": "GlucoseData",
                "input_stream": "PatientHealth.glucose",
                "validations": [{"expression": "glucose is not null and glucose > 20"}],
            }
        ],
    }


def test_mdmp_gate_allows_execution_for_valid_dataframe() -> None:
    calls = {"count": 0}

    @mdmp_gate(_contract_payload(), min_grade="clinical_grade")
    def process(df: pd.DataFrame) -> int:
        calls["count"] += 1
        return len(df)

    df = pd.DataFrame({"timestamp": [1, 2], "glucose": [110.0, 120.0]})
    assert process(df) == 2
    assert calls["count"] == 1


def test_mdmp_gate_blocks_execution_when_grade_below_threshold() -> None:
    @mdmp_gate(_contract_payload(), min_grade="clinical_grade")
    def process(df: pd.DataFrame) -> int:
        return len(df)

    df = pd.DataFrame({"timestamp": [1, 2], "glucose": [40.0, 45.0]})
    with pytest.raises(MDMPGateError):
        process(df)


def test_mdmp_gate_warn_mode_with_named_dataframe_arg() -> None:
    @mdmp_gate(_contract_payload(), min_grade="clinical_grade", fail_mode="warn", dataframe_arg="df")
    def process(*, df: pd.DataFrame) -> int:
        return len(df)

    df = pd.DataFrame({"timestamp": [1, 2], "glucose": [40.0, 45.0]})
    with pytest.warns(RuntimeWarning):
        assert process(df=df) == 2
