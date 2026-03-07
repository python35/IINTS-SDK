from __future__ import annotations

import pandas as pd

from iints.mdmp.backend import (
    active_mdmp_backend,
    mdmp_grade_meets_minimum,
    run_mdmp_validation,
)
from iints.data.contracts import parse_contract


def _contract_payload() -> dict:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "security": "PII_ENCRYPTED",
                "metadata": {
                    "required_columns": ["glucose"],
                    "column_types": {"glucose": "float"},
                    "ranges": {"glucose": {"min": 40, "max": 400}},
                },
            }
        ],
        "processes": [
            {
                "name": "GlucoseData",
                "input_stream": "PatientHealth.glucose",
                "validations": [
                    {
                        "expression": "glucose is not null and glucose > 20",
                        "on_fail": "DISCARD_AND_LOG",
                    }
                ],
            }
        ],
    }


def test_backend_defaults_to_iints_or_mdmp_core() -> None:
    assert active_mdmp_backend() in {"iints", "mdmp_core"}


def test_run_mdmp_validation_normalizes_result_shape() -> None:
    contract = parse_contract(_contract_payload())
    df = pd.DataFrame({"glucose": [100.0, 120.0, 140.0]})
    result = run_mdmp_validation(contract, df, apply_builtin_transforms=True)

    assert result.row_count == 3
    assert result.mdmp_grade in {"draft", "research_grade", "clinical_grade", "ai_ready", "raw"}
    assert isinstance(result.to_dict(), dict)


def test_mdmp_grade_order_supports_ai_ready() -> None:
    assert mdmp_grade_meets_minimum("ai_ready", "clinical_grade") is True
    assert mdmp_grade_meets_minimum("draft", "ai_ready") is False
