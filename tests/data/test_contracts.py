from __future__ import annotations

import json

from iints.data.contracts import compile_contract, parse_contract


def _demo_contract() -> dict:
    return {
        "version": 1,
        "streams": [
            {
                "name": "PatientHealth",
                "source": "sdk.iints_af.v1",
                "security": "PII_ENCRYPTED",
            }
        ],
        "processes": [
            {
                "name": "GlucoseData",
                "input_stream": "PatientHealth.glucose",
                "features": [
                    {
                        "name": "rolling_avg",
                        "operation": "moving_average",
                        "source": "PatientHealth.glucose",
                        "window": "15m",
                    },
                    {
                        "name": "trend_velocity",
                        "operation": "derivative",
                        "source": "PatientHealth.glucose",
                    },
                ],
                "labels": [
                    {
                        "name": "hyper_event",
                        "expression": "glucose > 180",
                        "classes": ["Critical", "Normal"],
                    }
                ],
                "validations": [
                    {
                        "expression": "glucose is not null and glucose > 20",
                        "on_fail": "DISCARD_AND_LOG",
                    }
                ],
            }
        ],
    }


def test_parse_contract_builds_structured_model() -> None:
    model = parse_contract(_demo_contract())
    assert model.version == 1
    assert len(model.streams) == 1
    assert model.streams[0].name == "PatientHealth"
    assert len(model.processes) == 1
    assert len(model.processes[0].features) == 2
    assert model.processes[0].labels[0].name == "hyper_event"


def test_compile_contract_includes_fingerprint() -> None:
    compiled = compile_contract(_demo_contract())
    assert compiled["version"] == 1
    assert "fingerprint_sha256" in compiled
    assert len(compiled["fingerprint_sha256"]) == 64
    json.dumps(compiled)  # must stay JSON-serializable


def test_contract_fingerprint_is_deterministic() -> None:
    first = compile_contract(_demo_contract())
    second = compile_contract(_demo_contract())
    assert first["fingerprint_sha256"] == second["fingerprint_sha256"]
