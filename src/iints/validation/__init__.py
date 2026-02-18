from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import ValidationError

from iints.core.simulator import StressEvent
from iints.validation.schemas import ScenarioModel, StressEventModel, PatientConfigModel


def _convert_legacy_scenario(data: Dict[str, Any]) -> Dict[str, Any]:
    if "events" in data and "stress_events" not in data:
        events = []
        for event in data.get("events", []):
            event_type = event.get("type")
            if event_type == "carb_error":
                event = dict(event)
                event["reported_value"] = 0
                event_type = "meal"
            events.append(
                {
                    "start_time": event.get("time"),
                    "event_type": event_type,
                    "value": event.get("value"),
                    "reported_value": event.get("reported_value"),
                    "absorption_delay_minutes": event.get("absorption_delay_minutes", 0),
                    "duration": event.get("duration", 0),
                }
            )
        return {
            "scenario_name": data.get("scenario_name", "Legacy Scenario"),
            "scenario_version": data.get("scenario_version", "legacy"),
            "description": data.get("description", ""),
            "stress_events": events,
        }
    return data


def load_scenario(path: Union[str, Path]) -> ScenarioModel:
    scenario_path = Path(path)
    data = json.loads(scenario_path.read_text())
    data = _convert_legacy_scenario(data)
    return ScenarioModel.model_validate(data)


def validate_scenario_dict(data: Dict[str, Any]) -> ScenarioModel:
    data = _convert_legacy_scenario(data)
    return ScenarioModel.model_validate(data)


def scenario_warnings(model: ScenarioModel) -> List[str]:
    warnings: List[str] = []
    for idx, event in enumerate(model.stress_events):
        prefix = f"event[{idx}]"
        if event.event_type in {"meal", "missed_meal"} and event.value is not None:
            if event.value > 200:
                warnings.append(f"{prefix}: meal value {event.value}g is unusually high")
        if event.absorption_delay_minutes > 120:
            warnings.append(f"{prefix}: absorption_delay_minutes {event.absorption_delay_minutes} is unusual")
        if event.duration > 240:
            warnings.append(f"{prefix}: duration {event.duration} is unusual")
    return warnings


def build_stress_events(payloads: List[Dict[str, Any]]) -> List[StressEvent]:
    events: List[StressEvent] = []
    for event_data in payloads:
        events.append(
            StressEvent(
                start_time=event_data["start_time"],
                event_type=event_data["event_type"],
                value=event_data.get("value"),
                reported_value=event_data.get("reported_value"),
                absorption_delay_minutes=event_data.get("absorption_delay_minutes", 0),
                duration=event_data.get("duration", 0),
                isf=event_data.get("isf"),
                icr=event_data.get("icr"),
                basal_rate=event_data.get("basal_rate"),
                dia_minutes=event_data.get("dia_minutes"),
            )
        )
    return events


def scenario_to_payloads(model: ScenarioModel) -> List[Dict[str, Any]]:
    return [event.model_dump() for event in model.stress_events]


def validate_patient_config_dict(data: Dict[str, Any]) -> PatientConfigModel:
    return PatientConfigModel.model_validate(data)


def load_patient_config(path: Union[str, Path]) -> PatientConfigModel:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    return validate_patient_config_dict(data)


def load_patient_config_by_name(name: str) -> PatientConfigModel:
    filename = f"{name}.yaml" if not name.endswith(".yaml") else name
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        content = files("iints.data.virtual_patients").joinpath(filename).read_text()
    else:
        from importlib import resources
        content = resources.read_text("iints.data.virtual_patients", filename)
    data = yaml.safe_load(content)
    return validate_patient_config_dict(data)


def format_validation_error(error: ValidationError) -> List[str]:
    lines: List[str] = []
    for entry in error.errors():
        loc = ".".join(str(item) for item in entry.get("loc", []))
        msg = entry.get("msg", "Invalid value")
        lines.append(f"{loc}: {msg}")
    return lines
