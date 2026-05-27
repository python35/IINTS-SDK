from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import ValidationError

from iints.validation import format_validation_error, scenario_warnings, validate_patient_config_dict, validate_scenario_dict


@dataclass(frozen=True)
class RunDoctorCheck:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class RunDoctorReport:
    status: str
    checks: List[RunDoctorCheck]
    summary: Dict[str, Any]

    @property
    def failed(self) -> bool:
        return any(check.status == "fail" for check in self.checks)

    @property
    def warned(self) -> bool:
        return any(check.status == "warning" for check in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
        }


def _check(status: str, name: str, message: str, **details: Any) -> RunDoctorCheck:
    return RunDoctorCheck(name=name, status=status, message=message, details=details)


def _nearest_existing_parent(path: Path) -> Optional[Path]:
    current = path.expanduser().resolve()
    if current.exists():
        return current if current.is_dir() else current.parent
    for parent in current.parents:
        if parent.exists():
            return parent
    return None


def _first_meal_minute(scenario: Any) -> Optional[int]:
    meal_times = [
        int(event.start_time)
        for event in scenario.stress_events
        if event.event_type in {"meal", "missed_meal"} and event.start_time is not None
    ]
    return min(meal_times) if meal_times else None


def _naive_drift_projection(initial_glucose: float, decay_rate: float, time_step_minutes: int, horizon: int) -> float:
    if time_step_minutes <= 0:
        return float(initial_glucose)
    multiplier = max(0.0, 1.0 - decay_rate * float(time_step_minutes))
    steps = max(0, int(horizon // time_step_minutes))
    glucose = float(initial_glucose)
    for _ in range(steps):
        glucose *= multiplier
    return glucose


def inspect_run_setup(
    *,
    algo_path: Optional[Path] = None,
    patient_config_path: Optional[Path] = None,
    scenario_path: Optional[Path] = None,
    duration_minutes: int = 1440,
    time_step_minutes: int = 5,
    output_dir: Optional[Path] = None,
    patient_config_name: Optional[str] = None,
) -> RunDoctorReport:
    """Preflight an IINTS run before spending time on a broken simulation."""

    checks: List[RunDoctorCheck] = []
    summary: Dict[str, Any] = {
        "duration_minutes": duration_minutes,
        "time_step_minutes": time_step_minutes,
        "patient_config_name": patient_config_name,
    }

    if duration_minutes <= 0:
        checks.append(_check("fail", "duration", "Duration must be greater than zero.", duration_minutes=duration_minutes))
    elif duration_minutes > 1440 * 7:
        checks.append(
            _check(
                "warning",
                "duration",
                "Very long run requested; use checkpoints or a Jetson endurance workflow.",
                duration_minutes=duration_minutes,
            )
        )
    else:
        checks.append(_check("pass", "duration", "Duration is in a usable range.", duration_minutes=duration_minutes))

    if time_step_minutes <= 0:
        checks.append(
            _check("fail", "time_step", "Time step must be greater than zero.", time_step_minutes=time_step_minutes)
        )
    elif time_step_minutes > 15:
        checks.append(
            _check(
                "warning",
                "time_step",
                "Large time steps can hide fast glucose changes and safety events.",
                time_step_minutes=time_step_minutes,
            )
        )
    else:
        checks.append(_check("pass", "time_step", "Time step is appropriate for a CGM-style simulation."))

    if algo_path is None:
        checks.append(_check("warning", "algorithm", "No algorithm path supplied; run may use a built-in preset."))
    else:
        resolved_algo = algo_path.expanduser()
        if not resolved_algo.is_file():
            checks.append(_check("fail", "algorithm", "Algorithm file was not found.", path=str(resolved_algo)))
        else:
            text = resolved_algo.read_text(encoding="utf-8", errors="replace")
            if "predict_insulin" not in text:
                checks.append(
                    _check(
                        "warning",
                        "algorithm",
                        "Algorithm file exists but no predict_insulin method was detected.",
                        path=str(resolved_algo),
                    )
                )
            else:
                checks.append(_check("pass", "algorithm", "Algorithm file exists and exposes predict_insulin."))

    scenario_model = None
    if scenario_path is None:
        checks.append(_check("warning", "scenario", "No scenario path supplied; run may use a preset scenario."))
    else:
        resolved_scenario = scenario_path.expanduser()
        if not resolved_scenario.is_file():
            checks.append(_check("fail", "scenario", "Scenario file was not found.", path=str(resolved_scenario)))
        else:
            try:
                raw_scenario = json.loads(resolved_scenario.read_text(encoding="utf-8"))
                scenario_model = validate_scenario_dict(raw_scenario)
                warnings = scenario_warnings(scenario_model)
                summary["first_meal_minute"] = _first_meal_minute(scenario_model)
                if warnings:
                    checks.append(_check("warning", "scenario", "Scenario validates with warnings.", warnings=warnings))
                else:
                    checks.append(_check("pass", "scenario", "Scenario file validates."))
            except ValidationError as exc:
                checks.append(
                    _check(
                        "fail",
                        "scenario",
                        "Scenario validation failed.",
                        errors=format_validation_error(exc),
                    )
                )
            except Exception as exc:
                checks.append(_check("fail", "scenario", "Scenario could not be read.", error=str(exc)))

    patient_model = None
    raw_patient: Dict[str, Any] = {}
    if patient_config_path is None:
        checks.append(_check("warning", "patient", "No patient config path supplied; run may use a packaged profile."))
    else:
        resolved_patient = patient_config_path.expanduser()
        if not resolved_patient.is_file():
            checks.append(_check("fail", "patient", "Patient config file was not found.", path=str(resolved_patient)))
        else:
            try:
                loaded = yaml.safe_load(resolved_patient.read_text(encoding="utf-8"))
                raw_patient = loaded if isinstance(loaded, dict) else {}
                patient_model = validate_patient_config_dict(raw_patient)
                checks.append(_check("pass", "patient", "Patient config validates."))
            except ValidationError as exc:
                checks.append(
                    _check(
                        "fail",
                        "patient",
                        "Patient config validation failed.",
                        errors=format_validation_error(exc),
                    )
                )
            except Exception as exc:
                checks.append(_check("fail", "patient", "Patient config could not be read.", error=str(exc)))

    if patient_model is not None:
        decay_rate = float(patient_model.glucose_decay_rate)
        decay_per_step = decay_rate * float(time_step_minutes)
        summary["glucose_decay_rate"] = decay_rate
        summary["glucose_decay_per_step"] = decay_per_step
        if decay_per_step > 0.10:
            checks.append(
                _check(
                    "fail",
                    "glucose_decay",
                    "glucose_decay_rate x time_step exceeds 10%; this can make demo patients collapse unrealistically fast.",
                    glucose_decay_rate=decay_rate,
                    time_step_minutes=time_step_minutes,
                    decay_per_step=decay_per_step,
                )
            )
        elif decay_per_step > 0.03:
            checks.append(
                _check(
                    "warning",
                    "glucose_decay",
                    "glucose drift is aggressive for a clinic-safe demo; consider <=0.001/min or an endurance profile.",
                    glucose_decay_rate=decay_rate,
                    time_step_minutes=time_step_minutes,
                    decay_per_step=decay_per_step,
                )
            )
        else:
            checks.append(_check("pass", "glucose_decay", "Glucose drift rate is suitable for a stable demo."))

        first_meal = summary.get("first_meal_minute")
        horizon = int(first_meal) if isinstance(first_meal, int) and first_meal > 0 else min(60, duration_minutes)
        projected = _naive_drift_projection(
            float(patient_model.initial_glucose),
            decay_rate,
            time_step_minutes,
            horizon,
        )
        summary["naive_no_input_glucose_at_horizon_mgdl"] = round(projected, 3)
        summary["naive_no_input_horizon_minutes"] = horizon
        if projected < 54.0:
            checks.append(
                _check(
                    "fail",
                    "pre_meal_projection",
                    "Naive no-meal/no-insulin projection reaches severe hypoglycemia before the first meal/horizon.",
                    projected_glucose_mgdl=round(projected, 3),
                    horizon_minutes=horizon,
                    initial_glucose_mgdl=float(patient_model.initial_glucose),
                )
            )
        elif projected < 70.0:
            checks.append(
                _check(
                    "warning",
                    "pre_meal_projection",
                    "Naive no-meal/no-insulin projection falls below 70 mg/dL before the first meal/horizon.",
                    projected_glucose_mgdl=round(projected, 3),
                    horizon_minutes=horizon,
                )
            )
        else:
            checks.append(
                _check(
                    "pass",
                    "pre_meal_projection",
                    "No immediate pre-meal hypoglycemia risk detected by the naive drift preview.",
                    projected_glucose_mgdl=round(projected, 3),
                    horizon_minutes=horizon,
                )
            )
        if "glucose_decay_rate" in raw_patient:
            checks.append(
                _check(
                    "pass",
                    "field_naming",
                    "glucose_decay_rate is accepted as a per-minute drift field for compatibility.",
                )
            )

    if output_dir is not None:
        target = output_dir.expanduser()
        parent = _nearest_existing_parent(target)
        if parent is None:
            checks.append(_check("fail", "output_dir", "No existing parent directory found.", path=str(target)))
        elif not os.access(parent, os.W_OK):
            checks.append(_check("fail", "output_dir", "Output directory parent is not writable.", parent=str(parent)))
        else:
            checks.append(_check("pass", "output_dir", "Output directory can be created or reused.", path=str(target)))

    status = "fail" if any(check.status == "fail" for check in checks) else "warning" if any(
        check.status == "warning" for check in checks
    ) else "pass"
    return RunDoctorReport(status=status, checks=checks, summary=summary)
