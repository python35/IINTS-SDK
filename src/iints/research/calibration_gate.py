from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ForecastCalibrationGate:
    name: str
    min_within_20_mgdl_pct: Optional[float] = None
    max_false_hypo_alarm_rate_pct: Optional[float] = None
    max_missed_hypo_rate_pct: Optional[float] = None
    min_interval_95_coverage_pct: Optional[float] = None


def load_calibration_gate_profiles(path: Optional[Path] = None) -> Dict[str, ForecastCalibrationGate]:
    if path is None:
        import sys

        if sys.version_info >= (3, 9):
            from importlib.resources import files

            content = files("iints.presets").joinpath("forecast_calibration_profiles.yaml").read_text()
        else:
            from importlib import resources

            content = resources.read_text("iints.presets", "forecast_calibration_profiles.yaml")
    else:
        content = path.read_text()

    payload = yaml.safe_load(content) or {}
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("forecast_calibration_profiles.yaml must contain a 'profiles' mapping")

    result: Dict[str, ForecastCalibrationGate] = {}
    for name, raw in profiles.items():
        if not isinstance(raw, dict):
            continue
        result[str(name)] = ForecastCalibrationGate(
            name=str(name),
            min_within_20_mgdl_pct=raw.get("min_within_20_mgdl_pct"),
            max_false_hypo_alarm_rate_pct=raw.get("max_false_hypo_alarm_rate_pct"),
            max_missed_hypo_rate_pct=raw.get("max_missed_hypo_rate_pct"),
            min_interval_95_coverage_pct=raw.get("min_interval_95_coverage_pct"),
        )
    return result


def evaluate_calibration_gate(
    report: Dict[str, Any],
    gate: ForecastCalibrationGate,
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}

    def _check_min(metric: str, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        value = report.get(metric)
        if value is None:
            checks[metric] = {"passed": False, "value": None, "threshold": threshold, "reason": "missing_metric"}
            return
        passed = float(value) >= float(threshold)
        checks[metric] = {
            "passed": passed,
            "value": float(value),
            "threshold": float(threshold),
            "reason": "ok" if passed else f"{value:.3f} < {float(threshold):.3f}",
        }

    def _check_max(metric: str, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        value = report.get(metric)
        if value is None:
            checks[metric] = {"passed": False, "value": None, "threshold": threshold, "reason": "missing_metric"}
            return
        passed = float(value) <= float(threshold)
        checks[metric] = {
            "passed": passed,
            "value": float(value),
            "threshold": float(threshold),
            "reason": "ok" if passed else f"{value:.3f} > {float(threshold):.3f}",
        }

    _check_min("within_20_mgdl_pct", gate.min_within_20_mgdl_pct)
    _check_max("false_hypo_alarm_rate_pct", gate.max_false_hypo_alarm_rate_pct)
    _check_max("missed_hypo_rate_pct", gate.max_missed_hypo_rate_pct)
    _check_min("interval_95_coverage_pct", gate.min_interval_95_coverage_pct)
    return checks

