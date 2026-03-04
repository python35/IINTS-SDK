from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.analysis.safety_index import compute_safety_index

Comparator = Callable[[float, float], bool]


@dataclass(frozen=True)
class ValidationRule:
    metric: str
    op: str
    threshold: float
    label: str
    required: bool = True


@dataclass(frozen=True)
class ValidationProfile:
    profile_id: str
    description: str
    min_duration_minutes: int
    checks: List[ValidationRule]


@dataclass(frozen=True)
class ValidationCheckResult:
    rule: ValidationRule
    value: Optional[float]
    passed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.rule.metric,
            "label": self.rule.label,
            "op": self.rule.op,
            "threshold": self.rule.threshold,
            "required": self.rule.required,
            "value": self.value,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RunValidationReport:
    profile: ValidationProfile
    passed: bool
    required_checks_passed: int
    required_checks_total: int
    score: float
    checks: List[ValidationCheckResult]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": {
                "id": self.profile.profile_id,
                "description": self.profile.description,
                "min_duration_minutes": self.profile.min_duration_minutes,
            },
            "passed": self.passed,
            "required_checks_passed": self.required_checks_passed,
            "required_checks_total": self.required_checks_total,
            "score": self.score,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": self.metrics,
        }


_COMPARATORS: Dict[str, Comparator] = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
    "==": lambda value, threshold: value == threshold,
}


def _load_profiles_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is not None:
        return yaml.safe_load(path.read_text()) or {}

    import sys

    if sys.version_info >= (3, 9):
        from importlib.resources import files

        content = files("iints.presets").joinpath("validation_profiles.yaml").read_text()
    else:
        from importlib import resources

        content = resources.read_text("iints.presets", "validation_profiles.yaml")
    return yaml.safe_load(content) or {}


def load_validation_profiles(path: Optional[Path] = None) -> Dict[str, ValidationProfile]:
    raw = _load_profiles_yaml(path)
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("validation_profiles.yaml must contain a top-level 'profiles' mapping")

    profiles: Dict[str, ValidationProfile] = {}
    for profile_id, profile_raw in raw_profiles.items():
        if not isinstance(profile_raw, dict):
            raise ValueError(f"profile '{profile_id}' must be a mapping")

        checks_raw = profile_raw.get("checks", [])
        if not isinstance(checks_raw, list):
            raise ValueError(f"profile '{profile_id}' has invalid 'checks' (must be a list)")

        checks: List[ValidationRule] = []
        for idx, check_raw in enumerate(checks_raw):
            if not isinstance(check_raw, dict):
                raise ValueError(f"profile '{profile_id}' check[{idx}] must be a mapping")
            op = str(check_raw.get("op", "")).strip()
            if op not in _COMPARATORS:
                raise ValueError(
                    f"profile '{profile_id}' check[{idx}] has unsupported operator '{op}'"
                )
            metric = str(check_raw.get("metric", "")).strip()
            if not metric:
                raise ValueError(f"profile '{profile_id}' check[{idx}] is missing metric")
            if "threshold" not in check_raw:
                raise ValueError(f"profile '{profile_id}' check[{idx}] is missing threshold")

            checks.append(
                ValidationRule(
                    metric=metric,
                    op=op,
                    threshold=float(check_raw["threshold"]),
                    label=str(check_raw.get("label") or metric),
                    required=bool(check_raw.get("required", True)),
                )
            )

        profiles[profile_id] = ValidationProfile(
            profile_id=profile_id,
            description=str(profile_raw.get("description", "")).strip(),
            min_duration_minutes=int(profile_raw.get("min_duration_minutes", 0)),
            checks=checks,
        )
    return profiles


def compute_run_metrics(
    results_df: pd.DataFrame,
    *,
    safety_report: Optional[Dict[str, Any]] = None,
    duration_minutes: Optional[int] = None,
) -> Dict[str, float]:
    if "glucose_actual_mgdl" not in results_df.columns:
        raise ValueError("results dataframe is missing required column 'glucose_actual_mgdl'")

    working = results_df.copy()
    if "time_minutes" not in working.columns:
        working["time_minutes"] = working.index.to_series() * 5.0

    if duration_minutes is None:
        if len(working) >= 2:
            duration_minutes = int(float(working["time_minutes"].max()) - float(working["time_minutes"].min()))
        else:
            duration_minutes = int(len(working) * 5)
    duration_minutes = max(int(duration_minutes), 1)

    calculator = ClinicalMetricsCalculator()
    clinical = calculator.calculate(
        glucose=working["glucose_actual_mgdl"].astype(float),
        timestamp=working["time_minutes"].astype(float),
        duration_hours=duration_minutes / 60.0,
    )

    metrics: Dict[str, float] = {
        "duration_minutes": float(duration_minutes),
        "steps": float(len(working)),
        "tir_70_180": float(clinical.tir_70_180),
        "tir_below_70": float(clinical.tir_below_70),
        "tir_below_54": float(clinical.tir_below_54),
        "tir_above_180": float(clinical.tir_above_180),
        "tir_above_250": float(clinical.tir_above_250),
        "mean_glucose": float(clinical.mean_glucose),
        "cv": float(clinical.cv),
        "gmi": float(clinical.gmi),
        "data_coverage": float(clinical.data_coverage),
    }

    if "safety_triggered" in working.columns:
        triggered_count = int(working["safety_triggered"].fillna(False).astype(bool).sum())
    else:
        triggered_count = 0

    report = safety_report or {}
    interventions = int(report.get("bolus_interventions_count", triggered_count))
    metrics["supervisor_interventions"] = float(interventions)
    metrics["supervisor_interventions_per_hour"] = float(interventions) / max(duration_minutes / 60.0, 1e-6)
    metrics["terminated_early"] = 1.0 if bool(report.get("terminated_early", False)) else 0.0

    if "supervisor_latency_ms" in working.columns:
        metrics["mean_supervisor_latency_ms"] = float(working["supervisor_latency_ms"].dropna().mean())
    else:
        metrics["mean_supervisor_latency_ms"] = 0.0

    if safety_report is not None:
        try:
            safety_idx = compute_safety_index(
                results_df=working,
                safety_report=report,
                duration_minutes=duration_minutes,
            )
            metrics["safety_index"] = float(safety_idx.score)
        except Exception:
            metrics["safety_index"] = float("nan")
    else:
        metrics["safety_index"] = float("nan")

    return metrics


def evaluate_run(
    results_df: pd.DataFrame,
    *,
    profile: ValidationProfile,
    safety_report: Optional[Dict[str, Any]] = None,
    duration_minutes: Optional[int] = None,
) -> RunValidationReport:
    metrics = compute_run_metrics(
        results_df,
        safety_report=safety_report,
        duration_minutes=duration_minutes,
    )

    checks: List[ValidationCheckResult] = []
    for rule in profile.checks:
        value = metrics.get(rule.metric)
        if value is None or pd.isna(value):
            checks.append(
                ValidationCheckResult(
                    rule=rule,
                    value=None,
                    passed=not rule.required,
                    reason=f"metric '{rule.metric}' is unavailable",
                )
            )
            continue

        comparator = _COMPARATORS[rule.op]
        passed = comparator(float(value), rule.threshold)
        checks.append(
            ValidationCheckResult(
                rule=rule,
                value=float(value),
                passed=passed,
                reason="ok" if passed else f"expected {rule.op} {rule.threshold}",
            )
        )

    duration_ok = metrics["duration_minutes"] >= float(profile.min_duration_minutes)
    duration_check = ValidationCheckResult(
        rule=ValidationRule(
            metric="duration_minutes",
            op=">=",
            threshold=float(profile.min_duration_minutes),
            label="Minimum run duration (minutes)",
            required=True,
        ),
        value=metrics["duration_minutes"],
        passed=duration_ok,
        reason="ok" if duration_ok else f"expected >= {profile.min_duration_minutes}",
    )
    checks.insert(0, duration_check)

    required_checks = [check for check in checks if check.rule.required]
    required_passed = sum(1 for check in required_checks if check.passed)
    required_total = len(required_checks)
    passed = required_passed == required_total
    score = 100.0 if required_total == 0 else (required_passed / required_total) * 100.0

    return RunValidationReport(
        profile=profile,
        passed=passed,
        required_checks_passed=required_passed,
        required_checks_total=required_total,
        score=score,
        checks=checks,
        metrics=metrics,
    )
