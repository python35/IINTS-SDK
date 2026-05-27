from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from .realism_validator import RealismReport


RealDataGateStatus = Literal["passed", "needs_review", "blocked"]


@dataclass(frozen=True)
class RealDataGateProfile:
    """Strict research gate for traces used as realism evidence or AI training data."""

    name: str = "real_data_research_strict_v1"
    min_duration_hours: float = 20.0
    max_failed_checks: int = 0
    max_warning_checks: int = 0
    max_rapid_change_count: int = 0
    max_long_gap_count: int = 0
    max_tir_below_54_pct: float = 0.5
    max_tir_below_70_pct: float = 5.0
    min_meals_for_full_day: int = 2
    require_reference_profile: bool = True
    require_external_reference: bool = True
    bundled_or_synthetic_dataset_ids: tuple[str, ...] = ("sample", "demo", "bundled_demo", "synthetic")


@dataclass(frozen=True)
class RealDataGateResult:
    profile_name: str
    status: RealDataGateStatus
    passed: bool
    score: float
    critical_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "status": self.status,
            "passed": self.passed,
            "score": self.score,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "required_actions": self.required_actions,
            "metrics": self.metrics,
        }


STRICT_REAL_DATA_RESEARCH_PROFILE = RealDataGateProfile()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def review_real_data_realism(
    report: RealismReport,
    *,
    profile: RealDataGateProfile = STRICT_REAL_DATA_RESEARCH_PROFILE,
) -> RealDataGateResult:
    """Apply a stricter gate than the general realism validator.

    The normal validator answers "does this look plausible?". This gate answers
    "is this strong enough to use as public realism evidence or local-AI training
    evidence without an explicit caveat?".
    """

    critical: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []

    failed_checks = [check for check in report.checks if check.status == "failed"]
    warning_checks = [check for check in report.checks if check.status == "warning"]

    if report.verdict != "likely_realistic":
        critical.append(f"General realism verdict is '{report.verdict}', not 'likely_realistic'.")
        actions.append("Tune the scenario/profile or calibrate against external data until the base verdict is likely_realistic.")
    if len(failed_checks) > profile.max_failed_checks:
        critical.append(f"{len(failed_checks)} failed realism check(s) exceeded the allowed {profile.max_failed_checks}.")
        actions.append("Resolve failed realism checks before using this trace as evidence or training data.")
    if len(warning_checks) > profile.max_warning_checks:
        warnings.append(f"{len(warning_checks)} warning realism check(s) exceeded the preferred {profile.max_warning_checks}.")
        actions.append("Investigate realism warnings; strict AI training bundles should be warning-free or explicitly caveated.")

    metrics = report.metrics
    duration_hours = _as_float(metrics.get("duration_hours"))
    if duration_hours < profile.min_duration_hours:
        critical.append(
            f"Duration is {duration_hours:.2f} h; strict daily evidence requires at least {profile.min_duration_hours:.1f} h."
        )
        actions.append("Use a near-full-day trace for daily physiology, TIR, and meal-pattern claims.")

    rapid_changes = _as_int(metrics.get("rapid_change_count"))
    long_gaps = _as_int(metrics.get("long_gap_count"))
    impossible_values = _as_int(metrics.get("impossible_value_count"))
    if impossible_values:
        critical.append(f"{impossible_values} impossible glucose value(s) were detected.")
        actions.append("Fix unit conversion, parser mapping, or corrupted CGM rows.")
    if rapid_changes > profile.max_rapid_change_count:
        critical.append(
            f"{rapid_changes} rapid-change event(s) exceeded the strict limit of {profile.max_rapid_change_count}."
        )
        actions.append("Review sensor artifact filtering and avoid training controllers on implausible jumps.")
    if long_gaps > profile.max_long_gap_count:
        critical.append(f"{long_gaps} long gap(s) exceeded the strict limit of {profile.max_long_gap_count}.")
        actions.append("Fill, split, or reject traces with long missing-data intervals before AI training.")

    below_54 = _as_float(metrics.get("tir_below_54_pct"))
    below_70 = _as_float(metrics.get("tir_below_70_pct"))
    if below_54 > profile.max_tir_below_54_pct:
        critical.append(f"Time <54 mg/dL is {below_54:.2f}%, above strict limit {profile.max_tir_below_54_pct:.2f}%.")
        actions.append("Do not use severe-hypoglycemia-heavy traces as 'normal realism' references.")
    if below_70 > profile.max_tir_below_70_pct:
        warnings.append(f"Time <70 mg/dL is {below_70:.2f}%, above preferred limit {profile.max_tir_below_70_pct:.2f}%.")
        actions.append("Label this trace as hypo-exposed or stress-test data rather than normal free-living evidence.")

    meal_count = _as_int(metrics.get("meal_count"))
    insulin_count = _as_int(metrics.get("insulin_event_count"))
    if duration_hours >= profile.min_duration_hours and meal_count < profile.min_meals_for_full_day:
        warnings.append(
            f"Only {meal_count} meal event(s) were annotated in a full-day trace; expected at least {profile.min_meals_for_full_day}."
        )
        actions.append("Prefer traces with meal and insulin context for physiological realism claims.")
    if meal_count >= profile.min_meals_for_full_day and insulin_count == 0:
        critical.append("Meal annotations are present but no insulin events were preserved.")
        actions.append("Reject or repair traces where bolus/basal information was lost during import.")

    reference_profile = report.reference_profile
    if profile.require_reference_profile and reference_profile is None:
        critical.append("No empirical reference profile was attached to the realism report.")
        actions.append("Run realism validation with a dataset-specific reference profile such as free_living_t1d.")
    elif reference_profile is not None:
        dataset_ids = set(reference_profile.dataset_ids)
        bundled_ids = set(profile.bundled_or_synthetic_dataset_ids)
        if profile.require_external_reference and dataset_ids and dataset_ids.issubset(bundled_ids):
            critical.append(
                f"Reference profile '{reference_profile.id}' is backed only by bundled/synthetic IDs: {sorted(dataset_ids)}."
            )
            actions.append("Back public realism claims with AZT1D, HUPA-UCM, AIDE, OhioT1DM, or another external dataset.")
        if not dataset_ids:
            warnings.append(f"Reference profile '{reference_profile.id}' does not declare dataset IDs.")
            actions.append("Add dataset provenance IDs to the reference registry.")

    reference_failed = 0
    reference_warnings = 0
    for check in report.checks:
        if check.code == "reference_envelope":
            reference_failed = _as_int(check.metrics.get("failed_metrics"))
            reference_warnings = _as_int(check.metrics.get("warning_metrics"))
            break
    if reference_failed:
        critical.append(f"Reference envelope has {reference_failed} failed metric(s).")
        actions.append("Retune the simulator or route this trace to stress-test data, not normal-realism evidence.")
    if reference_warnings:
        warnings.append(f"Reference envelope has {reference_warnings} warning metric(s).")
        actions.append("Inspect reference-envelope warnings before publishing or training a controller.")

    score = max(0.0, 1.0 - 0.18 * len(critical) - 0.06 * len(warnings))
    if critical:
        status: RealDataGateStatus = "blocked"
    elif warnings:
        status = "needs_review"
    else:
        status = "passed"

    gate_metrics = {
        "base_verdict": report.verdict,
        "base_realism_score": report.realism_score,
        "duration_hours": duration_hours,
        "failed_checks": len(failed_checks),
        "warning_checks": len(warning_checks),
        "rapid_change_count": rapid_changes,
        "long_gap_count": long_gaps,
        "tir_below_54_pct": below_54,
        "tir_below_70_pct": below_70,
        "meal_count": meal_count,
        "insulin_event_count": insulin_count,
        "reference_profile": reference_profile.id if reference_profile else None,
    }
    return RealDataGateResult(
        profile_name=profile.name,
        status=status,
        passed=status == "passed",
        score=round(score, 4),
        critical_failures=critical,
        warnings=warnings,
        required_actions=actions,
        metrics=gate_metrics,
    )
