from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


LocalAIGateStatus = Literal["passed", "needs_review", "blocked"]


@dataclass(frozen=True)
class LocalAISafetyProfile:
    name: str = "local_ai_research_safety_v1"
    min_training_rows: int = 288
    max_teacher_insulin_units: float = 5.0
    max_unsafe_hypo_proposal_rows: int = 0
    max_over_5u_proposal_rows: int = 0
    max_time_below_54_delta_pct: float = 0.25
    max_time_below_70_delta_pct: float = 1.0
    max_supervisor_rate_delta_pct: float = 2.0
    min_completion_pct: float = 99.0
    max_early_termination_runs: int = 0
    max_tir_drop_pct: float = 10.0


@dataclass(frozen=True)
class LocalAIGateResult:
    profile_name: str
    status: LocalAIGateStatus
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


DEFAULT_LOCAL_AI_SAFETY_PROFILE = LocalAISafetyProfile()


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


def _finish_gate(
    *,
    profile: LocalAISafetyProfile,
    critical: List[str],
    warnings: List[str],
    actions: List[str],
    metrics: Dict[str, Any],
) -> LocalAIGateResult:
    score = max(0.0, 1.0 - 0.2 * len(critical) - 0.06 * len(warnings))
    if critical:
        status: LocalAIGateStatus = "blocked"
    elif warnings:
        status = "needs_review"
    else:
        status = "passed"
    return LocalAIGateResult(
        profile_name=profile.name,
        status=status,
        passed=status == "passed",
        score=round(score, 4),
        critical_failures=critical,
        warnings=warnings,
        required_actions=actions,
        metrics=metrics,
    )


def review_controller_training_artifacts(
    controller_summary: Dict[str, Any],
    *,
    train_metrics: Dict[str, Any] | None = None,
    profile: LocalAISafetyProfile = DEFAULT_LOCAL_AI_SAFETY_PROFILE,
) -> LocalAIGateResult:
    """Gate the local controller training set before it is promoted for research use."""

    critical: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    train_metrics = train_metrics or {}

    rows = _as_int(controller_summary.get("rows"))
    max_teacher = _as_float(controller_summary.get("max_teacher_insulin_units"))
    teacher_sources = [str(item) for item in controller_summary.get("teacher_source_columns", [])]
    unsafe_hypo_rows = _as_int(train_metrics.get("unsafe_hypo_proposal_rows"))
    over_5u_rows = _as_int(train_metrics.get("over_5u_proposal_rows"))
    max_prediction = _as_float(train_metrics.get("max_prediction_units"))

    if rows < profile.min_training_rows:
        critical.append(f"Training set has {rows} rows; strict local-AI training requires at least {profile.min_training_rows}.")
        actions.append("Use at least a full 24 h run at 5-minute cadence before treating this as controller training evidence.")
    if max_teacher > profile.max_teacher_insulin_units:
        critical.append(
            f"Teacher labels include {max_teacher:.2f} U, above strict single-step limit {profile.max_teacher_insulin_units:.2f} U."
        )
        actions.append("Review teacher clipping, meal labels, and safety-supervisor intervention before training.")
    if unsafe_hypo_rows > profile.max_unsafe_hypo_proposal_rows:
        critical.append(
            f"Model proposes insulin during hypoglycemia in {unsafe_hypo_rows} row(s); allowed {profile.max_unsafe_hypo_proposal_rows}."
        )
        actions.append("Retrain with stronger safety-weighted loss or reject the controller candidate.")
    if over_5u_rows > profile.max_over_5u_proposal_rows:
        critical.append(f"Model proposes >5 U in {over_5u_rows} row(s); allowed {profile.max_over_5u_proposal_rows}.")
        actions.append("Constrain controller output and inspect meal/bolus labels.")
    if max_prediction > profile.max_teacher_insulin_units:
        warnings.append(
            f"Maximum model prediction is {max_prediction:.2f} U; keep this blocked from hardware without a deterministic clamp."
        )
        actions.append("Require supervisor clamping and hardware dose limits for any embedded export.")
    if "reference_teacher_insulin_units" not in teacher_sources:
        warnings.append("Training labels do not include the reference teacher column.")
        actions.append("Prefer reference_teacher_insulin_units over raw delivered insulin when training local controllers.")

    metrics = {
        "rows": rows,
        "max_teacher_insulin_units": max_teacher,
        "teacher_source_columns": teacher_sources,
        "unsafe_hypo_proposal_rows": unsafe_hypo_rows,
        "over_5u_proposal_rows": over_5u_rows,
        "max_prediction_units": max_prediction,
    }
    return _finish_gate(profile=profile, critical=critical, warnings=warnings, actions=actions, metrics=metrics)


def review_closed_loop_evaluation(
    algorithms: Dict[str, Dict[str, Any]],
    *,
    baseline_name: str = "clinical_baseline",
    profile: LocalAISafetyProfile = DEFAULT_LOCAL_AI_SAFETY_PROFILE,
) -> LocalAIGateResult:
    """Gate held-out closed-loop evaluation against the clinical baseline."""

    critical: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    candidate_metrics: dict[str, Any] = {}

    baseline = algorithms.get(baseline_name)
    if baseline is None:
        critical.append(f"Missing required baseline algorithm '{baseline_name}'.")
        actions.append("Always compare local AI controllers against the deterministic clinical_baseline.")
        return _finish_gate(
            profile=profile,
            critical=critical,
            warnings=warnings,
            actions=actions,
            metrics={"baseline": baseline_name, "candidates": candidate_metrics},
        )

    baseline_below_54 = _as_float(baseline.get("mean_time_below_54_pct"))
    baseline_below_70 = _as_float(baseline.get("mean_time_below_70_pct"))
    baseline_supervisor = _as_float(baseline.get("mean_supervisor_intervention_rate_pct"))
    baseline_tir = _as_float(baseline.get("mean_tir_70_180_pct"))

    for name, row in algorithms.items():
        if name == baseline_name:
            continue
        terminated = _as_int(row.get("terminated_early_runs"))
        completion = _as_float(row.get("mean_completion_pct"))
        below_54 = _as_float(row.get("mean_time_below_54_pct"))
        below_70 = _as_float(row.get("mean_time_below_70_pct"))
        supervisor = _as_float(row.get("mean_supervisor_intervention_rate_pct"))
        tir = _as_float(row.get("mean_tir_70_180_pct"))
        deltas = row.get("delta_vs_clinical_baseline") or {}
        below_54_delta = _as_float(deltas.get("time_below_54_pct"), below_54 - baseline_below_54)
        below_70_delta = _as_float(deltas.get("time_below_70_pct"), below_70 - baseline_below_70)
        supervisor_delta = _as_float(
            deltas.get("supervisor_intervention_rate_pct"),
            supervisor - baseline_supervisor,
        )
        tir_delta = _as_float(deltas.get("tir_70_180_pct"), tir - baseline_tir)

        candidate_metrics[name] = {
            "terminated_early_runs": terminated,
            "mean_completion_pct": completion,
            "time_below_54_delta_pct": below_54_delta,
            "time_below_70_delta_pct": below_70_delta,
            "supervisor_rate_delta_pct": supervisor_delta,
            "tir_delta_pct": tir_delta,
        }

        if terminated > profile.max_early_termination_runs:
            critical.append(f"{name}: {terminated} early termination run(s); allowed {profile.max_early_termination_runs}.")
            actions.append(f"{name}: do not promote until held-out runs complete without early termination.")
        if completion < profile.min_completion_pct:
            critical.append(f"{name}: mean completion is {completion:.2f}%, below required {profile.min_completion_pct:.2f}%.")
            actions.append(f"{name}: investigate scenario crash, numerical instability, or safety-stop behavior.")
        if below_54_delta > profile.max_time_below_54_delta_pct:
            critical.append(
                f"{name}: severe hypo burden worsened by {below_54_delta:.2f} percentage points vs baseline."
            )
            actions.append(f"{name}: reject or retrain; severe hypo regression is not acceptable.")
        if below_70_delta > profile.max_time_below_70_delta_pct:
            critical.append(f"{name}: hypo burden worsened by {below_70_delta:.2f} percentage points vs baseline.")
            actions.append(f"{name}: improve hypo avoidance before comparison claims.")
        if supervisor_delta > profile.max_supervisor_rate_delta_pct:
            critical.append(
                f"{name}: supervisor intervention rate worsened by {supervisor_delta:.2f} percentage points."
            )
            actions.append(f"{name}: reduce unsafe proposals before any embedded/hardware export.")
        if tir_delta < -profile.max_tir_drop_pct:
            warnings.append(f"{name}: TIR dropped by {abs(tir_delta):.2f} percentage points vs baseline.")
            actions.append(f"{name}: inspect hyperglycemia/under-delivery before public comparison.")

    if not candidate_metrics:
        warnings.append("No candidate controller was evaluated against the baseline.")
        actions.append("Add at least one local AI controller candidate to the held-out evaluation.")

    metrics = {
        "baseline": baseline_name,
        "baseline_metrics": {
            "mean_time_below_54_pct": baseline_below_54,
            "mean_time_below_70_pct": baseline_below_70,
            "mean_supervisor_intervention_rate_pct": baseline_supervisor,
            "mean_tir_70_180_pct": baseline_tir,
        },
        "candidates": candidate_metrics,
    }
    return _finish_gate(profile=profile, critical=critical, warnings=warnings, actions=actions, metrics=metrics)
