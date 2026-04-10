from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable

import pandas as pd

from iints.analysis.study_engine import slugify_study_token
from iints.research.evaluation import forecast_error_report
from iints.validation.run_validation import compute_run_metrics

CORE_METRIC_KEYS = {
    "tir_70_180": "mean_tir_70_180",
    "tir_below_70": "mean_tir_below_70",
    "tir_below_54": "mean_tir_below_54",
    "tir_above_180": "mean_tir_above_180",
    "tir_above_250": "mean_tir_above_250",
    "supervisor_interventions": "mean_supervisor_interventions",
    "mean_glucose": "mean_glucose",
    "cv": "mean_cv",
    "gmi": "mean_gmi",
}
PAIRWISE_METRICS = (
    "tir_70_180",
    "tir_below_70",
    "tir_above_180",
    "supervisor_interventions",
    "mean_glucose",
    "cv",
)
CALIBRATION_METRICS = (
    "mae",
    "rmse",
    "bias",
    "within_10_mgdl_pct",
    "within_20_mgdl_pct",
    "false_hypo_alarm_rate_pct",
    "missed_hypo_rate_pct",
    "interval_95_coverage_pct",
    "mean_predicted_std_mgdl",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if path.is_file():
        return _load_json(path)
    return None


def _resolve_reference_metrics(path: Path) -> Path:
    if path.is_dir():
        carelink_metrics = path / "carelink_metrics.json"
        if carelink_metrics.is_file():
            return carelink_metrics
        summary_json = path / "reference_metrics.json"
        if summary_json.is_file():
            return summary_json
        raise FileNotFoundError(
            f"Could not find carelink_metrics.json or reference_metrics.json under {path}"
        )
    return path


def _find_run_dirs(root: Path) -> list[Path]:
    if root.is_file() and root.name == "results.csv":
        return [root.parent]
    if (root / "results.csv").is_file():
        return [root]
    run_dirs = sorted({path.parent for path in root.rglob("results.csv")})
    return [path for path in run_dirs if path.is_dir()]


def _clean_numeric(values: Sequence[float | int | None]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def _mean(values: Sequence[float | int | None]) -> float | None:
    cleaned = _clean_numeric(values)
    if not cleaned:
        return None
    return float(mean(cleaned))


def _std(values: Sequence[float | int | None]) -> float | None:
    cleaned = _clean_numeric(values)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(stdev(cleaned))


def _percentile(values: Sequence[float | int | None], quantile: float) -> float | None:
    cleaned = _clean_numeric(values)
    if not cleaned:
        return None
    return float(pd.Series(cleaned).quantile(quantile))


def _ci95_half_width(values: Sequence[float | int | None]) -> float | None:
    cleaned = _clean_numeric(values)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(1.96 * stdev(cleaned) / sqrt(len(cleaned)))


def _stats(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    cleaned = _clean_numeric(values)
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean_value = float(mean(cleaned))
    std_value = 0.0 if len(cleaned) == 1 else float(stdev(cleaned))
    half_width = 0.0 if len(cleaned) == 1 else float(1.96 * std_value / sqrt(len(cleaned)))
    return {
        "count": len(cleaned),
        "mean": mean_value,
        "median": float(median(cleaned)),
        "std": std_value,
        "min": float(min(cleaned)),
        "max": float(max(cleaned)),
        "ci95_low": mean_value - half_width,
        "ci95_high": mean_value + half_width,
    }


def _comparison_mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
    return _mean(values)


def _cohens_d(left: Sequence[float | int | None], right: Sequence[float | int | None]) -> float | None:
    left_values = _clean_numeric(left)
    right_values = _clean_numeric(right)
    if not left_values or not right_values:
        return None
    left_mean = float(mean(left_values))
    right_mean = float(mean(right_values))
    left_var = 0.0 if len(left_values) < 2 else float(stdev(left_values)) ** 2
    right_var = 0.0 if len(right_values) < 2 else float(stdev(right_values)) ** 2
    pooled_denominator = len(left_values) + len(right_values) - 2
    if pooled_denominator <= 0:
        return None
    pooled_variance = (
        ((len(left_values) - 1) * left_var) + ((len(right_values) - 1) * right_var)
    ) / pooled_denominator
    if pooled_variance <= 0:
        return None
    return (left_mean - right_mean) / sqrt(pooled_variance)


def _difference_ci95(
    left: Sequence[float | int | None],
    right: Sequence[float | int | None],
) -> tuple[float | None, float | None, float | None]:
    left_values = _clean_numeric(left)
    right_values = _clean_numeric(right)
    if not left_values or not right_values:
        return None, None, None

    left_mean = float(mean(left_values))
    right_mean = float(mean(right_values))
    difference = left_mean - right_mean
    left_std = 0.0 if len(left_values) < 2 else float(stdev(left_values))
    right_std = 0.0 if len(right_values) < 2 else float(stdev(right_values))
    standard_error = sqrt((left_std**2 / max(len(left_values), 1)) + (right_std**2 / max(len(right_values), 1)))
    half_width = 1.96 * standard_error
    return difference, difference - half_width, difference + half_width


def _first_numeric(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        numeric = float(value)
        if isfinite(numeric):
            return numeric
    return None


def quality_badges_for_metrics(metrics: dict[str, float], *, certified: bool | None = None) -> list[str]:
    badges: list[str] = []
    tir = float(metrics.get("tir_70_180", 0.0))
    below_70 = float(metrics.get("tir_below_70", 0.0))
    below_54 = float(metrics.get("tir_below_54", 0.0))
    above_180 = float(metrics.get("tir_above_180", 0.0))
    above_250 = float(metrics.get("tir_above_250", 0.0))
    cv = float(metrics.get("cv", 0.0))
    interventions = float(metrics.get("supervisor_interventions", 0.0))
    terminated = float(metrics.get("terminated_early", 0.0))

    if tir >= 70.0:
        badges.append("strong_tir")
    if below_70 <= 4.0:
        badges.append("low_hypo_exposure")
    if below_54 <= 1.0:
        badges.append("low_severe_hypo_exposure")
    if cv <= 36.0:
        badges.append("stable_variability")
    if interventions >= 10.0:
        badges.append("supervisor_heavy")
    if above_180 >= 25.0:
        badges.append("hyper_exposed")
    if above_250 >= 5.0:
        badges.append("severe_hyper_exposed")
    if terminated >= 1.0:
        badges.append("terminated_early")
    if certified:
        badges.append("certified_data")
    if not badges:
        badges.append("needs_review")
    return badges


@dataclass(frozen=True)
class StudyRunSummary:
    run_dir: str
    run_id: str
    scenario_name: str
    algorithm: str
    condition_group: str
    seed: int | None
    metrics: dict[str, float]
    certification_grade: str | None
    certified_for_medical_research: bool | None
    quality_badges: list[str]
    baseline_reference: str | None
    baseline_tir_delta_vs_reference: float | None
    baseline_intervention_delta_vs_reference: float | None
    study_preset: str | None = None
    study_arm: str | None = None
    algorithm_id: str | None = None
    algorithm_role: str | None = None
    profile_id: str | None = None
    scenario_slug: str | None = None
    supervisor_enabled: bool | None = None
    corruption_modes: list[str] | None = None
    predictor_uncertainty_mean: float | None = None
    predictor_uncertainty_p95: float | None = None
    predictor_uncertainty_max: float | None = None
    calibration_report: dict[str, Any] | None = None
    calibration_gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "algorithm": self.algorithm,
            "condition_group": self.condition_group,
            "seed": self.seed,
            "metrics": self.metrics,
            "certification_grade": self.certification_grade,
            "certified_for_medical_research": self.certified_for_medical_research,
            "quality_badges": self.quality_badges,
            "baseline_reference": self.baseline_reference,
            "baseline_tir_delta_vs_reference": self.baseline_tir_delta_vs_reference,
            "baseline_intervention_delta_vs_reference": self.baseline_intervention_delta_vs_reference,
            "study_preset": self.study_preset,
            "study_arm": self.study_arm,
            "algorithm_id": self.algorithm_id,
            "algorithm_role": self.algorithm_role,
            "profile_id": self.profile_id,
            "scenario_slug": self.scenario_slug,
            "supervisor_enabled": self.supervisor_enabled,
            "corruption_modes": list(self.corruption_modes or []),
            "predictor_uncertainty_mean": self.predictor_uncertainty_mean,
            "predictor_uncertainty_p95": self.predictor_uncertainty_p95,
            "predictor_uncertainty_max": self.predictor_uncertainty_max,
            "calibration_report": self.calibration_report,
            "calibration_gate": self.calibration_gate,
        }


@dataclass(frozen=True)
class StudySummary:
    study_dir: str
    run_count: int
    aggregate: dict[str, float | int | None]
    aggregate_stats: dict[str, dict[str, float | int | None]]
    certification_comparison: dict[str, float | int | None]
    baseline_summary: dict[str, Any]
    failure_analysis: dict[str, Any]
    external_validation: dict[str, Any] | None
    study_protocol: dict[str, Any] | None
    evidence_rows: list[dict[str, Any]]
    runs: list[StudyRunSummary]
    by_algorithm: dict[str, Any]
    by_profile: dict[str, Any]
    by_arm: dict[str, Any]
    by_scenario: dict[str, Any]
    safety_summary: dict[str, Any]
    pairwise_baseline_deltas: dict[str, Any]
    calibration_summary: dict[str, Any] | None = None
    uncertainty_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_dir": self.study_dir,
            "run_count": self.run_count,
            "aggregate": self.aggregate,
            "aggregate_stats": self.aggregate_stats,
            "certification_comparison": self.certification_comparison,
            "baseline_summary": self.baseline_summary,
            "failure_analysis": self.failure_analysis,
            "external_validation": self.external_validation,
            "study_protocol": self.study_protocol,
            "evidence_rows": self.evidence_rows,
            "runs": [run.to_dict() for run in self.runs],
            "by_algorithm": self.by_algorithm,
            "by_profile": self.by_profile,
            "by_arm": self.by_arm,
            "by_scenario": self.by_scenario,
            "safety_summary": self.safety_summary,
            "pairwise_baseline_deltas": self.pairwise_baseline_deltas,
            "calibration_summary": self.calibration_summary,
            "uncertainty_summary": self.uncertainty_summary,
        }


@dataclass(frozen=True)
class StudyComparison:
    left_label: str
    right_label: str
    delta: dict[str, float | int | None]
    effect_estimates: dict[str, dict[str, float | None]]
    left_summary: dict[str, Any]
    right_summary: dict[str, Any]
    by_algorithm: dict[str, Any]
    by_profile: dict[str, Any]
    by_scenario: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_label": self.left_label,
            "right_label": self.right_label,
            "delta": self.delta,
            "effect_estimates": self.effect_estimates,
            "left_summary": self.left_summary,
            "right_summary": self.right_summary,
            "by_algorithm": self.by_algorithm,
            "by_profile": self.by_profile,
            "by_scenario": self.by_scenario,
        }


def _load_protocol_reference(study_dir: Path) -> dict[str, Any] | None:
    candidates = [
        study_dir / "study_design.json",
        study_dir / "protocol" / "study_design.json",
        study_dir / "protocol" / "study_protocol.json",
    ]
    for candidate in candidates:
        payload = _maybe_json(candidate)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["source_path"] = str(candidate)
            return payload
    return None


def _infer_protocol_metadata(runs: list[StudyRunSummary], protocol_reference: dict[str, Any] | None) -> list[StudyRunSummary]:
    if not protocol_reference:
        return runs
    algorithms = protocol_reference.get("algorithms", []) if isinstance(protocol_reference, dict) else []
    algorithm_by_name = {
        str(item.get("display_name")): item
        for item in algorithms
        if isinstance(item, dict) and item.get("display_name")
    }
    normalized_runs: list[StudyRunSummary] = []
    for run in runs:
        mapping = algorithm_by_name.get(run.algorithm)
        if not isinstance(mapping, dict):
            normalized_runs.append(run)
            continue
        normalized_runs.append(
            replace(
                run,
                algorithm_id=run.algorithm_id or mapping.get("algorithm_id"),
                algorithm_role=run.algorithm_role or mapping.get("role"),
            )
        )
    return normalized_runs


def _build_external_validation(aggregate: dict[str, float | int | None], reference_path: Path) -> dict[str, Any]:
    payload = _load_json(_resolve_reference_metrics(reference_path))
    reference_mean_glucose = _first_numeric(payload, ["mean_glucose_mgdl", "mean_glucose"])
    reference_cv = _first_numeric(payload, ["cv_pct", "cv"])
    reference_tir = _first_numeric(payload, ["time_in_range_70_180_pct", "tir_70_180"])
    reference_below_70 = _first_numeric(payload, ["time_below_70_pct", "tir_below_70"])
    reference_above_180 = _first_numeric(payload, ["time_above_180_pct", "tir_above_180"])

    mean_glucose = _first_numeric(aggregate, ["mean_glucose"])
    mean_cv = _first_numeric(aggregate, ["mean_cv"])
    mean_tir = _first_numeric(aggregate, ["mean_tir_70_180"])
    mean_below_70 = _first_numeric(aggregate, ["mean_tir_below_70"])
    mean_above_180 = _first_numeric(aggregate, ["mean_tir_above_180"])

    delta_glucose = None if mean_glucose is None or reference_mean_glucose is None else mean_glucose - reference_mean_glucose
    delta_cv = None if mean_cv is None or reference_cv is None else mean_cv - reference_cv
    delta_tir = None if mean_tir is None or reference_tir is None else mean_tir - reference_tir
    delta_below_70 = None if mean_below_70 is None or reference_below_70 is None else mean_below_70 - reference_below_70
    delta_above_180 = None if mean_above_180 is None or reference_above_180 is None else mean_above_180 - reference_above_180

    plausibility_verdict = "needs_review"
    if (
        delta_glucose is not None
        and delta_cv is not None
        and delta_tir is not None
        and abs(delta_glucose) <= 20.0
        and abs(delta_cv) <= 10.0
        and abs(delta_tir) <= 12.0
    ):
        plausibility_verdict = "close_match"
    elif delta_glucose is not None and abs(delta_glucose) <= 35.0:
        plausibility_verdict = "broadly_plausible"

    return {
        "reference_path": str(_resolve_reference_metrics(reference_path)),
        "reference_mean_glucose_mgdl": reference_mean_glucose,
        "reference_cv_pct": reference_cv,
        "reference_tir_70_180_pct": reference_tir,
        "reference_time_below_70_pct": reference_below_70,
        "reference_time_above_180_pct": reference_above_180,
        "delta_mean_glucose_mgdl": delta_glucose,
        "delta_cv_pct": delta_cv,
        "delta_tir_70_180_pct": delta_tir,
        "delta_time_below_70_pct": delta_below_70,
        "delta_time_above_180_pct": delta_above_180,
        "plausibility_verdict": plausibility_verdict,
    }


def _uncertainty_metrics_from_results(results_df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    column = "predictor_uncertainty_std_mgdl"
    if column not in results_df.columns:
        return None, None, None
    values = pd.to_numeric(results_df[column], errors="coerce").dropna().tolist()
    if not values:
        return None, None, None
    return float(pd.Series(values).mean()), float(pd.Series(values).quantile(0.95)), float(max(values))


def _calibration_from_results(results_df: pd.DataFrame) -> dict[str, Any] | None:
    observed_column = "glucose_actual_mgdl"
    predicted_column = "predicted_glucose_ai_30min"
    if observed_column not in results_df.columns or predicted_column not in results_df.columns:
        return None
    predicted_std = None
    if "predictor_uncertainty_std_mgdl" in results_df.columns:
        predicted_std = pd.to_numeric(results_df["predictor_uncertainty_std_mgdl"], errors="coerce").to_numpy(dtype=float)
    report = forecast_error_report(
        results_df[observed_column].to_numpy(dtype=float),
        results_df[predicted_column].to_numpy(dtype=float),
        predicted_std,
    )
    return report


def _evidence_row(run: StudyRunSummary) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "scenario_name": run.scenario_name,
        "scenario_slug": run.scenario_slug,
        "condition_group": run.condition_group,
        "study_arm": run.study_arm,
        "study_preset": run.study_preset,
        "algorithm": run.algorithm,
        "algorithm_id": run.algorithm_id,
        "algorithm_role": run.algorithm_role,
        "profile_id": run.profile_id,
        "seed": run.seed,
        "tir_70_180": run.metrics.get("tir_70_180"),
        "tir_below_70": run.metrics.get("tir_below_70"),
        "tir_below_54": run.metrics.get("tir_below_54"),
        "tir_above_180": run.metrics.get("tir_above_180"),
        "tir_above_250": run.metrics.get("tir_above_250"),
        "mean_glucose": run.metrics.get("mean_glucose"),
        "cv": run.metrics.get("cv"),
        "gmi": run.metrics.get("gmi"),
        "terminated_early": run.metrics.get("terminated_early"),
        "supervisor_interventions": run.metrics.get("supervisor_interventions"),
        "certification_grade": run.certification_grade,
        "certified_for_medical_research": run.certified_for_medical_research,
        "baseline_reference": run.baseline_reference,
        "baseline_tir_delta_vs_reference": run.baseline_tir_delta_vs_reference,
        "baseline_intervention_delta_vs_reference": run.baseline_intervention_delta_vs_reference,
        "supervisor_enabled": run.supervisor_enabled,
        "corruption_modes": ",".join(run.corruption_modes or []),
        "predictor_uncertainty_mean": run.predictor_uncertainty_mean,
        "predictor_uncertainty_p95": run.predictor_uncertainty_p95,
        "predictor_uncertainty_max": run.predictor_uncertainty_max,
        "quality_badges": ",".join(run.quality_badges),
        "run_dir": run.run_dir,
    }


def _top_runs(
    runs: list[StudyRunSummary],
    *,
    metric: str,
    reverse: bool,
    limit: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(
        runs,
        key=lambda run: float(run.metrics.get(metric, 0.0)),
        reverse=reverse,
    )[:limit]
    return [
        {
            "run_id": run.run_id,
            "scenario_name": run.scenario_name,
            "scenario_slug": run.scenario_slug,
            "algorithm": run.algorithm,
            "condition_group": run.condition_group,
            "profile_id": run.profile_id,
            "value": float(run.metrics.get(metric, 0.0)),
            "run_dir": run.run_dir,
        }
        for run in ranked
    ]


def _aggregate_core_metrics(runs: list[StudyRunSummary]) -> tuple[dict[str, float | int | None], dict[str, dict[str, float | int | None]]]:
    aggregate: dict[str, float | int | None] = {"run_count": len(runs)}
    aggregate_stats: dict[str, dict[str, float | int | None]] = {}
    for metric_key, aggregate_key in CORE_METRIC_KEYS.items():
        values = [run.metrics.get(metric_key) for run in runs]
        aggregate[aggregate_key] = _mean(values)
        aggregate_stats[metric_key] = _stats(values)

    uncertainty_mean_values = [run.predictor_uncertainty_mean for run in runs]
    uncertainty_p95_values = [run.predictor_uncertainty_p95 for run in runs]
    uncertainty_max_values = [run.predictor_uncertainty_max for run in runs]
    aggregate["mean_predictor_uncertainty_mean"] = _mean(uncertainty_mean_values)
    aggregate["mean_predictor_uncertainty_p95"] = _mean(uncertainty_p95_values)
    aggregate["mean_predictor_uncertainty_max"] = _mean(uncertainty_max_values)
    aggregate_stats["predictor_uncertainty_mean"] = _stats(uncertainty_mean_values)
    aggregate_stats["predictor_uncertainty_p95"] = _stats(uncertainty_p95_values)
    aggregate_stats["predictor_uncertainty_max"] = _stats(uncertainty_max_values)
    return aggregate, aggregate_stats


def _aggregate_calibration_reports(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_reports = [report for report in reports if isinstance(report, dict)]
    if not valid_reports:
        return None
    summary: dict[str, Any] = {"run_count": len(valid_reports)}
    for metric in CALIBRATION_METRICS:
        summary[f"mean_{metric}"] = _mean([_first_numeric(report, [metric]) for report in valid_reports])
    gates = [report.get("calibration_gate") for report in valid_reports if isinstance(report.get("calibration_gate"), dict)]
    if gates:
        summary["gate_pass_count"] = sum(1 for gate in gates if gate.get("passed") is True)
        summary["gate_fail_count"] = sum(1 for gate in gates if gate.get("passed") is False)
        summary["gate_profiles"] = sorted({str(gate.get("profile")) for gate in gates if gate.get("profile")})
    return summary


def _summarize_group(runs: list[StudyRunSummary]) -> dict[str, Any]:
    aggregate, aggregate_stats = _aggregate_core_metrics(runs)
    return {
        "run_count": len(runs),
        "aggregate": aggregate,
        "aggregate_stats": aggregate_stats,
        "certified_runs": sum(1 for run in runs if run.certified_for_medical_research),
        "uncertainty": {
            "mean": aggregate.get("mean_predictor_uncertainty_mean"),
            "p95": aggregate.get("mean_predictor_uncertainty_p95"),
            "max": aggregate.get("mean_predictor_uncertainty_max"),
        },
        "calibration": _aggregate_calibration_reports([run.calibration_report for run in runs if run.calibration_report]),
    }


def _group_summary(runs: list[StudyRunSummary], key_fn: Callable[[StudyRunSummary], str | None]) -> dict[str, Any]:
    groups: dict[str, list[StudyRunSummary]] = {}
    for run in runs:
        key = key_fn(run) or "unknown"
        groups.setdefault(key, []).append(run)
    return {group: _summarize_group(items) for group, items in sorted(groups.items())}


def _pairwise_baseline_deltas(runs: list[StudyRunSummary]) -> dict[str, Any]:
    keyed_runs: dict[tuple[str, str, str, int | None], dict[str, StudyRunSummary]] = {}
    for run in runs:
        arm = run.study_arm or run.condition_group
        profile = run.profile_id or "unknown"
        scenario = run.scenario_slug or run.scenario_name
        key = (arm, profile, scenario, run.seed)
        algo_key = run.algorithm_role or run.algorithm
        keyed_runs.setdefault(key, {})[algo_key] = run
        keyed_runs[key][run.algorithm] = run

    candidate_name = next((run.algorithm for run in runs if run.algorithm_role == "candidate"), None)
    by_baseline: dict[str, dict[str, list[float]]] = {}
    pair_count: dict[str, int] = {}
    for mapping in keyed_runs.values():
        candidate = next((item for item in mapping.values() if isinstance(item, StudyRunSummary) and item.algorithm_role == "candidate"), None)
        if candidate is None:
            continue
        baselines = [
            item
            for item in mapping.values()
            if isinstance(item, StudyRunSummary) and item.algorithm_role in {"baseline", "comparison"}
        ]
        seen: set[str] = set()
        for baseline in baselines:
            if baseline.algorithm in seen:
                continue
            seen.add(baseline.algorithm)
            pair_count[baseline.algorithm] = pair_count.get(baseline.algorithm, 0) + 1
            bucket = by_baseline.setdefault(baseline.algorithm, {metric: [] for metric in PAIRWISE_METRICS})
            for metric in PAIRWISE_METRICS:
                candidate_value = candidate.metrics.get(metric)
                baseline_value = baseline.metrics.get(metric)
                if candidate_value is None or baseline_value is None:
                    continue
                bucket[metric].append(float(candidate_value) - float(baseline_value))

    return {
        "candidate_algorithm": candidate_name,
        "baselines": {
            baseline: {
                "pair_count": pair_count.get(baseline, 0),
                "mean_deltas": {metric: _mean(values) for metric, values in metrics.items()},
                "delta_stats": {metric: _stats(values) for metric, values in metrics.items()},
            }
            for baseline, metrics in sorted(by_baseline.items())
        },
    }


def _uncertainty_distribution(runs: list[StudyRunSummary]) -> dict[str, Any] | None:
    values = [run.predictor_uncertainty_mean for run in runs]
    if not _clean_numeric(values):
        return None
    return {
        "count": len(_clean_numeric(values)),
        "mean": _mean(values),
        "p95": _percentile(values, 0.95),
        "max": max(_clean_numeric(values)),
    }


def _worst_tir_subset(runs: list[StudyRunSummary]) -> list[StudyRunSummary]:
    if not runs:
        return []
    limit = max(1, len(runs) // 4)
    return sorted(runs, key=lambda run: float(run.metrics.get("tir_70_180", 0.0)))[:limit]


def _build_safety_summary(runs: list[StudyRunSummary], certification_comparison: dict[str, Any]) -> dict[str, Any]:
    supervisor_on = [run for run in runs if run.supervisor_enabled is not False]
    supervisor_off = [run for run in runs if run.supervisor_enabled is False]
    heavy_intervention_runs = [run for run in runs if float(run.metrics.get("supervisor_interventions", 0.0)) >= 10.0]
    safe_runs = [
        run
        for run in runs
        if float(run.metrics.get("terminated_early", 0.0)) < 1.0 and float(run.metrics.get("tir_below_54", 0.0)) <= 0.0
    ]
    summary: dict[str, Any] = {
        "certified_vs_uncertified": certification_comparison,
        "supervisor_on_vs_off": {
            "supervisor_on_runs": len(supervisor_on),
            "supervisor_off_runs": len(supervisor_off),
            "mean_tir_70_180_supervisor_on": _mean([run.metrics.get("tir_70_180") for run in supervisor_on]),
            "mean_tir_70_180_supervisor_off": _mean([run.metrics.get("tir_70_180") for run in supervisor_off]),
            "mean_interventions_supervisor_on": _mean([run.metrics.get("supervisor_interventions") for run in supervisor_on]),
            "mean_interventions_supervisor_off": _mean([run.metrics.get("supervisor_interventions") for run in supervisor_off]),
        },
        "severe_hypo_run_count": sum(1 for run in runs if float(run.metrics.get("tir_below_54", 0.0)) > 0.0),
        "terminated_early_run_count": sum(1 for run in runs if float(run.metrics.get("terminated_early", 0.0)) >= 1.0),
        "mean_interventions_by_arm": {
            group: _mean([run.metrics.get("supervisor_interventions") for run in runs if (run.study_arm or run.condition_group) == group])
            for group in sorted({run.study_arm or run.condition_group for run in runs})
        },
        "mean_interventions_by_algorithm": {
            algorithm: _mean([run.metrics.get("supervisor_interventions") for run in runs if run.algorithm == algorithm])
            for algorithm in sorted({run.algorithm for run in runs})
        },
    }
    summary["uncertainty_safe_runs"] = _uncertainty_distribution(safe_runs)
    summary["uncertainty_heavy_intervention_runs"] = _uncertainty_distribution(heavy_intervention_runs)
    summary["uncertainty_worst_tir_runs"] = _uncertainty_distribution(_worst_tir_subset(runs))
    return summary


def _build_calibration_summary(runs: list[StudyRunSummary]) -> dict[str, Any] | None:
    overall = _aggregate_calibration_reports([run.calibration_report for run in runs if run.calibration_report])
    if overall is None:
        return None
    return {
        "overall": overall,
        "by_algorithm": {
            key: value["calibration"]
            for key, value in _group_summary(runs, lambda run: run.algorithm).items()
            if value.get("calibration") is not None
        },
        "by_profile": {
            key: value["calibration"]
            for key, value in _group_summary(runs, lambda run: run.profile_id).items()
            if value.get("calibration") is not None
        },
        "by_scenario": {
            key: value["calibration"]
            for key, value in _group_summary(runs, lambda run: run.scenario_slug or run.scenario_name).items()
            if value.get("calibration") is not None
        },
    }


def _build_uncertainty_summary(runs: list[StudyRunSummary]) -> dict[str, Any] | None:
    overall = _uncertainty_distribution(runs)
    if overall is None:
        return None
    safe_runs = [
        run
        for run in runs
        if float(run.metrics.get("terminated_early", 0.0)) < 1.0 and float(run.metrics.get("tir_below_54", 0.0)) <= 0.0
    ]
    heavy_intervention_runs = [run for run in runs if float(run.metrics.get("supervisor_interventions", 0.0)) >= 10.0]
    return {
        "overall": overall,
        "safe_runs": _uncertainty_distribution(safe_runs),
        "heavy_intervention_runs": _uncertainty_distribution(heavy_intervention_runs),
        "worst_tir_runs": _uncertainty_distribution(_worst_tir_subset(runs)),
    }


def analyze_run_directory(run_dir: Path) -> StudyRunSummary:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing results.csv in {run_dir}")

    results_df = pd.read_csv(results_csv)
    run_metadata = _maybe_json(run_dir / "run_metadata.json") or {}
    config = run_metadata.get("config", {}) if isinstance(run_metadata.get("config"), dict) else {}
    config_json = _maybe_json(run_dir / "config.json") or {}
    if not config and isinstance(config_json, dict):
        config = config_json
    scenario_config = config.get("scenario", {}) if isinstance(config.get("scenario"), dict) else {}
    algorithm_config = config.get("algorithm", {}) if isinstance(config.get("algorithm"), dict) else config.get("algorithm")
    safety_report = (
        _maybe_json(run_dir / "audit" / "audit_summary.json")
        or _maybe_json(run_dir / "audit" / "safety_summary.json")
        or {}
    )
    certification = (
        _maybe_json(run_dir / "certification.json")
        or _maybe_json(run_dir / "audit" / "certification.json")
        or _maybe_json(run_dir / "ai" / "report.signed.mdmp")
    )

    metrics = compute_run_metrics(
        results_df,
        safety_report=safety_report,
        duration_minutes=int(config.get("duration_minutes", 0) or 0) or None,
    )

    baseline_json = _maybe_json(run_dir / "baseline" / "baseline_comparison.json") or {}
    baseline_reference = baseline_json.get("reference") if isinstance(baseline_json.get("reference"), str) else None
    baseline_rows: list[dict[str, Any]] = (
        [row for row in baseline_json.get("rows", []) if isinstance(row, dict)]
        if isinstance(baseline_json.get("rows"), list)
        else []
    )
    if isinstance(algorithm_config, dict):
        metadata = algorithm_config.get("metadata", {})
        current_algorithm = str(metadata.get("name") or algorithm_config.get("class") or run_dir.name)
    else:
        current_algorithm = str(algorithm_config or run_dir.name)
    current_row = next((row for row in baseline_rows if row.get("algorithm") == current_algorithm), None)
    reference_row = next((row for row in baseline_rows if row.get("algorithm") == baseline_reference), None)

    baseline_tir_delta = None
    baseline_intervention_delta = None
    if isinstance(current_row, dict) and isinstance(reference_row, dict):
        if "tir_70_180" in current_row and "tir_70_180" in reference_row:
            baseline_tir_delta = float(current_row["tir_70_180"]) - float(reference_row["tir_70_180"])
        if "bolus_interventions" in current_row and "bolus_interventions" in reference_row:
            baseline_intervention_delta = float(current_row["bolus_interventions"]) - float(reference_row["bolus_interventions"])

    certification_grade = None
    certified = None
    if isinstance(certification, dict):
        certification_grade = certification.get("mdmp_grade") or certification.get("grade")
        if "certified_for_medical_research" in certification:
            certified = bool(certification["certified_for_medical_research"])
        elif certification_grade is not None:
            certified = str(certification_grade) in {"research_grade", "clinical_grade", "ai_ready"}

    scenario_name = str(scenario_config.get("scenario_name") or config.get("scenario_name") or run_dir.name)
    run_id = str(run_metadata.get("run_id") or run_dir.name)
    seed = run_metadata.get("seed", config.get("seed"))
    condition_group = str(
        config.get("condition_group")
        or config.get("study_condition")
        or scenario_config.get("condition_group")
        or scenario_config.get("study_arm")
        or scenario_name
    )

    predictor_uncertainty_mean, predictor_uncertainty_p95, predictor_uncertainty_max = _uncertainty_metrics_from_results(results_df)
    calibration_report = _maybe_json(run_dir / "forecast_evaluation.json") or _calibration_from_results(results_df)
    calibration_gate = calibration_report.get("calibration_gate") if isinstance(calibration_report, dict) else None

    study_preset = config.get("study_protocol_preset") or scenario_config.get("study_protocol_preset")
    study_arm = config.get("study_condition") or scenario_config.get("study_arm") or config.get("study_arm")
    algorithm_id = config.get("algorithm_id") or run_metadata.get("algorithm_id")
    algorithm_role = config.get("algorithm_role") or run_metadata.get("algorithm_role")
    profile_id = config.get("profile_id") or run_metadata.get("profile_id")
    scenario_slug = config.get("scenario_slug") or scenario_config.get("scenario_slug")
    supervisor_enabled = config.get("supervisor_enabled")
    if supervisor_enabled is None:
        supervisor_enabled = scenario_config.get("supervisor_enabled")
    corruption_modes = config.get("corruption_modes") or scenario_config.get("corruption_modes") or []
    if not isinstance(corruption_modes, list):
        corruption_modes = []

    return StudyRunSummary(
        run_dir=str(run_dir),
        run_id=run_id,
        scenario_name=scenario_name,
        algorithm=current_algorithm,
        condition_group=condition_group,
        seed=int(seed) if seed is not None else None,
        metrics=metrics,
        certification_grade=str(certification_grade) if certification_grade is not None else None,
        certified_for_medical_research=certified,
        quality_badges=quality_badges_for_metrics(metrics, certified=certified),
        baseline_reference=baseline_reference,
        baseline_tir_delta_vs_reference=baseline_tir_delta,
        baseline_intervention_delta_vs_reference=baseline_intervention_delta,
        study_preset=str(study_preset) if study_preset is not None else None,
        study_arm=str(study_arm) if study_arm is not None else None,
        algorithm_id=str(algorithm_id) if algorithm_id is not None else slugify_study_token(current_algorithm),
        algorithm_role=str(algorithm_role) if algorithm_role is not None else None,
        profile_id=str(profile_id) if profile_id is not None else None,
        scenario_slug=str(scenario_slug) if scenario_slug is not None else None,
        supervisor_enabled=bool(supervisor_enabled) if supervisor_enabled is not None else None,
        corruption_modes=[str(item) for item in corruption_modes],
        predictor_uncertainty_mean=predictor_uncertainty_mean,
        predictor_uncertainty_p95=predictor_uncertainty_p95,
        predictor_uncertainty_max=predictor_uncertainty_max,
        calibration_report=calibration_report,
        calibration_gate=calibration_gate if isinstance(calibration_gate, dict) else None,
    )


def analyze_study_directory(
    study_dir: Path,
    *,
    external_reference_metrics: Path | None = None,
) -> StudySummary:
    run_dirs = _find_run_dirs(study_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {study_dir}")

    runs = [analyze_run_directory(path) for path in run_dirs]
    protocol_reference = _load_protocol_reference(study_dir)
    runs = _infer_protocol_metadata(runs, protocol_reference)

    aggregate, aggregate_stats = _aggregate_core_metrics(runs)
    certified_runs = [run for run in runs if run.certified_for_medical_research]
    uncertified_runs = [run for run in runs if run.certified_for_medical_research is False or run.certification_grade is None]

    certification_comparison = {
        "certified_runs": len(certified_runs),
        "uncertified_runs": len(uncertified_runs),
        "mean_tir_70_180_certified": _mean([run.metrics.get("tir_70_180") for run in certified_runs]) if certified_runs else None,
        "mean_tir_70_180_uncertified": _mean([run.metrics.get("tir_70_180") for run in uncertified_runs]) if uncertified_runs else None,
        "mean_supervisor_interventions_certified": _mean([run.metrics.get("supervisor_interventions") for run in certified_runs]) if certified_runs else None,
        "mean_supervisor_interventions_uncertified": _mean([run.metrics.get("supervisor_interventions") for run in uncertified_runs]) if uncertified_runs else None,
        "mean_tir_below_70_certified": _mean([run.metrics.get("tir_below_70") for run in certified_runs]) if certified_runs else None,
        "mean_tir_below_70_uncertified": _mean([run.metrics.get("tir_below_70") for run in uncertified_runs]) if uncertified_runs else None,
    }
    certification_comparison["tir_delta_certified_minus_uncertified"] = (
        None
        if certification_comparison["mean_tir_70_180_certified"] is None or certification_comparison["mean_tir_70_180_uncertified"] is None
        else float(certification_comparison["mean_tir_70_180_certified"]) - float(certification_comparison["mean_tir_70_180_uncertified"])
    )

    baseline_rows: list[dict[str, Any]] = []
    for run in runs:
        baseline_json = _maybe_json(Path(run.run_dir) / "baseline" / "baseline_comparison.json") or {}
        if isinstance(baseline_json.get("rows"), list):
            baseline_rows.extend(row for row in baseline_json["rows"] if isinstance(row, dict))

    baseline_summary: dict[str, Any] = {
        "mean_tir_70_180_by_algorithm": {},
        "mean_bolus_interventions_by_algorithm": {},
        "run_quality_badge_counts": {},
        "condition_group_counts": {},
        "algorithm_role_counts": {},
    }
    if baseline_rows:
        algorithms = sorted({str(row.get("algorithm")) for row in baseline_rows if row.get("algorithm")})
        baseline_summary["mean_tir_70_180_by_algorithm"] = {
            algorithm: _comparison_mean([row for row in baseline_rows if row.get("algorithm") == algorithm], "tir_70_180")
            for algorithm in algorithms
        }
        baseline_summary["mean_bolus_interventions_by_algorithm"] = {
            algorithm: _comparison_mean([row for row in baseline_rows if row.get("algorithm") == algorithm], "bolus_interventions")
            for algorithm in algorithms
        }

    badge_counts: dict[str, int] = {}
    for run in runs:
        for badge in run.quality_badges:
            badge_counts[badge] = badge_counts.get(badge, 0) + 1
        baseline_summary["condition_group_counts"][run.condition_group] = (
            int(baseline_summary["condition_group_counts"].get(run.condition_group, 0)) + 1
        )
        role = run.algorithm_role or "unknown"
        baseline_summary["algorithm_role_counts"][role] = int(baseline_summary["algorithm_role_counts"].get(role, 0)) + 1
    baseline_summary["run_quality_badge_counts"] = badge_counts

    failure_analysis = {
        "terminated_early_runs": sum(1 for run in runs if float(run.metrics.get("terminated_early", 0.0)) >= 1.0),
        "severe_hypo_runs": sum(1 for run in runs if float(run.metrics.get("tir_below_54", 0.0)) > 0.0),
        "hypo_exposed_runs": sum(1 for run in runs if float(run.metrics.get("tir_below_70", 0.0)) > 4.0),
        "severe_hyper_runs": sum(1 for run in runs if float(run.metrics.get("tir_above_250", 0.0)) > 5.0),
        "supervisor_heavy_runs": sum(1 for run in runs if "supervisor_heavy" in run.quality_badges),
        "needs_review_runs": sum(1 for run in runs if "needs_review" in run.quality_badges),
        "worst_tir_runs": _top_runs(runs, metric="tir_70_180", reverse=False),
        "highest_intervention_runs": _top_runs(runs, metric="supervisor_interventions", reverse=True),
        "highest_hypo_runs": _top_runs(runs, metric="tir_below_54", reverse=True),
        "highest_hyper_runs": _top_runs(runs, metric="tir_above_250", reverse=True),
    }

    evidence_rows = [_evidence_row(run) for run in runs]
    by_algorithm = _group_summary(runs, lambda run: run.algorithm)
    by_profile = _group_summary(runs, lambda run: run.profile_id)
    by_arm = _group_summary(runs, lambda run: run.study_arm or run.condition_group)
    by_scenario = _group_summary(runs, lambda run: run.scenario_slug or run.scenario_name)
    pairwise_baseline_deltas = _pairwise_baseline_deltas(runs)
    safety_summary = _build_safety_summary(runs, certification_comparison)
    calibration_summary = _build_calibration_summary(runs)
    uncertainty_summary = _build_uncertainty_summary(runs)
    external_validation = (
        _build_external_validation(aggregate, external_reference_metrics)
        if external_reference_metrics is not None
        else None
    )

    return StudySummary(
        study_dir=str(study_dir),
        run_count=len(runs),
        aggregate=aggregate,
        aggregate_stats=aggregate_stats,
        certification_comparison=certification_comparison,
        baseline_summary=baseline_summary,
        failure_analysis=failure_analysis,
        external_validation=external_validation,
        study_protocol=protocol_reference,
        evidence_rows=evidence_rows,
        runs=runs,
        by_algorithm=by_algorithm,
        by_profile=by_profile,
        by_arm=by_arm,
        by_scenario=by_scenario,
        safety_summary=safety_summary,
        pairwise_baseline_deltas=pairwise_baseline_deltas,
        calibration_summary=calibration_summary,
        uncertainty_summary=uncertainty_summary,
    )


def load_study_summary(path: str | Path) -> StudySummary:
    resolved = Path(path)
    if resolved.is_file():
        payload = _load_json(resolved)
    else:
        summary_json = resolved / "study_summary.json"
        if summary_json.is_file():
            payload = _load_json(summary_json)
        else:
            return analyze_study_directory(resolved)

    runs = [
        StudyRunSummary(
            run_dir=str(item["run_dir"]),
            run_id=str(item["run_id"]),
            scenario_name=str(item["scenario_name"]),
            algorithm=str(item["algorithm"]),
            condition_group=str(item.get("condition_group", item.get("scenario_name", "default"))),
            seed=item.get("seed"),
            metrics=dict(item["metrics"]),
            certification_grade=item.get("certification_grade"),
            certified_for_medical_research=item.get("certified_for_medical_research"),
            quality_badges=list(item.get("quality_badges", [])),
            baseline_reference=item.get("baseline_reference"),
            baseline_tir_delta_vs_reference=item.get("baseline_tir_delta_vs_reference"),
            baseline_intervention_delta_vs_reference=item.get("baseline_intervention_delta_vs_reference"),
            study_preset=item.get("study_preset"),
            study_arm=item.get("study_arm"),
            algorithm_id=item.get("algorithm_id"),
            algorithm_role=item.get("algorithm_role"),
            profile_id=item.get("profile_id"),
            scenario_slug=item.get("scenario_slug"),
            supervisor_enabled=item.get("supervisor_enabled"),
            corruption_modes=list(item.get("corruption_modes", [])),
            predictor_uncertainty_mean=item.get("predictor_uncertainty_mean"),
            predictor_uncertainty_p95=item.get("predictor_uncertainty_p95"),
            predictor_uncertainty_max=item.get("predictor_uncertainty_max"),
            calibration_report=dict(item["calibration_report"]) if isinstance(item.get("calibration_report"), dict) else None,
            calibration_gate=dict(item["calibration_gate"]) if isinstance(item.get("calibration_gate"), dict) else None,
        )
        for item in payload.get("runs", [])
    ]
    return StudySummary(
        study_dir=str(payload.get("study_dir", resolved)),
        run_count=int(payload.get("run_count", len(runs))),
        aggregate=dict(payload.get("aggregate", {})),
        aggregate_stats=dict(payload.get("aggregate_stats", {})),
        certification_comparison=dict(payload.get("certification_comparison", {})),
        baseline_summary=dict(payload.get("baseline_summary", {})),
        failure_analysis=dict(payload.get("failure_analysis", {})),
        external_validation=dict(payload["external_validation"]) if isinstance(payload.get("external_validation"), dict) else None,
        study_protocol=dict(payload["study_protocol"]) if isinstance(payload.get("study_protocol"), dict) else None,
        evidence_rows=list(payload.get("evidence_rows", [])),
        runs=runs,
        by_algorithm=dict(payload.get("by_algorithm", {})),
        by_profile=dict(payload.get("by_profile", {})),
        by_arm=dict(payload.get("by_arm", {})),
        by_scenario=dict(payload.get("by_scenario", {})),
        safety_summary=dict(payload.get("safety_summary", {})),
        pairwise_baseline_deltas=dict(payload.get("pairwise_baseline_deltas", {})),
        calibration_summary=dict(payload["calibration_summary"]) if isinstance(payload.get("calibration_summary"), dict) else None,
        uncertainty_summary=dict(payload["uncertainty_summary"]) if isinstance(payload.get("uncertainty_summary"), dict) else None,
    )


def _subgroup_effects(
    left_runs: list[StudyRunSummary],
    right_runs: list[StudyRunSummary],
    key_fn: Callable[[StudyRunSummary], str | None],
) -> dict[str, Any]:
    groups = sorted({key_fn(run) or "unknown" for run in left_runs + right_runs})
    metrics = ("tir_70_180", "supervisor_interventions", "mean_glucose", "tir_below_70", "predictor_uncertainty_mean")
    result: dict[str, Any] = {}
    for group in groups:
        left_group = [run for run in left_runs if (key_fn(run) or "unknown") == group]
        right_group = [run for run in right_runs if (key_fn(run) or "unknown") == group]
        metric_payload: dict[str, Any] = {}
        for metric in metrics:
            if metric.startswith("predictor_uncertainty"):
                left_values = [getattr(run, metric) for run in left_group]
                right_values = [getattr(run, metric) for run in right_group]
            else:
                left_values = [run.metrics.get(metric) for run in left_group]
                right_values = [run.metrics.get(metric) for run in right_group]
            difference, ci95_low, ci95_high = _difference_ci95(left_values, right_values)
            metric_payload[metric] = {
                "left_n": len(_clean_numeric(left_values)),
                "right_n": len(_clean_numeric(right_values)),
                "difference_in_means": difference,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "cohens_d": _cohens_d(left_values, right_values),
            }
        result[group] = metric_payload
    return result


def compare_studies(left: str | Path, right: str | Path, *, left_label: str | None = None, right_label: str | None = None) -> StudyComparison:
    left_summary = load_study_summary(left)
    right_summary = load_study_summary(right)
    left_payload = left_summary.to_dict()
    right_payload = right_summary.to_dict()

    def _delta(metric: str, section: str = "aggregate") -> float | int | None:
        left_value = left_payload.get(section, {}).get(metric)
        right_value = right_payload.get(section, {}).get(metric)
        if left_value is None or right_value is None:
            return None
        return float(left_value) - float(right_value)

    delta = {
        "run_count": left_payload["run_count"] - right_payload["run_count"],
        "mean_tir_70_180": _delta("mean_tir_70_180"),
        "mean_tir_below_70": _delta("mean_tir_below_70"),
        "mean_tir_below_54": _delta("mean_tir_below_54"),
        "mean_tir_above_180": _delta("mean_tir_above_180"),
        "mean_tir_above_250": _delta("mean_tir_above_250"),
        "mean_supervisor_interventions": _delta("mean_supervisor_interventions"),
        "mean_glucose": _delta("mean_glucose"),
        "mean_cv": _delta("mean_cv"),
        "mean_gmi": _delta("mean_gmi"),
        "mean_predictor_uncertainty_mean": _delta("mean_predictor_uncertainty_mean"),
        "certified_runs": _delta("certified_runs", section="certification_comparison"),
        "uncertified_runs": _delta("uncertified_runs", section="certification_comparison"),
        "mean_tir_70_180_certified_gap": _delta("mean_tir_70_180_certified", section="certification_comparison"),
        "mean_tir_70_180_uncertified_gap": _delta("mean_tir_70_180_uncertified", section="certification_comparison"),
        "terminated_early_runs": _delta("terminated_early_runs", section="failure_analysis"),
        "severe_hypo_runs": _delta("severe_hypo_runs", section="failure_analysis"),
        "supervisor_heavy_runs": _delta("supervisor_heavy_runs", section="failure_analysis"),
    }
    if isinstance(left_payload.get("calibration_summary"), dict) and isinstance(right_payload.get("calibration_summary"), dict):
        left_overall = left_payload["calibration_summary"].get("overall", {})
        right_overall = right_payload["calibration_summary"].get("overall", {})
        for metric in ("mean_mae", "mean_rmse", "mean_within_20_mgdl_pct", "mean_interval_95_coverage_pct"):
            left_value = left_overall.get(metric)
            right_value = right_overall.get(metric)
            if left_value is not None and right_value is not None:
                delta[f"calibration_{metric}"] = float(left_value) - float(right_value)

    effect_estimates: dict[str, dict[str, float | None]] = {}
    for metric in ("tir_70_180", "supervisor_interventions", "mean_glucose", "tir_below_70"):
        left_values = [run.metrics.get(metric) for run in left_summary.runs]
        right_values = [run.metrics.get(metric) for run in right_summary.runs]
        difference, ci95_low, ci95_high = _difference_ci95(left_values, right_values)
        effect_estimates[metric] = {
            "difference_in_means": difference,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "cohens_d": _cohens_d(left_values, right_values),
        }

    left_runs = left_summary.runs
    right_runs = right_summary.runs
    return StudyComparison(
        left_label=left_label or Path(left).name,
        right_label=right_label or Path(right).name,
        delta=delta,
        effect_estimates=effect_estimates,
        left_summary=left_payload,
        right_summary=right_payload,
        by_algorithm=_subgroup_effects(left_runs, right_runs, lambda run: run.algorithm),
        by_profile=_subgroup_effects(left_runs, right_runs, lambda run: run.profile_id),
        by_scenario=_subgroup_effects(left_runs, right_runs, lambda run: run.scenario_slug or run.scenario_name),
    )
