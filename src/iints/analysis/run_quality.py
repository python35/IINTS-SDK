from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from iints.analysis.safety_visualizer import write_safety_visualizer
from iints.data.realism_dashboard import write_realism_dashboard
from iints.data.realism_validator import validate_realism_dataset, write_realism_report


CORE_RESULT_COLUMNS = (
    "time_minutes",
    "glucose_actual_mgdl",
    "delivered_insulin_units",
)


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)


def standardize_simulation_for_realism(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert an IINTS results CSV into the generic realism-check schema."""
    if "glucose_actual_mgdl" not in results_df.columns:
        raise ValueError("Realism scoring requires a glucose_actual_mgdl column.")

    frame = pd.DataFrame(index=results_df.index)
    frame["timestamp"] = _series_or_default(results_df, "time_minutes")
    frame["glucose"] = _series_or_default(results_df, "glucose_actual_mgdl")
    frame["carbs"] = _series_or_default(results_df, "carb_intake_grams")
    frame["insulin"] = _series_or_default(results_df, "delivered_insulin_units")
    return frame


def _auto_realism_reference(realism_frame: pd.DataFrame, requested: str | None) -> str | None:
    """Use empirical daily references only when the run is actually day-scale."""
    if requested != "auto":
        return requested

    timestamps = pd.to_numeric(realism_frame.get("timestamp", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(timestamps) < 2:
        return None

    duration_hours = float((timestamps.max() - timestamps.min()) / 60.0)
    carbs = pd.to_numeric(realism_frame.get("carbs", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    meal_count = int((carbs >= 10.0).sum())
    if duration_hours >= 18.0 and meal_count >= 3:
        return "free_living_t1d"
    return None


def _write_run_quality_markdown(
    output_path: Path,
    *,
    run_label: str,
    realism_report: Any,
    selected_reference: str | None,
    reference_selection: str | None,
    quality_summary: Dict[str, Any],
) -> Path:
    failed = [check for check in realism_report.checks if check.status == "failed"]
    warnings = [check for check in realism_report.checks if check.status == "warning"]
    lines = [
        f"# IINTS Run Quality Review - {run_label}",
        "",
        "This review is generated automatically to catch physiologically implausible simulation artifacts before a run is used for demos, reports, or AI research.",
        "",
        "## Summary",
        "",
        f"- Verdict: `{realism_report.verdict}`",
        f"- Realism score: `{realism_report.realism_score:.2f}`",
        f"- Result quality grade: `{quality_summary['grade']}`",
        f"- Result quality score: `{quality_summary['score']:.1f}/100`",
        f"- Reference selection: `{reference_selection}`",
        f"- Applied reference: `{selected_reference or 'none'}`",
        f"- Mean glucose: `{realism_report.metrics.get('mean_glucose_mgdl')} mg/dL`",
        f"- CV: `{realism_report.metrics.get('cv_pct')}%`",
        f"- Max glucose rate: `{realism_report.metrics.get('max_abs_rate_mgdl_per_min')} mg/dL/min`",
        f"- Longest near-flat stretch: `{realism_report.metrics.get('longest_low_motion_minutes')} min`",
        "",
        "## Interpretation",
        "",
        realism_report.summary,
        "",
    ]
    if quality_summary["review_reasons"]:
        lines.extend(["## Result Quality Gate", ""])
        for reason in quality_summary["review_reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
    if failed or warnings:
        lines.extend(["## Review Items", ""])
        for check in failed + warnings:
            lines.append(f"- `{check.status}` - {check.title}: {check.detail}")
        lines.append("")
    else:
        lines.extend(["## Review Items", "", "- No failed or warning realism checks were detected.", ""])

    lines.extend([
        "## Check Breakdown",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ])
    for check in realism_report.checks:
        detail = str(check.detail).replace("|", "\\|")
        lines.append(f"| {check.title} | `{check.status}` | {detail} |")

    lines.extend([
        "",
        "## Research Use Note",
        "",
        "A `likely_realistic` verdict means the trace passes the SDK's plausibility checks. It does not make the SDK a medical device and does not prove clinical validity. For local-AI training or public claims, use external reference profiles and the strict real-data gate.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _count_safety_interventions(results_df: pd.DataFrame, safety_report: Optional[Dict[str, Any]]) -> int:
    if safety_report:
        for key in (
            "bolus_interventions_count",
            "interventions_count",
            "safety_interventions_count",
            "total_overrides",
        ):
            if key in safety_report:
                try:
                    return int(safety_report[key])
                except (TypeError, ValueError):
                    pass
    if "safety_triggered" not in results_df.columns:
        return 0
    return int(results_df["safety_triggered"].fillna(False).astype(bool).sum())


def build_result_quality_summary(
    results_df: pd.DataFrame,
    *,
    realism_report: Any,
    safety_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a deterministic reviewer-facing quality gate for one run."""
    review_reasons: list[str] = []
    score = float(realism_report.realism_score) * 100.0

    missing_columns = [column for column in CORE_RESULT_COLUMNS if column not in results_df.columns]
    if missing_columns:
        review_reasons.append(f"Missing core result columns: {', '.join(missing_columns)}.")
        score -= 25.0

    glucose = pd.to_numeric(results_df.get("glucose_actual_mgdl", pd.Series(dtype=float)), errors="coerce")
    nan_glucose_rows = int(glucose.isna().sum())
    if nan_glucose_rows:
        review_reasons.append(f"{nan_glucose_rows} glucose row(s) are NaN or non-numeric.")
        score -= min(20.0, nan_glucose_rows * 2.0)

    timestamps = pd.to_numeric(results_df.get("time_minutes", pd.Series(dtype=float)), errors="coerce")
    duplicate_timestamp_rows = int(timestamps.duplicated().sum()) if not timestamps.empty else 0
    if duplicate_timestamp_rows:
        review_reasons.append(f"{duplicate_timestamp_rows} duplicate timestamp row(s) detected.")
        score -= min(15.0, duplicate_timestamp_rows * 1.5)

    terminated_early = bool((safety_report or {}).get("terminated_early", False))
    if terminated_early:
        review_reasons.append("Simulation terminated early.")
        score -= 35.0

    intervention_count = _count_safety_interventions(results_df, safety_report)
    if intervention_count:
        review_reasons.append(f"{intervention_count} safety intervention(s) occurred.")
        score -= min(20.0, intervention_count * 0.5)

    fail_soft_count = int((safety_report or {}).get("input_validator_fail_soft_count", 0) or 0)
    if fail_soft_count:
        review_reasons.append(f"{fail_soft_count} input-validator fail-soft event(s) occurred.")
        score -= min(20.0, fail_soft_count * 2.0)

    if realism_report.verdict == "likely_unrealistic":
        review_reasons.append("Realism verdict is likely_unrealistic.")
        score -= 30.0
    elif realism_report.verdict == "needs_review":
        review_reasons.append("Realism verdict needs_review.")
        score -= 10.0

    score = max(0.0, min(100.0, score))
    if terminated_early or missing_columns or realism_report.verdict == "likely_unrealistic":
        grade = "do_not_use"
    elif score < 75.0 or realism_report.verdict == "needs_review" or fail_soft_count:
        grade = "review_before_use"
    else:
        grade = "research_ready"

    return {
        "grade": grade,
        "score": round(score, 3),
        "review_reasons": review_reasons,
        "row_count": int(len(results_df)),
        "missing_columns": missing_columns,
        "nan_glucose_rows": nan_glucose_rows,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "safety_intervention_count": intervention_count,
        "input_validator_fail_soft_count": fail_soft_count,
        "terminated_early": terminated_early,
        "realism_verdict": realism_report.verdict,
        "realism_score": float(realism_report.realism_score),
    }


def write_run_quality_artifacts(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    run_label: Optional[str] = None,
    safety_report: Optional[Dict[str, Any]] = None,
    realism_reference: Optional[str] = "auto",
) -> Dict[str, Any]:
    """Write reviewer-facing quality artifacts for one run.

    The artifacts are intentionally non-blocking: if realism scoring cannot run
    because a CSV is incomplete, the simulation still completes and a warning is
    returned to the caller.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    label = run_label or output_path.name
    outputs: Dict[str, Any] = {}

    try:
        realism_frame = standardize_simulation_for_realism(results_df)
        selected_reference = _auto_realism_reference(realism_frame, realism_reference)
        realism_report = validate_realism_dataset(realism_frame, reference=selected_reference)
        realism_json = output_path / "realism_report.json"
        realism_html = output_path / "realism_dashboard.html"
        realism_markdown = output_path / "run_quality_review.md"
        quality_json = output_path / "run_quality_summary.json"
        write_realism_report(realism_report, realism_json)
        quality_summary = build_result_quality_summary(
            results_df,
            realism_report=realism_report,
            safety_report=safety_report,
        )
        quality_json.write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")
        write_realism_dashboard(
            realism_report,
            realism_frame,
            realism_html,
            title="IINTS Run Realism Review",
            source_label=label,
        )
        _write_run_quality_markdown(
            realism_markdown,
            run_label=label,
            realism_report=realism_report,
            selected_reference=selected_reference,
            reference_selection=realism_reference,
            quality_summary=quality_summary,
        )
        realism_summary = {
            "verdict": realism_report.verdict,
            "realism_score": realism_report.realism_score,
            "summary": realism_report.summary,
            "reference": selected_reference,
            "reference_selection": realism_reference,
            "quality_grade": quality_summary["grade"],
            "quality_score": quality_summary["score"],
        }
        if safety_report is not None:
            safety_report["realism_review"] = realism_summary
            safety_report["run_quality"] = quality_summary
        outputs.update(
            {
                "realism_report_json": str(realism_json),
                "realism_dashboard_html": str(realism_html),
                "run_quality_review_md": str(realism_markdown),
                "run_quality_summary_json": str(quality_json),
                "realism_review": realism_summary,
                "run_quality": quality_summary,
            }
        )
    except Exception as exc:
        outputs["realism_warning"] = str(exc)

    safety_outputs = write_safety_visualizer(
        results_df,
        output_path / "safety_visualizer.html",
        output_json=output_path / "safety_visualizer.json",
        safety_report=safety_report,
        title="IINTS Safety Contract Visualizer",
    )
    outputs["safety_visualizer_html"] = safety_outputs["html"]
    outputs["safety_visualizer_json"] = safety_outputs.get("json")
    return outputs
