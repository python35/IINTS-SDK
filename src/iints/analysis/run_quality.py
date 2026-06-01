from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from iints.analysis.safety_visualizer import write_safety_visualizer
from iints.data.realism_dashboard import write_realism_dashboard
from iints.data.realism_validator import validate_realism_dataset, write_realism_report


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
        write_realism_report(realism_report, realism_json)
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
        )
        realism_summary = {
            "verdict": realism_report.verdict,
            "realism_score": realism_report.realism_score,
            "summary": realism_report.summary,
            "reference": selected_reference,
            "reference_selection": realism_reference,
        }
        if safety_report is not None:
            safety_report["realism_review"] = realism_summary
        outputs.update(
            {
                "realism_report_json": str(realism_json),
                "realism_dashboard_html": str(realism_html),
                "run_quality_review_md": str(realism_markdown),
                "realism_review": realism_summary,
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
