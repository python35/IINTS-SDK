from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from iints.analysis.study_analysis import StudySummary, load_study_summary
from iints.utils.plotting import IINTS_BLUE, IINTS_GOLD, IINTS_NAVY, IINTS_RED, IINTS_TEAL, apply_plot_style


def _kpi(ax: Any, title: str, value: str, subtitle: str) -> None:
    ax.axis("off")
    ax.set_facecolor("#f8fbfd")
    ax.text(0.02, 0.86, title, fontsize=12, fontweight="bold", color=IINTS_NAVY, transform=ax.transAxes)
    ax.text(0.02, 0.45, value, fontsize=24, fontweight="bold", color=IINTS_BLUE, transform=ax.transAxes)
    ax.text(0.02, 0.18, subtitle, fontsize=10, color=IINTS_NAVY, transform=ax.transAxes)


def _load_summary(summary_input: str | Path | StudySummary) -> StudySummary:
    if isinstance(summary_input, StudySummary):
        return summary_input
    return load_study_summary(summary_input)


def _format_metric(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}{suffix}"


def _candidate_vs_baseline_panel(ax: Any, payload: dict[str, Any]) -> None:
    pairwise = payload.get("pairwise_baseline_deltas", {}) if isinstance(payload.get("pairwise_baseline_deltas"), dict) else {}
    baselines = pairwise.get("baselines", {}) if isinstance(pairwise.get("baselines"), dict) else {}
    if baselines:
        labels = list(baselines.keys())
        values = [
            float((details.get("mean_deltas", {}) or {}).get("tir_70_180") or 0.0)
            for details in baselines.values()
        ]
        colors = [IINTS_BLUE if value >= 0 else IINTS_RED for value in values]
        ax.bar(range(len(labels)), values, color=colors)
        ax.axhline(0.0, color=IINTS_NAVY, linewidth=1.0, alpha=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Candidate - baseline TIR (%)")
        candidate = pairwise.get("candidate_algorithm") or "Candidate"
        ax.set_title(f"{candidate} vs Baselines", color=IINTS_NAVY, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        return

    baseline = payload.get("baseline_summary", {}) if isinstance(payload.get("baseline_summary"), dict) else {}
    tir_by_algorithm = baseline.get("mean_tir_70_180_by_algorithm", {}) if isinstance(baseline.get("mean_tir_70_180_by_algorithm"), dict) else {}
    labels = [label for label, value in tir_by_algorithm.items() if value is not None]
    values = [float(tir_by_algorithm[label]) for label in labels]
    if labels:
        ax.bar(range(len(labels)), values, color=IINTS_BLUE)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Mean TIR 70-180 (%)")
    else:
        ax.text(0.5, 0.5, "No baseline data", ha="center", va="center", color=IINTS_NAVY, transform=ax.transAxes)
    ax.set_title("Baseline Comparison", color=IINTS_NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


def _profile_heatmap(ax: Any, payload: dict[str, Any]) -> None:
    by_profile = payload.get("by_profile", {}) if isinstance(payload.get("by_profile"), dict) else {}
    if not by_profile:
        ax.axis("off")
        ax.text(0.5, 0.5, "No profile-level study data", ha="center", va="center", color=IINTS_NAVY, transform=ax.transAxes)
        return

    profile_labels = list(by_profile.keys())
    metrics = ["mean_tir_70_180", "mean_tir_below_70", "mean_supervisor_interventions"]
    column_labels = ["TIR", "<70", "Interventions"]
    matrix = []
    for profile in profile_labels:
        aggregate = by_profile[profile].get("aggregate", {}) if isinstance(by_profile[profile], dict) else {}
        matrix.append([
            float(aggregate.get("mean_tir_70_180") or 0.0),
            float(aggregate.get("mean_tir_below_70") or 0.0),
            float(aggregate.get("mean_supervisor_interventions") or 0.0),
        ])
    data = np.array(matrix, dtype=float)
    image = ax.imshow(data, aspect="auto", cmap="Blues")
    ax.set_title("Profile Heatmap", color=IINTS_NAVY, fontweight="bold")
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels)
    ax.set_yticks(range(len(profile_labels)))
    ax.set_yticklabels(profile_labels)
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            ax.text(col_idx, row_idx, f"{data[row_idx, col_idx]:.1f}", ha="center", va="center", color=IINTS_NAVY, fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _safety_panel(ax: Any, payload: dict[str, Any]) -> None:
    safety = payload.get("safety_summary", {}) if isinstance(payload.get("safety_summary"), dict) else {}
    on_off = safety.get("supervisor_on_vs_off", {}) if isinstance(safety.get("supervisor_on_vs_off"), dict) else {}
    labels = ["Severe hypo", "Early stop", "On interv.", "Off interv."]
    values = [
        float(safety.get("severe_hypo_run_count") or 0.0),
        float(safety.get("terminated_early_run_count") or 0.0),
        float(on_off.get("mean_interventions_supervisor_on") or 0.0),
        float(on_off.get("mean_interventions_supervisor_off") or 0.0),
    ]
    ax.bar(range(len(labels)), values, color=[IINTS_RED, IINTS_RED, IINTS_TEAL, IINTS_BLUE])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Safety Outcomes", color=IINTS_NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


def _calibration_panel(ax: Any, payload: dict[str, Any]) -> None:
    calibration = payload.get("calibration_summary", {}) if isinstance(payload.get("calibration_summary"), dict) else {}
    overall = calibration.get("overall", {}) if isinstance(calibration.get("overall"), dict) else {}
    by_algorithm = calibration.get("by_algorithm", {}) if isinstance(calibration.get("by_algorithm"), dict) else {}

    if by_algorithm:
        labels = list(by_algorithm.keys())
        mae = [float((by_algorithm[label] or {}).get("mean_mae") or 0.0) for label in labels]
        rmse = [float((by_algorithm[label] or {}).get("mean_rmse") or 0.0) for label in labels]
        positions = np.arange(len(labels))
        width = 0.36
        ax.bar(positions - width / 2, mae, width=width, color=IINTS_BLUE, label="MAE")
        ax.bar(positions + width / 2, rmse, width=width, color=IINTS_GOLD, label="RMSE")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylabel("mg/dL")
    elif overall:
        labels = ["MAE", "RMSE", "Coverage"]
        values = [
            float(overall.get("mean_mae") or 0.0),
            float(overall.get("mean_rmse") or 0.0),
            float(overall.get("mean_interval_95_coverage_pct") or 0.0),
        ]
        ax.bar(range(len(labels)), values, color=[IINTS_BLUE, IINTS_GOLD, IINTS_TEAL])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center", color=IINTS_NAVY, transform=ax.transAxes)
        return

    ax.set_title("Calibration Panel", color=IINTS_NAVY, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)


def _uncertainty_panel(ax: Any, payload: dict[str, Any]) -> None:
    uncertainty = payload.get("uncertainty_summary", {}) if isinstance(payload.get("uncertainty_summary"), dict) else {}
    if not uncertainty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center", color=IINTS_NAVY, transform=ax.transAxes)
        return

    labels = ["Overall", "Safe", "Heavy", "Worst TIR"]
    buckets = [
        uncertainty.get("overall", {}),
        uncertainty.get("safe_runs", {}),
        uncertainty.get("heavy_intervention_runs", {}),
        uncertainty.get("worst_tir_runs", {}),
    ]
    mean_values = [float((bucket or {}).get("mean") or 0.0) for bucket in buckets]
    p95_values = [float((bucket or {}).get("p95") or 0.0) for bucket in buckets]
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, mean_values, width=width, color=IINTS_TEAL, label="Mean std")
    ax.bar(positions + width / 2, p95_values, width=width, color=IINTS_RED, label="P95 std")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Predictor std (mg/dL)")
    ax.set_title("Uncertainty vs Risk", color=IINTS_NAVY, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)


def _notes_panel(ax: Any, payload: dict[str, Any]) -> None:
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    certification = payload.get("certification_comparison", {}) if isinstance(payload.get("certification_comparison"), dict) else {}
    failure_analysis = payload.get("failure_analysis", {}) if isinstance(payload.get("failure_analysis"), dict) else {}
    external_validation = payload.get("external_validation")
    calibration = payload.get("calibration_summary", {}) if isinstance(payload.get("calibration_summary"), dict) else {}
    uncertainty = payload.get("uncertainty_summary", {}) if isinstance(payload.get("uncertainty_summary"), dict) else {}
    overall_calibration = calibration.get("overall", {}) if isinstance(calibration.get("overall"), dict) else {}
    overall_uncertainty = uncertainty.get("overall", {}) if isinstance(uncertainty.get("overall"), dict) else {}

    lines = [
        "Key Findings",
        "",
        f"- Mean glucose: {_format_metric(aggregate.get('mean_glucose'), ' mg/dL')}",
        f"- Mean CV: {_format_metric(aggregate.get('mean_cv'), '%')}",
        f"- Certified TIR advantage: {_format_metric(certification.get('tir_delta_certified_minus_uncertified'), '%')}",
        f"- Severe hypo runs: {failure_analysis.get('severe_hypo_runs', 'n/a')}",
        f"- Early terminations: {failure_analysis.get('terminated_early_runs', 'n/a')}",
    ]
    if overall_uncertainty:
        lines.extend(
            [
                "",
                "Uncertainty summary:",
                f"- Mean predictor std: {_format_metric(overall_uncertainty.get('mean'), ' mg/dL')}",
                f"- P95 predictor std: {_format_metric(overall_uncertainty.get('p95'), ' mg/dL')}",
            ]
        )
    if overall_calibration:
        lines.extend(
            [
                "",
                "Calibration summary:",
                f"- Mean MAE: {_format_metric(overall_calibration.get('mean_mae'), ' mg/dL')}",
                f"- Mean RMSE: {_format_metric(overall_calibration.get('mean_rmse'), ' mg/dL')}",
                f"- Interval coverage: {_format_metric(overall_calibration.get('mean_interval_95_coverage_pct'), '%')}",
            ]
        )
    if isinstance(external_validation, dict):
        lines.extend(
            [
                "",
                "External plausibility:",
                f"- Verdict: {external_validation.get('plausibility_verdict', 'n/a')}",
                f"- Mean glucose delta: {external_validation.get('delta_mean_glucose_mgdl', 'n/a')}",
                f"- TIR delta: {external_validation.get('delta_tir_70_180_pct', 'n/a')}",
            ]
        )
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        color=IINTS_NAVY,
        bbox={"facecolor": "white", "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.55", "alpha": 0.95},
        transform=ax.transAxes,
    )


def generate_study_poster(
    summary_input: str | Path | StudySummary,
    *,
    output_path: str | Path = "results/study_poster.png",
    title: str = "IINTS Study Results",
    subtitle: str = "Simulation evidence across runs, safety behavior, and certification quality.",
    summary_output_path: str | Path | None = None,
) -> dict[str, str]:
    summary = _load_summary(summary_input)
    payload = summary.to_dict()
    aggregate = payload["aggregate"]
    certification = payload["certification_comparison"]
    runs = payload["runs"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    apply_plot_style(dpi=180, font_scale=1.02)
    fig = plt.figure(figsize=(16, 13), facecolor="#f8fbfd")
    grid = fig.add_gridspec(4, 4, height_ratios=[0.9, 1.5, 1.5, 1.4], hspace=0.4, wspace=0.35)

    fig.suptitle(title, fontsize=24, fontweight="bold", color=IINTS_NAVY, y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11, color=IINTS_NAVY)

    _kpi(fig.add_subplot(grid[0, 0]), "Run count", str(payload["run_count"]), "Total bundles analyzed")
    _kpi(fig.add_subplot(grid[0, 1]), "Mean TIR 70-180", _format_metric(aggregate.get("mean_tir_70_180"), "%"), "Average time in range")
    _kpi(fig.add_subplot(grid[0, 2]), "Mean interventions", _format_metric(aggregate.get("mean_supervisor_interventions")), "Supervisor actions per run")
    _kpi(fig.add_subplot(grid[0, 3]), "Certified vs uncertified", f"{certification.get('certified_runs', 0)} / {certification.get('uncertified_runs', 0)}", "Research-grade split")

    ax_tir = fig.add_subplot(grid[1, 0:2])
    scenario_names = [str(run["scenario_name"]) for run in runs]
    tir_values = [float(run["metrics"].get("tir_70_180", 0.0)) for run in runs]
    ax_tir.bar(range(len(runs)), tir_values, color=IINTS_TEAL)
    ax_tir.axhline(70, color=IINTS_GOLD, linestyle="--", linewidth=1.3)
    ax_tir.set_title("Time In Range Per Run", color=IINTS_NAVY, fontweight="bold")
    ax_tir.set_ylabel("TIR 70-180 (%)")
    ax_tir.set_xticks(range(len(runs)))
    ax_tir.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax_tir.grid(axis="y", alpha=0.2)

    ax_baseline = fig.add_subplot(grid[1, 2:4])
    _candidate_vs_baseline_panel(ax_baseline, payload)

    ax_heatmap = fig.add_subplot(grid[2, 0:2])
    _profile_heatmap(ax_heatmap, payload)

    ax_safety = fig.add_subplot(grid[2, 2])
    _safety_panel(ax_safety, payload)

    ax_calibration = fig.add_subplot(grid[2, 3])
    _calibration_panel(ax_calibration, payload)

    ax_uncertainty = fig.add_subplot(grid[3, 0:2])
    _uncertainty_panel(ax_uncertainty, payload)

    ax_notes = fig.add_subplot(grid[3, 2:4])
    _notes_panel(ax_notes, payload)

    fig.text(
        0.5,
        0.03,
        "Built from IINTS run bundles with study protocol metadata, safety aggregation, and subgroup evidence.",
        ha="center",
        fontsize=10,
        color=IINTS_NAVY,
    )
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_output = Path(summary_output_path) if summary_output_path is not None else output.with_suffix(".json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"poster_png": str(output), "poster_summary_json": str(summary_output)}
