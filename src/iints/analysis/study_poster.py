from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

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
    baseline = payload["baseline_summary"]
    failure_analysis = payload.get("failure_analysis", {})
    external_validation = payload.get("external_validation")
    runs = payload["runs"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    apply_plot_style(dpi=180, font_scale=1.05)
    fig = plt.figure(figsize=(15, 9), facecolor="#f8fbfd")
    grid = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.5, 1.8], hspace=0.35, wspace=0.28)

    fig.suptitle(title, fontsize=24, fontweight="bold", color=IINTS_NAVY, y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11, color=IINTS_NAVY)

    _kpi(fig.add_subplot(grid[0, 0]), "Run count", str(payload["run_count"]), "Total bundles analyzed")
    _kpi(fig.add_subplot(grid[0, 1]), "Mean TIR 70-180", f"{aggregate['mean_tir_70_180']:.1f}%" if aggregate["mean_tir_70_180"] is not None else "n/a", "Average time in range")
    _kpi(fig.add_subplot(grid[0, 2]), "Mean interventions", f"{aggregate['mean_supervisor_interventions']:.1f}" if aggregate["mean_supervisor_interventions"] is not None else "n/a", "Supervisor actions per run")
    _kpi(fig.add_subplot(grid[0, 3]), "Certified vs uncertified", f"{certification['certified_runs']} / {certification['uncertified_runs']}", "Research-grade split")

    ax_tir = fig.add_subplot(grid[1, 0:2])
    scenario_names = [str(run["scenario_name"]) for run in runs]
    tir_values = [float(run["metrics"]["tir_70_180"]) for run in runs]
    ax_tir.bar(range(len(runs)), tir_values, color=IINTS_TEAL)
    ax_tir.axhline(70, color=IINTS_GOLD, linestyle="--", linewidth=1.3)
    ax_tir.set_title("Time In Range Per Run", color=IINTS_NAVY, fontweight="bold")
    ax_tir.set_ylabel("TIR 70-180 (%)")
    ax_tir.set_xticks(range(len(runs)))
    ax_tir.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax_tir.grid(axis="y", alpha=0.2)

    ax_interventions = fig.add_subplot(grid[1, 2:4])
    intervention_values = [float(run["metrics"]["supervisor_interventions"]) for run in runs]
    ax_interventions.bar(range(len(runs)), intervention_values, color=IINTS_RED)
    ax_interventions.set_title("Supervisor Interventions Per Run", color=IINTS_NAVY, fontweight="bold")
    ax_interventions.set_ylabel("Interventions")
    ax_interventions.set_xticks(range(len(runs)))
    ax_interventions.set_xticklabels(scenario_names, rotation=25, ha="right")
    ax_interventions.grid(axis="y", alpha=0.2)

    ax_baseline = fig.add_subplot(grid[2, 0:2])
    tir_by_algorithm = baseline.get("mean_tir_70_180_by_algorithm", {})
    if tir_by_algorithm:
        labels = list(tir_by_algorithm.keys())
        values = [float(tir_by_algorithm[item]) for item in labels if tir_by_algorithm[item] is not None]
        labels = [item for item in labels if tir_by_algorithm[item] is not None]
        ax_baseline.bar(range(len(labels)), values, color=IINTS_BLUE)
        ax_baseline.set_xticks(range(len(labels)))
        ax_baseline.set_xticklabels(labels, rotation=20, ha="right")
        ax_baseline.set_ylabel("Mean TIR 70-180 (%)")
    ax_baseline.set_title("Baseline Comparison", color=IINTS_NAVY, fontweight="bold")
    ax_baseline.grid(axis="y", alpha=0.2)

    ax_notes = fig.add_subplot(grid[2, 2:4])
    ax_notes.axis("off")
    badge_counts = baseline.get("run_quality_badge_counts", {})
    top_badges = sorted(badge_counts.items(), key=lambda item: (-item[1], item[0]))
    lines = [
        "Key Findings",
        "",
        f"- Mean glucose: {aggregate['mean_glucose']:.1f} mg/dL" if aggregate["mean_glucose"] is not None else "- Mean glucose: n/a",
        f"- Mean CV: {aggregate['mean_cv']:.1f}%" if aggregate["mean_cv"] is not None else "- Mean CV: n/a",
        f"- Certified TIR advantage: {certification['tir_delta_certified_minus_uncertified']:.1f}%" if certification["tir_delta_certified_minus_uncertified"] is not None else "- Certified TIR advantage: n/a",
        f"- Severe hypo runs: {failure_analysis.get('severe_hypo_runs', 'n/a')}",
        f"- Early terminations: {failure_analysis.get('terminated_early_runs', 'n/a')}",
        "",
        "Top run badges:",
    ]
    if top_badges:
        lines.extend([f"- {badge}: {count}" for badge, count in top_badges[:6]])
    else:
        lines.append("- no badge counts available")
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
    ax_notes.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        color=IINTS_NAVY,
        bbox={"facecolor": "white", "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.55", "alpha": 0.95},
        transform=ax_notes.transAxes,
    )

    fig.text(
        0.5,
        0.03,
        "Built from IINTS run bundles with simulation, certification, and evidence aggregation.",
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
