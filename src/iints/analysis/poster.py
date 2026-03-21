from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator
from iints.utils.plotting import (
    IINTS_BLUE,
    IINTS_GOLD,
    IINTS_NAVY,
    IINTS_ORANGE,
    IINTS_RED,
    IINTS_TEAL,
    apply_plot_style,
)


@dataclass(frozen=True)
class PosterScenario:
    label: str
    run_dir: str
    results_csv: str
    duration_hours: float
    total_steps: int
    tir_70_180: float
    tir_below_70: float
    tir_above_180: float
    mean_glucose: float
    max_glucose: float
    min_glucose: float
    supervisor_events: int
    meal_events: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _discover_run_dirs(results_root: Path, limit: int) -> list[Path]:
    bundles: list[Path] = []
    for child in results_root.iterdir():
        if child.is_dir() and (child / "results.csv").is_file():
            bundles.append(child)
    bundles.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return bundles[:limit]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _glucose_column(df: pd.DataFrame) -> str:
    for candidate in ("glucose_actual_mgdl", "glucose_to_algo_mgdl", "glucose"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a glucose column in results.csv")


def _time_column(df: pd.DataFrame) -> str:
    if "time_minutes" in df.columns:
        return "time_minutes"
    raise ValueError("results.csv must include a 'time_minutes' column")


def _override_mask(df: pd.DataFrame) -> pd.Series:
    if "safety_triggered" in df.columns:
        return df["safety_triggered"].fillna(False).astype(bool)
    if {"algo_recommended_insulin_units", "delivered_insulin_units"}.issubset(df.columns):
        return (df["algo_recommended_insulin_units"] - df["delivered_insulin_units"]) > 1e-9
    return pd.Series([False] * len(df), index=df.index)


def _meal_mask(df: pd.DataFrame) -> pd.Series:
    if "carb_intake_grams" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["carb_intake_grams"].fillna(0).astype(float) > 0


def _build_scenario(run_dir: Path, *, label: str | None = None) -> tuple[PosterScenario, pd.DataFrame]:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"results.csv not found in run directory: {run_dir}")

    df = pd.read_csv(results_csv)
    glucose_column = _glucose_column(df)
    time_column = _time_column(df)
    metrics = ClinicalMetricsCalculator().calculate(
        glucose=df[glucose_column],
        timestamp=df[time_column],
    )

    override_mask = _override_mask(df)
    meal_mask = _meal_mask(df)
    duration_hours = float(df[time_column].max()) / 60.0 if len(df) else 0.0

    scenario = PosterScenario(
        label=label or run_dir.name.replace("_", " "),
        run_dir=str(run_dir),
        results_csv=str(results_csv),
        duration_hours=duration_hours,
        total_steps=int(len(df)),
        tir_70_180=float(metrics.tir_70_180),
        tir_below_70=float(metrics.tir_below_70),
        tir_above_180=float(metrics.tir_above_180),
        mean_glucose=float(metrics.mean_glucose),
        max_glucose=float(df[glucose_column].max()),
        min_glucose=float(df[glucose_column].min()),
        supervisor_events=int(override_mask.sum()),
        meal_events=int(meal_mask.sum()),
    )
    return scenario, df


def _plot_single_panel(ax: Any, df: pd.DataFrame, scenario: PosterScenario) -> None:
    glucose_column = _glucose_column(df)
    time_column = _time_column(df)
    time_hours = df[time_column] / 60.0
    override_mask = _override_mask(df)
    meal_mask = _meal_mask(df)

    ax.axhspan(70, 180, alpha=0.14, color=IINTS_TEAL, zorder=0)
    ax.axhline(70, color=IINTS_RED, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(180, color=IINTS_ORANGE, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.plot(time_hours, df[glucose_column], color=IINTS_BLUE, linewidth=2.4, label="Glucose", zorder=2)

    if meal_mask.any():
        ax.scatter(
            time_hours[meal_mask],
            df.loc[meal_mask, glucose_column],
            color=IINTS_GOLD,
            s=38,
            marker="D",
            label="Meal event",
            zorder=3,
        )

    if override_mask.any():
        ax.scatter(
            time_hours[override_mask],
            df.loc[override_mask, glucose_column],
            color=IINTS_RED,
            s=42,
            marker="o",
            label="Supervisor intervention",
            zorder=4,
        )

    ax.set_title(scenario.label, color=IINTS_NAVY, fontweight="bold", pad=12)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_ylim(40, max(300, math.ceil(scenario.max_glucose / 25.0) * 25))
    ax.set_xlim(float(time_hours.min()), float(time_hours.max()) if len(time_hours) else 24.0)
    ax.grid(alpha=0.2)

    info_lines = [
        f"TIR 70-180: {scenario.tir_70_180:.1f}%",
        f"Time <70: {scenario.tir_below_70:.1f}%",
        f"Overrides: {scenario.supervisor_events}",
        f"Meals: {scenario.meal_events}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=IINTS_NAVY,
        bbox={"facecolor": "white", "edgecolor": "#cfd8dc", "boxstyle": "round,pad=0.45", "alpha": 0.92},
    )


def generate_results_poster(
    run_dirs: Sequence[str | Path] | None = None,
    *,
    labels: Sequence[str] | None = None,
    output_path: str | Path = "./results/posters/iints_results_poster.png",
    poster_title: str = "288 Decisions. Every Day. We Test Them All.",
    subtitle: str = "Three IINTS-AF scenarios showing control, stress handling, and supervisor protection.",
    results_root: str | Path = "./results",
    auto_limit: int = 3,
    summary_output_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Generate a poster-style summary image from one or more IINTS run bundles.

    Args:
        run_dirs: Explicit run bundle directories. If omitted, the latest bundles
            under `results_root` are used.
        labels: Optional labels aligned one-to-one with `run_dirs`.
        output_path: PNG path for the generated poster.
        poster_title: Main poster headline.
        subtitle: Supporting subtitle beneath the headline.
        results_root: Root directory to search when `run_dirs` is omitted.
        auto_limit: Maximum number of auto-discovered run bundles.
        summary_output_path: Optional JSON sidecar path. Defaults to `<output>.json`.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    explicit_runs = [Path(item) for item in (run_dirs or [])]
    if not explicit_runs:
        discovered = _discover_run_dirs(Path(results_root), auto_limit)
        if not discovered:
            raise FileNotFoundError(
                f"No run directories with results.csv were found under {results_root}."
            )
        explicit_runs = discovered

    if len(explicit_runs) > 3:
        raise ValueError("Poster generation currently supports up to 3 run directories.")

    label_list = list(labels or [])
    if label_list and len(label_list) != len(explicit_runs):
        raise ValueError("If labels are provided, their count must match the number of run directories.")

    scenarios: list[PosterScenario] = []
    frames: list[pd.DataFrame] = []
    for idx, run_dir in enumerate(explicit_runs):
        scenario, frame = _build_scenario(run_dir, label=label_list[idx] if label_list else None)
        scenarios.append(scenario)
        frames.append(frame)

    apply_plot_style(dpi=180, font_scale=1.15)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 8.6), squeeze=False)
    axis_list = list(axes[0])

    fig.patch.set_facecolor("#f8fbfd")
    fig.suptitle(poster_title, fontsize=24, fontweight="bold", color=IINTS_NAVY, y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=11.5, color=IINTS_NAVY)

    for ax, scenario, frame in zip(axis_list, scenarios, frames):
        _plot_single_panel(ax, frame, scenario)

    handles, labels_for_legend = axis_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_for_legend, loc="upper center", ncol=min(3, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.905))

    footer_lines = [
        f"Scenarios shown: {len(scenarios)}",
        f"Total simulated steps: {sum(item.total_steps for item in scenarios)}",
        f"Total supervisor interventions: {sum(item.supervisor_events for item in scenarios)}",
        "Generated directly from IINTS-AF run bundles.",
    ]
    fig.text(
        0.5,
        0.04,
        "   |   ".join(footer_lines),
        ha="center",
        va="center",
        fontsize=10,
        color=IINTS_NAVY,
    )

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.9))
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary_path = Path(summary_output_path) if summary_output_path is not None else output.with_suffix(".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "poster_title": poster_title,
                "subtitle": subtitle,
                "poster_path": str(output),
                "scenarios": [scenario.to_dict() for scenario in scenarios],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "poster_png": str(output),
        "summary_json": str(summary_path),
    }

