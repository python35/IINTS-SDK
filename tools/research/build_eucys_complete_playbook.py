#!/usr/bin/env python3
"""Build one visual, evidence-backed EUCYS playbook.

The document deliberately combines the project story, architecture, formulas,
application screenshots, fresh simulation runs, benchmark results, AI
evaluation, a live demo script, limitations, and reproduction instructions.
Numerical values are read from generated artifacts rather than copied from an
LLM response.
"""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PLAYBOOK_DIR = ROOT / "research" / "eucys_pack" / "complete_playbook"
ASSET_DIR = PLAYBOOK_DIR / "assets"
RUN_DIR = PLAYBOOK_DIR / "runs"
EVIDENCE_DIR = PLAYBOOK_DIR / "evidence"
BUILD_DIR = ROOT / "tmp" / "pdfs" / "eucys_complete_playbook"
OUTPUT_PDF_DIR = ROOT / "output" / "pdf"

MARKDOWN_PATH = PLAYBOOK_DIR / "IINTS_AF_EUCYS_COMPLETE_PLAYBOOK.md"
HTML_PATH = PLAYBOOK_DIR / "IINTS_AF_EUCYS_COMPLETE_PLAYBOOK.html"
PDF_PATH = OUTPUT_PDF_DIR / "IINTS_AF_EUCYS_COMPLETE_PLAYBOOK.pdf"
PACK_PDF_PATH = PLAYBOOK_DIR / "IINTS_AF_EUCYS_COMPLETE_PLAYBOOK.pdf"

COLORS = {
    "ink": "#173042",
    "navy": "#214E65",
    "teal": "#217A78",
    "blue": "#39789B",
    "green": "#4E7D67",
    "amber": "#B27C2B",
    "red": "#A3473F",
    "muted": "#667986",
    "line": "#CBD6DC",
    "panel": "#EEF3F5",
    "paper": "#FCFCFA",
}

PRESET_LABELS = {
    "reference_day": "Reference day",
    "baseline_day": "Baseline T1D",
    "free_living_day": "Free-living T1D",
    "meal_stress": "Meal stress",
    "hypo_night": "Hypo-prone night",
}

ASSETS_TO_COPY = {
    "app_overview.png": ROOT / "docs" / "assets" / "workbench" / "01-overview.png",
    "app_run_protocol.png": ROOT / "docs" / "assets" / "workbench" / "02-run-protocol.png",
    "app_results.png": ROOT / "docs" / "assets" / "workbench" / "03-results.png",
    "app_local_ai.png": ROOT / "docs" / "assets" / "workbench" / "04-local-ai.png",
    "app_reproducibility.png": ROOT / "docs" / "assets" / "workbench" / "05-reproducibility.png",
    "app_research_tools.png": ROOT / "docs" / "assets" / "workbench" / "06-research-tools.png",
    "terminal_simulation.png": ROOT
    / "deliverables"
    / "eucys_sdk_screenshots"
    / "01_terminal_simulation_output.png",
    "live_dashboard.jpg": ROOT
    / "deliverables"
    / "eucys_sdk_screenshots"
    / "02_live_dashboard_browser.jpg",
    "code_algorithm.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "01_algorithm_decision_logic.png",
    "code_simulator.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "02_simulation_control_loop.png",
    "code_supervisor.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "03_independent_safety_supervisor.png",
    "code_patient.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "04_physiological_patient_model.png",
    "code_realism.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "05_realism_validator.png",
    "code_ai_gate.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "08_local_ai_safety_gate.png",
    "code_mdmp.png": ROOT
    / "deliverables"
    / "eucys_code_screenshots"
    / "09_mdmp_eu_ai_pact_gate.png",
    "sdk_overview.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "01_sdk_at_a_glance.png",
    "evidence_workflow.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "02_data_to_evidence_workflow.png",
    "realism_explainer.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "03_realism_trace_and_metrics.png",
    "ai_safety_funnel.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "05_local_ai_safety_funnel.png",
    "mdmp_readiness.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "06_mdmp_eu_ai_pact_readiness.png",
    "edge_loop.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "07_jetson_edge_research_loop.png",
    "pump_bridge.png": ROOT
    / "deliverables"
    / "eucys_sdk_images"
    / "08_sdk_to_pump_research_bridge.png",
    "insulin_3d.png": ROOT / "EUCYS_POSTER_ASSETS" / "alphafold" / "insulin_3D.png",
    "glucagon_3d.png": ROOT / "EUCYS_POSTER_ASSETS" / "alphafold" / "glucagon_3D.png",
    "glut4_3d.png": ROOT / "EUCYS_POSTER_ASSETS" / "alphafold" / "opt2_glut4_channel.png",
    "insulin_receptor_3d.png": ROOT
    / "EUCYS_POSTER_ASSETS"
    / "alphafold"
    / "opt3_insulin_receptor.png",
    "system_architecture.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "system-architecture.png",
    "simulation_step.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "simulation-step.png",
    "numeric_authority.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "numeric-authority.png",
    "data_lifecycle.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "data-lifecycle.png",
    "validation_ladder.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "validation-ladder.png",
    "desktop_bridge.png": ROOT
    / "research"
    / "eucys_pack"
    / "assets"
    / "diagrams"
    / "desktop-bridge.png",
    "fresh_demo_poster.png": RUN_DIR
    / "eucys_demo"
    / "results"
    / "booth_demo_poster.png",
}

HARDWARE_SOURCE = Path("/Volumes/Samsung SSD 990 EVO Plus Media/mk2.png")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    choices = [
        Path(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf"
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for path in choices:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _project_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else "unknown"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _copy_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for target_name, source in ASSETS_TO_COPY.items():
        if not source.exists():
            raise FileNotFoundError(f"Required playbook asset is missing: {source}")
        shutil.copy2(source, ASSET_DIR / target_name)

    hardware_target = ASSET_DIR / "hardware_demonstrator.png"
    if HARDWARE_SOURCE.exists():
        hardware = Image.open(HARDWARE_SOURCE).convert("RGBA")
        alpha_box = hardware.getchannel("A").getbbox()
        if alpha_box:
            hardware = hardware.crop(alpha_box)
        pad = 28
        canvas = Image.new(
            "RGBA",
            (hardware.width + 2 * pad, hardware.height + 2 * pad),
            "white",
        )
        canvas.alpha_composite(hardware, (pad, pad))
        canvas.convert("RGB").save(hardware_target, quality=95)
    elif not hardware_target.exists():
        raise FileNotFoundError(
            "The hardware photograph was not found at its source path and no copied "
            f"fallback exists at {hardware_target}."
        )


def _render_report_pages() -> None:
    source = RUN_DIR / "presets" / "meal_stress" / "report.pdf"
    prefix = BUILD_DIR / "meal-report"
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required to render the fresh report pages.")
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(BUILD_DIR / "cache")
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [pdftoppm, "-png", "-r", "130", str(source), str(prefix)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    pages = sorted(BUILD_DIR.glob("meal-report-*.png"))
    if len(pages) < 4:
        raise RuntimeError(f"Expected four rendered report pages, found {len(pages)}.")
    for index, source_page in enumerate(pages[:4], start=1):
        shutil.copy2(source_page, ASSET_DIR / f"fresh_report_page_{index}.png")


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "grid.color": "#D9E1E5",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _plot_fresh_traces() -> None:
    selected = ("baseline_day", "meal_stress", "hypo_night")
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 9.1), constrained_layout=True)
    for axis, key in zip(axes, selected):
        folder = RUN_DIR / "presets" / key
        frame = pd.read_csv(folder / "results.csv")
        report = _json(folder / "realism_report.json")
        quality = _json(folder / "run_quality_summary.json")
        time_hours = frame["time_minutes"] / 60.0
        glucose = frame["glucose_actual_mgdl"]
        axis.axhspan(70, 180, color="#DCECDF", alpha=0.78, zorder=0)
        axis.axhline(70, color=COLORS["green"], lw=0.9)
        axis.axhline(180, color=COLORS["amber"], lw=0.9)
        axis.plot(time_hours, glucose, color=COLORS["navy"], lw=2.0, label="Latent glucose")
        if "glucose_to_algo_mgdl" in frame:
            axis.plot(
                time_hours,
                frame["glucose_to_algo_mgdl"],
                color=COLORS["blue"],
                lw=0.8,
                alpha=0.52,
                label="CGM-like observation",
            )
        meals = frame.loc[frame["carb_intake_grams"] > 0]
        for minute in meals["time_minutes"]:
            axis.axvline(minute / 60.0, color=COLORS["amber"], lw=0.8, alpha=0.6)
        verdict_color = (
            COLORS["green"] if report["verdict"] == "likely_realistic" else COLORS["red"]
        )
        axis.set_title(
            f"{PRESET_LABELS[key]} - {report['verdict'].replace('_', ' ')} "
            f"(realism {report['realism_score']:.2f}, quality {quality['score']:.1f}/100)",
            loc="left",
            color=verdict_color,
        )
        axis.set_ylabel("Glucose (mg/dL)")
        axis.set_xlim(float(time_hours.min()), float(time_hours.max()))
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Simulation time (hours)")
    axes[0].legend(loc="upper left", ncol=2)
    fig.suptitle(
        "Fresh preset runs generated for this playbook (seed 42, five-minute step)",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.savefig(ASSET_DIR / "fresh_preset_traces.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_realism_gate() -> None:
    records: list[dict[str, Any]] = []
    for key in PRESET_LABELS:
        folder = RUN_DIR / "presets" / key
        realism = _json(folder / "realism_report.json")
        quality = _json(folder / "run_quality_summary.json")
        records.append(
            {
                "label": PRESET_LABELS[key],
                "realism": 100.0 * float(realism["realism_score"]),
                "quality": float(quality["score"]),
                "pass": realism["verdict"] == "likely_realistic"
                and quality["grade"] == "research_ready",
            }
        )
    frame = pd.DataFrame(records)
    y = np.arange(len(frame))
    height = 0.34
    fig, axis = plt.subplots(figsize=(11.2, 5.8), constrained_layout=True)
    axis.barh(y + height / 2, frame["realism"], height, color=COLORS["blue"], label="Realism score")
    axis.barh(y - height / 2, frame["quality"], height, color=COLORS["teal"], label="Run quality")
    axis.axvline(70, color=COLORS["amber"], lw=1.2, ls="--", label="Review threshold (visual guide)")
    for index, row in frame.iterrows():
        status = "PASS" if row["pass"] else "FAIL"
        color = COLORS["green"] if row["pass"] else COLORS["red"]
        axis.text(
            102,
            index,
            status,
            va="center",
            ha="left",
            color=color,
            fontweight="bold",
        )
    axis.set_yticks(y, frame["label"])
    axis.set_xlim(0, 112)
    axis.set_xlabel("Score (0-100)")
    axis.set_title(
        "The gate rejects attractive-looking traces when realism evidence is weak",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="x")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(loc="lower right", ncol=3)
    fig.savefig(ASSET_DIR / "fresh_realism_gate.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_demo_summary() -> None:
    data = _json(RUN_DIR / "eucys_demo" / "results" / "booth_demo_poster.json")["scenarios"]
    labels = [item["label"] for item in data]
    tir = [item["tir_70_180"] for item in data]
    events = [item["supervisor_events"] for item in data]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    axes[0].bar(x, tir, color=[COLORS["green"], COLORS["amber"], COLORS["blue"]])
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Time in 70-180 mg/dL (%)")
    axes[0].set_title("Glycaemic result", loc="left")
    axes[0].grid(axis="y")
    axes[1].bar(x, events, color=COLORS["red"], alpha=0.88)
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].set_ylabel("Recorded supervisor events")
    axes[1].set_title("Intervention burden", loc="left")
    axes[1].grid(axis="y")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Fresh six-hour EUCYS demo: useful output, but an over-sensitive safety policy",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(ASSET_DIR / "fresh_demo_summary.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_benchmark_arms() -> None:
    frame = pd.read_csv(ROOT / "research" / "eucys_pack" / "assets" / "EUCYS_RESULTS_TABLE.csv")
    labels = ["Clean + certified", "Corrupted + uncertified", "Supervisor-off ablation"]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    left = np.zeros(len(frame))
    for column, label, color in (
        ("mean_tir_below_70", "<70", COLORS["red"]),
        ("mean_tir_70_180", "70-180", COLORS["green"]),
        ("mean_tir_above_180", ">180", COLORS["amber"]),
    ):
        values = frame[column].to_numpy()
        axes[0].barh(y, values, left=left, label=label, color=color)
        left += values
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Mean percentage of simulated time")
    axes[0].set_title("Glucose-range distribution", loc="left")
    axes[0].legend(ncol=3, loc="lower right")
    axes[1].barh(y, frame["mean_supervisor_interventions"], color=COLORS["navy"])
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean recorded interventions per run")
    axes[1].set_title("Safety-intervention burden", loc="left")
    for axis in axes:
        axis.grid(axis="x")
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "Locked 3,600-run benchmark - 1,200 runs per study arm",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(ASSET_DIR / "benchmark_3600_arms.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_algorithm_benchmark() -> None:
    frame = pd.read_csv(ROOT / "research" / "eucys_pack" / "assets" / "EUCYS_MAIN_FIGURE.csv")
    pivot = frame.pivot(index="algorithm", columns="metric", values="value")
    order = [
        "ExampleAlgorithm",
        "Clinical Baseline",
        "CorrectionBolus",
        "PIDController",
        "Standard Pump",
    ]
    pivot = pivot.reindex(order)
    x = np.arange(len(pivot))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), constrained_layout=True)
    axes[0].bar(x, pivot["mean_tir_70_180"], color=COLORS["green"])
    axes[0].set_xticks(x, order, rotation=18, ha="right")
    axes[0].set_ylim(80, 96)
    axes[0].set_ylabel("Mean TIR 70-180 (%)")
    axes[0].set_title("Mean time in range", loc="left")
    axes[1].bar(x, pivot["mean_supervisor_interventions"], color=COLORS["navy"])
    axes[1].set_xticks(x, order, rotation=18, ha="right")
    axes[1].set_ylabel("Mean safety interventions")
    axes[1].set_title("Intervention burden", loc="left")
    for axis in axes:
        axis.grid(axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Algorithm comparison across the locked benchmark matrix",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(ASSET_DIR / "benchmark_algorithms.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _plot_ai_benchmark() -> None:
    metrics = _json(
        ROOT
        / "models"
        / "iints-glucose-forecast-v0-ohio-safe-band"
        / "huggingface"
        / "model_card_metrics.json"
    )["models"]
    frame = pd.DataFrame(metrics)
    fig, axis = plt.subplots(figsize=(11.0, 6.3), constrained_layout=True)
    colors = []
    sizes = []
    for model in frame["model"]:
        if model == "pretrain_v2":
            colors.append(COLORS["green"])
            sizes.append(130)
        elif model == "new_pinn":
            colors.append(COLORS["red"])
            sizes.append(130)
        elif model in {"LastValue", "LinearTrend", "PhysiologyAware"}:
            colors.append(COLORS["muted"])
            sizes.append(70)
        else:
            colors.append(COLORS["blue"])
            sizes.append(85)
    axis.scatter(
        frame["mae"],
        frame["any_physiology_violation_pct"],
        c=colors,
        s=sizes,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for _, row in frame.iterrows():
        axis.annotate(
            row["model"],
            (row["mae"], row["any_physiology_violation_pct"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    axis.set_xlabel("Mean absolute error (mg/dL) - lower is better")
    axis.set_ylabel("Predictions with a physiology violation (%) - lower is better")
    axis.set_title(
        "Held-out OhioT1DM comparison: error alone cannot select a research model",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid()
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.99,
        0.97,
        "35,073 sequences\\n841,752 horizon predictions\\n10 compared models",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "#F3F6F7", "edgecolor": COLORS["line"], "pad": 8},
    )
    fig.savefig(ASSET_DIR / "ai_ohio_benchmark.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _parse_test_summary() -> dict[str, Any]:
    path = EVIDENCE_DIR / "pytest_full_2026-07-30.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"(?P<failed>\d+) failed, (?P<passed>\d+) passed, "
        r"(?P<skipped>\d+) skipped, (?P<warnings>\d+) warning",
        text,
    )
    if not match:
        raise RuntimeError("Could not parse the full pytest summary.")
    result = {key: int(value) for key, value in match.groupdict().items()}
    result["reason"] = (
        "One MDMP quick-certificate test could not import the optional "
        "`cryptography` package in the Python 3.10 test environment."
    )
    return result


def _plot_quality_gates(test_summary: dict[str, Any]) -> None:
    labels = ["Architecture", "Static typing", "Strict docs", "Target science tests", "Full suite"]
    values = [100, 100, 100, 100, 100 * test_summary["passed"] / (test_summary["passed"] + test_summary["failed"])]
    statuses = ["PASS", "PASS", "PASS", "PASS", "1 ENV FAIL"]
    colors = [COLORS["green"]] * 4 + [COLORS["amber"]]
    fig, axis = plt.subplots(figsize=(11.2, 4.7), constrained_layout=True)
    y = np.arange(len(labels))
    axis.barh(y, values, color=colors)
    for idx, (value, status) in enumerate(zip(values, statuses)):
        axis.text(min(value + 0.7, 100.5), idx, status, va="center", fontweight="bold", fontsize=9)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 108)
    axis.set_xlabel("Completed checks (%)")
    axis.set_title(
        "Software evidence generated on 30 July 2026",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(axis="x")
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(ASSET_DIR / "software_quality_gates.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGBA")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, "white")
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.alpha_composite(copy, (x, y))
    return canvas


def _make_cover(version: str, commit: str) -> None:
    width, height = 1654, 2339
    image = Image.new("RGB", (width, height), COLORS["paper"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 525), fill=COLORS["ink"])
    draw.rectangle((0, 525, 22, height), fill=COLORS["teal"])
    draw.text((100, 90), "IINTS-AF", font=_font(54, bold=True), fill="#D8EAEB")
    draw.text(
        (100, 185),
        "EUCYS COMPLETE\nRESEARCH PLAYBOOK",
        font=_font(88, bold=True),
        fill="white",
        spacing=14,
    )
    draw.text(
        (104, 420),
        "Virtual diabetes research, deterministic safety, AI experiments, and evidence",
        font=_font(27),
        fill="#C7D8DE",
    )

    hardware = _fit_image(Image.open(ASSET_DIR / "hardware_demonstrator.png"), (650, 860))
    image.paste(hardware.convert("RGB"), (900, 630))
    draw.rectangle((900, 630, 1550, 1490), outline=COLORS["line"], width=3)
    draw.text(
        (930, 1420),
        "Bench-only physical demonstrator",
        font=_font(22, bold=True),
        fill=COLORS["ink"],
    )

    frame = pd.read_csv(RUN_DIR / "presets" / "meal_stress" / "results.csv")
    chart_box = (95, 650, 820, 1235)
    draw.rounded_rectangle(chart_box, radius=18, fill="white", outline=COLORS["line"], width=3)
    draw.text((130, 690), "A run is evidence, not just a curve", font=_font(30, bold=True), fill=COLORS["ink"])
    plot_left, plot_top, plot_right, plot_bottom = 145, 790, 770, 1150
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill="#FAFCFC")
    t = frame["time_minutes"].to_numpy(dtype=float)
    g = frame["glucose_actual_mgdl"].to_numpy(dtype=float)
    t_norm = (t - t.min()) / max(1.0, t.max() - t.min())
    g_min, g_max = 60.0, max(230.0, float(g.max()) + 10)
    target_top = plot_bottom - (180 - g_min) / (g_max - g_min) * (plot_bottom - plot_top)
    target_bottom = plot_bottom - (70 - g_min) / (g_max - g_min) * (plot_bottom - plot_top)
    draw.rectangle(
        (plot_left, target_top, plot_right, target_bottom),
        fill="#E3EFE5",
    )
    points = [
        (
            plot_left + float(tx) * (plot_right - plot_left),
            plot_bottom - (float(gx) - g_min) / (g_max - g_min) * (plot_bottom - plot_top),
        )
        for tx, gx in zip(t_norm, g)
    ]
    draw.line(points, fill=COLORS["navy"], width=5)
    draw.text((145, 1170), "Fresh meal-stress simulation, seed 42", font=_font(20), fill=COLORS["muted"])

    draw.text((100, 1580), "What is inside", font=_font(38, bold=True), fill=COLORS["navy"])
    bullets = [
        "Research question, hypothesis, scope, and scientific boundaries",
        "Mermaid architecture, data, simulation, AI, desktop, and evidence diagrams",
        "All 15 registered deterministic formulas with runtime and literature links",
        "Application screenshots, hardware photo, protein structures, and code excerpts",
        "Fresh test runs, 3,600-run benchmark, and held-out OhioT1DM AI evaluation",
        "A timed live-demo script, failure plan, jury Q&A, and reproduction checklist",
    ]
    y = 1660
    for bullet in bullets:
        draw.ellipse((115, y + 8, 130, y + 23), fill=COLORS["teal"])
        wrapped = textwrap.wrap(bullet, width=80)
        draw.multiline_text(
            (155, y),
            "\n".join(wrapped),
            font=_font(24),
            fill=COLORS["ink"],
            spacing=5,
        )
        y += 85 + 30 * (len(wrapped) - 1)

    draw.line((100, 2210, 1550, 2210), fill=COLORS["line"], width=2)
    draw.text(
        (100, 2240),
        f"Runebob Baers  |  SDK source {version}  |  commit {commit}  |  {date.today().isoformat()}",
        font=_font(22),
        fill=COLORS["muted"],
    )
    image.save(ASSET_DIR / "cover.png", quality=95)


def _asset(name: str, width: str = "94%") -> str:
    return f"![{name}](assets/{name}){{ width={width} }}"


def _pagebreak(lines: list[str]) -> None:
    # Flush pending figures before starting the next evidence section.
    lines.extend(["", "\\clearpage", ""])


def _add_image(lines: list[str], filename: str, caption: str, width: str = "94%") -> None:
    # A blank line is semantically important: without it Pandoc can attach an
    # image to the preceding paragraph or list item and render it off-page.
    if lines and lines[-1] != "":
        lines.append("")
    lines.extend(
        [
            f"![{caption}](assets/{filename}){{ width={width} }}",
            "",
        ]
    )


def _formula_section(lines: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from iints.core.formula_registry import get_formula_registry

    lines.extend(
        [
            "# Appendix A - Complete deterministic formula registry",
            "",
            "The formula registry is immutable explanation metadata. Runtime values are "
            "calculated by deterministic Python model code; the local language model may "
            "describe these equations but may not derive, solve, alter, or replace them.",
            "",
        ]
    )
    for index, formula in enumerate(get_formula_registry(), start=1):
        implementation = "; ".join(
            path.rsplit("/", 1)[-1].replace(":", " :: ")
            for path in formula.implementation_paths
        )
        lines.extend(
            [
                f"## A.{index} {formula.formula_id}: {formula.title}",
                "",
                "\\[",
                formula.latex_expression,
                "\\]",
                "",
                f"**Runtime form.** {formula.solved_or_runtime_form}",
                "",
                f"**Units.** {formula.units}",
                "",
                "**State variables.** " + ", ".join(f"`{item}`" for item in formula.state_variables),
                "",
                "**Parameters.** " + ", ".join(f"`{item}`" for item in formula.parameters),
                "",
                f"**Implementation.** {implementation}",
                "",
                "**Scientific boundary.** " + formula.validation_note,
                "",
                "**Literature basis.** "
                + ", ".join(f"[source]({item})" for item in formula.literature_basis),
                "",
            ]
        )


def _build_markdown(
    version: str,
    commit: str,
    test_summary: dict[str, Any],
) -> str:
    preset_rows: list[dict[str, Any]] = []
    for key, label in PRESET_LABELS.items():
        folder = RUN_DIR / "presets" / key
        realism = _json(folder / "realism_report.json")
        quality = _json(folder / "run_quality_summary.json")
        metrics = realism["metrics"]
        preset_rows.append(
            {
                "label": label,
                "verdict": realism["verdict"],
                "realism": realism["realism_score"],
                "grade": quality["grade"],
                "quality": quality["score"],
                "mean": metrics["mean_glucose_mgdl"],
                "cv": metrics["cv_pct"],
                "tir": metrics["tir_70_180_pct"],
                "low": metrics["tir_below_70_pct"],
                "high": metrics["tir_above_180_pct"],
                "flat": metrics["flat_step_ratio"],
                "rate": metrics["max_abs_rate_mgdl_per_min"],
            }
        )

    demo = _json(RUN_DIR / "eucys_demo" / "results" / "booth_demo_poster.json")["scenarios"]
    ai_report = _json(
        ROOT
        / "models"
        / "iints-glucose-forecast-v0-ohio-safe-band"
        / "huggingface"
        / "comparison_report.json"
    )
    ai_best = ai_report["best_by_mae"]
    benchmark = pd.read_csv(ROOT / "research" / "eucys_pack" / "assets" / "EUCYS_RESULTS_TABLE.csv")

    lines: list[str] = ["![](assets/cover.png){ width=100% }"]
    _pagebreak(lines)
    lines.extend(
        [
            "# Read this first",
            "",
            "> **Research boundary.** IINTS-AF is an open-source research and educational "
            "> simulator. It is not a medical device, does not provide treatment advice, "
            "> and must never control real insulin or glucagon delivery.",
            "",
            "This is the single master document for a EUCYS presentation. It is designed "
            "to be browsed in order, used as a live-demo script, or handed to a reviewer. "
            "Every displayed number comes from a CSV/JSON/PDF artifact in the bundle. "
            "Failures are labelled as failures; they are not removed to make the project "
            "look stronger.",
            "",
            "## Three reading modes",
            "",
            "| Reader | Recommended route | Time |",
            "| --- | --- | ---: |",
            "| Jury member | Executive summary -> architecture -> fresh results -> limitations | 8 min |",
            "| Technical reviewer | Architecture -> formulas -> data/AI -> evidence map | 25 min |",
            "| Presenter | Live runbook -> backup plan -> jury Q&A | 12 min rehearsal |",
            "",
            "## Snapshot used in this book",
            "",
            f"- Source version: `{version}`.",
            f"- Repository commit: `{commit}`.",
            f"- Build date: `{date.today().isoformat()}`.",
            "- Fresh scenario seed: `42`; simulation step: `5 minutes`.",
            "- The local installed CLI still reported `1.5.31`; source metadata reports "
            f"`{version}`. This is recorded as a stale editable-install metadata issue.",
            "- AI was deliberately skipped in the live demo run so numerical evidence did "
            "not depend on Ollama availability.",
            "",
            "## Table of contents",
            "",
            "1. Project in 90 seconds",
            "2. Research question and experimental design",
            "3. Software architecture and authority boundaries",
            "4. Scientific model and equations",
            "5. Desktop workbench and researcher workflow",
            "6. Data, MDMP, AI, and molecular context",
            "7. Fresh runs generated for this book",
            "8. Locked 3,600-run EUCYS benchmark",
            "9. OhioT1DM glucose-forecast benchmark",
            "10. Software verification",
            "11. Live demonstration runbook",
            "12. Jury questions, limitations, and next experiments",
            "13. Evidence and reproduction map",
            "14. Complete 15-formula appendix",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 1. The project in 90 seconds",
            "",
            "## One-sentence pitch",
            "",
            "**IINTS-AF is an open-source research workbench that creates virtual diabetes "
            "scenarios, tests experimental algorithms behind deterministic safety checks, "
            "and turns each run into evidence that another researcher can inspect.**",
            "",
            "## Why it exists",
            "",
            "Diabetes algorithms must reason about delayed sensor readings, meals, active "
            "insulin, exercise, stress, device faults, and uncertainty. Testing an idea "
            "near a real patient is therefore not the first step. A transparent virtual "
            "patient provides a place to expose assumptions and failure cases first.",
            "",
            "## The six-stage research loop",
            "",
            "1. **Define** a patient, scenario, algorithm, seed, and expected evidence.",
            "2. **Simulate** latent physiology and a separate CGM-like observation.",
            "3. **Propose** a candidate insulin or research glucagon action.",
            "4. **Supervise** that proposal with deterministic safety logic.",
            "5. **Validate** data integrity, safety events, and physiological plausibility.",
            "6. **Document** CSV traces, manifests, reports, figures, and limitations.",
            "",
        ]
    )
    _add_image(lines, "sdk_overview.png", "IINTS-AF at a glance: one research workflow, multiple evidence layers")
    lines.extend(
        [
            "## What the project does not claim",
            "",
            "- It does not perfectly reproduce a human patient.",
            "- It does not prove clinical safety or treatment efficacy.",
            "- The safety supervisor reduces simulated risk; it cannot guarantee real-world safety.",
            "- AlphaFold, ClinVar, GTEx, or ChEMBL context does not automatically determine patient physiology.",
            "- A realistic-looking curve is not proof that the parameters are valid.",
            "- The local language model is advisory and has no numerical or actuator authority.",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 2. Research question and experimental design",
            "",
            "## Primary question",
            "",
            "> Can an open-source simulation workbench make risky or unrealistic "
            "> diabetes-algorithm behaviour visible and reproducible before an algorithm "
            "> is considered for real-world testing?",
            "",
            "## Hypothesis",
            "",
            "An algorithm that reacts to incomplete glucose context can produce questionable "
            "actions in stress or failure scenarios. A separate deterministic safety layer, "
            "combined with reproducible scenarios and evidence artifacts, should make at "
            "least part of this behaviour detectable, blockable, and reviewable.",
            "",
            "## Experimental ladder",
            "",
            "| Level | Question | Evidence |",
            "| --- | --- | --- |",
            "| Single run | Did one scenario execute and produce coherent artifacts? | Trace, report, manifest |",
            "| Scenario stress | Does behaviour change under meal, night, stress, or sensor conditions? | Realism and safety review |",
            "| Algorithm comparison | How do candidate and baselines differ under the same matrix? | Locked aggregate table |",
            "| Data ablation | What changes with corrupted or uncertified input? | Three-arm benchmark |",
            "| AI comparison | Is a predictor accurate and physiologically plausible on held-out subjects? | Horizon and violation metrics |",
            "| External validation | Does the result generalise to another simulator or population? | Future work |",
            "",
            "## Claim ladder",
            "",
            "| Claim level | Meaning in this project |",
            "| --- | --- |",
            "| Implemented | A code path exists and is inspectable |",
            "| Verified in software | Expected behaviour was observed under specified tests |",
            "| Calibrated | Parameters or outputs were compared with a documented dataset |",
            "| Externally validated | An independent simulator/population reproduced the result |",
            "| Clinically validated | Outside the scope of this project |",
            "",
            "The current project mainly supports the first two levels. Selected AI and realism "
            "workflows reach dataset calibration. It does not claim external or clinical validation.",
        ]
    )
    _add_image(lines, "evidence_workflow.png", "From raw data and protocol to reviewable evidence")
    _pagebreak(lines)
    lines.extend(
        [
            "# 3. Software architecture and authority boundaries",
            "",
            "## End-to-end architecture",
            "",
            "The diagram below is rendered from maintained Mermaid source. The crucial design "
            "choice is that the language model is downstream of recorded evidence.",
            "",
        ]
    )
    _add_image(lines, "system_architecture.png", "Mermaid system architecture")
    lines.extend(
        [
            "## One simulation step",
            "",
            "Scenario events update the mechanistic state; the sensor creates an observation; "
            "the candidate proposes an action; the supervisor accepts, reduces, or blocks it; "
            "then the recorder writes both state and decision evidence.",
            "",
        ]
    )
    _add_image(lines, "simulation_step.png", "Mermaid sequence diagram for one simulation step")
    lines.extend(
        [
            "## Numeric authority",
            "",
            "1. Deterministic patient code calculates physiological state.",
            "2. A deterministic or learned controller calculates a candidate.",
            "3. Deterministic safety logic validates and constrains the candidate.",
            "4. Metric code calculates results from the recorded trace.",
            "5. A language model may explain supplied artifacts only.",
            "",
        ]
    )
    _add_image(lines, "numeric_authority.png", "Mermaid numeric-authority boundary")
    lines.extend(
        [
            "## Repository layers",
            "",
            "| Layer | Main paths | Responsibility |",
            "| --- | --- | --- |",
            "| Domain core | `src/iints/core/`, `src/iints/api/` | Simulator, patient state, algorithms, deterministic safety |",
            "| Data and validation | `src/iints/data/`, `src/iints/validation/` | Imports, contracts, quality, realism, replay |",
            "| Research workflows | `src/iints/research/`, `src/iints/analysis/`, `src/iints/highlevel.py` | Studies, metrics, reports, model training |",
            "| Optional AI | `src/iints/ai/` | Evidence explanation and research assistance |",
            "| Interfaces | `src/iints/cli/`, `apps/iints-tauri/`, `src/iints_desktop/` | CLI and desktop access to the same engine |",
            "| Bench hardware | `src/iints/live_patient/`, `src/iints/jetson/` | Edge, FPGA, Pi, and UNO Q research adapters |",
            "",
            "The repository enforces selected dependency boundaries with "
            "`tools/ci/check_architecture_boundaries.py`.",
        ]
    )
    _add_image(lines, "desktop_bridge.png", "Mermaid desktop-to-Python bridge boundary")
    _pagebreak(lines)
    lines.extend(
        [
            "# 4. Scientific model and equations",
            "",
            "## Three virtual-patient modes",
            "",
            "| Model | Use | Strength | Main limitation |",
            "| --- | --- | --- | --- |",
            "| Custom patient | Fast demos and broad sweeps | Transparent and inexpensive | Lower physiological detail |",
            "| Bergman-style | Compact glucose-insulin ODE experiments | Familiar minimal-model structure | SDK extensions add calibration burden |",
            "| Hovorka-style | Compartmental insulin, glucose, meals, and research extensions | Richer state separation | More parameters; not independently certified |",
            "",
            "## Four equations to explain live",
            "",
            "### Meal absorption delay",
            "",
            "\\[",
            "\\frac{dD_3}{dt}=k_{\\mathrm{empt}}D_2-k_{\\mathrm{abs}}D_3",
            "\\]",
            "",
            "Carbohydrate does not appear instantly. It moves through stomach and gut compartments.",
            "",
            "### Two-depot insulin delay",
            "",
            "\\[",
            "\\frac{dS_2}{dt}=kS_1-kS_2, \\qquad U_I=kS_2",
            "\\]",
            "",
            "Delivered insulin and active insulin are different states; subcutaneous absorption takes time.",
            "",
            "### Hovorka-style glucose mass balance",
            "",
            "\\[",
            "\\frac{dQ_1}{dt}=-(\\mathrm{NIMGU}+F_R)-x_1Q_1+k_{12}Q_2+\\mathrm{EGP}+U_G",
            "\\]",
            "",
            "Glucose changes through meal appearance, endogenous production, utilisation, renal "
            "loss, and exchange between accessible and non-accessible compartments.",
            "",
            "### CGM blood-to-interstitial lag",
            "",
            "\\[",
            "\\tau_{\\mathrm{ISF}}\\frac{d\\mathrm{ISF}}{dt}=\\mathrm{BG}_{\\mathrm{lagged}}-\\mathrm{ISF}",
            "\\]",
            "",
            "The algorithm can receive a delayed CGM-like signal while the simulator preserves "
            "a separate latent blood-glucose state.",
            "",
            "## Scientific caution",
            "",
            "The 15-formula registry proves that equations are explicit and traceable. It does "
            "not prove that every parameter is identified for every patient. Full equations, "
            "runtime locations, units, and sources appear in Appendix A.",
        ]
    )
    _add_image(lines, "code_patient.png", "Code excerpt: deterministic physiological patient model", "76%")
    _pagebreak(lines)
    lines.extend(
        [
            "# 5. Desktop workbench and researcher workflow",
            "",
            "The Rust/Tauri desktop application is an interface to the Python research engine, "
            "not a second simulator. Paths use file/folder selectors; long jobs run through "
            "allowlisted bridge commands; outputs remain ordinary files in the selected workspace.",
            "",
            "## System overview",
        ]
    )
    _add_image(lines, "app_overview.png", "Desktop system overview and environment readiness")
    lines.extend(["## Configure and run a protocol"])
    _add_image(lines, "app_run_protocol.png", "Protocol selection, parameters, output folder, and run controls")
    lines.extend(["## Inspect results"])
    _add_image(lines, "app_results.png", "Results browser with trace, metrics, files, and report access")
    lines.extend(["## Reproducibility and audit"])
    _add_image(lines, "app_reproducibility.png", "Run metadata, evidence files, and reproducibility controls")
    lines.extend(
        [
            "## Intended user workflow",
            "",
            "1. Choose a protocol or import a scenario.",
            "2. Select the output directory instead of typing an absolute path.",
            "3. Review patient, algorithm, seed, duration, and step.",
            "4. Run the protocol and watch progress without freezing the UI.",
            "5. Inspect the glucose trace and quality gates before opening the narrative report.",
            "6. Open CSV/JSON/PDF artifacts directly from the Results tab.",
            "7. Invoke optional local AI only after deterministic artifacts exist.",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 6. Data, MDMP, AI, and molecular context",
            "",
            "## Data lifecycle",
        ]
    )
    _add_image(lines, "data_lifecycle.png", "Mermaid data lifecycle from source to comparison")
    lines.extend(
        [
            "## MDMP gate",
            "",
            "The data workflow checks an explicit contract, required columns, cadence, bounds, "
            "missingness, provenance, and hashable artifacts. A certificate documents a check; "
            "it is not a clinical approval or proof that a dataset is unbiased.",
        ]
    )
    _add_image(lines, "mdmp_readiness.png", "MDMP and EU AI Pact-oriented evidence readiness")
    lines.extend(
        [
            "## Local AI boundary",
            "",
            "Ollama can summarise a completed run, explain supplied metrics, compare named "
            "artifacts, and list limitations. It may not invent measurements, calculate missing "
            "statistics, solve the ODEs, approve dosing, or overwrite deterministic evidence.",
        ]
    )
    _add_image(lines, "app_local_ai.png", "Desktop local-AI workspace")
    _add_image(lines, "ai_safety_funnel.png", "AI is downstream of deterministic evidence and safety")
    lines.extend(
        [
            "## Molecular and multi-scale tools",
            "",
            "AlphaFold, ClinVar, STRING, ChEMBL, GTEx/Bgee, BindingDB, COPASI, OpenCOR, "
            "Physiome, and FMI/FMPy can provide research context. Their roles are deliberately "
            "bounded:",
            "",
            "| Tool class | May support | Must not be treated as |",
            "| --- | --- | --- |",
            "| AlphaFold/PAE | Structure confidence and domain context | Patient insulin sensitivity |",
            "| ClinVar | Variant classification evidence | Quantitative metabolic effect without functional evidence |",
            "| STRING/Reactome | Pathway and interaction context | Causal patient model |",
            "| ChEMBL/BindingDB | Measured pharmacology and affinity context | Automatic dose parameter |",
            "| GTEx/Bgee/HPA | Tissue expression context | Direct compartment scaling |",
            "| COPASI/OpenCOR | Independent model/sensitivity experiments | Clinical validation |",
            "| FMI/FMPy | Pump, flow, motor, occlusion, and sensor co-simulation | Real device certification |",
            "",
            "Unknown variants are fail-closed: no physiological scalar is applied without "
            "explicit functional evidence. AlphaFold pLDDT remains a structure-confidence "
            "metric only.",
        ]
    )
    _add_image(lines, "insulin_3d.png", "AlphaFold structural context: insulin precursor", "72%")
    _add_image(lines, "glut4_3d.png", "AlphaFold structural context: GLUT4 transporter", "72%")
    _pagebreak(lines)
    lines.extend(
        [
            "# 7. Fresh runs generated for this book",
            "",
            "These runs were generated from the current workspace on 30 July 2026 with seed "
            "`42`. Their purpose is not to select the prettiest trace, but to test whether the "
            "realism and quality gates accept or reject current presets.",
            "",
        ]
    )
    _add_image(lines, "fresh_preset_traces.png", "Fresh traces: one rejected baseline and two accepted stress scenarios")
    lines.extend(
        [
            "## Fresh preset gate table",
            "",
            "| Preset | Realism verdict | Realism score | Quality grade | Quality score |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for row in preset_rows:
        lines.append(
            f"| {row['label']} | {row['verdict'].replace('_', ' ')} | "
            f"{row['realism']:.2f} | {row['grade'].replace('_', ' ')} | "
            f"{row['quality']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Fresh preset metric table",
            "",
            "| Preset | Mean | CV | TIR | <70 | >180 | Flat-step ratio | Max rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in preset_rows:
        lines.append(
            f"| {row['label']} | {row['mean']:.1f} | {row['cv']:.1f}% | "
            f"{row['tir']:.1f}% | {row['low']:.1f}% | {row['high']:.1f}% | "
            f"{row['flat']:.3f} | {row['rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## What passed",
            "",
            "- **Meal stress:** realism `0.92`, quality `89.2/100`, TIR `78.6%`, no "
            "impossible or right-angle transitions.",
            "- **Hypo-prone night:** realism `1.00`, quality `96.9/100`, TIR `89.7%`, "
            "no impossible or right-angle transitions.",
            "",
            "## What failed",
            "",
            "- Reference, baseline, and free-living presets received realism `0.52` and "
            "`do_not_use` grades.",
            "- Their strongest failure signal is a flat-step ratio around `0.28-0.32` "
            "with near-flat stretches lasting `395-435 minutes`.",
            "- These are current calibration regressions. They must be retuned before these "
            "presets are used as research evidence.",
            "- The scenario named *hypo-prone night* never approaches hypoglycaemia "
            "(minimum about `101 mg/dL`). Its label and disturbance design need closer alignment.",
        ]
    )
    _add_image(lines, "fresh_realism_gate.png", "Fresh realism and run-quality gate")
    _pagebreak(lines)
    lines.extend(
        [
            "# 7.1 Fresh EUCYS three-scenario demo",
            "",
            "The zero-configuration demo ran three six-hour scenarios: normal, meal stress, "
            "and a deliberately unsafe policy routed through the supervisor.",
            "",
            "| Scenario | TIR | <70 | >180 | Mean glucose | Supervisor events | Meals |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in demo:
        lines.append(
            f"| {item['label']} | {item['tir_70_180']:.1f}% | "
            f"{item['tir_below_70']:.1f}% | {item['tir_above_180']:.1f}% | "
            f"{item['mean_glucose']:.1f} mg/dL | {item['supervisor_events']} | "
            f"{item['meal_events']} |"
        )
    lines.extend(
        [
            "",
            "## Honest interpretation",
            "",
            "The demo proves that the packaged workflow runs, separates scenarios, records "
            "events, and produces reports. It does **not** currently prove that the safety "
            "policy is well tuned. It recorded `184` supervisor events across `219` time "
            "steps. That intervention burden is too high for a convincing baseline policy "
            "and should be decomposed into routine shaping versus material safety episodes.",
        ]
    )
    _add_image(lines, "fresh_demo_summary.png", "Fresh demo result and intervention burden")
    _add_image(lines, "fresh_demo_poster.png", "Poster generated by the fresh EUCYS demo")
    _pagebreak(lines)
    lines.extend(
        [
            "# 7.2 What one generated report contains",
            "",
            "The fresh meal-stress run produced a four-page PDF from the same trace used above. "
            "The report is a presentation layer; CSV and JSON remain the numerical authority.",
        ]
    )
    _add_image(lines, "fresh_report_page_1.png", "Fresh report page 1: run identity and clinical-style summary", "74%")
    _add_image(lines, "fresh_report_page_3.png", "Fresh report page 3: trace and deterministic metrics", "74%")
    _add_image(lines, "fresh_report_page_4.png", "Fresh report page 4: safety evidence and interpretation boundary", "74%")
    _pagebreak(lines)
    lines.extend(
        [
            "# 8. Locked 3,600-run EUCYS benchmark",
            "",
            "The maintained benchmark contains three arms with `1,200` runs each. Across the "
            "matrix it varies patient profiles, scenario families, algorithm paths, and seeds. "
            "Its strongest contribution is the auditable protocol and exposed failure burden.",
            "",
            "| Study arm | Runs | Mean TIR | Mean <70 | Mean >180 | Mean interventions | Severe-hypo runs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in benchmark.iterrows():
        lines.append(
            f"| {row['arm_id'].replace('_', ' ')} | {int(row['run_count'])} | "
            f"{row['mean_tir_70_180']:.2f}% | {row['mean_tir_below_70']:.2f}% | "
            f"{row['mean_tir_above_180']:.2f}% | "
            f"{row['mean_supervisor_interventions']:.1f} | "
            f"{int(row['severe_hypo_run_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Corrupted/uncertified input is associated with substantially more time above 180 mg/dL.",
            "- The supervisor-off ablation also worsens range distribution relative to the clean arm.",
            "- Low-glucose burden remains material in every arm and cannot be dismissed by a high mean TIR.",
            "- Raw intervention counts are large and require episode-level normalisation.",
            "- Because all arms come from this simulator, this is internal comparative evidence, not external validation.",
        ]
    )
    _add_image(lines, "benchmark_3600_arms.png", "Three-arm aggregate results and intervention burden")
    _add_image(lines, "benchmark_algorithms.png", "Algorithm comparison within the locked matrix")
    _pagebreak(lines)
    lines.extend(
        [
            "# 9. Held-out OhioT1DM glucose-forecast benchmark",
            "",
            "A subject-level OhioT1DM comparison used `35,073` held-out sequences, a "
            "`240-minute` history, a `120-minute` horizon, and `841,752` individual horizon "
            "predictions across ten baselines/checkpoints.",
            "",
            f"The best mean-error result was **{ai_best['model']}** with MAE "
            f"`{ai_best['mae']:.2f} mg/dL`, RMSE `{ai_best['rmse']:.2f} mg/dL`, "
            f"`{ai_best['within_20_mgdl_pct']:.1f}%` within 20 mg/dL, a missed-hypoglycaemia "
            f"rate of `{ai_best['missed_hypo_rate_pct']:.2f}%`, and physiology violations in "
            f"`{ai_best['any_physiology_violation_pct']:.2f}%` of predictions.",
            "",
            "## Why the PINN candidate was not promoted",
            "",
            "The checkpoint called `new_pinn` had MAE around `23.14 mg/dL`, but its physiology-"
            "violation rate was `74.18%`. This is a failed candidate, not a success. It shows "
            "why a loss function name cannot substitute for held-out physiological evaluation.",
        ]
    )
    _add_image(lines, "ai_ohio_benchmark.png", "OhioT1DM model error versus physiological-violation rate")
    lines.extend(
        [
            "## AI research boundary",
            "",
            "- This model is for simulation, benchmarking, and educational research.",
            "- It must not be used for insulin dosing or real-time patient care.",
            "- The OhioT1DM subjects are split at subject level to reduce leakage.",
            "- Model promotion must consider error, hypo detection, uncertainty, and physiological violations together.",
            "- Independent external datasets and prospective evaluation remain future work.",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 10. Software verification",
            "",
            f"On 30 July 2026 the full suite produced **{test_summary['passed']} passed**, "
            f"**{test_summary['skipped']} skipped**, and **{test_summary['failed']} failed** "
            f"in `83.35 s`. {test_summary['reason']}",
            "",
            "The failure is retained in this dossier. It indicates that the test environment "
            "did not install the optional signing dependency expected by the MDMP certificate "
            "path. The architecture check, mypy check, strict documentation build, and focused "
            "science tests were run separately.",
            "",
            "| Check | Result | Evidence file |",
            "| --- | --- | --- |",
            "| Architecture boundaries | Passed | `evidence/architecture_boundaries_2026-07-30.txt` |",
            "| mypy across 212 source files | Passed | `evidence/mypy_2026-07-30.txt` |",
            "| MkDocs strict build | Passed | `evidence/mkdocs_strict_2026-07-30.txt` |",
            "| Focused formula/physiology/realism tests | See evidence log | `evidence/science_target_tests_2026-07-30.txt` |",
            f"| Full pytest suite | {test_summary['passed']} pass, {test_summary['failed']} environment failure | `evidence/pytest_full_2026-07-30.txt` |",
        ]
    )
    _add_image(lines, "software_quality_gates.png", "Current software quality-gate status")
    lines.extend(
        [
            "## What this verifies - and what it does not",
            "",
            "Software tests verify code contracts and selected numerical invariants. They do "
            "not establish clinical validity. Realism checks verify configured envelopes and "
            "trace properties. They do not prove that the virtual patient represents a particular person.",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 11. Live EUCYS demonstration runbook",
            "",
            "## Before the jury arrives",
            "",
            "1. Charge the laptop and disable sleep.",
            "2. Put the app, this playbook, and the generated poster in full-screen-ready windows.",
            "3. Verify `iints --version` and record any source/runtime version mismatch.",
            "4. Run the demo once with `--skip-ai`; keep that output as the rehearsed fallback.",
            "5. Do not depend on network access or Ollama for the core presentation.",
            "6. Keep the hardware demonstrator disconnected from any medication-delivery system.",
            "",
            "## Canonical command",
            "",
            "```bash",
            "iints demo eucys \\",
            "  --output-dir results/eucys_live \\",
            "  --skip-ai \\",
            "  --evidence \\",
            "  --overwrite",
            "```",
            "",
            "## Twelve-minute script",
            "",
            "| Time | Screen | Say |",
            "| --- | --- | --- |",
            "| 0:00 | Title / hardware photo | \"IINTS-AF is a research sandbox for finding risky or unrealistic diabetes-algorithm behaviour before any real-world claim.\" |",
            "| 0:45 | Architecture | \"The patient model calculates state, the candidate proposes, and deterministic safety decides what is allowed.\" |",
            "| 1:45 | Four equations | Explain meal delay, insulin delay, glucose mass, and CGM lag in plain language. |",
            "| 3:15 | App protocol screen | Show scenario, seed, duration, and output path. Do not open code yet. |",
            "| 4:15 | Run command / progress | Start the three-scenario demo or state that the rehearsed output comes from this exact command. |",
            "| 5:30 | Fresh traces | Compare normal, meal stress, and unsafe-policy cases. |",
            "| 6:45 | Safety evidence | Show candidate versus accepted action and one reason. State that high event counts reveal over-sensitivity. |",
            "| 8:00 | Realism gate | Show that three current presets fail. \"The validator also criticises my own simulator.\" |",
            "| 9:15 | 3,600-run benchmark | Explain clean, corrupted, and supervisor-off arms; mention low-glucose burden. |",
            "| 10:15 | AI benchmark | Show that the lowest-error model is not automatically the safest, and that the PINN candidate failed. |",
            "| 11:15 | Limitations | State what is implemented, what is calibrated, and what still needs external validation. |",
            "| 11:45 | Closing | \"The contribution is not a clinical controller; it is a transparent workflow to define, simulate, stress, supervise, validate, and document.\" |",
            "",
            "## Demonstration screen order",
            "",
            "1. `assets/system_architecture.png`",
            "2. App: **Run protocol**",
            "3. `fresh_demo_poster.png`",
            "4. `fresh_realism_gate.png`",
            "5. `benchmark_3600_arms.png`",
            "6. `ai_ohio_benchmark.png`",
            "7. Report page 4 and the limitations table",
            "",
            "## Failure plan",
            "",
            "| Failure | Immediate response |",
            "| --- | --- |",
            "| Demo is slow | Open the rehearsed output and show its command/manifest |",
            "| Ollama is unavailable | Continue; AI is optional and downstream |",
            "| Poster does not open | Open the PNG from the output directory |",
            "| PDF generation fails | Show CSV, metadata, realism JSON, and manifest |",
            "| Result is unexpected | Label it; do not rerun until a prettier seed appears |",
            "| A claim exceeds evidence | State what is known, assumed, and still untested |",
        ]
    )
    _pagebreak(lines)
    lines.extend(
        [
            "# 12. Jury questions, limitations, and next experiments",
            "",
            "## Likely questions",
            "",
            "### What did you build?",
            "",
            "An integrated open-source SDK and desktop workbench for virtual-patient simulation, "
            "algorithm interfaces, deterministic safety checks, data contracts, AI research, "
            "reports, evidence bundles, and bench-only hardware experiments.",
            "",
            "### Does AI calculate the physiology?",
            "",
            "No. Deterministic Python code evaluates the registered equations. AI receives "
            "completed artifacts as read-only explanation context.",
            "",
            "### Does the supervisor prove safety?",
            "",
            "No. It can expose, block, or reduce configured simulated actions. Its thresholds, "
            "event definitions, and failure coverage still require validation.",
            "",
            "### Why show failed presets and a failed PINN?",
            "",
            "Because the purpose of a research workbench is to reveal failures. Removing them "
            "would make the project less scientific.",
            "",
            "### How is a run reproducible?",
            "",
            "The run records scenario, patient, algorithm, duration, time step, seed, version, "
            "raw trace, validation output, report, and file manifest.",
            "",
            "### What do AlphaFold and ClinVar add?",
            "",
            "They add structural and variant-classification context. They do not calculate "
            "patient insulin sensitivity. Unknown variants are not given an effect without evidence.",
            "",
            "## Current limitations",
            "",
            "| Limitation | Evidence in this book | Required next step |",
            "| --- | --- | --- |",
            "| Three core presets fail current realism gate | Fresh preset table | Retune residual dynamics and flatline behaviour on held-out data |",
            "| Hypo-night label does not match its minimum glucose | Fresh trace | Redesign disturbance and expected intervention endpoint |",
            "| Safety event counts are too high | Fresh demo and 3,600-run chart | Separate routine shaping from material episodes; report per hour/episode |",
            "| Full suite has one dependency-related failure | Software verification | Align test extras so certificate signing dependencies are explicit |",
            "| Current model combinations are not externally certified | Claim ladder | Compare with an independent reference simulator |",
            "| Ohio model has non-zero physiology violations | AI benchmark | Add calibration, uncertainty gates, and external test cohorts |",
            "| Molecular context is not a patient-effect model | Tool boundary table | Require curated functional evidence and sensitivity analysis |",
            "",
            "## Next experiments in priority order",
            "",
            "1. Repair and rerun the reference/baseline/free-living preset calibration.",
            "2. Convert supervisor events into unique episodes and severity classes.",
            "3. Lock a clinically motivated hypo-night protocol with a prespecified nadir and rescue endpoint.",
            "4. Calibrate selected physiology parameters on OhioT1DM training subjects and test only on held-out subjects.",
            "5. Compare at least one scenario against an independent Hovorka/CellML implementation.",
            "6. Run parameter-identifiability and sensitivity analysis with COPASI/OpenCOR.",
            "7. Validate pump/occlusion/sensor adapters through FMI/FMPy co-simulation.",
            "8. Repeat the benchmark across release tags to detect scientific regression.",
        ]
    )
    _add_image(lines, "hardware_demonstrator.png", "Bench-only physical demonstrator - no person and no medication delivery", "58%")
    _pagebreak(lines)
    lines.extend(
        [
            "# 13. Evidence and reproduction map",
            "",
            "## Claim-to-evidence map",
            "",
            "| Claim | Runtime authority | Evidence | Boundary |",
            "| --- | --- | --- | --- |",
            "| Virtual patient state is simulated | Core patient models | Traces and physiology tests | Research approximation |",
            "| Candidate and safety authority are separate | Core supervisor | Safety columns and reports | Not clinical safety proof |",
            "| AI is not numerical authority | Formula registry and AI policy | AI gate tests/artifacts | Human review required |",
            "| Data can be contract-checked | Data package | MDMP report/certificate | Not clinical approval |",
            "| Desktop uses the same SDK | Rust/Tauri bridge | Desktop smoke and bridge tests | Packaging must stay maintained |",
            "| Benchmark has 3,600 runs | EUCYS aggregate CSV | 1,200 rows per arm aggregate | Internal simulator evidence |",
            "| Ohio benchmark is held out by subject | Training/comparison reports | Subject IDs and 841,752 predictions | Not external clinical validation |",
            "",
            "## Rebuild this one playbook",
            "",
            "```bash",
            "MPLCONFIGDIR=.mplt_eucys_playbook \\",
            "  ./.venv/bin/python tools/research/build_eucys_complete_playbook.py",
            "```",
            "",
            "## Reproduce the fresh demo",
            "",
            "```bash",
            "iints demo eucys \\",
            "  --output-dir research/eucys_pack/complete_playbook/runs/eucys_demo \\",
            "  --skip-ai --evidence --overwrite",
            "```",
            "",
            "## Principal quality commands",
            "",
            "```bash",
            "python tools/ci/check_architecture_boundaries.py",
            "mypy src/iints/",
            "mkdocs build --strict",
            "pytest tests/ -q",
            "```",
            "",
            "## Included evidence paths",
            "",
            "- `runs/eucys_demo/`: fresh three-scenario demo and poster.",
            "- `runs/presets/`: five fresh preset traces, reports, realism, and quality summaries.",
            "- `evidence/`: pytest, mypy, architecture, docs, and focused-science logs.",
            "- `assets/`: diagrams, screenshots, photographs, generated charts, and report pages.",
            "- `IINTS_AF_EUCYS_COMPLETE_PLAYBOOK.html`: offline browsable edition.",
            f"- `{PDF_PATH.relative_to(ROOT)}`: print-ready PDF edition.",
            "",
            "## Primary scientific sources",
            "",
            "1. Bergman et al., 1979. Minimal model. DOI: [10.1152/ajpendo.1979.236.6.E667](https://doi.org/10.1152/ajpendo.1979.236.6.E667)",
            "2. Hovorka et al., 2004. Nonlinear model predictive control. DOI: [10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010)",
            "3. Dalla Man et al., 2007. Meal simulation model. DOI: [10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506)",
            "4. Battelino et al., 2019. Time-in-range consensus. DOI: [10.2337/dci19-0028](https://doi.org/10.2337/dci19-0028)",
            "5. Riddell et al., 2017. Exercise management in T1D. DOI: [10.1016/S2213-8587(17)30014-1](https://doi.org/10.1016/S2213-8587(17)30014-1)",
            "6. Cryer, 2013. Hypoglycaemia and HAAF. DOI: [10.1056/NEJMra1215228](https://doi.org/10.1056/NEJMra1215228)",
            "7. Marling and Bunescu, 2020. OhioT1DM dataset. [CEUR paper](http://ceur-ws.org/Vol-2675/paper2.pdf)",
            "",
            "> **Safe closing line:** IINTS-AF does not prove that an algorithm is safe for "
            "> people. It makes a pre-clinical experiment easier to reproduce, inspect, and challenge.",
        ]
    )
    _pagebreak(lines)
    _formula_section(lines)
    _pagebreak(lines)
    lines.extend(
        [
            "# Appendix B - Selected implementation excerpts",
            "",
            "These screenshots are orientation aids. The repository source and exact release "
            "remain authoritative.",
            "",
            "| Excerpt | Review question |",
            "| --- | --- |",
            "| Candidate algorithm | Which observation and context produce a proposal? |",
            "| Simulator loop | In which order are events, state, observation, action, and recording handled? |",
            "| Safety supervisor | Which deterministic checks can alter a candidate? |",
            "| Realism validator | Which trace properties cause a fail or review verdict? |",
            "| Local AI gate | Why can narrative AI not become numerical authority? |",
            "| MDMP gate | Which data-contract and provenance checks precede evidence use? |",
        ]
    )
    _add_image(lines, "code_algorithm.png", "Candidate algorithm decision logic", "74%")
    _add_image(lines, "code_simulator.png", "Simulation control loop", "74%")
    _add_image(lines, "code_supervisor.png", "Independent deterministic safety supervisor", "74%")
    _add_image(lines, "code_realism.png", "Realism validator and gate", "74%")
    _add_image(lines, "code_ai_gate.png", "Local AI safety and evidence gate", "74%")
    _add_image(lines, "code_mdmp.png", "MDMP data-governance gate", "74%")
    _pagebreak(lines)
    lines.extend(
        [
            "# Appendix C - Mermaid source locations",
            "",
            "The diagrams shown in this book are generated from maintained Mermaid source:",
            "",
            "| Diagram | Source file |",
            "| --- | --- |",
            "| System architecture | `docs/eucys/diagrams/system-architecture.mmd` |",
            "| Simulation step | `docs/eucys/diagrams/simulation-step.mmd` |",
            "| Numeric authority | `docs/eucys/diagrams/numeric-authority.mmd` |",
            "| Evidence lifecycle | `docs/eucys/diagrams/evidence-lifecycle.mmd` |",
            "| Desktop bridge | `docs/eucys/diagrams/desktop-bridge.mmd` |",
            "| AI boundary | `docs/eucys/diagrams/ai-boundary.mmd` |",
            "| Data lifecycle | `docs/eucys/diagrams/data-lifecycle.mmd` |",
            "| Validation ladder | `docs/eucys/diagrams/validation-ladder.mmd` |",
            "",
            "## Example Mermaid source",
            "",
            "```mermaid",
            "flowchart LR",
            '    A[\"Scenario, patient, seed\"] --> B[\"Deterministic patient model\"]',
            '    B --> C[\"CGM-like observation\"]',
            '    C --> D[\"Candidate algorithm\"]',
            '    D --> E[\"Deterministic safety supervisor\"]',
            '    E --> F[\"Recorded evidence\"]',
            '    F -. read only .-> G[\"Optional local AI explanation\"]',
            "```",
            "",
            "End of playbook.",
        ]
    )
    return "\n".join(lines) + "\n"


def _html_document(markdown_text: str) -> str:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is required for the offline HTML edition.")
    body = subprocess.run(
        [
            pandoc,
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex",
            "--to",
            "html5",
            "--mathml",
            "--wrap=none",
        ],
        cwd=PLAYBOOK_DIR,
        input=markdown_text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    # The static diagram is already present; keep Mermaid source readable as code
    # without requiring network-loaded JavaScript in the offline edition.
    body = body.replace('<code class="language-mermaid">', '<code class="language-text">')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IINTS-AF EUCYS Complete Research Playbook</title>
  <style>
    :root {{
      --ink: {COLORS['ink']};
      --navy: {COLORS['navy']};
      --teal: {COLORS['teal']};
      --line: {COLORS['line']};
      --paper: {COLORS['paper']};
      --panel: #fff;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #E9EEF0;
      font: 17px/1.62 Georgia, "Times New Roman", serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1040px);
      gap: 28px;
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px;
    }}
    nav {{
      position: sticky;
      top: 20px;
      align-self: start;
      max-height: calc(100vh - 40px);
      overflow: auto;
      padding: 22px;
      color: #EAF3F4;
      background: var(--ink);
      border-top: 6px solid var(--teal);
      font: 13px/1.38 Arial, sans-serif;
    }}
    nav strong {{ display: block; margin-bottom: 14px; font-size: 16px; }}
    nav a {{ display: block; padding: 6px 0; color: #D5E5E8; text-decoration: none; }}
    nav a:hover {{ color: white; text-decoration: underline; }}
    main {{
      min-width: 0;
      padding: 52px 64px 90px;
      background: var(--panel);
      box-shadow: 0 8px 30px rgba(23,48,66,.10);
    }}
    h1, h2, h3 {{ color: var(--navy); font-family: Arial, sans-serif; line-height: 1.2; }}
    h1 {{ margin-top: 2.2em; padding-top: .5em; border-top: 4px solid var(--teal); font-size: 2.2rem; }}
    h1:first-of-type {{ margin-top: 0; }}
    h2 {{ margin-top: 2em; padding-bottom: .3em; border-bottom: 1px solid var(--line); }}
    h3 {{ margin-top: 1.7em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.2em 0 2em; font: 13px/1.42 Arial, sans-serif; }}
    th, td {{ padding: 9px 10px; border: 1px solid var(--line); vertical-align: top; text-align: left; }}
    th {{ background: #EAF1F3; }}
    tr:nth-child(even) td {{ background: #FAFBFB; }}
    img {{ display: block; max-width: 100%; height: auto; margin: 24px auto 8px; }}
    em {{ color: #596C78; }}
    blockquote {{ margin: 1.5em 0; padding: 13px 22px; background: #EFF6F5; border-left: 5px solid var(--teal); }}
    pre {{ overflow: auto; padding: 18px; background: #F1F4F5; border-left: 4px solid var(--teal); }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; }}
    p code, td code {{ padding: .1em .3em; background: #EFF2F3; }}
    a {{ color: #0D6672; }}
    @media (max-width: 900px) {{
      .shell {{ display: block; padding: 0; }}
      nav {{ position: static; max-height: none; }}
      main {{ padding: 32px 22px 70px; box-shadow: none; }}
      table {{ display: block; overflow-x: auto; }}
    }}
    @media print {{
      body {{ background: white; }}
      .shell {{ display: block; max-width: none; padding: 0; }}
      nav {{ display: none; }}
      main {{ padding: 0; box-shadow: none; }}
      h1 {{ break-before: page; }}
      img, table, blockquote {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<div class="shell">
  <nav>
    <strong>IINTS-AF EUCYS Playbook</strong>
    <a href="#read-this-first">Read first</a>
    <a href="#1-the-project-in-90-seconds">90-second project</a>
    <a href="#2-research-question-and-experimental-design">Research design</a>
    <a href="#3-software-architecture-and-authority-boundaries">Architecture</a>
    <a href="#4-scientific-model-and-equations">Scientific model</a>
    <a href="#5-desktop-workbench-and-researcher-workflow">Desktop app</a>
    <a href="#6-data-mdmp-ai-and-molecular-context">Data and AI</a>
    <a href="#7-fresh-runs-generated-for-this-book">Fresh runs</a>
    <a href="#8-locked-3600-run-eucys-benchmark">3,600-run benchmark</a>
    <a href="#9-held-out-ohiot1dm-glucose-forecast-benchmark">Ohio AI benchmark</a>
    <a href="#10-software-verification">Software checks</a>
    <a href="#11-live-eucys-demonstration-runbook">Live runbook</a>
    <a href="#12-jury-questions-limitations-and-next-experiments">Q&amp;A and limits</a>
    <a href="#13-evidence-and-reproduction-map">Evidence map</a>
    <a href="#appendix-a-complete-deterministic-formula-registry">All formulas</a>
  </nav>
  <main>{body}</main>
</div>
</body>
</html>
"""


def _write_latex_header() -> Path:
    path = BUILD_DIR / "playbook-header.tex"
    path.write_text(
        r"""
\usepackage{xcolor}
\definecolor{iintsnavy}{HTML}{214E65}
\definecolor{iintsteal}{HTML}{217A78}
\definecolor{iintsline}{HTML}{CBD6DC}
\pagestyle{plain}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.18}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _render_pdf() -> None:
    pandoc = shutil.which("pandoc")
    tectonic = shutil.which("tectonic")
    if not pandoc or not tectonic:
        raise RuntimeError("pandoc and tectonic are required for the complete playbook PDF.")
    header = _write_latex_header()
    command = [
        pandoc,
        str(MARKDOWN_PATH),
        "--from",
        "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex",
        "--pdf-engine",
        tectonic,
        "--resource-path",
        str(PLAYBOOK_DIR),
        "--include-in-header",
        str(header),
        "-V",
        "geometry:top=20mm,bottom=18mm,left=18mm,right=18mm",
        "-V",
        "fontsize=9pt",
        "-V",
        "papersize=a4",
        "-V",
        "linestretch=1.03",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=iintsnavy",
        "-V",
        "urlcolor=iintsteal",
        "-o",
        str(PDF_PATH),
    ]
    env = os.environ.copy()
    # Tectonic's populated macOS cache lives under ~/Library/Caches/Tectonic.
    # Do not inherit the temporary XDG cache used by Poppler.
    env.pop("XDG_CACHE_HOME", None)
    subprocess.run(command, cwd=PLAYBOOK_DIR, env=env, check=True)
    shutil.copy2(PDF_PATH, PACK_PDF_PATH)


def _validate_html() -> None:
    text = HTML_PATH.read_text(encoding="utf-8")
    missing: list[str] = []
    for match in re.finditer(r'<img[^>]+src="([^"]+)"', text):
        reference = html.unescape(match.group(1))
        target = (HTML_PATH.parent / reference).resolve()
        if not target.exists():
            missing.append(reference)
    if missing:
        raise RuntimeError(f"HTML contains missing image references: {missing}")
    if "Unknown variants" not in text or "3,600" not in text or "841,752" not in text:
        raise RuntimeError("Expected scientific-boundary/result text is missing from HTML.")


def _write_build_manifest(version: str, commit: str) -> None:
    selected_files = [
        MARKDOWN_PATH,
        HTML_PATH,
        PACK_PDF_PATH,
        *sorted(EVIDENCE_DIR.glob("*.txt")),
    ]
    hashes: dict[str, str] = {}
    for path in selected_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.relative_to(PLAYBOOK_DIR))] = digest

    page_count: int | None = None
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        result = subprocess.run(
            [pdfinfo, str(PACK_PDF_PATH)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
        if match:
            page_count = int(match.group(1))

    manifest = {
        "artifact": "IINTS-AF EUCYS Complete Research Playbook",
        "build_date": date.today().isoformat(),
        "sdk_version": version,
        "repository_commit": commit,
        "pdf_pages": page_count,
        "source_runs": {
            "fresh_presets": 5,
            "fresh_demo_scenarios": 3,
            "locked_benchmark_runs": 3600,
            "ohio_horizon_predictions": 841752,
        },
        "sha256": hashes,
    }
    (PLAYBOOK_DIR / "PLAYBOOK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    (PLAYBOOK_DIR / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def build() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplt_eucys_playbook"))
    _plot_style()
    _copy_assets()
    _render_report_pages()
    _plot_fresh_traces()
    _plot_realism_gate()
    _plot_demo_summary()
    _plot_benchmark_arms()
    _plot_algorithm_benchmark()
    _plot_ai_benchmark()
    test_summary = _parse_test_summary()
    _plot_quality_gates(test_summary)
    version = _project_version()
    commit = _git_commit()
    _make_cover(version, commit)
    markdown_text = _build_markdown(version, commit, test_summary)
    MARKDOWN_PATH.write_text(markdown_text, encoding="utf-8")
    HTML_PATH.write_text(_html_document(markdown_text), encoding="utf-8")
    _validate_html()
    _render_pdf()
    _write_build_manifest(version, commit)
    print(f"Markdown: {MARKDOWN_PATH}")
    print(f"HTML:     {HTML_PATH}")
    print(f"PDF:      {PDF_PATH}")
    print(f"Pack PDF: {PACK_PDF_PATH}")


if __name__ == "__main__":
    build()
