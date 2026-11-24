#!/usr/bin/env python3
"""Generate a clean scientific 4-panel visual dashboard for real MiniMed CareLink predictions.

Produces a high-resolution white-background figure showing:
  1. A 24-hour blind out-of-sample trace: actual Simplera CGM vs 30-min Gray-Box
     prediction (with clean NaN breaks across sensor gaps — never bridged!),
     90% confidence band, meal markers, and delivered insulin.
  2. Horizon RMSE comparison (30 / 60 / 120 min): Naive vs White-Box vs Gray-Box.
  3. 30-minute Clarke Error Grid with exact A/B/C/D/E boundaries and zone % score.
  4. Learned 24-hour personal circadian Fourier curve (zero-centered swing + mean offset).

Usage:
    $env:PYTHONPATH = "src"
    python tools/analysis/plot_minimed_prediction_dashboard.py "c:\\Users\\runeb\\Downloads\\Rune Bobbaers 24-11-2026.csv"
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iints.analysis.error_grid import clarke_zones
from iints.analysis.minimed_predictor import (
    MiniMedGrayBoxPredictor,
    PredictorConfig,
    Z_90_TWO_SIDED,
)
from iints.data.importer import import_carelink_timeline


def _insert_gap_nans(
    times: pd.Series,
    *arrays: np.ndarray,
    max_gap_minutes: float = 20.0,
) -> Tuple[List[pd.Timestamp], List[np.ndarray], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Insert NaN breaks wherever consecutive timestamps jump > max_gap_minutes."""
    t_list = list(times)
    n = len(t_list)
    if n == 0:
        return [], [np.array([], dtype=float) for _ in arrays], []

    out_t: List[pd.Timestamp] = [t_list[0]]
    out_arrs: List[List[float]] = [[float(a[0])] for a in arrays]
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for i in range(1, n):
        dt_min = (t_list[i] - t_list[i - 1]).total_seconds() / 60.0
        if dt_min > max_gap_minutes:
            gaps.append((t_list[i - 1], t_list[i]))
            mid_t = t_list[i - 1] + (t_list[i] - t_list[i - 1]) / 2
            out_t.append(mid_t)
            for lst in out_arrs:
                lst.append(float("nan"))
        out_t.append(t_list[i])
        for lst, a in zip(out_arrs, arrays):
            lst.append(float(a[i]))

    return out_t, [np.asarray(lst, dtype=float) for lst in out_arrs], gaps


def _draw_clarke_boundaries(ax: plt.Axes) -> None:
    """Draw exact Clarke Error Grid (1987) A/B/C/D/E boundaries matching error_grid.clarke_zones."""
    kw = {"color": "#64748B", "linewidth": 0.9, "linestyle": "--"}
    # Identity line
    ax.plot([0, 400], [0, 400], color="#0F172A", linewidth=1.0, alpha=0.45)

    # Zone A boundaries: rel_error <= 0.2 or (ref <= 70 and pred <= 70)
    ax.plot([0, 70], [70, 70], **kw)
    ax.plot([70, 70], [0, 56], **kw)
    ax.plot([70, 400], [56, 320], **kw)
    ax.plot([58.333, 333.333], [70, 400], **kw)

    # Zone C boundaries (exact match to error_grid.py lines 93-94)
    ax.plot([70, 290], [180, 400], **kw)
    ax.plot([130, 180], [0, 70], **kw)

    # Zone D & E boundaries
    ax.plot([0, 70], [180, 180], **kw)
    ax.plot([70, 70], [180, 400], **kw)
    ax.plot([180, 400], [70, 70], **kw)
    ax.plot([180, 180], [0, 70], **kw)
    ax.plot([240, 240], [70, 180], **kw)
    ax.plot([240, 400], [180, 180], **kw)

    # Zone labels placed strictly inside their true geometric regions
    ax.text(295, 305, "A", fontsize=11, fontweight="bold", color="#0F766E")
    ax.text(135, 220, "B", fontsize=10, fontweight="bold", color="#475569")
    ax.text(285, 200, "B", fontsize=10, fontweight="bold", color="#475569")
    ax.text(135, 345, "C", fontsize=10, fontweight="bold", color="#B45309")
    ax.text(158, 20, "C", fontsize=10, fontweight="bold", color="#B45309")
    ax.text(24, 125, "D", fontsize=10, fontweight="bold", color="#DC2626")
    ax.text(320, 120, "D", fontsize=10, fontweight="bold", color="#DC2626")
    ax.text(24, 300, "E", fontsize=10, fontweight="bold", color="#991B1B")
    ax.text(320, 28, "E", fontsize=10, fontweight="bold", color="#991B1B")


def build_dashboard(csv_path: Path, out_png: Path) -> Path:
    timeline = import_carelink_timeline(csv_path)
    ordered = timeline.sort_values("timestamp_dt").reset_index(drop=True)

    cfg = PredictorConfig()
    predictor = MiniMedGrayBoxPredictor(config=cfg)
    split_idx = int(len(ordered) * cfg.train_fraction)
    train_df = ordered.iloc[:split_idx].reset_index(drop=True)
    test_df = ordered.iloc[split_idx:].reset_index(drop=True)

    fit_info = predictor.fit(train_df)
    buckets = predictor._collect_rolling_forecasts(test_df, apply_residual=True)

    # Extract 30-min blind test arrays
    b30 = buckets[30]
    idx30 = np.asarray(b30["idx"], dtype=int)
    y_true_30 = np.asarray(b30["y_true"], dtype=float)
    y_gray_30 = np.asarray(b30["gray"], dtype=float)
    sd_gray_30 = np.asarray(b30["gray_std"], dtype=float)
    target_indices_30 = idx30 + int(round(30.0 / cfg.step_minutes))
    pred_times_30 = pd.to_datetime(test_df["timestamp_dt"].iloc[target_indices_30]).reset_index(drop=True)

    # Exact Clarke A+B percentage on all 30-min blind predictions
    zones_30 = clarke_zones(y_true_30, y_gray_30)
    clarke_ab_30 = float(np.mean(np.isin(zones_30, ["A", "B"])) * 100.0)

    # Pick the last 24-hour window in the blind test set and break cleanly across sensor gaps
    window_n = min(288, len(pred_times_30))
    t_raw = pred_times_30.iloc[-window_n:].reset_index(drop=True)
    yt_raw = y_true_30[-window_n:]
    yg_raw = y_gray_30[-window_n:]
    sd_raw = sd_gray_30[-window_n:]
    lo_raw = np.clip(yg_raw - Z_90_TWO_SIDED * sd_raw, 20.0, 450.0)
    hi_raw = np.clip(yg_raw + Z_90_TWO_SIDED * sd_raw, 20.0, 450.0)

    t_plot, (yt_plot, yg_plot, lo_plot, hi_plot), gaps = _insert_gap_nans(
        t_raw, yt_raw, yg_raw, lo_raw, hi_raw, max_gap_minutes=cfg.max_gap_minutes
    )

    t_start, t_end = t_raw.iloc[0], t_raw.iloc[-1]
    raw_win = test_df[(test_df["timestamp_dt"] >= t_start) & (test_df["timestamp_dt"] <= t_end)]

    # --- Clean scientific white 4-panel figure ---
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "font.size": 10,
        }
    )

    fig = plt.figure(figsize=(16, 10), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.28)

    # Panel 1 (Top): 24h blind forecast vs actual CGM with explicit sensor-gap breaks
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axhspan(70, 180, color="#ECFDF5", alpha=0.75, zorder=0, label="Doelbereik (70-180 mg/dL)")
    ax1.axhline(70, color="#DC2626", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.axhline(180, color="#D97706", linestyle="--", linewidth=0.9, alpha=0.7)

    for g_idx, (g_start, g_end) in enumerate(gaps):
        ax1.axvspan(
            g_start,
            g_end,
            color="#E2E8F0",
            alpha=0.65,
            zorder=1,
            label="Geen CGM / sensor-wissel" if g_idx == 0 else None,
        )

    ax1.fill_between(
        t_plot,
        lo_plot,
        hi_plot,
        color="#38BDF8",
        alpha=0.25,
        label="90% onzekerheidsband (p05-p95)",
        zorder=2,
    )
    ax1.plot(t_plot, yt_plot, color="#0F172A", linewidth=1.9, label="Werkelijke Simplera CGM", zorder=4)
    ax1.plot(
        t_plot,
        yg_plot,
        color="#0284C7",
        linewidth=1.6,
        label="30-min Gray-Box voorspelling (berekend 30 min eerder)",
        zorder=5,
    )

    meals = raw_win[raw_win["carbs"] > 0]
    if not meals.empty:
        ax1.scatter(
            meals["timestamp_dt"],
            meals["glucose"],
            color="#16A34A",
            marker="^",
            s=65,
            zorder=6,
            label="Maaltijd (koolhydraten)",
        )

    ax1_ins = ax1.twinx()
    ax1_ins.grid(False)
    boluses = raw_win[raw_win["bolus_units"] > 0]
    if not boluses.empty:
        ax1_ins.bar(
            boluses["timestamp_dt"],
            boluses["bolus_units"],
            width=0.008,
            color="#E11D48",
            alpha=0.45,
            label="MiniMed bolus (U)",
        )
    ax1_ins.set_ylabel("Bolus (U)", color="#BE123C")
    ax1_ins.set_ylim(0, max(12.0, float(boluses["bolus_units"].max()) * 2.4) if not boluses.empty else 12.0)

    # Merge legends from both y-axes so bolus bars are also labeled
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_ins.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        ncol=3,
        fontsize=8.8,
    )

    ax1.set_title(
        f"1. Blinde 24-uurs MiniMed 780G Predictie (30 min vooruit)  |  "
        f"Est. TDD = {fit_info['estimated_tdd_u_per_day']:.1f} U/dag, ISF = {fit_info['estimated_isf_mgdl_per_u']:.1f} mg/dL/U",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        pad=10,
    )
    ax1.set_ylabel("Glucose (mg/dL)")
    valid_hi = hi_raw[np.isfinite(hi_raw)]
    y_top = min(400.0, max(260.0, float(np.max(valid_hi)) + 25.0)) if valid_hi.size else 320.0
    ax1.set_ylim(40, y_top)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))

    # Panel 2 (Bottom-Left): RMSE Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    horizons = list(cfg.horizons_minutes)
    naive_rmses, wb_rmses, gray_rmses = [], [], []
    for h in horizons:
        bh = buckets[h]
        yt = np.asarray(bh["y_true"], dtype=float)
        naive_rmses.append(float(np.sqrt(np.mean((np.asarray(bh["naive"]) - yt) ** 2))))
        wb_rmses.append(float(np.sqrt(np.mean((np.asarray(bh["wb"]) - yt) ** 2))))
        gray_rmses.append(float(np.sqrt(np.mean((np.asarray(bh["gray"]) - yt) ** 2))))

    x = np.arange(len(horizons))
    w = 0.25
    ax2.bar(x - w, naive_rmses, width=w, color="#94A3B8", label="Naive (rechte lijn)")
    ax2.bar(x, wb_rmses, width=w, color="#38BDF8", label="White-Box (Hovorka+KF)")
    ax2.bar(x + w, gray_rmses, width=w, color="#0369A1", label="Gray-Box (Hybride)")
    for xi, val in zip(x + w, gray_rmses):
        ax2.text(xi, val + 2.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0369A1")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h} min" for h in horizons])
    ax2.set_ylabel("RMSE (mg/dL) — lager is beter")
    ax2.set_title("2. Foutvergelijking (Blinde Testset)", fontweight="bold", color="#0F172A")
    ax2.legend(fontsize=8.5, loc="upper left")

    # Panel 3 (Bottom-Middle): Canonical Clarke Error Grid (30 min)
    ax3 = fig.add_subplot(gs[1, 1])
    _draw_clarke_boundaries(ax3)
    ax3.scatter(y_true_30, y_gray_30, s=6, alpha=0.22, color="#0284C7", edgecolors="none")
    ax3.set_xlim(0, 400)
    ax3.set_ylim(0, 400)
    ax3.set_xlabel("Werkelijke CGM (mg/dL)")
    ax3.set_ylabel("Gray-Box Voorspelling (mg/dL)")
    ax3.set_title(
        f"3. Clarke Error Grid (30 min, A+B = {clarke_ab_30:.1f}%)",
        fontweight="bold",
        color="#0F172A",
    )

    # Panel 4 (Bottom-Right): Zero-centered 24-hour Personal Circadian Swing
    ax4 = fig.add_subplot(gs[1, 2])
    hours = np.linspace(0.0, 24.0, 241)
    w60 = predictor._residual_weights.get(60)
    if w60 is not None:
        mean_offset = float(w60[0])
        circ_swing = np.zeros_like(hours)
        for m in range(1, cfg.fourier_harmonics + 1):
            ang = 2.0 * math.pi * m * hours / 24.0
            circ_swing += float(w60[2 * m - 1]) * np.sin(ang) + float(w60[2 * m]) * np.cos(ang)
        ax4.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0)
        ax4.plot(hours, circ_swing, color="#0F766E", linewidth=2.2)
        ax4.fill_between(
            hours, 0.0, circ_swing, where=(circ_swing >= 0), color="#F59E0B", alpha=0.22, label="Hoger dan daggemiddelde"
        )
        ax4.fill_between(
            hours, 0.0, circ_swing, where=(circ_swing < 0), color="#0EA5E9", alpha=0.22, label="Lager dan daggemiddelde"
        )
        ax4.legend(fontsize=8.5, loc="best")
        ax4.set_title(
            f"4. 24-uurs Dagritme (offset: {mean_offset:+.1f} mg/dL)",
            fontweight="bold",
            color="#0F172A",
        )
    else:
        ax4.set_title("4. 24-uurs Dagritme (Fourier)", fontweight="bold", color="#0F172A")
    ax4.set_xlim(0, 24)
    ax4.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax4.set_xlabel("Uur van de dag")
    ax4.set_ylabel("Circadiaanse afwijking (mg/dL)")

    fig.suptitle(
        "IINTS-SDK — Persoonlijk MiniMed 780G Gray-Box Predictie Dashboard (In-Silico Onderzoek)",
        fontsize=14,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 4-panel MiniMed Gray-Box prediction dashboard.")
    parser.add_argument("csv_path", type=Path, help="Path to Medtronic CareLink CSV")
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("docs/presentation_visuals/minimed_prediction_dashboard.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()
    out = build_dashboard(args.csv_path, args.out_png)
    print(f"Dashboard opgeslagen op: {out.resolve()}")


if __name__ == "__main__":
    main()
