from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

from iints.research.visualizer import apply_scientific_plot_style


@dataclass(frozen=True)
class EUCYSFigureMetadata:
    """Metadata describing a single scientific figure for the jury dossier."""

    figure_id: str
    category: str
    title: str
    subtitle: str
    file_name: str
    png_path: Path | None
    description: str
    key_metrics: dict[str, str]
    scientific_citations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "category": self.category,
            "title": self.title,
            "subtitle": self.subtitle,
            "file_name": self.file_name,
            "png_path": str(self.png_path) if self.png_path is not None else None,
            "description": self.description,
            "key_metrics": self.key_metrics,
            "scientific_citations": self.scientific_citations,
        }


@dataclass(frozen=True)
class EUCYSJuryPortfolio:
    """Complete EUCYS Jury Portfolio with all generated assets."""

    output_dir: Path
    figures: list[EUCYSFigureMetadata]
    index_html_path: Path
    manifest_json_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "total_figures": len(self.figures),
            "figures": [f.to_dict() for f in self.figures],
            "index_html_path": str(self.index_html_path),
            "manifest_json_path": str(self.manifest_json_path),
        }


# ==============================================================================
# INDIVIDUAL SCIENTIFIC PLOT GENERATORS (NATURE / LANCET GRADE, 300 DPI)
# ==============================================================================

def plot_clarke_error_grid(output_path: Path | str,
                           reference=None,
                           predicted=None) -> Path:
    """Plot a Clarke Error Grid Analysis from paired measurements.

    Args:
        output_path: Destination PNG.
        reference: Reference glucose values (mg/dL). Required.
        predicted: Model predictions paired with ``reference``. Required.

    Raises:
        ValueError: If no paired data is supplied. Zone percentages are
            counted from the data; this function will not draw a grid from
            simulated scatter, because the resulting percentages would not be
            a clinical accuracy result.
    """
    from iints.analysis.error_grid import clarke_error_grid

    if reference is None or predicted is None:
        raise ValueError(
            "plot_clarke_error_grid requires paired (reference, predicted) "
            "glucose values. Supply model outputs against held-out reference "
            "data; the zone percentages are computed from them."
        )

    ega = clarke_error_grid(reference, predicted)
    ref_vals = np.asarray(reference, dtype=float).ravel()
    pred_vals = np.asarray(predicted, dtype=float).ravel()

    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Reference Blood Glucose (mg/dL)", fontsize=11.5, weight="bold")
    ax.set_ylabel("IINTS-AF Digital Twin Predicted Glucose (mg/dL)", fontsize=11.5, weight="bold")

    # Zone A (+- 20% or +- 20 mg/dL)
    zone_a_poly = Polygon(
        np.array([[0, 0], [70, 56], [400, 320], [400, 400], [58.33, 70], [0, 70]], dtype=float),
        closed=True, facecolor="#DCFCE7", edgecolor="#86EFAC", alpha=0.7, linewidth=1.0, zorder=1
    )
    ax.add_patch(zone_a_poly)

    # Zone B Polygon (upper and lower)
    zone_b_upper = Polygon(
        np.array([[0, 70], [58.33, 70], [400, 400], [400, 400], [0, 400]], dtype=float),
        closed=True, facecolor="#FEF9C3", edgecolor="#FDE047", alpha=0.4, linewidth=0.8, zorder=0
    )
    zone_b_lower = Polygon(
        np.array([[70, 0], [70, 56], [400, 320], [400, 0]], dtype=float),
        closed=True, facecolor="#FEF9C3", edgecolor="#FDE047", alpha=0.4, linewidth=0.8, zorder=0
    )
    ax.add_patch(zone_b_upper)
    ax.add_patch(zone_b_lower)

    # Zone Boundary Lines
    ax.plot([0, 400], [0, 400], color="#475569", linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)
    ax.plot([0, 58.33], [70, 70], color="#1E293B", linewidth=1.2, zorder=2)
    ax.plot([58.33, 400], [70, 480], color="#1E293B", linewidth=1.2, zorder=2)
    ax.plot([70, 400], [56, 320], color="#1E293B", linewidth=1.2, zorder=2)
    ax.plot([70, 70], [0, 56], color="#1E293B", linewidth=1.2, zorder=2)
    ax.plot([180, 400], [70, 70], color="#1E293B", linewidth=1.0, zorder=2)
    ax.plot([240, 240], [70, 180], color="#1E293B", linewidth=1.0, zorder=2)

    ax.scatter(ref_vals, pred_vals, color="#2563EB", alpha=0.55, s=22,
               edgecolors="#1D4ED8", linewidth=0.4,
               label=f"Paired forecasts (N={ega.n_pairs:,})", zorder=4)

    # Zone Labels with clear styling
    pct = ega.percentages
    ax.text(250, 250, f"Zone A\n(Clinically Accurate: {pct['A']:.1f}%)", fontsize=11, color="#047857", weight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#059669", alpha=0.95), zorder=5)
    ax.text(330, 170, f"Zone B\n(Benign: {pct['B']:.1f}%)", fontsize=10, color="#B45309", weight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#D97706", alpha=0.9), zorder=5)
    ax.text(45, 340, f"Zone E\n(Erroneous: {pct['E']:.1f}%)", fontsize=9.5, color="#DC2626", weight="bold", ha="center", zorder=5)
    ax.text(340, 40, f"Zone D\n(Missed: {pct['D']:.1f}%)", fontsize=9.5, color="#DC2626", weight="bold", ha="center", zorder=5)

    # Statistical Summary Card Box (Top Left) - all values counted from data.
    stats_text = (
        "Clarke EGA (computed from paired data):\n"
        f"• Zone A (No Action Required): {pct['A']:.1f}%\n"
        f"• Zone B (Benign Deviation):   {pct['B']:.1f}%\n"
        f"• Zone C/D/E (Clinical Hazard): {ega.hazardous_pct:.1f}%\n"
        f"• Pairs analysed: n = {ega.n_pairs:,}"
    )
    ax.text(18, 280, stats_text, fontsize=9, color="#0F172A", family="monospace", va="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#94A3B8", alpha=0.95), zorder=5)

    ax.grid(True, linestyle="--", color="#E2E8F0", alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    plt.title(f"Clarke Error Grid Analysis – {ega.n_pairs:,} paired in silico forecasts\n"
              "Zone percentages counted from the plotted pairs",
              fontsize=12.5, weight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_glycemic_tir_distribution(output_path: Path | str) -> Path:
    """Generate a labelled synthetic TIR layout fixture.

    This legacy helper is retained for visual regression work only. Its values
    are not study results; the EUCYS dossier no longer calls it by default.
    """
    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300, gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    # Panel a: Stacked Horizontal Bars
    cohorts = ["Healthy (N=15)", "Prediabetes (N=16)", "Type 2 Diabetes (N=14)", "IINTS-AF Closed-Loop"]
    v_low = np.array([0.0, 0.1, 0.4, 0.0])       # <54 mg/dL
    low = np.array([0.4, 0.7, 1.8, 0.8])         # 54-69 mg/dL
    tir = np.array([96.2, 88.5, 68.2, 92.4])     # 70-180 mg/dL
    high = np.array([3.2, 9.8, 24.5, 6.4])       # 181-250 mg/dL
    v_high = np.array([0.2, 0.9, 5.1, 0.4])      # >250 mg/dL

    y = np.arange(len(cohorts))
    height = 0.52

    # Bar fills with modern palette
    ax1.barh(y, v_low, height, label="Very Low (<54 mg/dL) [<1% Target]", color="#991B1B", zorder=3)
    ax1.barh(y, low, height, left=v_low, label="Low (54–69 mg/dL) [<4% Target]", color="#EF4444", zorder=3)
    ax1.barh(y, tir, height, left=v_low+low, label="In Range (70–180 mg/dL) [>70% Target]", color="#10B981", zorder=3)
    ax1.barh(y, high, height, left=v_low+low+tir, label="High (181–250 mg/dL) [<25% Target]", color="#F59E0B", zorder=3)
    ax1.barh(y, v_high, height, left=v_low+low+tir+high, label="Very High (>250 mg/dL) [<5% Target]", color="#EA580C", zorder=3)

    # Draw TIR percentage text on bars
    for idx, (t_val, vl, l) in enumerate(zip(tir, v_low, low)):
        ax1.text(vl + l + t_val / 2.0, y[idx], f"{t_val:.1f}%", ha="center", va="center", color="#FFFFFF", fontsize=10.5, fontweight="bold", zorder=4)

    ax1.axvline(70.0, color="#059669", linestyle="--", linewidth=1.5, alpha=0.9, label="ADA Target Threshold (70% TIR)", zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cohorts, fontsize=10.5, fontweight="bold", color="#0F172A")
    ax1.set_xlabel("Percentage of Time (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Clinical Glycemic Profile per Cohort\n(ATTD / ADA International Consensus Standards)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle="--", alpha=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8.5, framealpha=0.95)
    ax1.text(-0.20, 1.05, "a", transform=ax1.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    # Panel b: Clean Graphic Metric Cards (Custom layout, no clipped table)
    ax2.axis("off")
    ax2.text(0.5, 0.985, "IINTS-AF Closed-Loop Glycemic Targets Scorecard", ha="center", va="top", fontsize=12, fontweight="bold", color="#0F172A")
    ax2.text(0.5, 0.935, "Evaluated against International Consensus Guidelines (N=45 Cohorts)", ha="center", va="top", fontsize=9.2, color="#64748B")

    cards = [
        ("Time In Range (TIR: 70–180 mg/dL)", "92.4%", "Target: > 70.0%", "EXCEEDS (OPTIMAL)", "#059669", "#ECFDF5"),
        ("Time Below Range (TBR: <70 mg/dL)", "0.8%", "Target: < 4.0%", "EXCEEDS (SAFE)", "#059669", "#ECFDF5"),
        ("Severe Hypoglycemia (<54 mg/dL)", "0.0%", "Target: < 1.0%", "ZERO INCIDENTS", "#059669", "#ECFDF5"),
        ("Time Above Range (TAR: >180 mg/dL)", "6.8%", "Target: < 25.0%", "EXCEEDS (OPTIMAL)", "#059669", "#ECFDF5"),
        ("Glycemic Variability (CV %)", "28.4%", "Target: < 36.0%", "STABLE DYNAMICS", "#059669", "#ECFDF5"),
        ("Glucose Management Index", "124.6 mg/dL", "Target GMI: < 154 mg/dL", "6.3% eA1c", "#2563EB", "#EFF6FF"),
    ]

    card_y_starts = np.linspace(0.77, 0.02, len(cards))
    card_h = 0.105

    for (title, value, target, status, border_col, bg_col), cy in zip(cards, card_y_starts):
        # Draw card background
        rect = FancyBboxPatch((0.02, cy), 0.96, card_h, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=bg_col, edgecolor=border_col, linewidth=1.2, transform=ax2.transAxes)
        ax2.add_patch(rect)

        # Texts inside card
        ax2.text(0.04, cy + card_h * 0.65, title, fontsize=8.8, fontweight="bold", color="#0F172A", transform=ax2.transAxes)
        ax2.text(0.04, cy + card_h * 0.25, target, fontsize=7.8, color="#64748B", transform=ax2.transAxes)
        ax2.text(0.74, cy + card_h * 0.50, value, fontsize=11.0, fontweight="bold", color=border_col, ha="right", va="center", transform=ax2.transAxes)
        ax2.text(0.97, cy + card_h * 0.50, status, fontsize=7.2, fontweight="bold", color="#FFFFFF", ha="right", va="center", bbox=dict(boxstyle="round,pad=0.25", facecolor=border_col, edgecolor="none"), transform=ax2.transAxes)

    ax2.text(-0.05, 1.05, "b", transform=ax2.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    fig.suptitle("SYNTHETIC LAYOUT FIXTURE - NOT A STUDY RESULT", color="#991B1B", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_sc_islet_gsis_dynamics(output_path: Path | str) -> Path:
    """Generate a synthetic GSIS layout fixture, not wet-lab evidence."""
    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    # Panel a: Dynamic Perifusion Profile (C-Peptide release over time)
    t = np.linspace(0, 90, 180)
    c_pep = np.full_like(t, 0.40)
    mask1 = (t >= 20) & (t < 35)
    c_pep[mask1] += 3.2 * np.exp(-(t[mask1] - 25)**2 / 12)
    mask2 = (t >= 30) & (t < 60)
    c_pep[mask2] += 1.75 * (1 - np.exp(-(t[mask2] - 30) / 8))
    mask3 = (t >= 60)
    c_pep[mask3] = 0.40 + (c_pep[t < 60][-1] - 0.40) * np.exp(-(t[mask3] - 60) / 10)

    np.random.seed(42)
    primary_c_pep = c_pep * 1.12 + np.random.normal(0, 0.04, len(t))

    # Phase background bands
    ax1.axvspan(0, 20, color="#F1F5F9", alpha=0.7)
    ax1.axvspan(20, 60, color="#FEF3C7", alpha=0.5)
    ax1.axvspan(60, 90, color="#F1F5F9", alpha=0.7)

    # Phase Badges (Placed neatly at the top)
    ax1.text(10, 4.30, "Basal Glucose\n(2.8 mM / 50 mg/dL)", fontsize=8.5, ha="center", color="#475569", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.9))
    ax1.text(40, 4.30, "High Glucose Challenge\n(16.7 mM / 300 mg/dL)", fontsize=8.5, ha="center", color="#B45309", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF3C7", edgecolor="#D97706", alpha=0.9))
    ax1.text(75, 4.30, "Return to Basal\n(2.8 mM)", fontsize=8.5, ha="center", color="#475569", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.9))

    ax1.plot(t, primary_c_pep, label="Primary Cadaveric Human Islets (Gold Standard)", color="#64748B", linestyle="--", linewidth=2.0)
    ax1.plot(t, c_pep, label="IINTS-AF SC-Beta Islet Cluster (Stage 6)", color="#2563EB", linewidth=3.0)
    ax1.fill_between(t, c_pep - 0.15, c_pep + 0.15, color="#DBEAFE", alpha=0.5, label="95% Confidence Interval")

    # Annotate Phase 1 Secretion Peak cleanly offset
    ax1.annotate("Phase 1 Peak:\n3.60 ng/10⁶ cells/min", xy=(25, 3.6), xytext=(36, 3.2),
                 ha="left", fontsize=8.5, fontweight="bold", color="#1D4ED8",
                 arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF", edgecolor="#2563EB", alpha=0.9))

    ax1.set_xlabel("Dynamic Perifusion Time (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("C-Peptide Secretion (ng / 10⁶ cells / min)", fontsize=11, fontweight="bold")
    ax1.set_title("Glucose-Stimulated Insulin Secretion (GSIS) Perifusion Assay", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 4.7)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax1.text(-0.12, 1.05, "a", transform=ax1.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    # Panel b: Stage-6 Proteomics & Maturation Fingerprint
    markers = ["INS\n(Insulin)", "PDX1\n(TF)", "NKX6-1\n(Beta ID)", "MAFA\n(Maturation)", "GCG\n(Glucagon)", "SST\n(Somatostatin)"]
    expr_primary = np.array([100.0, 95.0, 92.0, 88.0, 20.0, 8.0])
    expr_sc_islet = np.array([94.0, 91.0, 89.0, 84.0, 18.0, 7.0])
    err_sc = np.array([3.2, 2.8, 3.5, 4.1, 1.5, 0.9])

    x = np.arange(len(markers))
    width = 0.36

    ax2.bar(x - width/2, expr_primary, width, label="Primary Human Islet", color="#64748B", edgecolor="#334155", linewidth=0.8)
    ax2.bar(x + width/2, expr_sc_islet, width, yerr=err_sc, capsize=4, label="SC-Beta Islet (IINTS-AF)", color="#059669", edgecolor="#047857", linewidth=0.8)

    # Stimulation Index Badge (Top Right)
    ax2.text(0.96, 0.94, "Stimulation Index (SI = High/Low):\nSI_SC = 3.68 ± 0.24 (Authentic GSIS)", transform=ax2.transAxes, ha="right", va="top", fontsize=9, fontweight="bold", color="#047857", bbox=dict(boxstyle="round,pad=0.35", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.95))

    ax2.set_xticks(x)
    ax2.set_xticklabels(markers, fontsize=9.5, fontweight="bold", color="#0F172A")
    ax2.set_ylabel("Relative Expression (% of Primary Islets)", fontsize=11, fontweight="bold")
    ax2.set_title("Proteomics & Maturation Marker Fingerprint\n(Stage 6 Differentiated Clusters)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax2.text(-0.10, 1.05, "b", transform=ax2.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    fig.suptitle("SYNTHETIC LAYOUT FIXTURE - NO WET-LAB DATA", color="#991B1B", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_regenerative_graft_survival(output_path: Path | str) -> Path:
    """Generate an uncalibrated synthetic graft layout fixture."""
    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(11, 5.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFFFF")

    days = np.linspace(0, 90, 91)
    np.random.seed(42)
    exogenous_insulin = 42.0 * np.exp(-days / 18) + np.random.normal(0, 0.35, len(days))
    exogenous_insulin = np.clip(exogenous_insulin, 0, 50)
    c_pep_fasting = 1.85 * (1.0 - np.exp(-days / 20)) + np.random.normal(0, 0.015, len(days))

    # Background Phases
    ax1.axvspan(0, 14, color="#F1F5F9", alpha=0.6)
    ax1.axvspan(14, 45, color="#FEF3C7", alpha=0.4)
    ax1.axvspan(45, 90, color="#ECFDF5", alpha=0.5)

    ax1.text(7, 46.5, "Phase 1: Vascularization\n& Engraftment", fontsize=8.2, ha="center", color="#64748B", fontweight="bold")
    ax1.text(29.5, 46.5, "Phase 2: Maturation\n& Insulin Weaning", fontsize=8.2, ha="center", color="#B45309", fontweight="bold")
    ax1.text(67.5, 46.5, "Phase 3: 100% Endogenous Independence", fontsize=8.2, ha="center", color="#047857", fontweight="bold")

    ax1.set_xlabel("Days Post-Transplantation (Subcutaneous / Omental Pouch)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Exogenous Insulin Dosing (Units / Day)", color="#DC2626", fontsize=11, fontweight="bold")
    line1 = ax1.plot(days, exogenous_insulin, color="#DC2626", linewidth=3.0, label="Exogenous Daily Insulin (U/day)")
    ax1.tick_params(axis="y", labelcolor="#DC2626")
    ax1.set_ylim(-2, 52)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fasting Endogenous C-Peptide (ng/mL)", color="#059669", fontsize=11, fontweight="bold")
    line2 = ax2.plot(days, c_pep_fasting, color="#059669", linewidth=3.0, linestyle="-", label="Endogenous C-Peptide (ng/mL)")
    ax2.tick_params(axis="y", labelcolor="#059669")
    ax2.set_ylim(-0.1, 2.5)
    ax2.spines["top"].set_visible(False)

    # Day 45 Milestone Annotation
    ax1.axvline(45, color="#059669", linestyle="--", linewidth=2.0)
    ax1.text(46, 28, "Day 45 Milestone:\n100% Exogenous Insulin\nIndependence (0 U/day)\nC-Peptide: 1.85 ng/mL", fontsize=9.5, color="#047857", fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.95))

    lines = line1 + line2
    labels = [str(l.get_label()) for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.95, fontsize=9.5)

    plt.title("Synthetic 90-Day Graft Scenario\n(Uncalibrated layout fixture - not a treatment result)", fontsize=13, fontweight="bold", pad=15)
    ax1.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_edge_hardware_latency_budget(output_path: Path | str) -> Path:
    """Generate a synthetic hardware-layout fixture, not benchmark evidence."""
    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax1.set_facecolor("#FFFFFF")
    ax2.set_facecolor("#FFFFFF")

    # Panel a: Platform Latency Comparison (Horizontal Bar, Log Scale)
    platforms = [
        "Cloud REST API (Remote)",
        "Desktop CPU (x86_64)",
        "NVIDIA Jetson Orin Nano (15W)",
        "Xilinx Zynq FPGA Co-Proc",
        "IINTS-AF Rust Deterministic Core",
    ]
    latencies_ms = [485.0, 18.2, 4.20, 0.85, 0.40]
    colors = ["#EF4444", "#F59E0B", "#10B981", "#2563EB", "#7C3AED"]

    y_pos = np.arange(len(platforms))
    bars = ax1.barh(y_pos, latencies_ms, color=colors, height=0.55, edgecolor="#0F172A", linewidth=0.8, zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Inference & Safety Supervisor Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(platforms, fontsize=9.5, fontweight="bold", color="#0F172A")
    ax1.set_title("Platform Execution Latency vs Hard Safety Bound", fontsize=12, fontweight="bold")
    ax1.axvline(1000, color="#DC2626", linestyle="--", linewidth=1.5, label="Max Medical Safety Bound (1,000 ms)", zorder=2)
    ax1.axvline(5.0, color="#059669", linestyle=":", linewidth=1.5, label="Deterministic Edge Goal (5.0 ms)", zorder=2)

    for bar, val in zip(bars, latencies_ms):
        x_text = val * 1.25 if val < 100 else 120.0
        t_col = "#0F172A" if val < 100 else "#FFFFFF"
        ax1.text(x_text, bar.get_y() + bar.get_height()/2.0, f"{val:.2f} ms", va="center", fontsize=9.5, fontweight="bold", color=t_col, zorder=4)

    ax1.grid(axis="x", linestyle="--", alpha=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax1.text(-0.25, 1.05, "a", transform=ax1.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    # Panel b: Jetson Orin Nano Subsystem Pipeline Breakdown (Horizontal Stacked Bar)
    stages = ["Sensor Ingestion", "GlucoFM Encoder", "Digital Twin ODE", "Dual-Guard Safety", "ML-DSA Sign"]
    times = [0.35, 2.10, 0.85, 0.50, 0.40]
    stage_cols = ["#3B82F6", "#6366F1", "#10B981", "#F59E0B", "#EF4444"]

    ax2.set_xlim(0, 4.8)
    ax2.set_ylim(-0.8, 1.2)
    ax2.set_xlabel("Cumulative Latency on Jetson Orin Nano (ms)", fontsize=11, fontweight="bold")
    ax2.set_yticks([])

    left = 0.0
    for name, t_ms, col in zip(stages, times, stage_cols):
        ax2.barh(0, t_ms, height=0.45, left=left, color=col, edgecolor="#0F172A", linewidth=0.8, label=f"{name} ({t_ms:.2f} ms)", zorder=3)
        if t_ms >= 1.50:
            ax2.text(left + t_ms/2.0, 0, f"{name}\n{t_ms:.2f} ms", ha="center", va="center", color="#FFFFFF", fontsize=8.5, fontweight="bold", zorder=4)
        left += t_ms

    # Total & Duty Cycle Callout
    duty_badge = (
        "Total Edge Cycle: 4.20 ms\n"
        "5-Min Clinical Tick: 300,000 ms\n"
        "Hardware Duty Cycle: 0.0014%\n"
        "Power Draw: 12.4W (Passive Cooling)"
    )
    ax2.text(2.4, -0.55, duty_badge, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.95))

    ax2.set_title("Jetson Orin Nano Execution Pipeline (Total = 4.20 ms)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=8, framealpha=0.95)
    ax2.text(-0.05, 1.05, "b", transform=ax2.transAxes, fontsize=16, fontweight="bold", color="#0F172A")

    fig.suptitle("SYNTHETIC LAYOUT FIXTURE - NO HARDWARE BENCHMARK INPUT", color="#991B1B", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


def plot_quantum_safe_mdmp_security(output_path: Path | str) -> Path:
    """Generate an implementation-status diagram without conformity claims."""
    apply_scientific_plot_style()
    out_file = Path(output_path).expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "IINTS-AF Regulatory & Cryptographic Governance Architecture", ha="center", va="top", fontsize=14, fontweight="bold", color="#0F172A")
    ax.text(0.5, 0.925, "Implemented controls, self-assessment topics, and explicit roadmap gaps", ha="center", va="top", fontsize=10, color="#64748B")

    cards_data = [
        ("Packet Integrity & Signing", "Ed25519 + SHA-256 + ChaCha20-Poly1305", "Every CGM & infusion control packet is signed and authenticated with classical primitives. Post-quantum (ML-DSA / FIPS 204) signing is a roadmap item, not implemented.", "IMPLEMENTED (classical)", "#2563EB", "#EFF6FF"),
        ("Data Provenance & Lineage", "W3C PROV-O & RO-Crate", "Immutable SHA-256 hash-chained study bundles and run parameters provide a mathematically verifiable, tamper-evident audit trail.", "TRACEABLE AUDIT", "#7C3AED", "#FAF5FF"),
        ("EU AI Act Review", "Regulation (EU) 2024/1689", "Research-software self-assessment topics are documented. This is not a legal classification, conformity assessment, or CE mark.", "SELF-ASSESSMENT", "#059669", "#ECFDF5"),
        ("In Silico Credibility", "ASME V&V 40 concepts", "Evidence contracts support traceable comparison, but context-of-use validation and regulatory qualification remain future work.", "PARTIAL / RESEARCH", "#D97706", "#FFFBEB"),
        ("Deterministic Safety Layer", "Research implementation", "Fixed safety logic can override candidate outputs in simulation. It is not certified IEC 62304 Class C medical software.", "IMPLEMENTED IN SIMULATION", "#DC2626", "#FEF2F2"),
        ("Patient Data Sovereignty", "GDPR / HIPAA De-identification", "Local-first edge execution ensures zero biometric telemetry egress to the cloud without explicit, cryptographically signed patient consent.", "AIR-GAPPED READY", "#0D9488", "#F0FDFA"),
    ]

    cols = 3
    rows = 2
    x_coords = [0.02, 0.35, 0.68]
    y_coords = [0.47, 0.04]
    w_card = 0.30
    h_card = 0.38

    import textwrap

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(cards_data):
                break
            title, standard, desc, status, accent_col, bg_col = cards_data[idx]
            cx = x_coords[c]
            cy = y_coords[r]

            # Card background
            card_rect = FancyBboxPatch((cx, cy), w_card, h_card, boxstyle="round,pad=0.015,rounding_size=0.03", facecolor=bg_col, edgecolor=accent_col, linewidth=1.4)
            ax.add_patch(card_rect)

            # Header bar
            header_rect = FancyBboxPatch((cx, cy + h_card * 0.74), w_card, h_card * 0.26, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=accent_col, edgecolor="none")
            ax.add_patch(header_rect)

            # Header Title & Standard
            ax.text(cx + 0.015, cy + h_card * 0.88, title, fontsize=9.8, fontweight="bold", color="#FFFFFF")
            ax.text(cx + 0.015, cy + h_card * 0.78, standard, fontsize=8, color="#F8FAFC")

            # Body Text formatted with clean word wrapping
            wrapped_desc = textwrap.fill(desc, width=34)
            ax.text(cx + 0.015, cy + h_card * 0.65, wrapped_desc, fontsize=8.5, color="#1E293B", va="top", linespacing=1.35)

            # Status Badge at Bottom
            ax.text(cx + w_card - 0.015, cy + 0.045, status, fontsize=7.8, fontweight="bold", color="#FFFFFF", ha="right", va="center", bbox=dict(boxstyle="round,pad=0.3", facecolor=accent_col, edgecolor="none"))

            idx += 1

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return out_file


# ==============================================================================
# MASTER PORTFOLIO & PLAYBOOK GENERATOR
# ==============================================================================

def generate_complete_eucys_jury_portfolio(
    output_dir: Path | str = "results/eucys_jury_dossier",
    ega_pairs: tuple | None = None,
    arena_evaluation_artifacts: Sequence[Path | str] | None = None,
    confounder_evidence: Path | str | None = None,
    dual_sensor_evidence: Path | str | None = None,
    safety_trace: Path | str | None = None,
) -> EUCYSJuryPortfolio:
    """
    Generate the publication-grade PNG figures, metadata manifest, and interactive HTML dossier.

    Args:
        output_dir: Destination directory.
        ega_pairs: Optional ``(reference, predicted)`` values from a held-out
            evaluation. Evidence-dependent figures are skipped when their
            corresponding artifact is absent; the generator never fabricates
            a replacement score.
    """
    from iints.research.visualizer import (
        plot_foundation_arena_radar,
        plot_confounder_cosine_analysis,
        plot_glucofm_dual_stream_decomposition,
        plot_cgmacros_dualsensor_comparison,
        plot_fda_safety_mitigation_timeline,
    )

    out_dir = Path(output_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures: list[EUCYSFigureMetadata] = []

    # 1. Arena Radar Plot - comparable evidence only.
    p1 = (
        plot_foundation_arena_radar(
            fig_dir / "01_foundation_arena_radar.png",
            arena_evaluation_artifacts,
        )
        if arena_evaluation_artifacts
        else None
    )
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-01",
        category="AI & Foundation Models",
        title="Foundation-model benchmark",
        subtitle=(
            "Within-benchmark normalized comparison from supplied artifacts"
            if p1 else "Not generated - no comparable evaluation artifacts supplied"
        ),
        file_name="01_foundation_arena_radar.png",
        png_path=p1,
        description=(
            "Every value is loaded from a model evaluation using the same "
            "benchmark ID and a group-disjoint split. Radial scores are "
            "relative min-max values, not clinical percentages."
            if p1 else
            "Skipped. Supply at least two iints.foundation-arena.evaluation.v1 "
            "artifacts with three shared measured metrics."
        ),
        key_metrics={"Status": "evidence-backed" if p1 else "not generated"},
        scientific_citations=["Metwally et al. (arXiv:2605.30865v2, 2026)"],
    ))

    # 2. Confounder Cosine Analysis - pair-level evidence only.
    p2 = (
        plot_confounder_cosine_analysis(
            fig_dir / "02_confounder_cosine_analysis.png",
            confounder_evidence,
        )
        if confounder_evidence is not None
        else None
    )
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-02",
        category="AI & Foundation Models",
        title="Physiological confounder benchmark",
        subtitle=(
            "Pair-level embedding similarity under controlled simulated physiology"
            if p2 else "Not generated - no confounder evidence supplied"
        ),
        file_name="02_confounder_cosine_analysis.png",
        png_path=p2,
        description=(
            "Similarity summaries and points are computed directly from the "
            "supplied long-format evidence table."
            if p2 else
            "Skipped. Supply model_name, si_ratio, and "
            "embedding_cosine_similarity for every evaluated pair."
        ),
        key_metrics={"Status": "evidence-backed" if p2 else "not generated"},
        scientific_citations=["Metwally et al. (arXiv:2605.30865v2, 2026)"],
    ))

    # 3. GlucoFM implementation schematic. This is a method figure, not data.
    p3 = plot_glucofm_dual_stream_decomposition(fig_dir / "03_glucofm_dual_stream_decomposition.png")
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-03",
        category="AI & Foundation Models",
        title="IINTS GlucoFM v2 reproduction architecture",
        subtitle="Independent paper-aligned implementation; method schematic only",
        file_name="03_glucofm_dual_stream_decomposition.png",
        png_path=p3,
        description="A 288-position masked CGM grid is causally separated into state and event streams. Both use 24 one-hour patches; 64D stream tokens are fused into a 128D representation and processed by three transformer layers.",
        key_metrics={"Grid": "288 x 5 min", "Patches": "24 x 12 per stream", "Embedding": "128D"},
        scientific_citations=["Metwally et al. (arXiv:2605.30865v2, 2026)"],
    ))

    # 4. Dual-sensor comparison - paired evidence only.
    p4 = (
        plot_cgmacros_dualsensor_comparison(
            fig_dir / "04_paired_dual_sensor.png",
            dual_sensor_evidence,
        )
        if dual_sensor_evidence is not None
        else None
    )
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-04",
        category="Clinical & Sensor Dynamics",
        title="Paired dual-sensor comparison",
        subtitle=("Descriptive medians from supplied paired data" if p4 else "Not generated - no paired sensor evidence supplied"),
        file_name="04_paired_dual_sensor.png",
        png_path=p4,
        description=(
            "Computed from timestamped Dexcom/Libre pairs grouped by the "
            "cohort labels in the supplied artifact."
            if p4 else
            "Skipped. The dossier does not substitute generated sensor curves."
        ),
        key_metrics={"Status": "evidence-backed" if p4 else "not generated"},
        scientific_citations=["CGMacros source publication and dataset documentation"],
    ))

    # 5. Clarke Error Grid Analysis
    # Requires paired (reference, predicted) glucose values from an actual
    # held-out evaluation. The figure is skipped rather than drawn from
    # simulated scatter: a grid without real pairs is not an accuracy result.
    if ega_pairs is None:
        p5 = None
        ega_result = None
    else:
        from iints.analysis.error_grid import clarke_error_grid

        ega_result = clarke_error_grid(ega_pairs[0], ega_pairs[1])
        p5 = plot_clarke_error_grid(fig_dir / "05_clarke_error_grid_analysis.png",
                                    reference=ega_pairs[0], predicted=ega_pairs[1])
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-05",
        category="Clinical & Sensor Dynamics",
        title="Clarke Error Grid Analysis (EGA)",
        subtitle=(f"Clinical accuracy across {ega_result.n_pairs:,} paired in silico measurements"
                  if ega_result else "Not generated - no paired evaluation data supplied"),
        file_name="05_clarke_error_grid_analysis.png",
        png_path=p5,
        description=(
            "Predicted glucose versus reference values across Clarke clinical "
            f"accuracy zones: {ega_result.percentages['A']:.1f}% in Zone A, "
            f"{ega_result.hazardous_pct:.1f}% in the hazardous zones C/D/E."
            if ega_result else
            "Skipped: requires paired (reference, predicted) values from a "
            "held-out evaluation. Pass ega_pairs to generate this figure."
        ),
        key_metrics=({
            "Zone A (Accurate)": f"{ega_result.percentages['A']:.1f}%",
            "Zone B (Benign)": f"{ega_result.percentages['B']:.1f}%",
            "Zone C/D/E (Dangerous)": f"{ega_result.hazardous_pct:.1f}%",
            "Pairs": f"{ega_result.n_pairs:,}",
        } if ega_result else {"Status": "not generated - no paired data"}),
        scientific_citations=["Clarke et al. (Diabetes Care, 1987)"],
    ))

    # 6. No default trace is substituted for a measured run.
    p6 = None
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-06",
        category="Clinical & Sensor Dynamics",
        title="International Consensus Glycemic Targets (TIR / TBR / TAR)",
        subtitle="Not generated - no run-level glucose evidence supplied",
        file_name="06_glycemic_tir_clinical_distribution.png",
        png_path=p6,
        description="Skipped. TIR, TBR, TAR, mean glucose, and CV must be calculated from an identified results file rather than hardcoded demonstration values.",
        key_metrics={"Status": "not generated - run evidence required"},
        scientific_citations=["Battelino et al. (Diabetes Care, 2019)", "ADA Standards of Care (2026)"],
    ))

    # 7. Safety benchmark - explicit in-silico trace only.
    p7 = (
        plot_fda_safety_mitigation_timeline(
            fig_dir / "07_safety_benchmark_trace.png", safety_trace
        )
        if safety_trace is not None
        else None
    )
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-07",
        category="FDA & Device Safety",
        title="Deterministic safety benchmark",
        subtitle=("In-silico comparator and supervisor traces" if p7 else "Not generated - no safety trace supplied"),
        file_name="07_safety_benchmark_trace.png",
        png_path=p7,
        description=("The plot is explicitly labelled as simulation evidence and does not claim FDA validation or reproduction of a patient event." if p7 else "Skipped. Supply an in-silico benchmark trace; no recall outcome is fabricated."),
        key_metrics={"Status": "simulation evidence" if p7 else "not generated"},
        scientific_citations=["FDA MAUDE and recall records may inform scenario design; they do not validate simulator outcomes"],
    ))

    # 8. Wet-lab claims require wet-lab evidence; none is generated by default.
    p8 = None
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-08",
        category="Regenerative & Molecular",
        title="Stem-Cell Derived Beta-Islet GSIS & Biomarkers",
        subtitle="Not generated - no GSIS assay and proteomics evidence supplied",
        file_name="08_sc_islet_gsis_cpeptide_dynamics.png",
        png_path=p8,
        description="Skipped. The SDK can ingest proteomics evidence, but it cannot label generated curves as in-vitro validation.",
        key_metrics={"Status": "not generated - experimental evidence required"},
        scientific_citations=["Pagliuca et al. (Cell, 2014)", "Rezania et al. (Nature Biotech, 2014)"],
    ))

    # 9. A transplantation trajectory is not generated without a trace.
    p9 = None
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-09",
        category="Regenerative & Molecular",
        title="90-Day SC-Islet Graft Engraftment & Insulin Independence",
        subtitle="Not generated - no calibrated longitudinal graft trace supplied",
        file_name="09_regenerative_graft_longterm_survival.png",
        png_path=p9,
        description="Skipped. The dossier will not infer insulin independence, C-peptide, or graft viability from an uncalibrated demonstration curve.",
        key_metrics={"Status": "not generated - calibrated evidence required"},
        scientific_citations=["Shapiro et al. (NEJM, 2000)", "Vertex VX-880 Phase 1/2 Data"],
    ))

    # 10. Hardware latency must come from benchmark output.
    p10 = None
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-10",
        category="Hardware & Edge Computing",
        title="Deterministic Edge Latency Budget Breakdown",
        subtitle="Not generated - no hardware benchmark artifact supplied",
        file_name="10_edge_hardware_latency_budget.png",
        png_path=p10,
        description="Skipped. Platform latency, stage timing, power, and duty cycle must come from a traceable benchmark run.",
        key_metrics={"Status": "not generated - benchmark evidence required"},
        scientific_citations=["IEEE Trans Biomedical Circuits (2025)", "ASME V&V 40"],
    ))

    # 11. Quantum-Safe MDMP Security
    p11 = plot_quantum_safe_mdmp_security(fig_dir / "11_quantum_safe_mdmp_security.png")
    figures.append(EUCYSFigureMetadata(
        figure_id="FIG-11",
        category="Security & Regulatory",
        title="MDMP Packet Integrity & EU AI Act Governance",
        subtitle="Signing and provenance status against the High-Risk Annex III checklist",
        file_name="11_quantum_safe_mdmp_security.png",
        png_path=p11,
        description="Status table for packet signing (Ed25519, classical), authenticated encryption (ChaCha20-Poly1305), immutable RO-Crate provenance, and self-assessed EU AI Act Annex III readiness. Post-quantum signing is a roadmap item and is not implemented; the EU AI Act status is a self-assessment, not a conformity assessment by a notified body.",
        key_metrics={"Signing": "Ed25519 (classical)", "Post-quantum": "roadmap, not implemented", "EU AI Act Status": "self-assessed readiness"},
        scientific_citations=["NIST FIPS 204 (2024)", "EU AI Act Regulation (EU) 2024/1689"],
    ))

    # Save metadata manifest JSON
    manifest_json_path = out_dir / "eucys_portfolio_manifest.json"
    manifest_data = {
        "portfolio_title": "IINTS-AF EUCYS 2026 European Jury Scientific Portfolio & Dossier",
        "total_figures": len(figures),
        "figures": [f.to_dict() for f in figures],
    }
    manifest_json_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # Generate Standalone Interactive HTML Dossier
    index_html_path = out_dir / "index.html"
    html_content = _build_eucys_dossier_html(figures, manifest_data)
    index_html_path.write_text(html_content, encoding="utf-8")

    return EUCYSJuryPortfolio(
        output_dir=out_dir,
        figures=figures,
        index_html_path=index_html_path,
        manifest_json_path=manifest_json_path,
    )


def _build_eucys_dossier_html(figures: list[EUCYSFigureMetadata], manifest: dict[str, Any]) -> str:
    """Build a rich, responsive, interactive single-file HTML dossier with full gallery view."""

    cards_html = []
    categories = sorted(list(set(f.category for f in figures)))

    for f in figures:
        metrics_items = "".join(f'<div class="metric-pill"><span class="m-k">{k}:</span> <span class="m-v">{v}</span></div>' for k, v in f.key_metrics.items())
        citations_items = "".join(f'<li>{c}</li>' for c in f.scientific_citations)

        if f.png_path is None:
            # Skipped figure (e.g. Clarke EGA with no paired evaluation data
            # supplied) - render the metadata card without an image rather
            # than crashing or fabricating a placeholder plot.
            image_block = '<div class="img-container img-container--skipped">Not generated</div>'
        else:
            png_bytes = f.png_path.read_bytes()
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            image_block = f"""
            <div class="img-container" onclick="openModal('{f.figure_id}', '{b64}', '{f.title}', '{f.subtitle}')">
                <img src="data:image/png;base64,{b64}" alt="{f.title}" loading="lazy" />
                <div class="zoom-overlay">🔍 Click to Enlarge (300 DPI)</div>
            </div>
            """

        cards_html.append(f"""
        <div class="figure-card" data-category="{f.category}">
            <div class="card-header">
                <span class="card-id">{f.figure_id}</span>
                <span class="card-cat">{f.category}</span>
            </div>
            {image_block}
            <div class="card-body">
                <h3>{f.title}</h3>
                <p class="subtitle">{f.subtitle}</p>
                <div class="metrics-row">{metrics_items}</div>
                <p class="desc">{f.description}</p>
                <div class="citations-box">
                    <span class="cit-label">Scientific Basis:</span>
                    <ul>{citations_items}</ul>
                </div>
            </div>
        </div>
        """)

    filter_buttons = ['<button class="filter-btn active" onclick="filterCat(\'all\', this)">All Figures (' + str(len(figures)) + ')</button>']
    for cat in categories:
        count = sum(1 for f in figures if f.category == cat)
        filter_buttons.append(f'<button class="filter-btn" onclick="filterCat(\'{cat}\', this)">{cat} ({count})</button>')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IINTS-AF • EUCYS 2026 Jury Scientific Portfolio & Playbook</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --surface: #ffffff;
            --bg: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border: #e2e8f0;
            --success: #059669;
            --accent: #7c3aed;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            padding: 24px;
        }}
        .header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #2563eb 100%);
            color: white;
            padding: 36px 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(15,23,42,0.25);
        }}
        .header h1 {{ font-size: 28px; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.5px; }}
        .header p {{ font-size: 15px; opacity: 0.92; max-width: 900px; line-height: 1.6; }}
        .badge-row {{ margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
        .header-badge {{
            background: rgba(255,255,255,0.18);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.25);
        }}
        .filter-bar {{
            display: flex;
            gap: 10px;
            margin-bottom: 24px;
            overflow-x: auto;
            padding-bottom: 8px;
        }}
        .filter-btn {{
            background: white;
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 24px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }}
        .filter-btn:hover {{ border-color: var(--primary); color: var(--primary); }}
        .filter-btn.active {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            box-shadow: 0 2px 8px rgba(37,99,235,0.3);
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
            gap: 24px;
        }}
        @media (max-width: 768px) {{
            .gallery-grid {{ grid-template-columns: 1fr; }}
        }}
        .figure-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .figure-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        }}
        .card-header {{
            padding: 12px 18px;
            background: #f8fafc;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-id {{
            font-weight: 800;
            font-size: 13px;
            color: var(--primary);
            background: #eff6ff;
            padding: 3px 10px;
            border-radius: 6px;
            border: 1px solid #bfdbfe;
        }}
        .card-cat {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .img-container {{
            position: relative;
            background: #ffffff;
            cursor: pointer;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 8px;
        }}
        .img-container img {{
            width: 100%;
            height: auto;
            max-height: 380px;
            object-fit: contain;
            display: block;
            border-radius: 6px;
        }}
        .zoom-overlay {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(15,23,42,0.88);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.2s ease;
            pointer-events: none;
            backdrop-filter: blur(4px);
        }}
        .img-container:hover .zoom-overlay {{ opacity: 1; }}
        .card-body {{
            padding: 18px;
            display: flex;
            flex-direction: column;
            flex: 1;
        }}
        .card-body h3 {{ font-size: 17px; font-weight: 700; margin-bottom: 4px; color: var(--text-main); }}
        .subtitle {{ font-size: 13px; color: var(--text-muted); margin-bottom: 12px; }}
        .metrics-row {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }}
        .metric-pill {{
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 12px;
        }}
        .m-k {{ font-weight: 600; color: var(--text-muted); }}
        .m-v {{ font-weight: 800; color: var(--primary); }}
        .desc {{ font-size: 13.5px; color: #334155; line-height: 1.55; margin-bottom: 14px; flex: 1; }}
        .citations-box {{
            background: #f8fafc;
            border-left: 3px solid var(--primary);
            padding: 8px 12px;
            border-radius: 0 6px 6px 0;
            font-size: 11.5px;
            color: var(--text-muted);
        }}
        .cit-label {{ font-weight: 700; display: block; margin-bottom: 2px; color: var(--text-main); }}
        .citations-box ul {{ padding-left: 16px; margin: 0; }}

        /* MODAL */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15,23,42,0.88);
            backdrop-filter: blur(8px);
            align-items: center;
            justify-content: center;
            padding: 24px;
        }}
        .modal.active {{ display: flex; }}
        .modal-content {{
            background: white;
            border-radius: 16px;
            max-width: 1250px;
            width: 100%;
            max-height: 94vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        }}
        .modal-header {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .modal-title h2 {{ font-size: 18px; font-weight: 700; color: var(--text-main); }}
        .modal-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: var(--text-muted);
            padding: 4px 8px;
        }}
        .modal-body {{ padding: 24px; text-align: center; }}
        .modal-body img {{ max-width: 100%; height: auto; max-height: 72vh; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>IINTS-AF • EUCYS 2026 European Jury Scientific Portfolio & Playbook</h1>
        <p>Interactive research dossier for EUCYS 2026. Method diagrams are shown separately from measured evidence; missing evidence remains visibly ungenerated rather than being replaced by demonstration values.</p>
        <div class="badge-row">
            <div class="header-badge">★ EUCYS 2026 Finalist Project</div>
            <div class="header-badge">Independent GlucoFM v2 reproduction</div>
            <div class="header-badge">Evidence-only result figures</div>
            <div class="header-badge">Group-disjoint evaluation contract</div>
            <div class="header-badge">Research and education only</div>
        </div>
    </div>

    <div class="filter-bar">
        {"".join(filter_buttons)}
    </div>

    <div class="gallery-grid" id="galleryGrid">
        {"".join(cards_html)}
    </div>

    <div class="modal" id="imageModal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <div class="modal-header">
                <div class="modal-title">
                    <h2 id="modalTitle">Figure</h2>
                    <p id="modalSubtitle" style="font-size: 13px; color: var(--text-muted);"></p>
                </div>
                <button class="modal-close" onclick="closeModal(event)">✕</button>
            </div>
            <div class="modal-body">
                <img id="modalImg" src="" alt="Full view" />
            </div>
        </div>
    </div>

    <script>
        function filterCat(cat, btn) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cards = document.querySelectorAll('.figure-card');
            cards.forEach(card => {{
                if (cat === 'all' || card.getAttribute('data-category') === cat) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        function openModal(id, b64, title, subtitle) {{
            document.getElementById('modalTitle').innerText = id + ": " + title;
            document.getElementById('modalSubtitle').innerText = subtitle;
            document.getElementById('modalImg').src = "data:image/png;base64," + b64;
            document.getElementById('imageModal').classList.add('active');
        }}

        function closeModal(e) {{
            document.getElementById('imageModal').classList.remove('active');
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});
    </script>
</body>
</html>
"""


__all__ = [
    "EUCYSFigureMetadata",
    "EUCYSJuryPortfolio",
    "plot_clarke_error_grid",
    "plot_glycemic_tir_distribution",
    "plot_sc_islet_gsis_dynamics",
    "plot_regenerative_graft_survival",
    "plot_edge_hardware_latency_budget",
    "plot_quantum_safe_mdmp_security",
    "generate_complete_eucys_jury_portfolio",
]
