from __future__ import annotations

from pathlib import Path
import pytest

from iints.research.visualizer import (
    generate_all_scientific_visualizations,
    plot_foundation_arena_radar,
    plot_confounder_cosine_analysis,
    plot_glucofm_dual_stream_decomposition,
    plot_cgmacros_dualsensor_comparison,
    plot_fda_safety_mitigation_timeline,
    generate_interactive_dashboard_html,
)


def test_plot_foundation_arena_radar(tmp_path: Path):
    out = plot_foundation_arena_radar(tmp_path / "radar.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_confounder_cosine_analysis(tmp_path: Path):
    out = plot_confounder_cosine_analysis(tmp_path / "confounder.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_glucofm_dual_stream_decomposition(tmp_path: Path):
    out = plot_glucofm_dual_stream_decomposition(tmp_path / "glucofm.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_cgmacros_dualsensor_comparison(tmp_path: Path):
    out = plot_cgmacros_dualsensor_comparison(tmp_path / "dualsensor.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_fda_safety_mitigation_timeline(tmp_path: Path):
    out = plot_fda_safety_mitigation_timeline(tmp_path / "safety.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_generate_all_scientific_visualizations(tmp_path: Path):
    artifacts = generate_all_scientific_visualizations(tmp_path / "viz_suite")
    assert artifacts.output_dir.exists()
    assert artifacts.arena_radar_png.exists()
    assert artifacts.confounder_cosine_png.exists()
    assert artifacts.glucofm_decomposition_png.exists()
    assert artifacts.cgmacros_dualsensor_png.exists()
    assert artifacts.fda_safety_timeline_png.exists()
    assert artifacts.interactive_dashboard_html.exists()

    content = artifacts.interactive_dashboard_html.read_text(encoding="utf-8")
    assert "IINTS-AF" in content and "Scientific Visualization Suite" in content
    assert "data:image/png;base64," in content
