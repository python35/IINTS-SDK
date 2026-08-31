from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from iints.research.eucys_playbook_generator import (
    EUCYSFigureMetadata,
    EUCYSJuryPortfolio,
    plot_clarke_error_grid,
    plot_glycemic_tir_distribution,
    plot_sc_islet_gsis_dynamics,
    plot_regenerative_graft_survival,
    plot_edge_hardware_latency_budget,
    plot_quantum_safe_mdmp_security,
    generate_complete_eucys_jury_portfolio,
)


def test_plot_clarke_error_grid():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "clarke.png"
        res = plot_clarke_error_grid(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_glycemic_tir_distribution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "tir.png"
        res = plot_glycemic_tir_distribution(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_sc_islet_gsis_dynamics():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "gsis.png"
        res = plot_sc_islet_gsis_dynamics(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_regenerative_graft_survival():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "graft.png"
        res = plot_regenerative_graft_survival(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_edge_hardware_latency_budget():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "edge.png"
        res = plot_edge_hardware_latency_budget(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_plot_quantum_safe_mdmp_security():
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = Path(tmp_dir) / "security.png"
        res = plot_quantum_safe_mdmp_security(out)
        assert res.exists()
        assert res.stat().st_size > 1000


def test_generate_complete_eucys_jury_portfolio():
    with tempfile.TemporaryDirectory() as tmp_dir:
        port = generate_complete_eucys_jury_portfolio(output_dir=tmp_dir)
        assert isinstance(port, EUCYSJuryPortfolio)
        assert len(port.figures) >= 11
        assert port.index_html_path.exists()
        assert port.manifest_json_path.exists()

        # Check manifest content
        manifest_data = json.loads(port.manifest_json_path.read_text())
        assert manifest_data["total_figures"] == len(port.figures)
        assert len(manifest_data["figures"]) >= 11

        # Check HTML content
        html = port.index_html_path.read_text()
        assert "EUCYS 2026" in html
        assert "Clarke Error Grid Analysis" in html
        assert "data:image/png;base64," in html
