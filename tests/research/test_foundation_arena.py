from __future__ import annotations

from pathlib import Path
import pytest

pytest.importorskip("torch", reason="PyTorch not installed")

from iints.research.foundation_arena import run_foundation_model_arena


def test_foundation_model_arena_execution(tmp_path: Path):
    out_dir = tmp_path / "arena_test"
    report = run_foundation_model_arena(output_dir=out_dir, n_benchmark_trials=10)

    assert report.total_models_evaluated >= 4
    model_names = [m.model_name for m in report.models]
    assert any("GlucoFM" in name for name in model_names)
    assert any("CGM-JEPA" in name for name in model_names)
    assert any("GluFormer" in name for name in model_names)
    assert any("IINTS-AF" in name for name in model_names)

    # IINTS-AF Digital Twin must have 0% confounder vulnerability
    twin_metric = next(m for m in report.models if "IINTS-AF" in m.model_name)
    assert twin_metric.confounder_vulnerability_pct == 0.0

    # Observational models must have high confounder vulnerability
    glucofm_metric = next(m for m in report.models if "GlucoFM" in m.model_name)
    assert glucofm_metric.confounder_vulnerability_pct > 90.0

    # Ensure artifacts were created
    assert (out_dir / "foundation_arena_comparison.csv").is_file()
    assert (out_dir / "foundation_arena_summary.json").is_file()
    assert (out_dir / "FOUNDATION_MODEL_ARENA_REPORT.md").is_file()
