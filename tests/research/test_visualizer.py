from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch not installed")

from iints.research.visualizer import (
    generate_all_scientific_visualizations,
    plot_cgmacros_dualsensor_comparison,
    plot_confounder_cosine_analysis,
    plot_fda_safety_mitigation_timeline,
    plot_foundation_arena_radar,
    plot_glucofm_dual_stream_decomposition,
)


def _arena_artifact(path: Path, name: str, offset: float) -> Path:
    payload = {
        "schema_version": "iints.foundation-arena.evaluation.v1",
        "model": {
            "name": name,
            "architecture": "test encoder",
            "latent_dimension": 128,
            "implementation_kind": "test",
            "checkpoint_sha256": ("a" if name == "model-a" else "b") * 64,
        },
        "evaluation": {
            "benchmark_id": "unit-test-v1",
            "task": "forecast",
            "cohort_id": "fixture",
            "split_id": "held-out-subjects",
            "split_strategy": "subject-disjoint",
            "group_disjoint": True,
            "n_groups": 4,
            "n_samples": 40,
            "seed": 42,
        },
        "metrics": {
            "mae_mgdl": {"value": 20.0 + offset, "unit": "mg/dL", "direction": "lower"},
            "hypo_auprc": {"value": 0.50 - offset / 100.0, "unit": "", "direction": "higher"},
            "coverage": {"value": 0.90 - offset / 100.0, "unit": "fraction", "direction": "higher"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evidence_files(tmp_path: Path) -> tuple[list[Path], Path, Path, Path]:
    arena = [
        _arena_artifact(tmp_path / "model_a.json", "model-a", 0.0),
        _arena_artifact(tmp_path / "model_b.json", "model-b", 2.0),
    ]
    confounder = tmp_path / "confounder.csv"
    pd.DataFrame(
        {
            "model_name": ["model-a", "model-a", "model-b", "model-b"],
            "si_ratio": [2.0, 3.0, 2.0, 3.0],
            "embedding_cosine_similarity": [0.50, 0.55, 0.70, 0.72],
        }
    ).to_csv(confounder, index=False)

    dual_sensor = tmp_path / "dual_sensor.csv"
    timestamps = pd.date_range("2026-01-01", periods=12, freq="2h")
    pd.DataFrame(
        {
            "timestamp": list(timestamps) * 2,
            "dexcom_mgdl": np.tile(np.linspace(90, 145, 12), 2),
            "libre_mgdl": np.tile(np.linspace(88, 141, 12), 2),
            "cohort": ["T1D"] * 12 + ["reference"] * 12,
        }
    ).to_csv(dual_sensor, index=False)

    safety = tmp_path / "safety.csv"
    minute = np.arange(0, 61, 5)
    pd.DataFrame(
        {
            "time_minutes": minute,
            "unsupervised_glucose_mgdl": 110.0 - minute * 0.9,
            "supervised_glucose_mgdl": 110.0 - np.minimum(minute, 20) * 0.4,
        }
    ).to_csv(safety, index=False)
    return arena, confounder, dual_sensor, safety


def test_evidence_plots_fail_closed_without_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two measured"):
        plot_foundation_arena_radar(tmp_path / "radar.png")
    with pytest.raises(ValueError, match="requires an evidence"):
        plot_confounder_cosine_analysis(tmp_path / "confounder.png")
    with pytest.raises(ValueError, match="requires a paired"):
        plot_cgmacros_dualsensor_comparison(tmp_path / "dual.png")
    with pytest.raises(ValueError, match="requires an in-silico"):
        plot_fda_safety_mitigation_timeline(tmp_path / "safety.png")


def test_evidence_backed_plots(tmp_path: Path) -> None:
    arena, confounder, dual_sensor, safety = _evidence_files(tmp_path)
    outputs = [
        plot_foundation_arena_radar(tmp_path / "radar.png", arena),
        plot_confounder_cosine_analysis(tmp_path / "confounder.png", confounder),
        plot_cgmacros_dualsensor_comparison(tmp_path / "dual.png", dual_sensor),
        plot_fda_safety_mitigation_timeline(tmp_path / "safety.png", safety),
    ]
    assert all(output.exists() and output.stat().st_size > 1000 for output in outputs)


def test_plot_glucofm_method_schematic(tmp_path: Path) -> None:
    output = plot_glucofm_dual_stream_decomposition(tmp_path / "glucofm.png")
    assert output.exists()
    assert output.stat().st_size > 1000


def test_generate_suite_without_evidence_marks_results_missing(tmp_path: Path) -> None:
    artifacts = generate_all_scientific_visualizations(tmp_path / "method_only")
    assert artifacts.glucofm_decomposition_png.exists()
    assert artifacts.arena_radar_png is None
    assert artifacts.confounder_cosine_png is None
    assert artifacts.cgmacros_dualsensor_png is None
    assert artifacts.fda_safety_timeline_png is None
    content = artifacts.interactive_dashboard_html.read_text(encoding="utf-8")
    assert "never replaces it with synthetic scores" in content
    assert content.count("Not generated") == 4


def test_generate_suite_with_evidence(tmp_path: Path) -> None:
    arena, confounder, dual_sensor, safety = _evidence_files(tmp_path)
    artifacts = generate_all_scientific_visualizations(
        tmp_path / "full_suite",
        arena_evaluation_artifacts=arena,
        confounder_evidence=confounder,
        cgmacros_evidence=dual_sensor,
        safety_trace=safety,
    )
    generated = [
        artifacts.arena_radar_png,
        artifacts.confounder_cosine_png,
        artifacts.glucofm_decomposition_png,
        artifacts.cgmacros_dualsensor_png,
        artifacts.fda_safety_timeline_png,
        artifacts.interactive_dashboard_html,
    ]
    assert all(path is not None and path.exists() for path in generated)
