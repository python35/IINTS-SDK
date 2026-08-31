from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.research.cgm_jepa import (
    CGMJEPAConfig,
    CGMJEPAEncoder,
    extract_cgm_jepa_embeddings,
    load_cgm_jepa_model,
)
from iints.research.cgm_jepa_bridge import (
    bridge_simulation_to_jepa,
    prepare_cgm_jepa_window,
)
from iints.research.cgm_jepa_experiment import (
    add_sensor_noise_and_dropouts,
    run_cgm_jepa_parameter_experiment,
    simulate_physiological_cgm_24h,
)

runner = CliRunner()


def test_cgm_jepa_encoder_forward():
    model = CGMJEPAEncoder(CGMJEPAConfig())
    dummy_input = torch.randn(4, 288)  # Batch of 4 24h windows
    emb = model(dummy_input, pool="mean")

    assert emb.shape == (4, 96)
    assert not torch.isnan(emb).any()


def test_extract_cgm_jepa_embeddings():
    # Single trace
    trace_1d = np.random.uniform(70, 180, size=288)
    emb_1d = extract_cgm_jepa_embeddings(trace_1d)
    assert emb_1d.shape == (96,)

    # Batch of traces
    trace_2d = np.random.uniform(70, 180, size=(5, 288))
    emb_2d = extract_cgm_jepa_embeddings(trace_2d)
    assert emb_2d.shape == (5, 96)


def test_prepare_cgm_jepa_window():
    # Irregular 100-step simulation
    df = pd.DataFrame({
        "time_minutes": np.linspace(0, 1440, 100),
        "glucose_actual_mgdl": np.random.uniform(80, 150, 100),
    })
    window = prepare_cgm_jepa_window(df)
    assert len(window) == 288
    assert np.all(np.isfinite(window))


def test_bridge_simulation_to_jepa(tmp_path: Path):
    sim_dir = tmp_path / "mock_sim_run"
    sim_dir.mkdir(parents=True)
    steps_df = pd.DataFrame({
        "time_minutes": np.arange(288) * 5.0,
        "glucose_actual_mgdl": 110.0 + 30.0 * np.sin(np.arange(288) / 20.0),
    })
    steps_df.to_csv(sim_dir / "results.csv", index=False)

    out_dir = tmp_path / "jepa_embedding_out"
    res = bridge_simulation_to_jepa(sim_dir, output_dir=out_dir)

    assert res.embedding_dim == 96
    assert len(res.embedding_vector) == 96
    assert (out_dir / "cgm_jepa_embedding.json").is_file()
    assert (out_dir / "cgm_jepa_embedding.csv").is_file()


def test_simulate_physiological_cgm_and_noise():
    trace_sensitive = simulate_physiological_cgm_24h(insulin_sensitivity_factor=1.8, seed=42)
    trace_resistant = simulate_physiological_cgm_24h(insulin_sensitivity_factor=0.5, seed=42)

    assert len(trace_sensitive) == 288
    assert len(trace_resistant) == 288
    # Insulin resistant patient should have higher mean glucose and higher post-meal peaks
    assert np.mean(trace_resistant) > np.mean(trace_sensitive)

    noisy = add_sensor_noise_and_dropouts(trace_sensitive, noise_std_mgdl=10.0, dropout_fraction=0.05)
    assert len(noisy) == 288
    assert np.isnan(noisy).any()  # contains dropouts


def test_run_cgm_jepa_parameter_experiment(tmp_path: Path):
    out_dir = tmp_path / "cgm_jepa_study_test"
    res = run_cgm_jepa_parameter_experiment(
        output_dir=out_dir,
        num_simulations=20,
        sweep_param="insulin_sensitivity",
        param_range=(0.5, 1.8),
    )

    assert res.num_simulations == 20
    assert res.linear_probe_r2 > 0.80
    assert res.spearman_monotonicity_rho > 0.80
    assert res.noise_robustness_cosine_sim > 0.85
    assert (out_dir / "cgm_jepa_embeddings_100runs.csv").is_file()
    assert (out_dir / "cgm_jepa_experiment_summary.json").is_file()
    assert (out_dir / "CGM_JEPA_SCIENTIFIC_REPORT.md").is_file()


def test_cli_cgm_jepa_commands(tmp_path: Path):
    # Test embed CLI
    sim_csv = tmp_path / "sim.csv"
    pd.DataFrame({
        "time_minutes": np.arange(288) * 5.0,
        "glucose_actual_mgdl": 120.0 + 15.0 * np.sin(np.arange(288) / 15.0),
    }).to_csv(sim_csv, index=False)

    emb_out = tmp_path / "cli_emb_out"
    res_embed = runner.invoke(
        app,
        [
            "research",
            "cgm-jepa-embed",
            "--input",
            str(sim_csv),
            "--output-dir",
            str(emb_out),
        ],
    )
    assert res_embed.exit_code == 0
    assert "CGM-JEPA Embedding Generated Successfully" in res_embed.output
    assert (emb_out / "cgm_jepa_embedding.json").is_file()

    # Test experiment CLI
    exp_out = tmp_path / "cli_exp_out"
    res_exp = runner.invoke(
        app,
        [
            "research",
            "cgm-jepa-experiment",
            "--output-dir",
            str(exp_out),
            "--n-simulations",
            "10",
        ],
    )
    assert res_exp.exit_code == 0
    assert "Experiment Completed Successfully" in res_exp.output
    assert (exp_out / "CGM_JEPA_SCIENTIFIC_REPORT.md").is_file()

    # Test confounder CLI
    conf_out = tmp_path / "cli_conf_out"
    res_conf = runner.invoke(
        app,
        [
            "research",
            "cgm-jepa-confounder",
            "--output-dir",
            str(conf_out),
            "--num-pairs",
            "6",
        ],
    )
    assert res_conf.exit_code == 0
    assert "Physiological Confounder Vulnerability Summary" in res_conf.output
    assert (conf_out / "PHYSIOLOGICAL_CONFOUNDER_REPORT.md").is_file()
    assert (conf_out / "confounder_benchmark_pairs.csv").is_file()
