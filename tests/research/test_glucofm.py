from __future__ import annotations

import numpy as np
import pytest
import torch

from iints.research.glucofm import (
    GlucoFMConfig,
    GlucoFMDualStreamEncoder,
    GlucoFMDownstreamProbes,
    build_glucofm_foundation_model,
    embed_cgm_with_glucofm,
)


def test_glucofm_signal_decomposition():
    encoder = GlucoFMDualStreamEncoder()
    # Create test signal with slow baseline (100) + fast meal spike (50)
    cgm = torch.ones(2, 288) * 100.0
    cgm[:, 120:150] += 50.0

    state, event = encoder.decompose_signal(cgm)
    assert state.shape == (2, 288)
    assert event.shape == (2, 288)
    # Event should contain the fast spike
    assert torch.max(event) > 20.0


def test_glucofm_forward_pass_and_embedding_dimension():
    config = GlucoFMConfig(stream_dimension=64, fused_dimension=128, encoder_layers=2, attention_heads=4)
    encoder = GlucoFMDualStreamEncoder(config)

    cgm_input = torch.randn(3, 288) * 30.0 + 120.0
    mask = torch.ones(3, 288)
    # Inject missingness into last 20 steps
    mask[:, -20:] = 0.0

    z_fused = encoder(cgm_input, mask)
    assert z_fused.shape == (3, 128)
    assert not torch.isnan(z_fused).any()


def test_glucofm_downstream_probes():
    encoder, probes = build_glucofm_foundation_model()
    cgm_input = torch.randn(2, 288) * 25.0 + 110.0
    z_fused = encoder(cgm_input)

    # 5 macronutrients: [carbs, protein, fat, fiber, kcal]
    meal_macros = torch.tensor([[45.0, 15.0, 10.0, 5.0, 350.0], [70.0, 25.0, 20.0, 8.0, 550.0]])
    outputs = probes(z_fused, meal_macros)

    assert "homa_ir" in outputs
    assert outputs["homa_ir"].shape == (2, 1)
    assert "diabetes_logits" in outputs
    assert outputs["diabetes_logits"].shape == (2, 3)
    assert "ppgr_forecast_2h" in outputs
    assert outputs["ppgr_forecast_2h"].shape == (2, 24)


def test_embed_cgm_with_glucofm():
    # Architecture smoke test only (see embed_cgm_with_glucofm's docstring):
    # untrained weights cannot produce research-grade embeddings, so this
    # must go through the explicit encoder + allow_untrained=True path
    # rather than the checkpoint-required default.
    raw_series = [100.0 + 10.0 * np.sin(i / 10.0) for i in range(288)]
    encoder = GlucoFMDualStreamEncoder()
    embedding = embed_cgm_with_glucofm(raw_series, encoder=encoder, allow_untrained=True)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (encoder.config.fused_dimension,)
    assert not np.isnan(embedding).any()
