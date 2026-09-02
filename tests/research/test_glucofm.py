from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from iints.research.glucofm import (
    CausalMaskAwareGaussianFilter,
    GlucoFMConfig,
    GlucoFMCheckpointMetadata,
    GlucoFMDualStreamEncoder,
    GlucoFMDownstreamProbes,
    GlucoFMPretrainer,
    align_cgm_window,
    build_glucofm_foundation_model,
    embed_cgm_with_glucofm,
    embed_cgm_with_glucofm_result,
    load_glucofm_checkpoint,
    save_glucofm_checkpoint,
)
from iints.research.glucofm_training import pretrain_glucofm


def _small_config() -> GlucoFMConfig:
    return GlucoFMConfig(
        stream_dimension=16,
        fused_dimension=32,
        attention_heads=4,
        encoder_layers=1,
        feedforward_dimension=64,
        predictor_layers=1,
        state_waveform_dimension=12,
        state_difference_dimension=4,
        state_statistics_dimension=8,
        event_waveform_dimension=8,
        event_roc_dimension=8,
        event_statistics_dimension=8,
        dropout=0.0,
    )


def test_glucofm_paper_aligned_defaults() -> None:
    config = GlucoFMConfig()
    assert config.sequence_length == 288
    assert config.sampling_interval_minutes == 5
    assert config.patch_size == 12
    assert config.patch_count == 24
    assert config.stream_dimension == 64
    assert config.fused_dimension == 128
    assert config.encoder_layers == 3
    assert config.attention_heads == 4
    assert config.feedforward_dimension == 256


def test_align_window_preserves_missingness_and_averages_duplicates() -> None:
    timestamps = pd.to_datetime(
        ["2026-01-01 09:07", "2026-01-01 09:09", "2026-01-01 09:17"]
    )
    window = align_cgm_window([100.0, 120.0, 150.0], timestamps)
    assert window.values[0] == pytest.approx(110.0)
    assert window.values[2] == pytest.approx(150.0)
    assert window.observation_mask.sum() == 2
    assert window.observation_mask[1] == 0
    assert window.values[1] == 0
    assert window.duplicate_measurements_averaged == 1
    assert window.absolute_grid_indices[0] == (9 * 60 + 7) // 5


def test_state_filter_is_causal() -> None:
    layer = CausalMaskAwareGaussianFilter(max_lag=12)
    first = torch.linspace(80.0, 140.0, 288).unsqueeze(0)
    second = first.clone()
    second[:, 180:] += 200.0
    mask = torch.ones_like(first)
    state_a = layer(first, mask)
    state_b = layer(second, mask)
    assert torch.equal(state_a[:, :180], state_b[:, :180])


def test_pretrainer_has_frozen_target_and_finite_backward() -> None:
    model = GlucoFMPretrainer(config=_small_config())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    values = torch.linspace(85.0, 180.0, 288).repeat(2, 1)
    mask = torch.ones_like(values)
    result = model(values, mask)
    assert torch.isfinite(result.loss)
    assert result.masked_patches.shape == (2, 24)
    result.loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.online_encoder.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


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


def test_checkpoint_roundtrip_is_deterministic_and_traceable(tmp_path: Path) -> None:
    torch.manual_seed(7)
    encoder = GlucoFMDualStreamEncoder(_small_config())
    metadata = GlucoFMCheckpointMetadata(
        trained=True,
        training_epochs=2,
        dataset_sha256="c" * 64,
        dataset_description="unit-test daily windows",
        code_revision="test-revision",
    )
    checkpoint = save_glucofm_checkpoint(tmp_path / "encoder.pt", encoder, metadata)
    loaded, loaded_metadata = load_glucofm_checkpoint(checkpoint)
    assert loaded_metadata == metadata
    trace = np.linspace(80.0, 180.0, 288, dtype=np.float32)
    first = embed_cgm_with_glucofm_result(trace, checkpoint=checkpoint)
    second = embed_cgm_with_glucofm_result(trace, checkpoint=checkpoint)
    np.testing.assert_array_equal(first.embedding, second.embedding)
    assert first.embedding.shape == (32,)
    assert len(first.checkpoint_sha256) == 64
    assert loaded.checkpoint_metadata == metadata


def test_untrained_embedding_is_rejected_by_default() -> None:
    trace = np.linspace(90.0, 140.0, 288, dtype=np.float32)
    with pytest.raises(ValueError, match="Untrained random"):
        embed_cgm_with_glucofm(trace, encoder=GlucoFMDualStreamEncoder(_small_config()))


def test_one_epoch_subject_disjoint_pretraining_smoke(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for subject_index in range(3):
        timestamps = pd.date_range("2026-01-01", periods=576, freq="5min")
        glucose = (
            110.0
            + 8.0 * np.sin(np.arange(576) / 30.0)
            + float(subject_index * 5)
        )
        for timestamp, value in zip(timestamps, glucose):
            rows.append(
                {
                    "subject_id": f"subject-{subject_index}",
                    "timestamp": timestamp,
                    "glucose_mgdl": value,
                }
            )
    source = tmp_path / "training.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    result = pretrain_glucofm(
        source,
        tmp_path / "trained",
        glucose_column="glucose_mgdl",
        timestamp_column="timestamp",
        subject_column="subject_id",
        epochs=1,
        batch_size=2,
        device="cpu",
        config=_small_config(),
    )
    assert result.completed_epochs == 1
    assert result.train_subjects + result.validation_subjects == 3
    assert result.train_subjects >= 1
    assert result.validation_subjects >= 1
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["split_kind"] == "subject-disjoint"
    assert report["official_checkpoint"] is False
    assert report["checkpoint_sha256"]
    loaded, metadata = load_glucofm_checkpoint(result.checkpoint_path)
    assert loaded.config.fused_dimension == 32
    assert metadata.trained is True
