from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


GLUCOFM_MODEL_FAMILY = "iints-glucofm-v2-reproduction"
GLUCOFM_CHECKPOINT_FORMAT = "iints.glucofm.encoder.v1"
GLUCOFM_PAPER_REVISION = "arXiv:2605.30865v2"
GLUCOFM_PAPER_URL = "https://arxiv.org/abs/2605.30865v2"
GLUCOFM_IMPLEMENTATION_KIND = "independent-paper-reproduction"


@dataclass(frozen=True)
class GlucoFMConfig:
    """Paper-aligned defaults for the GlucoFM v2 architecture.

    This is an independent implementation from the published method. It is not
    an official Google checkpoint or an exact source-code reproduction.
    """

    sequence_length: int = 288
    sampling_interval_minutes: int = 5
    patch_size: int = 12
    stream_dimension: int = 64
    fused_dimension: int = 128
    attention_heads: int = 4
    encoder_layers: int = 3
    feedforward_dimension: int = 256
    predictor_layers: int = 1
    dropout: float = 0.1
    state_waveform_dimension: int = 64
    state_difference_dimension: int = 16
    state_statistics_dimension: int = 48
    event_waveform_dimension: int = 48
    event_roc_dimension: int = 48
    event_statistics_dimension: int = 32
    gaussian_sigma_min: float = 2.0
    gaussian_sigma_max: float = 12.0
    gaussian_sigma_initial: float = 6.0
    gaussian_max_lag: int = 36
    roc_max_backoff: int = 9
    mask_ratio_min: float = 0.50
    mask_ratio_max: float = 0.60
    ema_momentum_initial: float = 0.997
    mcr_weight: float = 1.0
    temporal_dynamics_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.sequence_length != 288 or self.sampling_interval_minutes != 5:
            raise ValueError("GlucoFM v2 uses a fixed 288-position, 5-minute daily grid.")
        if self.patch_size != 12:
            raise ValueError("GlucoFM v2 uses 12-position (one-hour) patches in both streams.")
        if self.sequence_length % self.patch_size:
            raise ValueError("sequence_length must be divisible by patch_size")
        if self.fused_dimension % self.attention_heads:
            raise ValueError("fused_dimension must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.mask_ratio_min <= self.mask_ratio_max < 1.0:
            raise ValueError("mask ratios must satisfy 0 < min <= max < 1")
        if not self.gaussian_sigma_min < self.gaussian_sigma_initial < self.gaussian_sigma_max:
            raise ValueError("initial Gaussian sigma must lie inside its constrained range")

    @property
    def patch_count(self) -> int:
        return self.sequence_length // self.patch_size

    @property
    def fused_dim(self) -> int:
        """Compatibility name for older IINTS callers."""

        return self.fused_dimension

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignedCGMWindow:
    """One 24-hour chronological CGM grid with its physical observation mask."""

    values: np.ndarray
    observation_mask: np.ndarray
    absolute_grid_indices: np.ndarray
    start_timestamp: str | None
    binning: str
    duplicate_measurements_averaged: int = 0

    @property
    def observed_count(self) -> int:
        return int(np.sum(self.observation_mask))

    @property
    def coverage(self) -> float:
        return float(np.mean(self.observation_mask))


@dataclass(frozen=True)
class GlucoFMCheckpointMetadata:
    """Provenance stored alongside every reusable encoder checkpoint."""

    trained: bool
    training_epochs: int
    dataset_sha256: str
    dataset_description: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_family: str = GLUCOFM_MODEL_FAMILY
    implementation_kind: str = GLUCOFM_IMPLEMENTATION_KIND
    paper_revision: str = GLUCOFM_PAPER_REVISION
    code_revision: str = "unknown"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GlucoFMCheckpointMetadata:
        return cls(
            trained=bool(payload.get("trained", False)),
            training_epochs=int(payload.get("training_epochs", 0)),
            dataset_sha256=str(payload.get("dataset_sha256", "unknown")),
            dataset_description=str(payload.get("dataset_description", "unknown")),
            created_at_utc=str(payload.get("created_at_utc", "unknown")),
            model_family=str(payload.get("model_family", GLUCOFM_MODEL_FAMILY)),
            implementation_kind=str(
                payload.get("implementation_kind", GLUCOFM_IMPLEMENTATION_KIND)
            ),
            paper_revision=str(payload.get("paper_revision", GLUCOFM_PAPER_REVISION)),
            code_revision=str(payload.get("code_revision", "unknown")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class GlucoFMEmbeddingResult:
    embedding: np.ndarray
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_metadata: GlucoFMCheckpointMetadata
    input_observed_count: int
    input_coverage: float
    input_start_timestamp: str | None

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.checkpoint_metadata.model_family,
            "implementation_kind": self.checkpoint_metadata.implementation_kind,
            "paper_revision": self.checkpoint_metadata.paper_revision,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_metadata": self.checkpoint_metadata.to_dict(),
            "input_observed_count": self.input_observed_count,
            "input_coverage": self.input_coverage,
            "input_start_timestamp": self.input_start_timestamp,
            "embedding_dimension": int(self.embedding.shape[0]),
            "research_only": True,
            "medical_device": False,
        }


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _start_minute_of_day(timestamp: pd.Timestamp) -> int:
    return int(timestamp.hour * 60 + timestamp.minute)


def align_cgm_window(
    glucose_values: Sequence[float] | np.ndarray | pd.Series,
    timestamps: Sequence[Any] | np.ndarray | pd.Series | None = None,
    *,
    start_time_minutes: int = 0,
    binning: str = "floor",
    config: GlucoFMConfig | None = None,
) -> AlignedCGMWindow:
    """Align readings without turning missing positions into observations."""

    cfg = config or GlucoFMConfig()
    values = np.asarray(glucose_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("At least one CGM value is required; empty windows are not imputed.")
    if binning not in {"floor", "nearest"}:
        raise ValueError("binning must be 'floor' or 'nearest'")

    aligned = np.zeros(cfg.sequence_length, dtype=np.float32)
    mask = np.zeros(cfg.sequence_length, dtype=np.float32)
    duplicate_count = 0
    start_timestamp: str | None = None

    if timestamps is None:
        if values.size != cfg.sequence_length:
            raise ValueError(
                "Timestamp-free GlucoFM input must already contain exactly 288 "
                "chronological 5-minute positions. Pass timestamps for irregular data."
            )
        finite = np.isfinite(values)
        aligned[finite] = values[finite]
        mask[finite] = 1.0
        start_index = int(start_time_minutes // cfg.sampling_interval_minutes) % cfg.sequence_length
    else:
        raw_times = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        if len(raw_times) != len(values):
            raise ValueError("glucose_values and timestamps must have equal length")
        valid = np.isfinite(values) & raw_times.notna().to_numpy()
        if not np.any(valid):
            raise ValueError("No finite timestamped CGM observations were found")

        frame = pd.DataFrame(
            {"timestamp": raw_times[valid].to_numpy(), "glucose": values[valid]}
        ).sort_values("timestamp", kind="stable")
        first = pd.Timestamp(frame.iloc[0]["timestamp"])
        start_timestamp = first.isoformat()
        start_index = (
            _start_minute_of_day(first) // cfg.sampling_interval_minutes
        ) % cfg.sequence_length
        elapsed_minutes = (
            frame["timestamp"] - first
        ).dt.total_seconds().to_numpy(dtype=float) / 60.0
        raw_positions = elapsed_minutes / float(cfg.sampling_interval_minutes)
        positions = (
            np.floor(raw_positions) if binning == "floor" else np.rint(raw_positions)
        ).astype(int)
        frame = frame.assign(grid_position=positions)
        frame = frame[
            (frame["grid_position"] >= 0)
            & (frame["grid_position"] < cfg.sequence_length)
        ]
        if frame.empty:
            raise ValueError("No CGM observations fall inside the first 24-hour window")
        grouped = frame.groupby("grid_position", sort=True)["glucose"]
        means = grouped.mean()
        duplicate_count = int(len(frame) - len(means))
        grid_positions = means.index.to_numpy(dtype=int)
        aligned[grid_positions] = means.to_numpy(dtype=np.float32)
        mask[grid_positions] = 1.0

    if not np.any(mask):
        raise ValueError("The aligned window contains no physical CGM observations")
    absolute = (start_index + np.arange(cfg.sequence_length, dtype=np.int64)) % cfg.sequence_length
    return AlignedCGMWindow(
        values=aligned,
        observation_mask=mask,
        absolute_grid_indices=absolute,
        start_timestamp=start_timestamp,
        binning=binning,
        duplicate_measurements_averaged=duplicate_count,
    )


def mask_aware_normalize(
    values: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each window using physical observations only."""

    mask = observation_mask.to(dtype=values.dtype)
    counts = mask.sum(dim=-1, keepdim=True)
    if torch.any(counts <= 0):
        raise ValueError("Every GlucoFM window must contain at least one observed value")
    mean = (values * mask).sum(dim=-1, keepdim=True) / counts.clamp_min(epsilon)
    variance = (((values - mean) ** 2) * mask).sum(dim=-1, keepdim=True)
    variance = variance / counts.clamp_min(epsilon)
    std = torch.sqrt(variance.clamp_min(epsilon))
    normalized = ((values - mean) / std) * mask
    return normalized, mean.squeeze(-1), std.squeeze(-1)


class CausalMaskAwareGaussianFilter(nn.Module):
    """Learnable one-sided Gaussian filter from GlucoFM v2 equations 11-13."""

    def __init__(
        self,
        sigma_min: float = 2.0,
        sigma_max: float = 12.0,
        sigma_initial: float = 6.0,
        max_lag: int = 36,
    ) -> None:
        super().__init__()
        if not sigma_min < sigma_initial < sigma_max:
            raise ValueError("sigma_initial must lie between sigma_min and sigma_max")
        fraction = (sigma_initial - sigma_min) / (sigma_max - sigma_min)
        rho_initial = math.log(fraction / (1.0 - fraction))
        self.rho = nn.Parameter(torch.tensor(rho_initial, dtype=torch.float32))
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.max_lag = int(max_lag)

    @property
    def sigma(self) -> torch.Tensor:
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            self.rho
        )

    def kernel(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lag = torch.arange(self.max_lag + 1, device=device, dtype=dtype)
        sigma = self.sigma.to(device=device, dtype=dtype)
        weights = torch.exp(-(lag**2) / (2.0 * sigma**2))
        return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)

    def forward(self, values: torch.Tensor, observation_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or observation_mask.shape != values.shape:
            raise ValueError("values and observation_mask must both have shape [batch, time]")
        mask = observation_mask.to(dtype=values.dtype)
        weights = self.kernel(device=values.device, dtype=values.dtype)
        states: list[torch.Tensor] = []
        for position in range(values.shape[1]):
            first = max(0, position - self.max_lag)
            width = position - first + 1
            local_weights = weights[:width].flip(0).unsqueeze(0)
            local_mask = mask[:, first : position + 1]
            denominator = (local_mask * local_weights).sum(dim=1)
            numerator = (
                values[:, first : position + 1] * local_mask * local_weights
            ).sum(dim=1)
            states.append(
                torch.where(
                    denominator > 0,
                    numerator / denominator.clamp_min(1e-8),
                    torch.zeros_like(numerator),
                )
            )
        return torch.stack(states, dim=1)


def nearest_observed_rate_of_change(
    values: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    max_backoff: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equation 10: use the closest observed predecessor within nine grid steps."""

    if values.ndim != 2 or values.shape != observation_mask.shape:
        raise ValueError("values and observation_mask must both have shape [batch, time]")
    mask = observation_mask > 0.5
    roc = torch.zeros_like(values)
    valid_roc = torch.zeros_like(mask)
    for lag in range(1, max_backoff + 1):
        candidates = mask[:, lag:] & mask[:, :-lag] & ~valid_roc[:, lag:]
        candidate_values = (values[:, lag:] - values[:, :-lag]) / float(lag)
        roc[:, lag:] = torch.where(candidates, candidate_values, roc[:, lag:])
        valid_roc[:, lag:] |= candidates
    return roc, valid_roc.to(dtype=values.dtype)


def _patchify(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    return values.reshape(values.shape[0], -1, patch_size)


def _masked_patch_statistics(
    patches: torch.Tensor,
    patch_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = patch_mask.sum(dim=-1)
    mean = (patches * patch_mask).sum(dim=-1) / count.clamp_min(1.0)
    variance = (((patches - mean.unsqueeze(-1)) ** 2) * patch_mask).sum(dim=-1)
    variance = variance / count.clamp_min(1.0)
    std = torch.sqrt(variance.clamp_min(0.0))
    valid = count > 0
    mean = torch.where(valid, mean, torch.zeros_like(mean))
    std = torch.where(valid, std, torch.zeros_like(std))
    return mean, std


class GlucoFMStreamEncoder(nn.Module):
    """Paper-aligned state/event patch embedder (pre-Transformer stream encoder)."""

    def __init__(self, config: GlucoFMConfig, stream: str) -> None:
        super().__init__()
        if stream not in {"state", "event"}:
            raise ValueError("stream must be 'state' or 'event'")
        self.config = config
        self.stream = stream
        if stream == "state":
            self.waveform = nn.Linear(config.patch_size, config.state_waveform_dimension)
            self.dynamic = nn.Linear(config.patch_size, config.state_difference_dimension)
            self.statistics = nn.Linear(2, config.state_statistics_dimension)
            input_dimension = (
                config.state_waveform_dimension
                + config.state_difference_dimension
                + config.state_statistics_dimension
            )
        else:
            self.waveform = nn.Linear(config.patch_size, config.event_waveform_dimension)
            self.dynamic = nn.Linear(config.patch_size, config.event_roc_dimension)
            self.statistics = nn.Linear(2, config.event_statistics_dimension)
            input_dimension = (
                config.event_waveform_dimension
                + config.event_roc_dimension
                + config.event_statistics_dimension
            )
        self.projection = nn.Sequential(
            nn.Linear(input_dimension, config.stream_dimension),
            nn.GELU(),
            nn.LayerNorm(config.stream_dimension),
        )

    def forward(
        self,
        waveform_patches: torch.Tensor,
        dynamic_patches: torch.Tensor,
        statistics: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.waveform(waveform_patches),
                self.dynamic(dynamic_patches),
                self.statistics(statistics),
            ],
            dim=-1,
        )
        return self.projection(features)


@dataclass(frozen=True)
class GlucoFMTokenOutput:
    state_tokens: torch.Tensor
    event_tokens: torch.Tensor
    fused_tokens: torch.Tensor
    temporal_features: torch.Tensor
    patch_density: torch.Tensor
    normalized_values: torch.Tensor
    state_signal: torch.Tensor
    event_signal: torch.Tensor


class GlucoFMDualStreamEncoder(nn.Module):
    """Independent paper-aligned implementation of the GlucoFM v2 encoder."""

    def __init__(self, config: GlucoFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or GlucoFMConfig()
        cfg = self.config
        self.state_event_filter = CausalMaskAwareGaussianFilter(
            sigma_min=cfg.gaussian_sigma_min,
            sigma_max=cfg.gaussian_sigma_max,
            sigma_initial=cfg.gaussian_sigma_initial,
            max_lag=cfg.gaussian_max_lag,
        )
        self.state_encoder = GlucoFMStreamEncoder(cfg, "state")
        self.event_encoder = GlucoFMStreamEncoder(cfg, "event")
        self.fusion_projection = nn.Sequential(
            nn.Linear(cfg.stream_dimension * 2, cfg.fused_dimension),
            nn.GELU(),
            nn.LayerNorm(cfg.fused_dimension),
        )
        self.patch_position = nn.Parameter(
            torch.zeros(1, cfg.patch_count, cfg.fused_dimension)
        )
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        self.circular_time_projection = nn.Linear(2, cfg.fused_dimension)
        self.circular_time_gate = nn.Parameter(torch.zeros(cfg.fused_dimension))
        context_layer = nn.TransformerEncoderLayer(
            d_model=cfg.fused_dimension,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dimension,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer, num_layers=cfg.encoder_layers
        )
        self.output_norm = nn.LayerNorm(cfg.fused_dimension)
        self.checkpoint_metadata: GlucoFMCheckpointMetadata | None = None

    def decompose_signal(
        self,
        cgm: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones_like(cgm) if observation_mask is None else observation_mask
        state = self.state_event_filter(cgm, mask)
        event = (cgm - state) * mask.to(dtype=cgm.dtype)
        return state, event

    def _temporal_features(self, absolute_grid_indices: torch.Tensor) -> torch.Tensor:
        patch_indices = _patchify(
            absolute_grid_indices.to(dtype=torch.float32), self.config.patch_size
        )[:, :, 0]
        angle = 2.0 * math.pi * patch_indices / float(self.config.sequence_length)
        circular = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        projected = self.circular_time_projection(circular)
        gated = torch.sigmoid(self.circular_time_gate).view(1, 1, -1) * projected
        return self.patch_position + gated

    def encode_patch_tokens(
        self,
        cgm_24h: torch.Tensor,
        observation_mask: torch.Tensor,
        absolute_grid_indices: torch.Tensor | None = None,
    ) -> GlucoFMTokenOutput:
        cfg = self.config
        if cgm_24h.ndim != 2 or cgm_24h.shape[1] != cfg.sequence_length:
            raise ValueError(
                f"Expected cgm_24h shape [batch, {cfg.sequence_length}], got "
                f"{tuple(cgm_24h.shape)}"
            )
        if observation_mask.shape != cgm_24h.shape:
            raise ValueError("observation_mask must match cgm_24h")
        if not torch.all((observation_mask == 0) | (observation_mask == 1)):
            raise ValueError("observation_mask must be binary")
        if absolute_grid_indices is None:
            absolute_grid_indices = torch.arange(
                cfg.sequence_length, device=cgm_24h.device
            ).unsqueeze(0).expand(cgm_24h.shape[0], -1)
        if absolute_grid_indices.shape != cgm_24h.shape:
            raise ValueError("absolute_grid_indices must match cgm_24h")

        normalized, _, _ = mask_aware_normalize(cgm_24h, observation_mask)
        state, event = self.decompose_signal(normalized, observation_mask)
        roc, roc_mask = nearest_observed_rate_of_change(
            normalized,
            observation_mask,
            max_backoff=cfg.roc_max_backoff,
        )

        physical_patches = _patchify(observation_mask, cfg.patch_size)
        normalized_patches = _patchify(normalized, cfg.patch_size)
        state_patches = _patchify(state, cfg.patch_size)
        event_patches = _patchify(event, cfg.patch_size)
        roc_patches = _patchify(roc, cfg.patch_size)
        roc_mask_patches = _patchify(roc_mask, cfg.patch_size)

        state_mean, state_std = _masked_patch_statistics(
            normalized_patches, physical_patches
        )
        roc_mean, roc_std = _masked_patch_statistics(roc_patches, roc_mask_patches)
        state_statistics = torch.stack([state_mean, state_std], dim=-1)
        event_statistics = torch.stack([roc_mean, roc_std], dim=-1)
        state_difference = torch.diff(
            state_patches, dim=-1, prepend=state_patches[:, :, :1]
        )

        state_tokens = self.state_encoder(
            state_patches, state_difference, state_statistics
        )
        event_tokens = self.event_encoder(event_patches, roc_patches, event_statistics)
        fused_tokens = self.fusion_projection(
            torch.cat([state_tokens, event_tokens], dim=-1)
        )
        temporal = self._temporal_features(absolute_grid_indices)
        density = physical_patches.mean(dim=-1)
        return GlucoFMTokenOutput(
            state_tokens=state_tokens,
            event_tokens=event_tokens,
            fused_tokens=fused_tokens + temporal,
            temporal_features=temporal,
            patch_density=density,
            normalized_values=normalized,
            state_signal=state,
            event_signal=event,
        )

    def encode_context(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.context_encoder(fused_tokens))

    def forward_tokens(
        self,
        cgm_24h: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        absolute_grid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = torch.ones_like(cgm_24h) if observation_mask is None else observation_mask
        token_output = self.encode_patch_tokens(cgm_24h, mask, absolute_grid_indices)
        return self.encode_context(token_output.fused_tokens)

    def forward(
        self,
        cgm_24h: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        absolute_grid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the frozen daily representation (global mean over 24 patches)."""

        return self.forward_tokens(
            cgm_24h, observation_mask, absolute_grid_indices
        ).mean(dim=1)


class GlucoFMDownstreamProbes(nn.Module):
    """Untrained research heads; callers must fit these on subject-disjoint data."""

    def __init__(self, fused_dim: int = 128, meal_context_dim: int = 5) -> None:
        super().__init__()
        self.probe_homa_ir = nn.Linear(fused_dim, 1)
        self.probe_diabetes_status = nn.Linear(fused_dim, 3)
        self.probe_hypo_sensitivity = nn.Linear(fused_dim, 1)
        self.ppgr_head_24 = nn.Sequential(
            nn.Linear(fused_dim + meal_context_dim, 128),
            nn.GELU(),
            nn.Linear(128, 24),
        )

    def forward(
        self,
        representation: torch.Tensor,
        meal_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {
            "homa_ir": self.probe_homa_ir(representation),
            "diabetes_logits": self.probe_diabetes_status(representation),
            "hypo_sensitivity": self.probe_hypo_sensitivity(representation),
        }
        if meal_context is not None:
            outputs["ppgr_forecast_2h"] = self.ppgr_head_24(
                torch.cat([representation, meal_context], dim=-1)
            )
        return outputs


@dataclass(frozen=True)
class GlucoFMPretrainingOutput:
    loss: torch.Tensor
    masked_context_loss: torch.Tensor
    temporal_dynamics_loss: torch.Tensor
    masked_patches: torch.Tensor
    patch_density: torch.Tensor


class GlucoFMPretrainer(nn.Module):
    """EMA target branch and the two GlucoFM v2 latent objectives."""

    def __init__(
        self,
        encoder: GlucoFMDualStreamEncoder | None = None,
        config: GlucoFMConfig | None = None,
    ) -> None:
        super().__init__()
        self.online_encoder = encoder or GlucoFMDualStreamEncoder(config)
        self.config = self.online_encoder.config
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.checkpoint_metadata = None
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.config.fused_dimension)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=self.config.fused_dimension,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_dimension,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context_predictor = nn.TransformerEncoder(
            predictor_layer, num_layers=self.config.predictor_layers
        )
        transition_input = (
            self.config.stream_dimension * 2 + self.config.fused_dimension
        )
        self.state_transition = nn.Sequential(
            nn.Linear(transition_input, self.config.stream_dimension), nn.GELU()
        )
        self.event_transition = nn.Sequential(
            nn.Linear(transition_input, self.config.stream_dimension), nn.GELU()
        )

    def sample_patch_mask(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        ratios = torch.empty(batch_size, device=device).uniform_(
            self.config.mask_ratio_min,
            self.config.mask_ratio_max,
            generator=generator,
        )
        random_scores = torch.rand(
            batch_size,
            self.config.patch_count,
            device=device,
            generator=generator,
        )
        patch_mask = torch.zeros_like(random_scores, dtype=torch.bool)
        for row, ratio in enumerate(ratios):
            count = max(1, int(round(float(ratio) * self.config.patch_count)))
            patch_mask[row, torch.argsort(random_scores[row])[:count]] = True
        return patch_mask

    def forward(
        self,
        cgm_24h: torch.Tensor,
        observation_mask: torch.Tensor,
        absolute_grid_indices: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GlucoFMPretrainingOutput:
        cfg = self.config
        if patch_mask is None:
            patch_mask = self.sample_patch_mask(
                cgm_24h.shape[0], device=cgm_24h.device, generator=generator
            )
        if patch_mask.shape != (cgm_24h.shape[0], cfg.patch_count):
            raise ValueError("patch_mask has an invalid shape")

        visible_steps = (~patch_mask).repeat_interleave(cfg.patch_size, dim=1)
        online_mask = observation_mask * visible_steps.to(observation_mask.dtype)
        online = self.online_encoder.encode_patch_tokens(
            cgm_24h, online_mask, absolute_grid_indices
        )
        online_context_input = torch.where(
            patch_mask.unsqueeze(-1),
            self.mask_token.expand(cgm_24h.shape[0], cfg.patch_count, -1),
            online.fused_tokens,
        )
        online_context = self.online_encoder.encode_context(online_context_input)
        predicted_context = self.context_predictor(online_context)

        with torch.no_grad():
            target = self.target_encoder.encode_patch_tokens(
                cgm_24h, observation_mask, absolute_grid_indices
            )
            target_context = self.target_encoder.encode_context(target.fused_tokens)

        mcr_per_patch = F.smooth_l1_loss(
            predicted_context, target_context, reduction="none"
        ).mean(dim=-1)
        mcr_weights = target.patch_density * patch_mask.to(target.patch_density.dtype)
        masked_context_loss = (mcr_per_patch * mcr_weights).sum() / mcr_weights.sum().clamp_min(
            1e-8
        )

        current_state = online.state_tokens[:, :-1]
        current_event = online.event_tokens[:, :-1]
        current_time = online.temporal_features[:, :-1]
        state_delta = self.state_transition(
            torch.cat([current_state, current_event, current_time], dim=-1)
        )
        event_delta = self.event_transition(
            torch.cat([current_event, current_state, current_time], dim=-1)
        )
        predicted_state = current_state + state_delta
        predicted_event = current_event + event_delta
        state_error = F.smooth_l1_loss(
            predicted_state, target.state_tokens[:, 1:], reduction="none"
        ).mean(dim=-1)
        event_error = F.smooth_l1_loss(
            predicted_event, target.event_tokens[:, 1:], reduction="none"
        ).mean(dim=-1)
        transition_weights = (
            (~patch_mask[:, :-1]).to(target.patch_density.dtype)
            * target.patch_density[:, :-1]
            * target.patch_density[:, 1:]
        )
        denominator = transition_weights.sum().clamp_min(1e-8)
        state_loss = (state_error * transition_weights).sum() / denominator
        event_loss = (event_error * transition_weights).sum() / denominator
        temporal_loss = 0.5 * (state_loss + event_loss)
        total = cfg.mcr_weight * masked_context_loss + cfg.temporal_dynamics_weight * temporal_loss
        return GlucoFMPretrainingOutput(
            loss=total,
            masked_context_loss=masked_context_loss,
            temporal_dynamics_loss=temporal_loss,
            masked_patches=patch_mask,
            patch_density=target.patch_density,
        )

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)


def glucofm_ema_momentum(
    step: int,
    total_steps: int,
    initial: float = 0.997,
) -> float:
    """Cosine schedule from the initial EMA momentum toward one."""

    if total_steps <= 0:
        return 1.0
    progress = min(max(step / total_steps, 0.0), 1.0)
    return 1.0 - (1.0 - initial) * 0.5 * (1.0 + math.cos(math.pi * progress))


def augment_glucofm_batch(
    values: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the four CGM-aware augmentations described in Appendix C.7."""

    augmented = values.clone()
    mask = observation_mask.clone()
    batch, length = augmented.shape
    operations = ("wander", "compression", "decimation", "disconnection")
    base_probability = {
        "wander": 0.25,
        "compression": 0.10,
        "decimation": 0.40,
        "disconnection": 0.05,
    }

    def random_scalar() -> float:
        return float(torch.rand((), generator=generator, device=values.device))

    for row in range(batch):
        order = torch.randperm(len(operations), generator=generator, device=values.device)
        probability_scale = 1.0
        for operation_index in order.tolist():
            operation = operations[operation_index]
            probability = base_probability[operation] * probability_scale
            if random_scalar() >= probability:
                continue
            if operation == "wander":
                amplitude = 5.0 + random_scalar() * 10.0
                frequency = 0.5 + random_scalar() * 1.5
                phase = random_scalar() * 2.0 * math.pi
                time = torch.arange(length, device=values.device, dtype=values.dtype)
                perturbation = amplitude * torch.sin(
                    2.0 * math.pi * frequency * time / float(length) + phase
                )
                augmented[row] += perturbation * mask[row]
            elif operation == "compression":
                width = 6 + int(random_scalar() * 7.0)
                start = int(random_scalar() * max(1, length - width + 1))
                minimum = 0.4 + random_scalar() * 0.3
                half = max((width - 1) / 2.0, 1.0)
                positions = torch.arange(width, device=values.device, dtype=values.dtype)
                v_shape = minimum + (1.0 - minimum) * torch.abs(positions - half) / half
                augmented[row, start : start + width] *= v_shape
            elif operation == "decimation" and int(mask[row].sum()) > 200:
                offset = int(random_scalar() * 3.0)
                keep = torch.zeros(length, device=values.device, dtype=mask.dtype)
                keep[offset::3] = 1.0
                mask[row] *= keep
                augmented[row] *= mask[row]
            elif operation == "disconnection":
                block_count = 1 + int(random_scalar() * 3.0)
                for _ in range(block_count):
                    width = 2 + int(random_scalar() * 11.0)
                    start = int(random_scalar() * max(1, length - width + 1))
                    mask[row, start : start + width] = 0.0
                    augmented[row, start : start + width] = 0.0
            probability_scale *= 0.25
    return augmented, mask


def build_glucofm_foundation_model(
    config: GlucoFMConfig | None = None,
) -> tuple[GlucoFMDualStreamEncoder, GlucoFMDownstreamProbes]:
    """Build untrained modules for research training; not valid for inference yet."""

    cfg = config or GlucoFMConfig()
    return (
        GlucoFMDualStreamEncoder(cfg),
        GlucoFMDownstreamProbes(fused_dim=cfg.fused_dimension),
    )


def save_glucofm_checkpoint(
    path: Path | str,
    encoder: GlucoFMDualStreamEncoder,
    metadata: GlucoFMCheckpointMetadata,
) -> Path:
    if not metadata.trained or metadata.training_epochs <= 0:
        raise ValueError("Reusable GlucoFM checkpoints must be marked as trained")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": GLUCOFM_CHECKPOINT_FORMAT,
        "model_family": GLUCOFM_MODEL_FAMILY,
        "config": encoder.config.to_dict(),
        "metadata": metadata.to_dict(),
        "state_dict": encoder.state_dict(),
    }
    torch.save(payload, destination)
    return destination


def _safe_torch_load(path: Path, device: str | torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Secure GlucoFM loading requires torch.load(weights_only=True); upgrade PyTorch."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("GlucoFM checkpoint payload must be a mapping")
    return payload


def load_glucofm_checkpoint(
    path: Path | str,
    *,
    device: str | torch.device = "cpu",
    require_trained: bool = True,
) -> tuple[GlucoFMDualStreamEncoder, GlucoFMCheckpointMetadata]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GlucoFM checkpoint not found: {checkpoint}")
    payload = _safe_torch_load(checkpoint, device)
    if payload.get("format") != GLUCOFM_CHECKPOINT_FORMAT:
        raise ValueError(
            "Unsupported checkpoint format. IINTS will not load arbitrary pickle/model files."
        )
    if payload.get("model_family") != GLUCOFM_MODEL_FAMILY:
        raise ValueError("Checkpoint belongs to a different model family")
    config_payload = payload.get("config")
    metadata_payload = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(config_payload, Mapping) or not isinstance(metadata_payload, Mapping):
        raise ValueError("Checkpoint is missing config or provenance metadata")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing its tensor state_dict")
    config = GlucoFMConfig(**dict(config_payload))
    metadata = GlucoFMCheckpointMetadata.from_mapping(metadata_payload)
    if require_trained and not metadata.trained:
        raise ValueError("Untrained GlucoFM weights cannot be used as scientific embeddings")
    encoder = GlucoFMDualStreamEncoder(config).to(device)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.checkpoint_metadata = metadata
    encoder.eval()
    return encoder, metadata


def embed_cgm_with_glucofm_result(
    cgm_series: Sequence[float] | np.ndarray | pd.Series,
    *,
    checkpoint: Path | str,
    timestamps: Sequence[Any] | np.ndarray | pd.Series | None = None,
    start_time_minutes: int = 0,
    binning: str = "floor",
    device: str | torch.device = "cpu",
) -> GlucoFMEmbeddingResult:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    encoder, metadata = load_glucofm_checkpoint(checkpoint_path, device=device)
    window = align_cgm_window(
        cgm_series,
        timestamps,
        start_time_minutes=start_time_minutes,
        binning=binning,
        config=encoder.config,
    )
    values = torch.from_numpy(window.values).unsqueeze(0).to(device)
    mask = torch.from_numpy(window.observation_mask).unsqueeze(0).to(device)
    absolute = torch.from_numpy(window.absolute_grid_indices).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = encoder(values, mask, absolute).squeeze(0).cpu().numpy()
    return GlucoFMEmbeddingResult(
        embedding=embedding,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_metadata=metadata,
        input_observed_count=window.observed_count,
        input_coverage=window.coverage,
        input_start_timestamp=window.start_timestamp,
    )


def embed_cgm_with_glucofm(
    cgm_series: Sequence[float] | np.ndarray | pd.Series,
    encoder: GlucoFMDualStreamEncoder | None = None,
    *,
    checkpoint: Path | str | None = None,
    timestamps: Sequence[Any] | np.ndarray | pd.Series | None = None,
    start_time_minutes: int = 0,
    binning: str = "floor",
    allow_untrained: bool = False,
) -> np.ndarray:
    """Extract a deterministic 128D representation from explicit model weights.

    Scientific use requires ``checkpoint``. ``allow_untrained`` exists only for
    architecture smoke tests and deliberately cannot be enabled from the CLI.
    """

    if checkpoint is not None and encoder is not None:
        raise ValueError("Provide checkpoint or encoder, not both")
    if checkpoint is not None:
        return embed_cgm_with_glucofm_result(
            cgm_series,
            checkpoint=checkpoint,
            timestamps=timestamps,
            start_time_minutes=start_time_minutes,
            binning=binning,
        ).embedding
    if encoder is None:
        raise ValueError(
            "A trained GlucoFM reproduction checkpoint is required. "
            "Run 'iints research glucofm-pretrain' first."
        )
    if not allow_untrained and (
        encoder.checkpoint_metadata is None or not encoder.checkpoint_metadata.trained
    ):
        raise ValueError("Untrained random GlucoFM weights cannot produce research embeddings")
    window = align_cgm_window(
        cgm_series,
        timestamps,
        start_time_minutes=start_time_minutes,
        binning=binning,
        config=encoder.config,
    )
    encoder.eval()
    with torch.inference_mode():
        embedding = encoder(
            torch.from_numpy(window.values).unsqueeze(0),
            torch.from_numpy(window.observation_mask).unsqueeze(0),
            torch.from_numpy(window.absolute_grid_indices).unsqueeze(0),
        )
    return embedding.squeeze(0).cpu().numpy()


__all__ = [
    "GLUCOFM_MODEL_FAMILY",
    "GLUCOFM_CHECKPOINT_FORMAT",
    "GLUCOFM_PAPER_REVISION",
    "GLUCOFM_PAPER_URL",
    "GLUCOFM_IMPLEMENTATION_KIND",
    "GlucoFMConfig",
    "AlignedCGMWindow",
    "GlucoFMCheckpointMetadata",
    "GlucoFMEmbeddingResult",
    "CausalMaskAwareGaussianFilter",
    "GlucoFMStreamEncoder",
    "GlucoFMTokenOutput",
    "GlucoFMDualStreamEncoder",
    "GlucoFMDownstreamProbes",
    "GlucoFMPretrainingOutput",
    "GlucoFMPretrainer",
    "align_cgm_window",
    "mask_aware_normalize",
    "nearest_observed_rate_of_change",
    "glucofm_ema_momentum",
    "augment_glucofm_batch",
    "build_glucofm_foundation_model",
    "save_glucofm_checkpoint",
    "load_glucofm_checkpoint",
    "embed_cgm_with_glucofm_result",
    "embed_cgm_with_glucofm",
    "sha256_file",
]
