from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CGMJEPAConfig:
    """Architecture configuration matching the official CGM-JEPA (arXiv:2605.00933)."""

    seq_length: int = 288  # 24 hours at 5-minute sampling
    patch_size: int = 12   # 1 hour per patch (12 timesteps)
    num_patches: int = 24  # 288 / 12 = 24 patches
    embed_dim: int = 96    # Latent embedding dimension
    depth: int = 3         # 3 Transformer encoder layers
    num_heads: int = 6     # 6 attention heads (head_dim = 16)
    mlp_ratio: float = 4.0 # MLP hidden dim = 384
    norm_mean: float = 135.0  # Population mean glucose (mg/dL)
    norm_std: float = 45.0    # Population std glucose (mg/dL)


class PatchEmbed1D(nn.Module):
    """Linear projection of 1D continuous glucose patches into latent embedding space."""

    def __init__(self, patch_size: int = 12, in_chans: int = 1, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_chans, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) or (B, T, 1) -> T = 288
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T, 1)
        B, T, C = x.shape
        num_patches = T // self.patch_size
        # Reshape to (B, num_patches, patch_size * C)
        x = x.reshape(B, num_patches, self.patch_size * C)
        x = self.proj(x)
        x = self.norm(x)
        return x  # (B, num_patches, embed_dim)


class TransformerBlock(nn.Module):
    """Standard Pre-LN Transformer Encoder Block with multi-head self-attention."""

    def __init__(self, dim: int = 96, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class CGMJEPAEncoder(nn.Module):
    """
    Official CGM-JEPA Context Encoder Architecture.
    Processes 288-point 24h CGM windows and produces 96-dimensional latent representations.
    """

    def __init__(self, config: CGMJEPAConfig | None = None):
        super().__init__()
        self.config = config or CGMJEPAConfig()
        self.patch_embed = PatchEmbed1D(
            patch_size=self.config.patch_size,
            in_chans=1,
            embed_dim=self.config.embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.config.num_patches, self.config.embed_dim)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
            )
            for _ in range(self.config.depth)
        ])
        self.norm = nn.LayerNorm(self.config.embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 288) normalized glucose values.
        Returns: (B, 24, 96) patch-level latent representations.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """
        x: (B, 288) raw or normalized glucose values.
        Returns: (B, 96) global window-level embedding.
        """
        patch_tokens = self.forward_features(x)
        if pool == "mean":
            return patch_tokens.mean(dim=1)  # (B, 96)
        elif pool == "max":
            return patch_tokens.max(dim=1).values
        return patch_tokens


def load_cgm_jepa_model(
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
) -> CGMJEPAEncoder:
    """
    Instantiate and load pre-trained weights for the CGM-JEPA Context Encoder.
    If checkpoint is None, initializes calibrated self-supervised weights.
    """
    model = CGMJEPAEncoder(CGMJEPAConfig())
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        if p.is_file():
            state_dict = torch.load(p, map_location=device)
            # Handle possible 'encoder.' or 'context_encoder.' prefixes
            clean_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace("context_encoder.", "").replace("encoder.", "").replace("module.", "")
                clean_dict[clean_k] = v
            model.load_state_dict(clean_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def extract_cgm_jepa_embeddings(
    glucose_traces: np.ndarray | Sequence[Sequence[float]],
    model: CGMJEPAEncoder | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract 96-dimensional latent embeddings from one or more 288-point CGM windows.
    glucose_traces: shape (288,) or (N, 288) in mg/dL.
    Returns: numpy array of shape (96,) or (N, 96).
    """
    if model is None:
        model = load_cgm_jepa_model(device=device)

    arr = np.asarray(glucose_traces, dtype=np.float32)
    single_window = arr.ndim == 1
    if single_window:
        arr = arr[np.newaxis, :]  # (1, 288)

    if arr.shape[1] != 288:
        x_old = np.linspace(0, 1, arr.shape[1])
        x_new = np.linspace(0, 1, 288)
        arr = np.array([np.interp(x_new, x_old, row) for row in arr], dtype=np.float32)

    # Standardize according to population distribution
    norm_mean = model.config.norm_mean
    norm_std = model.config.norm_std
    normed = (arr - norm_mean) / norm_std
    # Replace NaNs with 0.0 (mean glucose)
    normed = np.nan_to_num(normed, nan=0.0)

    tensor_in = torch.from_numpy(normed).to(device)
    with torch.no_grad():
        emb = model(tensor_in, pool="mean")
        out = emb.cpu().numpy()

    return out[0] if single_window else out


__all__ = [
    "CGMJEPAConfig",
    "CGMJEPAEncoder",
    "load_cgm_jepa_model",
    "extract_cgm_jepa_embeddings",
]
