from __future__ import annotations

from dataclasses import dataclass

from .backends.ollama import DEFAULT_MINISTRAL_MODEL


@dataclass(frozen=True)
class LocalMistralModelProfile:
    tag: str
    label: str
    approx_download_gb: float
    recommended_system_ram_gb: int
    recommended_vram_gb: int | None
    fit: str
    notes: str
    aliases: tuple[str, ...] = ()


LOCAL_MISTRAL_MODEL_PROFILES: tuple[LocalMistralModelProfile, ...] = (
    LocalMistralModelProfile(
        tag="ministral-3:3b",
        label="Ministral 3 3B",
        approx_download_gb=3.0,
        recommended_system_ram_gb=16,
        recommended_vram_gb=6,
        fit="Entry-level laptop / small edge box",
        notes="Best starting point for CPU-only systems or modest GPUs. Fastest option, lowest memory pressure.",
        aliases=("ministral-3:3b",),
    ),
    LocalMistralModelProfile(
        tag=DEFAULT_MINISTRAL_MODEL,
        label="Ministral 3 8B",
        approx_download_gb=6.0,
        recommended_system_ram_gb=24,
        recommended_vram_gb=10,
        fit="Balanced desktop / strong laptop",
        notes="Recommended default for most users. Best trade-off between quality, speed, and local memory footprint.",
        aliases=("ministral", "ministral-3", "ministral-3:8b"),
    ),
    LocalMistralModelProfile(
        tag="ministral-3:14b",
        label="Ministral 3 14B",
        approx_download_gb=10.0,
        recommended_system_ram_gb=32,
        recommended_vram_gb=16,
        fit="High-end workstation",
        notes="Use when you have plenty of RAM or a strong GPU and want better reasoning depth at the cost of latency.",
        aliases=("ministral-3:14b",),
    ),
)


def list_local_mistral_models() -> list[LocalMistralModelProfile]:
    return list(LOCAL_MISTRAL_MODEL_PROFILES)
