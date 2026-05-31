from __future__ import annotations

from dataclasses import dataclass

from .backends.ollama import DEFAULT_MINISTRAL_MODEL

DEFAULT_MISTRAL_API_MODEL = "mistral-small-latest"
DEFAULT_MISTRAL_API_REASONING_EFFORT = "high"
STRONG_MISTRAL_API_MODEL = "mistral-medium-3-5"
DEFAULT_MISTRAL_OCR_MODEL = "mistral-ocr-latest"
DEFAULT_MISTRAL_MODERATION_MODEL = "mistral-moderation-2603"
DEFAULT_MISTRAL_TRANSCRIBE_MODEL = "voxtral-mini-latest"


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


@dataclass(frozen=True)
class MistralAPIMigrationProfile:
    deprecated_models: tuple[str, ...]
    replacement_model: str
    reasoning_effort: str | None
    retirement_date: str
    use_case: str
    notes: str


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


MISTRAL_API_MIGRATION_PROFILES: tuple[MistralAPIMigrationProfile, ...] = (
    MistralAPIMigrationProfile(
        deprecated_models=("devstral-small-2507", "devstral-small-latest"),
        replacement_model=DEFAULT_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-05-31",
        use_case="code and agentic helper",
        notes="Use Mistral Small 4 through the stable latest alias.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("devstral-medium-2507", "devstral-2512", "devstral-latest"),
        replacement_model=STRONG_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-05-31 / 2026-07-31",
        use_case="strong code and research assistant",
        notes="Use Mistral Medium 3.5 for higher-complexity code/research review.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("magistral-small-2509",),
        replacement_model=DEFAULT_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-07-31",
        use_case="small reasoning",
        notes="Mistral Small 4 now supports adjustable reasoning.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("magistral-medium-2509", "magistral-medium-latest"),
        replacement_model=STRONG_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-07-31",
        use_case="medium reasoning",
        notes="Mistral Medium 3.5 supports adjustable reasoning for harder tasks.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=(
            "mistral-large-2411",
            "pixtral-large-2411",
            "mistral-medium-2505",
            "mistral-medium-2508",
        ),
        replacement_model=STRONG_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-05-31 / 2026-08-31",
        use_case="general strong cloud model",
        notes="Medium 3.5 is the current stronger general-purpose replacement.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("mistral-small-2506",),
        replacement_model=DEFAULT_MISTRAL_API_MODEL,
        reasoning_effort=DEFAULT_MISTRAL_API_REASONING_EFFORT,
        retirement_date="2026-07-31",
        use_case="small general cloud model",
        notes="Use the Mistral Small 4 latest alias.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("open-mistral-nemo-2407",),
        replacement_model="ministral-8b-2512",
        reasoning_effort=None,
        retirement_date="2026-07-31",
        use_case="small/low-cost serverless model",
        notes="Serverless replacement for Nemo. Local SDK defaults still use Ollama `ministral-3:8b`.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("mistral-ocr-2505",),
        replacement_model=DEFAULT_MISTRAL_OCR_MODEL,
        reasoning_effort=None,
        retirement_date="2026-05-31",
        use_case="OCR",
        notes="Use OCR 3 through the latest alias.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("mistral-moderation-2411", "mistral-moderation-latest"),
        replacement_model=DEFAULT_MISTRAL_MODERATION_MODEL,
        reasoning_effort=None,
        retirement_date="2026-06-30",
        use_case="moderation",
        notes="Use Mistral Moderation 2.",
    ),
    MistralAPIMigrationProfile(
        deprecated_models=("voxtral-mini-2507",),
        replacement_model=DEFAULT_MISTRAL_TRANSCRIBE_MODEL,
        reasoning_effort=None,
        retirement_date="2026-05-31",
        use_case="audio transcription",
        notes="Use Voxtral Mini Transcribe 2.0 through the latest alias.",
    ),
)


def list_local_mistral_models() -> list[LocalMistralModelProfile]:
    return list(LOCAL_MISTRAL_MODEL_PROFILES)


def list_mistral_api_migrations() -> list[MistralAPIMigrationProfile]:
    return list(MISTRAL_API_MIGRATION_PROFILES)


def migrate_mistral_api_model(model_name: str) -> tuple[str, str | None, bool]:
    requested = model_name.strip()
    requested_key = requested.lower()
    for profile in MISTRAL_API_MIGRATION_PROFILES:
        if requested_key in {model.lower() for model in profile.deprecated_models}:
            return profile.replacement_model, profile.reasoning_effort, True
    return requested, None, False
