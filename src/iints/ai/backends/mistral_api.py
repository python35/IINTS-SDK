from __future__ import annotations

from ..model_catalog import (
    DEFAULT_MISTRAL_API_MODEL,
    DEFAULT_MISTRAL_API_REASONING_EFFORT,
    migrate_mistral_api_model,
)


class MistralAPIBackend:
    backend_name = "mistral_api"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MISTRAL_API_MODEL,
        reasoning_effort: str | None = DEFAULT_MISTRAL_API_REASONING_EFFORT,
        **kwargs,
    ) -> None:
        migrated_model, migrated_reasoning, migrated = migrate_mistral_api_model(model_name)
        self.requested_model_name = model_name
        self.model_name = migrated_model
        self.reasoning_effort = migrated_reasoning if migrated_reasoning is not None else reasoning_effort
        self.model_was_migrated = migrated

    def available(self) -> bool:
        return False

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        migration_hint = (
            f" Requested deprecated model '{self.requested_model_name}' was mapped to "
            f"'{self.model_name}'."
            if self.model_was_migrated
            else ""
        )
        reasoning_hint = (
            f" Default Mistral reasoning_effort is '{self.reasoning_effort}'."
            if self.reasoning_effort
            else ""
        )
        raise RuntimeError(
            "Cloud fallback is not enabled in this SDK build yet. "
            "Use mode='local' with Ollama for the open Ministral 3 model, or configure "
            f"your external Mistral client for '{self.model_name}'."
            f"{reasoning_hint}{migration_hint}"
        )
