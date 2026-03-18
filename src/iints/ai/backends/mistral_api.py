from __future__ import annotations


class MistralAPIBackend:
    backend_name = "mistral_api"

    def __init__(self, *args, **kwargs) -> None:
        self.model_name = "mistral_api_unconfigured"

    def available(self) -> bool:
        return False

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError(
            "Cloud fallback is not enabled in this SDK build yet. "
            "Use mode='local' with Ollama for the open Ministral 3 model."
        )
