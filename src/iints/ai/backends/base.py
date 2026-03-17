from __future__ import annotations

from typing import Protocol


class CompletionBackend(Protocol):
    backend_name: str
    model_name: str

    def available(self) -> bool:
        ...

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        ...
