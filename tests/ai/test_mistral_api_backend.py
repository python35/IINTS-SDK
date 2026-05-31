from __future__ import annotations

import pytest

from iints.ai.backends.mistral_api import MistralAPIBackend
from iints.ai.model_catalog import migrate_mistral_api_model


def test_deprecated_mistral_small_model_migrates_to_small_latest() -> None:
    model, reasoning_effort, migrated = migrate_mistral_api_model("devstral-small-latest")

    assert migrated is True
    assert model == "mistral-small-latest"
    assert reasoning_effort == "high"


def test_deprecated_mistral_medium_model_migrates_to_medium_35() -> None:
    model, reasoning_effort, migrated = migrate_mistral_api_model("magistral-medium-latest")

    assert migrated is True
    assert model == "mistral-medium-3-5"
    assert reasoning_effort == "high"


def test_current_mistral_api_model_is_left_unchanged() -> None:
    model, reasoning_effort, migrated = migrate_mistral_api_model("mistral-small-latest")

    assert migrated is False
    assert model == "mistral-small-latest"
    assert reasoning_effort is None


def test_mistral_api_backend_reports_migrated_model_in_error() -> None:
    backend = MistralAPIBackend(model_name="devstral-small-latest")

    assert backend.model_name == "mistral-small-latest"
    assert backend.reasoning_effort == "high"
    assert backend.model_was_migrated is True
    with pytest.raises(RuntimeError, match="devstral-small-latest"):
        backend.complete(system_prompt="system", user_prompt="prompt")
