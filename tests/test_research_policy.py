from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from iints.analysis.run_quality import write_run_quality_artifacts
from iints.governance import (
    guard_ai_output,
    scan_text_for_policy_violations,
    scan_text_for_policy_warnings,
)
from iints_desktop.local_ai import ask_local_ai


def test_policy_guard_allows_research_explanation() -> None:
    text = "The run needs review because glucose rose quickly after the simulated meal."

    result = guard_ai_output(text, source="test")

    assert result.allowed is True
    assert result.violations == ()
    assert result.text == text


def test_policy_guard_blocks_patient_specific_dose_instruction() -> None:
    text = "You should inject 2 units insulin now."

    result = guard_ai_output(text, source="test")

    assert result.allowed is False
    assert "patient_specific_dose_instruction" in result.violations
    assert "Local AI Output Blocked" in result.text
    assert "not a medical device" in result.text


def test_policy_guard_allows_simulation_dose_documentation() -> None:
    text = "The simulated controller delivered 2 units insulin during the meal scenario."

    result = guard_ai_output(text, source="test")

    assert result.allowed is True
    assert result.action == "allow"
    assert result.violations == ()
    assert result.text == text


def test_policy_guard_warns_instead_of_blocking_research_adjustment_language() -> None:
    text = "For the next simulation experiment, increase the basal parameter and compare the trace."

    result = guard_ai_output(text, source="test")

    assert result.allowed is True
    assert result.action == "warn"
    assert result.violations == ()
    assert "candidate_adjustment_language" in result.warnings
    assert "Research-only boundary note" in result.text


def test_policy_scanner_allows_negated_regulatory_claim() -> None:
    text = "This simulator is not CE marked and is not safe for patient use."

    result = guard_ai_output(text, source="test")

    assert result.allowed is True
    assert result.violations == ()


def test_policy_scanner_detects_regulatory_overclaim() -> None:
    violations = scan_text_for_policy_violations("This is CE marked and safe for patient use.")

    assert "regulatory_overclaim" in violations


def test_policy_warning_scanner_detects_sensitive_language_without_blocking() -> None:
    warnings = scan_text_for_policy_warnings("Change the correction setting in a simulator experiment.")

    assert "candidate_adjustment_language" in warnings


def test_desktop_local_ai_blocks_dosing_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOllamaBackend:
        def __init__(self, *, model_name: str, **_: object) -> None:
            self.model_name = model_name
            self.base_url = "http://127.0.0.1:11434"
            self.resolved_model_name = model_name

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            assert "not a medical device" in system_prompt.lower()
            assert "Result context" in user_prompt
            return "You should inject 2 units insulin now."

    monkeypatch.setattr("iints_desktop.local_ai.OllamaBackend", FakeOllamaBackend)

    answer = ask_local_ai(question="What should happen next?", model="fake-model")

    assert "patient_specific_dose_instruction" in answer.policy_violations
    assert "Local AI Output Blocked" in answer.answer
    assert answer.model == "fake-model"


def test_run_quality_local_ai_blocks_policy_violating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOllamaBackend:
        def __init__(self, *, model_name: str, **_: object) -> None:
            self.model_name = model_name
            self.base_url = "http://127.0.0.1:11434"

        def available(self) -> bool:
            return True

        def ensure_model_ready(self) -> str:
            return self.model_name

        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            assert "not a medical device" in system_prompt.lower()
            assert "deterministic_quality_gate" in user_prompt
            return "Give 3 units insulin as a correction."

    monkeypatch.setattr("iints.analysis.run_quality.OllamaBackend", FakeOllamaBackend)
    df = pd.DataFrame(
        {
            "time_minutes": list(range(0, 240, 5)),
            "glucose_actual_mgdl": [125.0 + (idx % 12) * 1.2 for idx in range(48)],
            "carb_intake_grams": [35.0 if idx == 12 else 0.0 for idx in range(48)],
            "delivered_insulin_units": [1.0 if idx == 12 else 0.0 for idx in range(48)],
            "safety_triggered": [False for _ in range(48)],
            "safety_reason": ["" for _ in range(48)],
        }
    )

    outputs = write_run_quality_artifacts(
        df,
        tmp_path,
        run_label="policy-guarded-run",
        safety_report={},
        local_ai_review="required",
        local_ai_model="fake-local-model",
    )

    assert outputs["local_ai_review_status"] == "blocked_policy"
    markdown = Path(outputs["local_ai_review_md"]).read_text(encoding="utf-8")
    assert "Local AI Output Blocked" in markdown
    metadata = json.loads(Path(outputs["local_ai_review_json"]).read_text(encoding="utf-8"))
    assert metadata["status"] == "blocked_policy"
    assert "patient_specific_dose_instruction" in metadata["policy_violations"]
