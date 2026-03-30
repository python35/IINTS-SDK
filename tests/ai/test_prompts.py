from __future__ import annotations

from iints.ai.prompts import MAX_PROMPT_PAYLOAD_CHARS, build_prompt


def test_build_prompt_truncates_large_payloads() -> None:
    payload = {
        "run": {
            "notes": "glucose-spike " * 4000,
            "summary": {"tir": 0.72, "lbgi": 1.1},
        }
    }

    _, user_prompt = build_prompt("generate_report", payload)

    assert "payload truncated for local AI inference" in user_prompt
    assert len(user_prompt) < MAX_PROMPT_PAYLOAD_CHARS + 1000


def test_review_realism_prompt_requests_structured_feedback_sections() -> None:
    _, user_prompt = build_prompt("review_realism", {"summary": {"tir_70_180": 75}})

    assert "What looks realistic" in user_prompt
    assert "What looks suspicious" in user_prompt
    assert "Priority fixes" in user_prompt
    assert "What to improve next" in user_prompt
