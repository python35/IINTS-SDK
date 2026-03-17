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
