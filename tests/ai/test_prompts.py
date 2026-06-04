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


def test_predict_insulin_prompt_is_research_only_and_cannot_increase_mpc_dose() -> None:
    _, user_prompt = build_prompt(
        "predict_insulin",
        {
            "current_glucose": 185,
            "active_insulin": 2.4,
            "insulin_effect": 0.02,
            "mpc_recommended_units": 0.45,
        },
    )

    lowered = user_prompt.lower()
    assert "research sandbox" in lowered
    assert "not controlling a real pump" in lowered
    assert "never increase insulin above the deterministic mpc dose" in lowered
    assert "simulator will still apply hard glucagon safety caps" in lowered
    assert "FINAL_DOSE" in user_prompt
    assert "FINAL_GLUCAGON_DOSE_MG" in user_prompt
