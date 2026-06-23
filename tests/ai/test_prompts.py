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


def test_predict_insulin_prompt_forbids_ai_numerical_authority() -> None:
    system_prompt, user_prompt = build_prompt(
        "predict_insulin",
        {
            "current_glucose": 185,
            "active_insulin": 2.4,
            "insulin_effect": 0.02,
            "mpc_recommended_units": 0.45,
        },
    )

    lowered = user_prompt.lower()
    assert "never calculate" in system_prompt.lower()
    assert "research sandbox" in lowered
    assert "do not calculate a dose" in lowered
    assert "fixed mpc and safety layers retain all numerical authority" in lowered
    assert '"ai_numeric_authority": false' in lowered
    assert "FINAL_DOSE" not in user_prompt
    assert "FINAL_GLUCAGON_DOSE_MG" not in user_prompt


def test_prompts_include_static_formula_registry_context() -> None:
    system_prompt, user_prompt = build_prompt(
        "generate_insights",
        {"summary": {"mean_glucose_mgdl": 145}, "trace_sample": [{"glucose_actual_mgdl": 145}]},
    )

    lowered = user_prompt.lower()
    assert "formula_registry" in lowered
    assert "f01_bergman_glucose_rhs" in lowered
    assert '"ai_formula_authority": false' in lowered
    assert "never invent a formula" in system_prompt.lower()
    assert "do not derive or solve formulas" in lowered
