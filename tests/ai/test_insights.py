from __future__ import annotations

from iints.ai.insights import AI_INSIGHT_CONTEXT_VERSION, build_insight_context
from iints.ai.prompts import build_prompt


def test_build_insight_context_flags_fast_and_high_glucose_trace() -> None:
    payload = {
        "trace_sample": [
            {"time_minutes": 0, "glucose_actual_mgdl": 120, "delivered_insulin_units": 0.0},
            {"time_minutes": 5, "glucose_actual_mgdl": 190, "delivered_insulin_units": 0.1},
            {
                "time_minutes": 10,
                "glucose_actual_mgdl": 260,
                "delivered_insulin_units": 0.2,
                "safety_triggered": True,
            },
        ]
    }

    context = build_insight_context("generate_insights", payload)

    assert context["context_version"] == AI_INSIGHT_CONTEXT_VERSION
    assert context["authority"]["ai_numeric_authority"] is False
    assert context["input_coverage"]["glucose_points"] == 3
    assert context["input_coverage"]["safety_event_like_records"] == 1
    assert "very_high_glucose_above_250" in context["deterministic_flags"]
    assert "safety_supervisor_activity_present" in context["deterministic_flags"]
    assert context["computed_trace_summary"]["max_observed_rate_mgdl_per_min"] == 14.0


def test_insights_prompt_embeds_deterministic_context_contract() -> None:
    system_prompt, user_prompt = build_prompt(
        "generate_insights",
        {"summary": {"mean_glucose_mgdl": 145}, "trace_sample": [{"glucose_actual_mgdl": 145}]},
    )

    lowered = user_prompt.lower()
    assert "insight_context" in lowered
    assert "key finding" in lowered
    assert "separate evidence from hypotheses" in system_prompt.lower()
    assert '"ai_numeric_authority": false' in lowered
    assert "do not create new metrics" in lowered
