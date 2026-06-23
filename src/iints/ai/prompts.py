from __future__ import annotations

import json
from typing import Any, Literal

from iints.core.formula_registry import formula_context_for_ai

from .insights import build_insight_context


TaskName = Literal[
    "explain_decision",
    "analyze_trends",
    "detect_anomalies",
    "generate_insights",
    "generate_report",
    "review_realism",
    "predict_insulin",
]
MAX_PROMPT_PAYLOAD_CHARS = 12000

SYSTEM_PROMPT = (
    "You are the IINTS-AF research assistant for closed-loop insulin delivery simulations "
    "and imported glucose datasets. "
    "Explain glycemic behavior clearly, conservatively, and in plain language. "
    "Do not give medical advice, treatment instructions, or patient-specific recommendations. "
    "Never calculate, estimate, derive, correct, or alter numerical values. "
    "All metrics and controller outputs must come from deterministic SDK code. "
    "Use supplied numbers exactly as written; if a value is missing, say that it is unavailable. "
    "Base your response on the supplied insight_context first, then the raw payload. "
    "Use the supplied formula_registry as immutable context; never invent a formula. "
    "Separate evidence from hypotheses; label hypotheses as non-diagnostic research ideas. "
    "State uncertainty when the input is incomplete. "
    "For research use only."
)

TASK_TEMPLATES: dict[TaskName, str] = {
    "explain_decision": (
        "Given this single decision step or noteworthy glucose snapshot, explain:\n"
        "1. What is happening in the data and why it stands out\n"
        "2. Whether there are safety signals, supervision, or notable context clues\n"
        "3. What a research user should pay attention to next\n\n"
        "Respond in 3 short paragraphs.\n\n"
        "Input JSON:\n{data}"
    ),
    "analyze_trends": (
        "Review this glucose-oriented payload and summarize the main glycemic trends.\n"
        "Focus on direction, stability, excursions, and likely triggers.\n"
        "Respond with:\n"
        "- Trend summary\n"
        "- Main risk signals\n"
        "- Short operational takeaway\n\n"
        "Payload JSON:\n{data}"
    ),
    "detect_anomalies": (
        "Inspect this run or imported-data summary and identify unusual patterns, inconsistent values, or clinically relevant anomalies.\n"
        "Respond with:\n"
        "- Detected anomalies\n"
        "- Why each anomaly matters\n"
        "- Whether follow-up validation is recommended\n\n"
        "Summary JSON:\n{data}"
    ),
    "generate_insights": (
        "Create a useful research insight brief from this IINTS-AF run or glucose dataset.\n"
        "Use the deterministic insight_context as the evidence backbone.\n"
        "Do not create new metrics or alter supplied values.\n\n"
        "Respond in markdown with these sections:\n"
        "1. Key finding\n"
        "2. Evidence from the SDK context\n"
        "3. Likely mechanisms to inspect (research hypotheses only)\n"
        "4. Safety/supervisor signals\n"
        "5. Data quality or realism concerns\n"
        "6. Next validation experiments\n"
        "7. Research-only limitation\n\n"
        "Input JSON:\n{data}"
    ),
    "generate_report": (
        "Write a concise markdown report for this IINTS-AF simulation run or imported personal glucose dataset.\n"
        "Include sections:\n"
        "1. Executive summary\n"
        "2. Glycemic behavior\n"
        "3. Safety, supervision, or device behavior\n"
        "4. Notable events, patterns, or anomalies\n"
        "5. Research-only conclusion\n\n"
        "Input JSON:\n{data}"
    ),
    "review_realism": (
        "Review this simulation or imported-data payload and judge whether the results look physiologically plausible for research use.\n"
        "Be conservative and do not overclaim. If the payload is incomplete, say so clearly.\n"
        "Respond in markdown with these sections:\n"
        "1. Overall realism verdict (Likely realistic / Needs review / Likely unrealistic)\n"
        "2. What looks realistic\n"
        "3. What looks suspicious\n"
        "4. Priority fixes\n"
        "5. What to improve next\n"
        "6. Suggested follow-up validation checks\n\n"
        "Focus on glycemic ranges, excursion patterns, insulin behavior, safety overrides, and whether the data looks internally coherent.\n\n"
        "Input JSON:\n{data}"
    ),
    "predict_insulin": (
        "Explain an immutable, deterministically calculated dose result inside the IINTS-AF research sandbox.\n"
        "You are not controlling a real pump, treating a patient, or giving medical advice.\n"
        "Do not calculate a dose, suggest another dose, modify a supplied number, or emit a new numeric result. "
        "The fixed MPC and safety layers retain all numerical authority. "
        "Explain only which deterministic rules were applied and why the recorded result was accepted, clamped, or held.\n\n"
        "Input JSON:\n{data}"
    ),
}


def _serialize_payload(payload: Any) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    if len(text) <= MAX_PROMPT_PAYLOAD_CHARS:
        return text

    head_chars = MAX_PROMPT_PAYLOAD_CHARS // 2
    tail_chars = MAX_PROMPT_PAYLOAD_CHARS - head_chars
    omitted = len(text) - MAX_PROMPT_PAYLOAD_CHARS
    return (
        f"{text[:head_chars]}\n"
        f"... [payload truncated for local AI inference, omitted {omitted} characters] ...\n"
        f"{text[-tail_chars:]}"
    )


def build_prompt(task: TaskName, payload: Any) -> tuple[str, str]:
    template = TASK_TEMPLATES[task]
    insight_context = build_insight_context(task, payload)
    guarded_payload = {
        "calculation_policy": {
            "mode": "deterministic_sdk_only",
            "ai_numeric_authority": False,
            "instruction": "Explain supplied values only; never calculate or alter numbers.",
        },
        "insight_context": insight_context,
        "formula_registry": formula_context_for_ai(),
        "payload": payload,
    }
    return SYSTEM_PROMPT, template.format(data=_serialize_payload(guarded_payload))
