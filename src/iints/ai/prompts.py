from __future__ import annotations

import json
from typing import Any, Literal


TaskName = Literal[
    "explain_decision",
    "analyze_trends",
    "detect_anomalies",
    "generate_report",
    "review_realism",
]
MAX_PROMPT_PAYLOAD_CHARS = 12000

SYSTEM_PROMPT = (
    "You are the IINTS-AF research assistant for closed-loop insulin delivery simulations "
    "and imported glucose datasets. "
    "Explain glycemic behavior clearly, conservatively, and in plain language. "
    "Do not give medical advice, treatment instructions, or patient-specific recommendations. "
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
        "2. Strong realism signals\n"
        "3. Questionable or unrealistic patterns\n"
        "4. Concrete feedback points the SDK developer can improve\n"
        "5. Suggested follow-up validation checks\n\n"
        "Focus on glycemic ranges, excursion patterns, insulin behavior, safety overrides, and whether the data looks internally coherent.\n\n"
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
    return SYSTEM_PROMPT, template.format(data=_serialize_payload(payload))
