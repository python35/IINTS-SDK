from __future__ import annotations

import json
from typing import Any, Literal


TaskName = Literal["explain_decision", "analyze_trends", "detect_anomalies", "generate_report"]

SYSTEM_PROMPT = (
    "You are the IINTS-AF research assistant for closed-loop insulin delivery simulations. "
    "Explain simulation behavior clearly, conservatively, and in plain language. "
    "Do not give medical advice, treatment instructions, or patient-specific recommendations. "
    "State uncertainty when the input is incomplete. "
    "For research use only."
)

TASK_TEMPLATES: dict[TaskName, str] = {
    "explain_decision": (
        "Given this single simulation step, explain:\n"
        "1. What the algorithm decided and why\n"
        "2. Whether the independent safety supervisor likely intervened\n"
        "3. Whether the decision appears safe in context\n\n"
        "Respond in 3 short paragraphs.\n\n"
        "Simulation step JSON:\n{data}"
    ),
    "analyze_trends": (
        "Review this glucose-oriented simulation payload and summarize the main glycemic trends.\n"
        "Focus on direction, stability, excursions, and likely triggers.\n"
        "Respond with:\n"
        "- Trend summary\n"
        "- Main risk signals\n"
        "- Short operational takeaway\n\n"
        "Simulation payload JSON:\n{data}"
    ),
    "detect_anomalies": (
        "Inspect this run summary and identify unusual patterns, inconsistent values, or clinically relevant anomalies.\n"
        "Respond with:\n"
        "- Detected anomalies\n"
        "- Why each anomaly matters\n"
        "- Whether follow-up validation is recommended\n\n"
        "Run summary JSON:\n{data}"
    ),
    "generate_report": (
        "Write a concise markdown report for this IINTS-AF simulation run.\n"
        "Include sections:\n"
        "1. Executive summary\n"
        "2. Glycemic behavior\n"
        "3. Safety and supervisor behavior\n"
        "4. Notable events or anomalies\n"
        "5. Research-only conclusion\n\n"
        "Simulation run JSON:\n{data}"
    ),
}


def _serialize_payload(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str)


def build_prompt(task: TaskName, payload: Any) -> tuple[str, str]:
    template = TASK_TEMPLATES[task]
    return SYSTEM_PROMPT, template.format(data=_serialize_payload(payload))
