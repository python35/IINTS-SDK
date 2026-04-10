from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from iints.analysis.study_engine import (
    DEFAULT_BASELINE_ALGORITHMS,
    DEFAULT_HYPOTHESES,
    DEFAULT_METRICS,
    DEFAULT_PROFILE_SET,
    StudyDesignPayload,
    build_study_design_payload,
)

DEFAULT_SCENARIOS = [
    "baseline_day",
    "meal_challenge",
    "exercise_challenge",
    "supervisor_override",
]

DEFAULT_CORRUPTION_MODES = [
    "timestamp_shift",
    "missing_block",
    "duplicate_rows",
    "glucose_spikes",
    "drop_meal_annotations",
    "unit_scale_error",
]


def _candidate_source_type(candidate_algorithm: str) -> str:
    return "path" if candidate_algorithm.endswith(".py") or "/" in candidate_algorithm or "\\" in candidate_algorithm else "plugin"


def build_study_protocol_payload(
    *,
    preset: str = "default",
    title: str = "IINTS Scientific Validation Protocol",
    primary_hypothesis: str | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    algorithms: list[str] | None = None,
    corruption_modes: list[str] | None = None,
    external_reference_label: str = "CareLink personal workbench metrics",
    profile_set: str = DEFAULT_PROFILE_SET,
    include_default_baselines: bool = True,
    extra_algorithms: list[str] | None = None,
) -> dict[str, Any]:
    legacy_algorithms = [item.strip() for item in (algorithms or []) if str(item).strip()]
    candidate_algorithm = legacy_algorithms[0] if legacy_algorithms else "your_algorithm"
    comparison_algorithms = legacy_algorithms[1:] + [item.strip() for item in (extra_algorithms or []) if item.strip()]

    design = build_study_design_payload(
        preset=preset,
        title=title,
        primary_hypothesis=primary_hypothesis,
        scenarios=scenarios,
        seeds=seeds,
        candidate_algorithm=candidate_algorithm,
        include_default_baselines=include_default_baselines,
        extra_algorithms=comparison_algorithms,
        profile_set=profile_set,
        external_reference_label=external_reference_label,
        candidate_source_type=_candidate_source_type(candidate_algorithm),
        candidate_source_ref=candidate_algorithm,
    )
    payload = design.to_dict()
    if corruption_modes:
        payload["corruption_plan"] = [
            item for item in payload.get("corruption_plan", []) if item.get("mode") in set(corruption_modes)
        ]
    return payload


def render_study_protocol_markdown(payload: dict[str, Any]) -> str:
    hypotheses = payload.get("hypotheses", [])
    profiles = payload.get("profiles", [])
    algorithms = payload.get("algorithms", [])
    scenarios = payload.get("scenarios", [])
    study_arms = payload.get("study_arms", [])
    metrics = payload.get("metrics", [])
    statistics_plan = payload.get("statistics_plan", {})
    external_validation = payload.get("external_validation", {})
    commands = payload.get("recommended_commands", [])

    lines = [
        f"# {payload.get('title', 'IINTS Scientific Validation Protocol')}",
        "",
        "## Research Question",
        "",
        str(payload.get("research_question", "")),
        "",
        "## Hypotheses",
        "",
    ]
    for hypothesis in hypotheses:
        lines.append(f"- **{hypothesis.get('id', 'H?')}**: {hypothesis.get('statement', '')}")

    lines.extend(
        [
            "",
            "## Profile Set",
            "",
            f"- Active profile set: `{payload.get('profile_set', DEFAULT_PROFILE_SET)}`",
            f"- Profile count: `{len(profiles)}`",
            "",
        ]
    )
    for profile in profiles:
        lines.append(
            f"- `{profile.get('profile_id', '')}` — {profile.get('label', '')}: {profile.get('description', '')}"
        )

    lines.extend(["", "## Algorithm Registry", ""])
    for algorithm in algorithms:
        lines.append(
            "- "
            f"`{algorithm.get('algorithm_id', '')}` — {algorithm.get('display_name', '')} "
            f"({algorithm.get('role', '')}, {algorithm.get('source_type', '')})"
        )

    lines.extend(["", "## Scenario Families", ""])
    for scenario in scenarios:
        lines.append(
            f"- `{scenario.get('slug', '')}` — {scenario.get('label', '')} "
            f"({scenario.get('recommended_duration_minutes', 'n/a')} min)"
        )

    lines.extend(["", "## Study Arms", ""])
    for arm in study_arms:
        lines.append(
            f"- `{arm.get('arm_id', '')}` — {arm.get('label', '')}; "
            f"certification=`{arm.get('expected_certification', '')}`, "
            f"supervisor_enabled=`{arm.get('supervisor_enabled', False)}`, "
            f"corruption_modes=`{', '.join(arm.get('corruption_modes', [])) or 'none'}`"
        )

    lines.extend(
        [
            "",
            "## Study Matrix",
            "",
            f"- Total matrix rows: `{len(payload.get('matrix_rows', []))}`",
            f"- Seeds: `{', '.join(str(item) for item in payload.get('seed_policy', {}).get('seeds', []))}`",
            "- Cross-product: profiles × scenarios × arms × algorithms × seeds",
            "",
            "## Controlled Corruption Operators",
            "",
        ]
    )
    for item in payload.get("corruption_plan", []):
        lines.append(f"- `{item.get('mode', '')}`: {item.get('purpose', '')}")

    lines.extend(
        [
            "",
            "## Outcome Metrics",
            "",
        ]
    )
    for metric in metrics:
        lines.append(f"- `{metric}`")

    lines.extend(
        [
            "",
            "## Statistics Plan",
            "",
            f"- Descriptive: `{', '.join(statistics_plan.get('descriptive', []))}`",
            f"- Comparative: `{', '.join(statistics_plan.get('comparative', []))}`",
            f"- Failure analysis: `{', '.join(statistics_plan.get('failure_analysis', []))}`",
            "",
            "## External Validation",
            "",
            f"- Reference: `{external_validation.get('reference_label', '')}`",
            f"- Compare: `{', '.join(external_validation.get('compare_metrics', []))}`",
            f"- Note: {external_validation.get('note', '')}",
            "",
            "## Reproducibility Checklist",
            "",
        ]
    )
    for item in payload.get("reproducibility_checklist", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Recommended Commands", ""])
    for command in commands:
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def write_study_protocol_bundle(
    output_dir: str | Path,
    *,
    preset: str = "default",
    title: str = "IINTS Scientific Validation Protocol",
    primary_hypothesis: str | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    algorithms: list[str] | None = None,
    corruption_modes: list[str] | None = None,
    external_reference_label: str = "CareLink personal workbench metrics",
    profile_set: str = DEFAULT_PROFILE_SET,
    include_default_baselines: bool = True,
    extra_algorithms: list[str] | None = None,
) -> dict[str, str]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    payload = build_study_protocol_payload(
        preset=preset,
        title=title,
        primary_hypothesis=primary_hypothesis,
        scenarios=scenarios,
        seeds=seeds,
        algorithms=algorithms,
        corruption_modes=corruption_modes,
        external_reference_label=external_reference_label,
        profile_set=profile_set,
        include_default_baselines=include_default_baselines,
        extra_algorithms=extra_algorithms,
    )

    markdown_path = target / "STUDY_PROTOCOL.md"
    design_json = target / "study_design.json"
    matrix_csv = target / "study_matrix.csv"
    algorithms_json = target / "algorithms.json"

    markdown_path.write_text(render_study_protocol_markdown(payload), encoding="utf-8")
    design_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    algorithms_json.write_text(json.dumps(payload.get("algorithms", []), indent=2), encoding="utf-8")

    with matrix_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "arm_id",
            "condition_group",
            "profile_id",
            "profile_label",
            "algorithm_id",
            "algorithm_label",
            "algorithm_role",
            "algorithm_source_type",
            "scenario_slug",
            "scenario_label",
            "seed",
            "recommended_duration_minutes",
            "supervisor_enabled",
            "expected_certification",
            "corruption_modes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("matrix_rows", []):
            writer.writerow(
                {
                    **row,
                    "corruption_modes": ",".join(row.get("corruption_modes", [])),
                }
            )

    return {
        "protocol_markdown": str(markdown_path),
        "study_design_json": str(design_json),
        "study_matrix_csv": str(matrix_csv),
        "algorithms_json": str(algorithms_json),
    }
