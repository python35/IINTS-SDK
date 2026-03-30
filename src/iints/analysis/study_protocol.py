from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_HYPOTHESES = [
    {
        "id": "H1",
        "statement": "Certified data produces more reliable closed-loop evaluation summaries than uncertified or deliberately corrupted data.",
    },
    {
        "id": "H2",
        "statement": "The safety supervisor reduces severe hypo exposure and early terminations without an unacceptable loss in time in range.",
    },
    {
        "id": "H3",
        "statement": "AI realism review flags suspicious runs that also appear in quantitative failure-analysis outputs.",
    },
]

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

DEFAULT_METRICS = [
    "tir_70_180",
    "tir_below_70",
    "tir_below_54",
    "tir_above_180",
    "tir_above_250",
    "mean_glucose",
    "cv",
    "gmi",
    "supervisor_interventions",
    "terminated_early",
]


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
) -> dict[str, Any]:
    normalized_preset = preset.strip().lower()
    if normalized_preset not in {"default", "eucys"}:
        raise ValueError("preset must be 'default' or 'eucys'")

    scenario_list = scenarios or list(DEFAULT_SCENARIOS)
    seed_list = seeds or [1, 2, 3, 4, 5]
    algorithm_list = algorithms or ["your_algorithm", "Standard PID", "Standard Pump"]
    corruption_list = corruption_modes or list(DEFAULT_CORRUPTION_MODES)
    hypotheses = list(DEFAULT_HYPOTHESES)
    if primary_hypothesis is not None:
        hypotheses[0] = {"id": "H1", "statement": primary_hypothesis}
    elif normalized_preset == "eucys":
        hypotheses[0] = {
            "id": "H1",
            "statement": "Certified data produces more reliable and defendable evaluation evidence than deliberately corrupted uncertified data.",
        }
        seed_list = seeds or list(range(1, 11))
        corruption_list = corruption_modes or ["timestamp_shift", "missing_block", "glucose_spikes", "unit_scale_error"]
        title = title if title != "IINTS Scientific Validation Protocol" else "IINTS EUCYS Scientific Validation Protocol"

    conditions = [
        {
            "name": "clean_certified",
            "description": "Nominal run with certification artifacts generated and grade checked.",
        },
        {
            "name": "corrupted_uncertified",
            "description": "Same scenario with one or more deliberate corruption operators applied.",
        },
        {
            "name": "supervisor_on",
            "description": "Safety supervisor enabled to measure intervention rate and severe hypo prevention.",
        },
        {
            "name": "supervisor_off",
            "description": "Ablation condition used only for controlled comparisons and failure analysis.",
        },
    ]

    recommended_commands = [
        "iints scenarios export-study-pack --output-dir scenarios/study_pack",
        "iints study-protocol --output-dir results/study_protocol",
        "iints data corrupt-for-study data/demo/diabetes_cgm.csv --output-csv data/demo/diabetes_cgm_corrupted.csv --mode timestamp_shift --mode missing_block",
        "iints analyze results/study --output-json results/study_summary.json --output-markdown results/study_summary.md",
        "iints compare-study results/study_clean results/study_corrupted --output-json results/study_comparison.json",
        "iints poster-study results/study_summary.json --output-path results/study_poster.png",
    ]
    if normalized_preset == "eucys":
        recommended_commands[0] = "iints scenarios export-study-pack --preset eucys --output-dir scenarios/eucys_pack"
        recommended_commands[1] = "iints study-protocol --preset eucys --output-dir results/study_protocol"

    return {
        "preset": normalized_preset,
        "title": title,
        "research_question": "Does certified data improve the reliability and interpretability of closed-loop insulin algorithm evaluation?",
        "hypotheses": hypotheses,
        "algorithms": algorithm_list,
        "scenarios": scenario_list,
        "conditions": conditions,
        "seed_policy": {
            "seeds": seed_list,
            "note": "Reuse the same seeds across all conditions to keep paired comparisons fair.",
        },
        "corruption_plan": [
            {
                "mode": mode,
                "purpose": {
                    "timestamp_shift": "Stress provenance and temporal consistency checks.",
                    "missing_block": "Simulate sensor outages or dropped rows.",
                    "duplicate_rows": "Simulate export duplication or stitching bugs.",
                    "glucose_spikes": "Stress plausibility filters and realism review.",
                    "drop_meal_annotations": "Test how missing meal context changes evaluation.",
                    "unit_scale_error": "Expose unit-mismatch errors and impossible ranges.",
                }.get(mode, "Controlled corruption operator for scientific ablation."),
            }
            for mode in corruption_list
        ],
        "metrics": DEFAULT_METRICS,
        "statistics_plan": {
            "descriptive": ["mean", "median", "std", "min", "max", "95% confidence interval"],
            "comparative": ["difference in means", "Cohen's d"],
            "failure_analysis": [
                "terminated_early_runs",
                "severe_hypo_runs",
                "supervisor_heavy_runs",
                "worst_tir_runs",
            ],
        },
        "external_validation": {
            "reference_label": external_reference_label,
            "compare_metrics": [
                "mean_glucose_mgdl",
                "cv_pct",
                "time_in_range_70_180_pct",
                "time_below_70_pct",
                "time_above_180_pct",
            ],
            "note": "Use imported CareLink-style real data only as a plausibility reference, not as a clinical efficacy claim.",
        },
        "exclusion_rules": [
            "Document early-terminated runs instead of silently discarding them.",
            "Exclude only runs with missing results.csv or unreadable metadata.",
            "Keep a manifest of every corrupted dataset generated for the study.",
        ],
        "reproducibility_checklist": [
            "Export the official study pack.",
            "Write the protocol bundle before running scenarios.",
            "Reuse the same seeds across conditions.",
            "Store certification JSON next to each run.",
            "Run analyze, compare-study, and poster-study on the final bundle.",
        ],
        "recommended_commands": recommended_commands,
    }


def render_study_protocol_markdown(payload: dict[str, Any]) -> str:
    hypotheses = payload.get("hypotheses", [])
    scenarios = payload.get("scenarios", [])
    algorithms = payload.get("algorithms", [])
    corruption_plan = payload.get("corruption_plan", [])
    commands = payload.get("recommended_commands", [])
    metrics = payload.get("metrics", [])
    statistics_plan = payload.get("statistics_plan", {})
    external_validation = payload.get("external_validation", {})

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
            "## Study Design",
            "",
            f"- Algorithms: `{', '.join(str(item) for item in algorithms)}`",
            f"- Scenarios: `{', '.join(str(item) for item in scenarios)}`",
            f"- Seeds: `{', '.join(str(item) for item in payload.get('seed_policy', {}).get('seeds', []))}`",
            "",
            "## Controlled Corruption Operators",
            "",
        ]
    )
    for item in corruption_plan:
        lines.append(f"- `{item.get('mode', '')}`: {item.get('purpose', '')}")

    lines.extend(
        [
            "",
            "## Outcome Metrics",
            "",
            "- " + "\n- ".join(str(metric) for metric in metrics),
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
    )

    markdown_path = target / "STUDY_PROTOCOL.md"
    design_json = target / "study_design.json"
    matrix_csv = target / "study_matrix.csv"

    markdown_path.write_text(render_study_protocol_markdown(payload), encoding="utf-8")
    design_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with matrix_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "algorithm", "seed", "clean_certified", "corrupted_uncertified", "supervisor_on", "supervisor_off"],
        )
        writer.writeheader()
        for scenario in payload["scenarios"]:
            for algorithm in payload["algorithms"]:
                for seed in payload["seed_policy"]["seeds"]:
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "algorithm": algorithm,
                            "seed": seed,
                            "clean_certified": "yes",
                            "corrupted_uncertified": "yes",
                            "supervisor_on": "yes",
                            "supervisor_off": "optional",
                        }
                    )

    return {
        "protocol_markdown": str(markdown_path),
        "study_design_json": str(design_json),
        "study_matrix_csv": str(matrix_csv),
    }
