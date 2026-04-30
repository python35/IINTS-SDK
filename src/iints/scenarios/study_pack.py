from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from iints.utils.csv_safety import sanitize_csv_mapping


def _official_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "slug": "baseline_day",
            "label": "Baseline Day",
            "recommended_duration_minutes": 1440,
            "scenario": {
                "scenario_name": "Official Study Pack - Baseline Day",
                "schema_version": "1.1",
                "scenario_version": "1.1",
                "description": "Baseline day with breakfast, lunch, dinner, and a small evening snack for reproducible full-day benchmarking.",
                "stress_events": [
                    {"start_time": 450, "event_type": "meal", "value": 44, "reported_value": 44, "duration": 45, "absorption_delay_minutes": 10},
                    {"start_time": 735, "event_type": "meal", "value": 62, "reported_value": 62, "duration": 75, "absorption_delay_minutes": 15},
                    {"start_time": 765, "event_type": "exercise", "value": 0.25, "duration": 25},
                    {"start_time": 1095, "event_type": "meal", "value": 78, "reported_value": 78, "duration": 90, "absorption_delay_minutes": 20},
                    {"start_time": 1290, "event_type": "meal", "value": 18, "reported_value": 18, "duration": 30, "absorption_delay_minutes": 5},
                ],
            },
        },
        {
            "slug": "meal_challenge",
            "label": "Meal Challenge",
            "recommended_duration_minutes": 1440,
            "scenario": {
                "scenario_name": "Official Study Pack - Meal Challenge",
                "schema_version": "1.1",
                "scenario_version": "1.1",
                "description": "Larger mixed meals with a heavy lunch and delayed dinner absorption to stress post-prandial control.",
                "stress_events": [
                    {"start_time": 435, "event_type": "meal", "value": 50, "reported_value": 50, "duration": 50, "absorption_delay_minutes": 10},
                    {"start_time": 720, "event_type": "meal", "value": 96, "reported_value": 90, "duration": 105, "absorption_delay_minutes": 20},
                    {"start_time": 930, "event_type": "meal", "value": 22, "reported_value": 22, "duration": 25, "absorption_delay_minutes": 5},
                    {"start_time": 1110, "event_type": "meal", "value": 108, "reported_value": 96, "duration": 135, "absorption_delay_minutes": 35},
                    {"start_time": 1260, "event_type": "meal", "value": 26, "reported_value": 20, "duration": 35, "absorption_delay_minutes": 10},
                ],
            },
        },
        {
            "slug": "exercise_challenge",
            "label": "Exercise Challenge",
            "recommended_duration_minutes": 1440,
            "scenario": {
                "scenario_name": "Official Study Pack - Exercise Challenge",
                "schema_version": "1.1",
                "scenario_version": "1.1",
                "description": "Exercise disturbance layered into an otherwise normal day to probe falling-glucose safety behavior.",
                "stress_events": [
                    {"start_time": 450, "event_type": "meal", "value": 46, "reported_value": 46, "duration": 45, "absorption_delay_minutes": 10},
                    {"start_time": 720, "event_type": "meal", "value": 64, "reported_value": 64, "duration": 70, "absorption_delay_minutes": 15},
                    {"start_time": 975, "event_type": "exercise", "value": 0.55, "duration": 60},
                    {"start_time": 1050, "event_type": "meal", "value": 24, "reported_value": 24, "duration": 30, "absorption_delay_minutes": 5},
                    {"start_time": 1140, "event_type": "meal", "value": 74, "reported_value": 74, "duration": 85, "absorption_delay_minutes": 15},
                ],
            },
        },
        {
            "slug": "supervisor_override",
            "label": "Supervisor Override",
            "recommended_duration_minutes": 1440,
            "scenario": {
                "scenario_name": "Official Study Pack - Supervisor Override",
                "schema_version": "1.1",
                "scenario_version": "1.1",
                "description": "Undercounted meals, activity, and a later sensor fault combine to reveal unsafe dosing pressure and supervisor actions.",
                "stress_events": [
                    {"start_time": 450, "event_type": "meal", "value": 48, "reported_value": 48, "duration": 45, "absorption_delay_minutes": 10},
                    {"start_time": 720, "event_type": "meal", "value": 104, "reported_value": 56, "duration": 105, "absorption_delay_minutes": 20},
                    {"start_time": 760, "event_type": "exercise", "value": 0.45, "duration": 40},
                    {"start_time": 1080, "event_type": "meal", "value": 92, "reported_value": 60, "duration": 95, "absorption_delay_minutes": 20},
                    {"start_time": 1160, "event_type": "sensor_error", "value": 190},
                    {"start_time": 1290, "event_type": "meal", "value": 14, "reported_value": 14, "duration": 20, "absorption_delay_minutes": 5},
                ],
            },
        },
    ]


def build_official_study_pack(*, seeds: list[int] | None = None) -> dict[str, Any]:
    resolved_seeds = seeds or list(range(1, 11))
    return {
        "name": "iints_official_study_pack",
        "version": "1.1",
        "preset": "official",
        "seeds": resolved_seeds,
        "scenarios": _official_scenarios(),
    }


def build_eucys_study_pack(*, seeds: list[int] | None = None) -> dict[str, Any]:
    resolved_seeds = seeds or list(range(1, 11))
    scenarios = _official_scenarios()
    study_arms = [
        {
            "arm_id": "clean_certified",
            "label": "Clean Certified",
            "data_condition": "clean",
            "expected_certification": "research_grade",
            "supervisor_enabled": True,
            "corruption_modes": [],
        },
        {
            "arm_id": "corrupted_uncertified",
            "label": "Corrupted Uncertified",
            "data_condition": "corrupted",
            "expected_certification": "uncertified",
            "supervisor_enabled": True,
            "corruption_modes": ["timestamp_shift", "missing_block", "glucose_spikes"],
        },
        {
            "arm_id": "supervisor_off_ablation",
            "label": "Supervisor Off Ablation",
            "data_condition": "clean",
            "expected_certification": "research_grade",
            "supervisor_enabled": False,
            "corruption_modes": [],
        },
    ]
    matrix_rows = [
        {
            "scenario_slug": scenario["slug"],
            "scenario_label": scenario["label"],
            "arm_id": arm["arm_id"],
            "data_condition": arm["data_condition"],
            "expected_certification": arm["expected_certification"],
            "supervisor_enabled": arm["supervisor_enabled"],
            "corruption_modes": arm["corruption_modes"],
            "recommended_duration_minutes": scenario["recommended_duration_minutes"],
            "seeds": resolved_seeds,
        }
        for scenario in scenarios
        for arm in study_arms
    ]
    return {
        "name": "iints_eucys_study_pack",
        "version": "1.1",
        "preset": "eucys",
        "primary_claim": "Certified data improves the reliability of closed-loop insulin algorithm evaluation.",
        "recommended_algorithms": ["your_algorithm", "Standard PID", "Standard Pump"],
        "seeds": resolved_seeds,
        "scenarios": scenarios,
        "study_arms": study_arms,
        "matrix_rows": matrix_rows,
    }


def build_eucys_arm_scenario(base_scenario: dict[str, Any], *, arm_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = deepcopy(base_scenario)
    stress_events = list(scenario.get("stress_events", []))
    metadata: dict[str, Any] = {"arm_id": arm_id, "operations": []}

    if arm_id == "clean_certified":
        scenario["scenario_name"] = f"{scenario.get('scenario_name', 'Scenario')} [Clean Certified]"
        return scenario, metadata

    if arm_id == "supervisor_off_ablation":
        scenario["scenario_name"] = f"{scenario.get('scenario_name', 'Scenario')} [Supervisor Off]"
        metadata["operations"].append({"mode": "supervisor_off", "applied": True})
        return scenario, metadata

    if arm_id != "corrupted_uncertified":
        raise ValueError(f"Unknown EUCYS study arm: {arm_id}")

    corrupted_events: list[dict[str, Any]] = []
    for idx, event in enumerate(stress_events):
        mutated = dict(event)
        if "start_time" in mutated:
            mutated["start_time"] = int(mutated["start_time"]) + 60
        if idx == 0 and mutated.get("event_type") == "meal":
            raw_reported = mutated.get("reported_value", mutated.get("value", 0))
            if isinstance(raw_reported, (int, float, str)):
                reported_value = float(raw_reported)
            else:
                reported_value = 0.0
            mutated["reported_value"] = max(0.0, reported_value - 20.0)
        corrupted_events.append(mutated)

    if stress_events:
        duplicate = dict(stress_events[0])
        if "start_time" in duplicate:
            duplicate["start_time"] = int(duplicate["start_time"]) + 30
        corrupted_events.append(duplicate)
        metadata["operations"].append({"mode": "duplicate_event", "applied": True})

    corrupted_events.append({"start_time": 300, "event_type": "sensor_error", "value": 220})
    metadata["operations"].extend(
        [
            {"mode": "timestamp_shift", "applied": True, "minutes": 60},
            {"mode": "meal_annotation_mismatch", "applied": True},
            {"mode": "sensor_error_injection", "applied": True},
        ]
    )

    scenario["stress_events"] = corrupted_events
    scenario["scenario_name"] = f"{scenario.get('scenario_name', 'Scenario')} [Corrupted Uncertified]"
    return scenario, metadata


def export_official_study_pack(output_dir: str | Path, *, seeds: list[int] | None = None) -> dict[str, str]:
    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    pack = build_official_study_pack(seeds=seeds)

    for scenario in pack["scenarios"]:
        scenario_path = resolved / f"{scenario['slug']}.json"
        scenario_path.write_text(json.dumps(scenario["scenario"], indent=2), encoding="utf-8")

    manifest_path = resolved / "study_pack_manifest.json"
    manifest_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    readme_path = resolved / "README.md"
    readme_path.write_text(
        "# IINTS Official Study Pack\n\n"
        "This folder contains a reusable set of study scenarios plus a recommended seed list.\n\n"
        "Suggested loop:\n\n"
        "```bash\n"
        "for seed in 1 2 3 4 5 6 7 8 9 10; do\n"
        "  iints run-full --algo algorithms/example_algorithm.py --scenario-path scenarios/study_pack/baseline_day.json --seed \"$seed\" --duration 1440 --output-dir \"results/study/run_$seed\"\n"
        "done\n"
        "```\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(resolved),
        "manifest_json": str(manifest_path),
        "readme": str(readme_path),
    }


def export_eucys_study_pack(output_dir: str | Path, *, seeds: list[int] | None = None) -> dict[str, str]:
    resolved = Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    pack = build_eucys_study_pack(seeds=seeds)

    for scenario in pack["scenarios"]:
        scenario_path = resolved / f"{scenario['slug']}.json"
        scenario_path.write_text(json.dumps(scenario["scenario"], indent=2), encoding="utf-8")

    manifest_path = resolved / "study_pack_manifest.json"
    manifest_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    matrix_path = resolved / "eucys_study_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_slug",
                "scenario_label",
                "arm_id",
                "data_condition",
                "expected_certification",
                "supervisor_enabled",
                "corruption_modes",
                "recommended_duration_minutes",
                "seeds",
            ],
        )
        writer.writeheader()
        for row in pack["matrix_rows"]:
            writer.writerow(
                sanitize_csv_mapping(
                {
                    **row,
                    "corruption_modes": ",".join(row["corruption_modes"]),
                    "seeds": ",".join(str(seed) for seed in row["seeds"]),
                }
                )
            )

    readme_path = resolved / "README.md"
    readme_path.write_text(
        "# IINTS EUCYS Study Pack\n\n"
        "This preset is designed for a scientific fair or jury conversation.\n\n"
        "It fixes:\n\n"
        "- one shared seed list\n"
        "- one shared scenario set\n"
        "- three study arms: clean certified, corrupted uncertified, and supervisor-off ablation\n\n"
        "Recommended flow:\n\n"
        "```bash\n"
        "iints study-protocol --preset eucys --output-dir results/study_protocol\n"
        "iints data corrupt-for-study data/demo/diabetes_cgm.csv --output-csv data/demo/diabetes_cgm_corrupted.csv --mode timestamp_shift --mode missing_block --mode glucose_spikes\n"
        "iints analyze results/study_clean --output-json results/study_clean/study_summary.json\n"
        "iints compare-study results/study_clean results/study_corrupted --output-json results/study_comparison.json\n"
        "```\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(resolved),
        "manifest_json": str(manifest_path),
        "matrix_csv": str(matrix_path),
        "readme": str(readme_path),
    }
