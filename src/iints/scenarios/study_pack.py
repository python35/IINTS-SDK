from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def _official_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "slug": "baseline_day",
            "label": "Baseline Day",
            "recommended_duration_minutes": 1440,
            "scenario": {
                "scenario_name": "Official Study Pack - Baseline Day",
                "schema_version": "1.1",
                "scenario_version": "1.0",
                "description": "Baseline day for reproducible controller benchmarking.",
                "stress_events": [
                    {"start_time": 60, "event_type": "meal", "value": 40, "reported_value": 40, "duration": 50},
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
                "scenario_version": "1.0",
                "description": "Larger mixed meals to stress control behavior.",
                "stress_events": [
                    {"start_time": 45, "event_type": "meal", "value": 75, "reported_value": 75, "duration": 75},
                    {"start_time": 180, "event_type": "meal", "value": 55, "reported_value": 55, "duration": 55},
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
                "scenario_version": "1.0",
                "description": "Exercise disturbance to probe falling-glucose safety behavior.",
                "stress_events": [
                    {"start_time": 120, "event_type": "exercise", "value": 0.6, "duration": 50},
                    {"start_time": 165, "event_type": "meal", "value": 30, "reported_value": 30, "duration": 45},
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
                "scenario_version": "1.0",
                "description": "A scenario intended to reveal unsafe dosing pressure and supervisor actions.",
                "stress_events": [
                    {"start_time": 30, "event_type": "meal", "value": 60, "reported_value": 60, "duration": 60},
                    {"start_time": 120, "event_type": "exercise", "value": 0.8, "duration": 60},
                    {"start_time": 220, "event_type": "sensor_error", "value": 180},
                ],
            },
        },
    ]


def build_official_study_pack(*, seeds: list[int] | None = None) -> dict[str, Any]:
    resolved_seeds = seeds or list(range(1, 11))
    return {
        "name": "iints_official_study_pack",
        "version": "1.0",
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
        "version": "1.0",
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
                {
                    **row,
                    "corruption_modes": ",".join(row["corruption_modes"]),
                    "seeds": ",".join(str(seed) for seed in row["seeds"]),
                }
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
