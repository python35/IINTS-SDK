from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.data.realism_validator import validate_realism_dataset
from iints.highlevel import run_simulation
from iints.presets import get_preset


REFERENCE_PATIENTS = {
    "free_living_t1d": "reference_free_living_t1d",
    "azt1d_daily": "reference_azt1d_t1d",
    "hupa_ucm_daily": "reference_hupa_ucm_t1d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search free-living scenarios that stay realistic across multiple empirical references."
    )
    parser.add_argument("--base-preset", default="free_living_t1d")
    parser.add_argument(
        "--references",
        default="free_living_t1d,azt1d_daily,hupa_ucm_daily",
    )
    parser.add_argument("--seeds", default="1,42,99")
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--search-seed", type=int, default=20260516)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/scenario_search/report.json"),
    )
    parser.add_argument(
        "--best-scenario-out",
        type=Path,
        default=Path("results/scenario_search/best_multi_reference_scenario.json"),
    )
    return parser.parse_args()


def _parse_csv_text(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _standardize(results):
    return results.rename(
        columns={
            "time_minutes": "timestamp",
            "glucose_actual_mgdl": "glucose",
            "carb_intake_grams": "carbs",
            "delivered_insulin_units": "insulin",
        }
    )[["timestamp", "glucose", "carbs", "insulin"]]


def normalize_scenario(events: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: int(event["start_time"]))
    return {
        "scenario_name": "Multi-Reference Free-Living Search",
        "scenario_version": "1.0",
        "stress_events": ordered,
    }


def mutate_events(
    events: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    mutated = copy.deepcopy(events)
    meal_events = [event for event in mutated if event["event_type"] == "meal"]
    exercise_events = [event for event in mutated if event["event_type"] == "exercise"]

    for index, event in enumerate(meal_events):
        nominal = [450, 735, 1080, 1290][index]
        event["start_time"] = int(max(360, min(1320, nominal + rng.choice([-30, -15, 0, 15, 30]))))
        base_value = [42, 56, 68, 18][index]
        event["value"] = float(max(12, base_value + rng.choice([-6, -3, 0, 3, 6])))
        report_fraction = rng.choice([0.92, 0.95, 0.98, 1.0])
        event["reported_value"] = round(float(event["value"]) * report_fraction, 1)
        event["absorption_delay_minutes"] = int(rng.choice([5, 10, 15, 20]))
        event["duration"] = int(rng.choice([25, 40, 50, 60, 75, 90]))

    for event in exercise_events:
        event["start_time"] = int(rng.choice([900, 930, 960, 990, 1020]))
        event["value"] = float(rng.choice([0.1, 0.15, 0.2, 0.25]))
        event["duration"] = int(rng.choice([20, 25, 30, 35]))
    return mutated


def _row_for_run(
    *,
    scenario: dict[str, Any],
    reference: str,
    patient_config: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    outputs = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=scenario,
        patient_config=patient_config,
        duration_minutes=1440,
        time_step=5,
        seed=seed,
        output_dir=output_dir,
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )
    report = validate_realism_dataset(_standardize(outputs["results"]), reference=reference)
    return {
        "reference": reference,
        "seed": seed,
        "verdict": report.verdict,
        "realism_score": round(report.realism_score, 4),
        "failed_checks": sum(1 for check in report.checks if check.status == "failed"),
        "warning_checks": sum(1 for check in report.checks if check.status == "warning"),
    }


def summarize_candidate(
    candidate_id: str,
    scenario: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_reference: dict[str, dict[str, Any]] = {}
    for reference in sorted({str(row["reference"]) for row in rows}):
        subset = [row for row in rows if row["reference"] == reference]
        verdict_counts = Counter(str(row["verdict"]) for row in subset)
        by_reference[reference] = {
            "runs": len(subset),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "likely_realistic_runs": verdict_counts.get("likely_realistic", 0),
            "mean_realism_score": round(mean(float(row["realism_score"]) for row in subset), 4),
        }
    return {
        "candidate_id": candidate_id,
        "scenario": scenario,
        "references": by_reference,
        "min_likely_realistic_runs": min(
            payload["likely_realistic_runs"] for payload in by_reference.values()
        ),
        "total_likely_realistic_runs": sum(
            payload["likely_realistic_runs"] for payload in by_reference.values()
        ),
        "mean_realism_score": round(mean(float(row["realism_score"]) for row in rows), 4),
        "failed_checks": sum(int(row["failed_checks"]) for row in rows),
        "warning_checks": sum(int(row["warning_checks"]) for row in rows),
    }


def scenario_rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(candidate["min_likely_realistic_runs"]),
        float(candidate["total_likely_realistic_runs"]),
        float(candidate["mean_realism_score"]),
        -float(candidate["failed_checks"]),
        -float(candidate["warning_checks"]),
    )


def build_report(
    *,
    base_preset: str,
    references: list[str],
    seeds: list[int],
    population_size: int,
    generations: int,
    elite_count: int,
    search_seed: int,
) -> dict[str, Any]:
    rng = random.Random(search_seed)
    base_events = copy.deepcopy(get_preset(base_preset)["scenario"]["stress_events"])
    population = [normalize_scenario(base_events)]
    while len(population) < population_size:
        population.append(normalize_scenario(mutate_events(base_events, rng)))

    history: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="iints_scenario_search_") as tmp:
        root = Path(tmp)
        for generation in range(generations):
            summaries: list[dict[str, Any]] = []
            for index, scenario in enumerate(population):
                candidate_id = f"g{generation}_c{index}"
                rows: list[dict[str, Any]] = []
                for reference in references:
                    for seed in seeds:
                        rows.append(
                            _row_for_run(
                                scenario=scenario,
                                reference=reference,
                                patient_config=REFERENCE_PATIENTS[reference],
                                seed=seed,
                                output_dir=root / candidate_id / reference / f"seed_{seed}",
                            )
                        )
                summaries.append(summarize_candidate(candidate_id, scenario, rows))
            ranked = sorted(summaries, key=scenario_rank_key, reverse=True)
            history.extend(ranked)
            elites = ranked[: max(1, min(elite_count, len(ranked)))]
            population = [copy.deepcopy(elite["scenario"]) for elite in elites]
            while len(population) < population_size:
                parent = rng.choice(elites)["scenario"]["stress_events"]
                population.append(normalize_scenario(mutate_events(parent, rng)))

    ranked_history = sorted(history, key=scenario_rank_key, reverse=True)
    return {
        "base_preset": base_preset,
        "references": references,
        "seeds": seeds,
        "population_size": population_size,
        "generations": generations,
        "best_candidate": ranked_history[0],
        "top_candidates": ranked_history[:10],
    }


def main() -> None:
    args = parse_args()
    report = build_report(
        base_preset=args.base_preset,
        references=_parse_csv_text(args.references),
        seeds=_parse_csv_ints(args.seeds),
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        search_seed=args.search_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    args.best_scenario_out.parent.mkdir(parents=True, exist_ok=True)
    args.best_scenario_out.write_text(
        json.dumps(report["best_candidate"]["scenario"], indent=2)
    )
    print(json.dumps({"report": str(args.out), "best_scenario": str(args.best_scenario_out), "best_candidate": report["best_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
