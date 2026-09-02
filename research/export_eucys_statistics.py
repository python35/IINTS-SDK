"""Recompute every reportable EUCYS statistic from the archived run records.

The report numbers used to be typed into ``research/EUCYS_REPORT.md`` by hand,
which is how they came to describe an interval method the SDK no longer uses.
This exporter derives them from ``study_summary.json`` instead, so the report
can be refreshed from one source and any interval it prints carries the method
and the number of independent clusters it rests on.

The design is a fully crossed matrix of (profile, scenario, arm, algorithm,
seed), so runs are not independent: each profile appears in hundreds of runs.
Intervals are therefore taken over profiles, and candidate-vs-baseline
contrasts are paired within (arm, profile, scenario, seed) blocks before the
profile-level interval is computed. See ``iints.analysis.clustered_inference``.

Usage:
    PYTHONPATH=src python research/export_eucys_statistics.py \
        --results-root results/eucys_2026 \
        --out research/eucys_pack/eucys_statistics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iints.analysis.clustered_inference import MIN_CLUSTERS_FOR_INTERVAL
from iints.analysis.study_analysis import StudySummary, compare_studies, load_study_summary

REPORTED_METRICS = ("tir_70_180", "tir_below_70", "tir_below_54", "tir_above_180", "tir_above_250", "mean_glucose", "cv", "supervisor_interventions")
ARM_DIRECTORIES = {
    "clean_certified": "study_clean",
    "corrupted_uncertified": "study_corrupted",
    "supervisor_off_ablation": "study_supervisor_off",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_text(block: dict[str, Any]) -> str:
    """One cell for a markdown table, or the reason no interval was reported."""
    low, high = block.get("ci95_low"), block.get("ci95_high")
    if low is None or high is None:
        return f"not reported ({block.get('ci95_omitted_because', 'unknown reason')})"
    return f"{low:.2f} to {high:.2f}"


def _describe_design(summary: StudySummary) -> dict[str, Any]:
    runs = summary.runs
    profiles = sorted({run.profile_id for run in runs if run.profile_id})
    scenarios = sorted({run.scenario_slug or run.scenario_name for run in runs})
    arms = sorted({run.study_arm or run.condition_group for run in runs})
    algorithms = sorted({run.algorithm for run in runs})
    seeds = sorted({run.seed for run in runs if run.seed is not None})
    cells = len(profiles) * len(scenarios) * len(arms) * len(algorithms) * len(seeds)
    return {
        "run_count": len(runs),
        "profiles": profiles,
        "scenarios": scenarios,
        "arms": arms,
        "algorithms": algorithms,
        "seeds": seeds,
        "fully_crossed": cells == len(runs),
        "cells_expected_if_fully_crossed": cells,
        # The sample size that governs the interval width. More profiles buy
        # power; more seeds do not, because they are within-cluster repeats.
        "independent_clusters": len(profiles),
        "cluster_level": "profile_id",
        "minimum_clusters_for_interval": MIN_CLUSTERS_FOR_INTERVAL,
    }


def _metric_rows(grouping: dict[str, Any], metric: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label, block in grouping.items():
        stats = block.get("aggregate_stats", {}).get(metric, {})
        if not stats:
            continue
        rows[label] = {
            "n_runs": stats.get("count"),
            "mean": stats.get("mean"),
            "ci95_low": stats.get("ci95_low"),
            "ci95_high": stats.get("ci95_high"),
            "ci_method": stats.get("ci_method"),
            "n_clusters": stats.get("n_clusters"),
            "cluster_level": stats.get("cluster_level"),
            "ci95_omitted_because": stats.get("ci95_omitted_because"),
            "interval_text": _interval_text(stats),
            # Kept so a reader can see how much of the old interval width came
            # from counting runs instead of patients.
            "pseudoreplicated_ci95_half_width": stats.get("pseudoreplicated_ci95_half_width"),
        }
    return rows


CONTRAST_METRICS = ("tir_70_180", "tir_below_70", "tir_above_180", "supervisor_interventions", "mean_glucose")


def _contrast_rows(summary: StudySummary, metric: str = "tir_70_180") -> dict[str, Any]:
    deltas = summary.pairwise_baseline_deltas or {}
    rows: dict[str, Any] = {}
    for baseline, block in (deltas.get("baselines") or {}).items():
        stats = block.get("delta_stats", {}).get(metric, {})
        if not stats:
            continue
        low, high = stats.get("ci95_low"), stats.get("ci95_high")
        rows[baseline] = {
            "n_paired_runs": stats.get("count"),
            "mean_delta": stats.get("mean"),
            "ci95_low": low,
            "ci95_high": high,
            "ci_method": stats.get("ci_method"),
            "n_clusters": stats.get("n_clusters"),
            "interval_text": _interval_text(stats),
            "excludes_zero": None if low is None or high is None else bool(low > 0 or high < 0),
            "ci95_omitted_because": stats.get("ci95_omitted_because"),
        }
    return {"candidate_algorithm": deltas.get("candidate_algorithm"), "baselines": rows}


def build_statistics(results_root: Path) -> dict[str, Any]:
    full_path = results_root / "study_summary.json"
    if not full_path.is_file():
        raise FileNotFoundError(f"No study_summary.json under {results_root}")

    full = load_study_summary(full_path)
    inputs = [{"path": str(full_path), "sha256": _sha256(full_path)}]

    arms: dict[str, Any] = {}
    for arm_name, directory in ARM_DIRECTORIES.items():
        arm_path = results_root / directory / "study_summary.json"
        if not arm_path.is_file():
            continue
        arm_summary = load_study_summary(arm_path)
        inputs.append({"path": str(arm_path), "sha256": _sha256(arm_path)})
        arms[arm_name] = {
            "design": _describe_design(arm_summary),
            "by_algorithm": _metric_rows(arm_summary.by_algorithm, "tir_70_180"),
            "paired_contrasts_tir": _contrast_rows(arm_summary),
        }

    # Arm contrasts are the evidence for H1 and H2, so they are derived here
    # rather than quoted from the stored comparison files.
    arm_contrasts: dict[str, Any] = {}
    clean_path = results_root / ARM_DIRECTORIES["clean_certified"] / "study_summary.json"
    for other in ("corrupted_uncertified", "supervisor_off_ablation"):
        other_path = results_root / ARM_DIRECTORIES[other] / "study_summary.json"
        if not (clean_path.is_file() and other_path.is_file()):
            continue
        comparison = compare_studies(
            clean_path, other_path, left_label="clean_certified", right_label=other
        ).to_dict()
        rows = {}
        for metric, block in (comparison.get("effect_estimates") or {}).items():
            if not isinstance(block, dict):
                continue
            low, high = block.get("ci95_low"), block.get("ci95_high")
            rows[metric] = {
                "difference_in_means": block.get("difference_in_means"),
                "ci95_low": low,
                "ci95_high": high,
                "ci_method": block.get("ci_method"),
                "n_clusters": block.get("n_clusters"),
                "interval_text": _interval_text(block),
                "excludes_zero": None if low is None or high is None else bool(low > 0 or high < 0),
                "ci95_omitted_because": block.get("ci95_omitted_because"),
            }
        arm_contrasts[f"clean_certified_vs_{other}"] = rows

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "arm_contrasts": arm_contrasts,
        "method": {
            "interval": "cluster-level t interval over virtual-patient profiles",
            "contrast": "paired within (arm, profile_id, scenario, seed) blocks, then a profile-level t interval over the per-profile mean differences",
            "rationale": "runs sharing a profile are repeated measurements of one virtual patient; an interval over runs would be pseudo-replicated",
            "multiplicity": "none applied; intervals are reported without a family-wise correction, so a ranking across five algorithms should not be read as a sequence of tests",
        },
        "design": _describe_design(full),
        "full_bundle": {
            "aggregate": {metric: _metric_rows({"all": {"aggregate_stats": full.aggregate_stats}}, metric).get("all") for metric in REPORTED_METRICS},
            "by_algorithm": _metric_rows(full.by_algorithm, "tir_70_180"),
            "by_arm": _metric_rows(full.by_arm, "tir_70_180"),
            "paired_contrasts_tir": _contrast_rows(full),
            "paired_contrasts_by_metric": {
                metric: _contrast_rows(full, metric)["baselines"] for metric in CONTRAST_METRICS
            },
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/eucys_2026"))
    parser.add_argument("--out", type=Path, default=Path("research/eucys_pack/eucys_statistics.json"))
    args = parser.parse_args()

    payload = build_statistics(args.results_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    design = payload["design"]
    print(f"runs={design['run_count']} clusters={design['independent_clusters']} fully_crossed={design['fully_crossed']}")
    for baseline, row in payload["full_bundle"]["paired_contrasts_tir"]["baselines"].items():
        print(f"{baseline:20} delta={row['mean_delta']:+.3f} pp  ci={row['interval_text']}  excludes_zero={row['excludes_zero']}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
