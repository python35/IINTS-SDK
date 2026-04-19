from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from iints.utils.csv_safety import sanitize_csv_dataframe

EUCYS_ARM_LAYOUT = {
    "clean_certified": "study_clean",
    "corrupted_uncertified": "study_corrupted",
    "supervisor_off_ablation": "study_supervisor_off",
}


def build_eucys_limitations_and_ethics_markdown() -> str:
    return "\n".join(
        [
            "# Limitations And Ethics",
            "",
            "IINTS-AF is a research and benchmarking platform, not a medical device.",
            "",
            "## What this work supports",
            "",
            "- Reproducible simulation-first benchmarking",
            "- Baseline-vs-candidate comparison under shared protocols",
            "- Safety-layer and supervisor-ablation analysis",
            "- Uncertainty and calibration reporting when predictor signals are available",
            "- Local, edge-ready demonstrations that stay aligned with the study language",
            "",
            "## What this work does not support",
            "",
            "- Real-world dosing advice",
            "- Clinical deployment claims",
            "- Replacement of medical supervision",
            "- Direct proof of patient benefit",
            "",
            "## Main limitations",
            "",
            "- The patient profiles are simulated abstractions rather than real patients.",
            "- Study outcomes depend on the selected profiles, scenario families, and baseline definitions.",
            "- Public datasets can support plausibility checks, but they do not convert the benchmark into a clinical validation study.",
            "- Safety conclusions are relative to the encoded safety supervisor and disturbance models.",
            "- Controller ranking can change when assumptions, populations, or features change.",
            "",
            "## Ethics Position",
            "",
            "- The platform is designed to reduce unsafe experimentation by moving evaluation earlier into simulation.",
            "- The project should be presented as decision-system research, not as a treatment claim.",
            "- Stronger glucose metrics should always be interpreted together with safety outcomes and uncertainty.",
            "- The honest scientific contribution is improved evaluation quality, not proof of clinical readiness.",
        ]
    )


def build_eucys_abstract_draft_markdown() -> str:
    return "\n".join(
        [
            "# EUCYS Abstract Draft",
            "",
            "Use this as the competition-facing draft that stays aligned with the benchmark bundle.",
            "Replace the bracketed placeholders only after the final benchmark bundle is ready.",
            "",
            "Type 1 diabetes care increasingly depends on software that interprets glucose measurements and helps determine insulin delivery. "
            "However, insulin-decision AI is difficult to evaluate responsibly: good glucose metrics alone are not enough, because safety, explainability, and reproducibility matter just as much. "
            "In this project, I developed **IINTS-AF**, a local, simulation-first evaluation platform for benchmarking insulin-decision algorithms under fixed study protocols.",
            "",
            "The central research question is whether a transparent, safety-first benchmark can evaluate candidate insulin-decision algorithms more rigorously than isolated demo runs or single-metric comparisons. "
            "The platform combines virtual patient simulation, baseline comparison, safety-layer analysis, protocol bundles, subgroup summaries, and exportable study artifacts. "
            "Instead of showing one favorable run, IINTS-AF compares a candidate algorithm against classical baselines such as `PID Controller`, `Standard Pump`, and `Correction Bolus` across predefined patient profiles, scenario families, and random seeds.",
            "",
            "In the final benchmark bundle, the platform evaluated **[RUN_COUNT]** runs across **[PROFILE_COUNT]** patient profiles, **[SCENARIO_COUNT]** scenario families, and **[ALGORITHM_COUNT]** algorithms. "
            "The candidate algorithm achieved **[CANDIDATE_TIR]%** time in range compared with **[BASELINE_TIR]%** for the strongest baseline, while the safety analysis showed **[SAFETY_RESULT]**. "
            "Together, these results support the claim that insulin-decision AI should be judged through safety-aware, reproducible benchmarking rather than through isolated performance figures alone.",
            "",
            "IINTS-AF is **not** a medical device and is **not** intended for clinical dosing. "
            "Its contribution is a transparent and reproducible evaluation framework that helps researchers test AI-guided insulin decision systems more rigorously before any real-world use is considered.",
        ]
    )


def build_eucys_poster_outline_markdown() -> str:
    return "\n".join(
        [
            "# EUCYS Poster Outline",
            "",
            "## Required Sections",
            "",
            "1. Problem",
            "2. Research question",
            "3. Architecture: sensor -> forecast/controller -> safety gate -> final action",
            "4. Study design: profiles, arms, seeds, baselines",
            "5. Main figure: candidate vs baselines on TIR, <70, >180, interventions",
            "6. Core results table",
            "7. Safety / ablation block",
            "8. Reproducibility block",
            "9. Limitations and ethics",
            "10. Conclusion",
            "",
            "## One-Sentence Rule",
            "",
            "The poster should make it obvious that the main contribution is a transparent, safety-first benchmark platform, not a clinical dosing product.",
        ]
    )


def build_eucys_jury_qa_markdown() -> str:
    return "\n".join(
        [
            "# EUCYS Jury Q&A",
            "",
            "## What is the main contribution?",
            "",
            "A transparent and reproducible benchmark platform for evaluating AI-guided insulin decision systems under safety-focused study conditions.",
            "",
            "## Is this a medical device?",
            "",
            "No. It is research software for simulation-first evaluation, not clinical dosing.",
            "",
            "## Why simulation first?",
            "",
            "Because unsafe systems should be filtered out before any real-world exposure is considered.",
            "",
            "## Why compare to classical baselines?",
            "",
            "Because a new method only matters if it is compared fairly against reasonable existing strategies.",
            "",
            "## What are the biggest limitations?",
            "",
            "Simulated patients are abstractions, dataset-based plausibility is not clinical proof, and benchmark conclusions depend on the selected scenarios and baselines.",
        ]
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any, suffix: str = "", digits: int = 1) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}{suffix}"


def _extract_protocol_counts(study_design: dict[str, Any]) -> dict[str, int]:
    return {
        "profile_count": len(study_design.get("profiles", [])) if isinstance(study_design.get("profiles"), list) else 0,
        "scenario_count": len(study_design.get("scenarios", [])) if isinstance(study_design.get("scenarios"), list) else 0,
        "algorithm_count": len(study_design.get("algorithms", [])) if isinstance(study_design.get("algorithms"), list) else 0,
    }


def _algorithm_aggregate(payload: dict[str, Any], algorithm: str) -> dict[str, Any]:
    by_algorithm = payload.get("by_algorithm", {})
    if not isinstance(by_algorithm, dict):
        return {}
    group = by_algorithm.get(algorithm, {})
    if not isinstance(group, dict):
        return {}
    aggregate = group.get("aggregate", {})
    return aggregate if isinstance(aggregate, dict) else {}


def _candidate_and_baseline_snapshot(clean_summary: dict[str, Any]) -> dict[str, Any]:
    pairwise = clean_summary.get("pairwise_baseline_deltas", {})
    baselines = pairwise.get("baselines", {}) if isinstance(pairwise, dict) else {}
    candidate = pairwise.get("candidate_algorithm") if isinstance(pairwise, dict) else None
    if not isinstance(candidate, str) or not candidate:
        by_algorithm = clean_summary.get("by_algorithm", {})
        if isinstance(by_algorithm, dict) and by_algorithm:
            candidate = next(iter(by_algorithm.keys()))
    baseline_names = list(baselines.keys()) if isinstance(baselines, dict) else []
    baseline_tir_rows = [
        (
            baseline_name,
            _as_float(_algorithm_aggregate(clean_summary, baseline_name).get("mean_tir_70_180")),
        )
        for baseline_name in baseline_names
    ]
    filtered_baseline_tir_rows: list[tuple[str, float]] = [
        (baseline_name, tir)
        for baseline_name, tir in baseline_tir_rows
        if tir is not None
    ]
    strongest_baseline = max(filtered_baseline_tir_rows, key=lambda item: item[1])[0] if filtered_baseline_tir_rows else None

    candidate_aggregate = _algorithm_aggregate(clean_summary, candidate) if candidate else {}
    strongest_baseline_aggregate = _algorithm_aggregate(clean_summary, strongest_baseline) if strongest_baseline else {}

    return {
        "candidate_algorithm": candidate,
        "strongest_baseline": strongest_baseline,
        "candidate_tir": _as_float(candidate_aggregate.get("mean_tir_70_180")),
        "strongest_baseline_tir": _as_float(strongest_baseline_aggregate.get("mean_tir_70_180")),
        "candidate_interventions": _as_float(candidate_aggregate.get("mean_supervisor_interventions")),
    }


def build_eucys_filled_abstract_markdown(
    *,
    root_summary: dict[str, Any],
    clean_summary: dict[str, Any],
    study_design: dict[str, Any],
) -> str:
    counts = _extract_protocol_counts(study_design)
    snapshot = _candidate_and_baseline_snapshot(clean_summary)
    clean_safety = clean_summary.get("safety_summary", {}) if isinstance(clean_summary.get("safety_summary"), dict) else {}

    run_count = root_summary.get("run_count", clean_summary.get("run_count", "n/a"))
    candidate_name = snapshot.get("candidate_algorithm") or "the candidate algorithm"
    strongest_baseline = snapshot.get("strongest_baseline") or "the strongest baseline"
    candidate_tir = _format_metric(snapshot.get("candidate_tir"), "%")
    baseline_tir = _format_metric(snapshot.get("strongest_baseline_tir"), "%")
    severe_hypo = int(clean_safety.get("severe_hypo_run_count", 0) or 0)
    candidate_interventions = _format_metric(snapshot.get("candidate_interventions"))
    safety_result = (
        f"{candidate_interventions} mean supervisor interventions in the clean certified arm and {severe_hypo} severe-hypo-equivalent run(s) across that arm"
    )

    return "\n".join(
        [
            "# EUCYS Abstract (Filled From Results Bundle)",
            "",
            "Type 1 diabetes care increasingly depends on software that interprets glucose measurements and helps determine insulin delivery. "
            "However, AI-based decision systems must be evaluated not only for glucose performance, but also for safety, explainability, and reproducibility. "
            "In this project, I developed **IINTS-AF**, a simulation-first benchmark platform for testing insulin-decision algorithms under fixed study protocols.",
            "",
            "The platform combines virtual patient simulation, baseline comparison, safety-layer analysis, protocol bundles, subgroup summaries, and exportable study artifacts. "
            "Instead of showing a single favorable run, IINTS-AF compares a candidate algorithm against classical baselines such as `PID Controller`, `Standard Pump`, and `Correction Bolus` across predefined profiles, scenario families, and random seeds.",
            "",
            f"In the current benchmark bundle, the platform evaluated **{run_count}** runs across **{counts['profile_count']}** patient profiles, **{counts['scenario_count']}** scenario families, and **{counts['algorithm_count']}** algorithms. "
            f"In the clean certified arm, **{candidate_name}** achieved **{candidate_tir}** time in range compared with **{baseline_tir}** for **{strongest_baseline}**, while the safety layer recorded **{safety_result}**. "
            "These results show that insulin-decision AI should be interpreted through safety-aware, reproducible benchmarking rather than through isolated performance figures alone.",
            "",
            "IINTS-AF is **not** a medical device and is **not** intended for clinical dosing. "
            "Its contribution is a transparent and reproducible evaluation framework for safer preclinical research on AI-guided insulin decision systems.",
        ]
    )


def generate_eucys_main_figure(
    clean_summary: dict[str, Any],
    *,
    output_path: str | Path,
    csv_output_path: str | Path | None = None,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError(
            "generate_eucys_main_figure() requires the optional reporting stack. "
            "Install 'iints-sdk-python35[reports]' or 'iints-sdk-python35[full]'."
        ) from exc

    by_algorithm = clean_summary.get("by_algorithm", {})
    if not isinstance(by_algorithm, dict) or not by_algorithm:
        raise ValueError("The clean certified summary does not contain by_algorithm data for the EUCYS main figure.")

    snapshot = _candidate_and_baseline_snapshot(clean_summary)
    candidate = snapshot.get("candidate_algorithm")
    pairwise = clean_summary.get("pairwise_baseline_deltas", {})
    baseline_names = list((pairwise.get("baselines") or {}).keys()) if isinstance(pairwise, dict) else []
    algorithm_order = [name for name in [candidate, *baseline_names] if isinstance(name, str) and name in by_algorithm]
    if not algorithm_order:
        algorithm_order = list(by_algorithm.keys())

    metric_specs = [
        ("mean_tir_70_180", "TIR 70-180 (%)"),
        ("mean_tir_below_70", "Time <70 (%)"),
        ("mean_tir_above_180", "Time >180 (%)"),
        ("mean_supervisor_interventions", "Safety interventions"),
    ]

    rows: list[dict[str, Any]] = []
    for algorithm in algorithm_order:
        aggregate = _algorithm_aggregate(clean_summary, algorithm)
        is_candidate = algorithm == candidate
        for metric_key, metric_label in metric_specs:
            rows.append(
                {
                    "algorithm": algorithm,
                    "is_candidate": is_candidate,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "value": _as_float(aggregate.get(metric_key)),
                }
            )

    figure_df = pd.DataFrame(rows)
    if csv_output_path is not None:
        csv_path = Path(csv_output_path).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        sanitize_csv_dataframe(figure_df).to_csv(csv_path, index=False)
    else:
        csv_path = None

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#1d5fa7" if algorithm == candidate else "#c59b2a" for algorithm in algorithm_order]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    axes_flat = list(axes.flatten())
    for ax, (metric_key, metric_label) in zip(axes_flat, metric_specs):
        values = [
            _as_float(_algorithm_aggregate(clean_summary, algorithm).get(metric_key)) or 0.0
            for algorithm in algorithm_order
        ]
        ax.bar(range(len(algorithm_order)), values, color=colors)
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xticks(range(len(algorithm_order)))
        ax.set_xticklabels(algorithm_order, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.2)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("EUCYS Main Figure - Candidate vs Baselines (Clean Certified Arm)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    result = {"main_figure_png": str(output)}
    if csv_path is not None:
        result["main_figure_csv"] = str(csv_path)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _copy_many(sources: list[Path], destination_dir: Path) -> list[str]:
    copied: list[str] = []
    for source in sources:
        copied_path = _copy_if_exists(source, destination_dir / source.name)
        if copied_path is not None:
            copied.append(copied_path)
    return copied


def generate_eucys_results_bundle(
    study_root: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    root = Path(study_root).expanduser().resolve()
    bundle_root = Path(output_dir).expanduser().resolve() if output_dir is not None else root / "EUCYS_RESULTS"
    bundle_root.mkdir(parents=True, exist_ok=True)

    protocol_dir = root / "protocol"
    comparisons_dir = root / "comparisons"
    root_summary_json = root / "study_summary.json"

    if not protocol_dir.is_dir():
        raise FileNotFoundError(f"Could not find protocol directory under {root}")

    arm_payloads: dict[str, dict[str, Any]] = {}
    arm_artifacts: dict[str, dict[str, str]] = {}
    table_rows: list[dict[str, Any]] = []

    for arm_id, source_dir_name in EUCYS_ARM_LAYOUT.items():
        source_dir = root / source_dir_name
        summary_json = source_dir / "study_summary.json"
        if not summary_json.is_file():
            raise FileNotFoundError(f"Missing required study summary: {summary_json}")

        payload = _read_json(summary_json)
        arm_payloads[arm_id] = payload

        arm_dest = bundle_root / "arms" / arm_id
        artifacts = {
            "summary_json": _copy_if_exists(summary_json, arm_dest / "study_summary.json"),
            "summary_md": _copy_if_exists(source_dir / "study_summary.md", arm_dest / "study_summary.md"),
            "evidence_csv": _copy_if_exists(source_dir / "evidence_table.csv", arm_dest / "evidence_table.csv"),
            "evidence_md": _copy_if_exists(source_dir / "evidence_table.md", arm_dest / "evidence_table.md"),
            "poster_png": _copy_if_exists(source_dir / "study_poster.png", arm_dest / "study_poster.png"),
            "poster_summary_json": _copy_if_exists(source_dir / "study_poster.json", arm_dest / "study_poster.json"),
        }
        arm_artifacts[arm_id] = {key: value for key, value in artifacts.items() if value is not None}

        aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
        safety = payload.get("safety_summary", {}) if isinstance(payload.get("safety_summary"), dict) else {}
        table_rows.append(
            {
                "arm_id": arm_id,
                "run_count": payload.get("run_count"),
                "mean_tir_70_180": aggregate.get("mean_tir_70_180"),
                "mean_tir_below_70": aggregate.get("mean_tir_below_70"),
                "mean_tir_above_180": aggregate.get("mean_tir_above_180"),
                "mean_glucose": aggregate.get("mean_glucose"),
                "mean_supervisor_interventions": aggregate.get("mean_supervisor_interventions"),
                "severe_hypo_run_count": safety.get("severe_hypo_run_count"),
                "terminated_early_run_count": safety.get("terminated_early_run_count"),
                "poster_png": arm_artifacts[arm_id].get("poster_png"),
            }
        )

    protocol_dest = bundle_root / "protocol"
    study_design_json = protocol_dir / "study_design.json"
    protocol_files = {
        "protocol_markdown": _copy_if_exists(protocol_dir / "STUDY_PROTOCOL.md", protocol_dest / "STUDY_PROTOCOL.md"),
        "study_design_json": _copy_if_exists(study_design_json, protocol_dest / "study_design.json"),
        "study_matrix_csv": _copy_if_exists(protocol_dir / "study_matrix.csv", protocol_dest / "study_matrix.csv"),
        "algorithms_json": _copy_if_exists(protocol_dir / "algorithms.json", protocol_dest / "algorithms.json"),
    }

    comparison_dest = bundle_root / "comparisons"
    comparison_files = _copy_many(sorted(comparisons_dir.glob("*")) if comparisons_dir.is_dir() else [], comparison_dest)

    root_summary_payload = _read_json(root_summary_json) if root_summary_json.is_file() else {}
    root_summary_copy = _copy_if_exists(root_summary_json, bundle_root / "study_summary.json")
    study_design_payload = _read_json(study_design_json) if study_design_json.is_file() else {}
    table_path = bundle_root / "EUCYS_RESULTS_TABLE.csv"
    sanitize_csv_dataframe(pd.DataFrame(table_rows)).to_csv(table_path, index=False)

    summary_lines = [
        "# EUCYS Results Bundle",
        "",
        f"- Source study root: `{root}`",
        f"- Results table: `{table_path.name}`",
        f"- Protocol markdown: `{Path(protocol_files['protocol_markdown']).name if protocol_files['protocol_markdown'] else 'missing'}`",
        f"- Comparison artifact count: `{len(comparison_files)}`",
        "",
        "## Included Arms",
        "",
    ]
    for row in table_rows:
        summary_lines.extend(
            [
                f"### {row['arm_id']}",
                f"- Run count: `{row['run_count']}`",
                f"- Mean TIR 70-180: `{row['mean_tir_70_180']}`",
                f"- Mean TIR <70: `{row['mean_tir_below_70']}`",
                f"- Mean TIR >180: `{row['mean_tir_above_180']}`",
                f"- Mean glucose: `{row['mean_glucose']}`",
                f"- Mean interventions: `{row['mean_supervisor_interventions']}`",
                f"- Severe hypo runs: `{row['severe_hypo_run_count']}`",
                f"- Early terminations: `{row['terminated_early_run_count']}`",
                "",
            ]
        )
    summary_path = bundle_root / "EUCYS_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    limitations_path = bundle_root / "EUCYS_LIMITATIONS.md"
    limitations_path.write_text(build_eucys_limitations_and_ethics_markdown(), encoding="utf-8")

    abstract_path = bundle_root / "EUCYS_ABSTRACT_DRAFT.md"
    abstract_path.write_text(build_eucys_abstract_draft_markdown(), encoding="utf-8")

    abstract_filled_path = bundle_root / "EUCYS_ABSTRACT_FILLED.md"
    abstract_filled_path.write_text(
        build_eucys_filled_abstract_markdown(
            root_summary=root_summary_payload,
            clean_summary=arm_payloads["clean_certified"],
            study_design=study_design_payload,
        ),
        encoding="utf-8",
    )

    poster_outline_path = bundle_root / "EUCYS_POSTER_OUTLINE.md"
    poster_outline_path.write_text(build_eucys_poster_outline_markdown(), encoding="utf-8")

    jury_qa_path = bundle_root / "EUCYS_JURY_QA.md"
    jury_qa_path.write_text(build_eucys_jury_qa_markdown(), encoding="utf-8")

    figure_outputs = generate_eucys_main_figure(
        arm_payloads["clean_certified"],
        output_path=bundle_root / "EUCYS_MAIN_FIGURE.png",
        csv_output_path=bundle_root / "EUCYS_MAIN_FIGURE.csv",
    )

    figure_manifest = {
        "bundle_kind": "eucys_results",
        "source_study_root": str(root),
        "root_summary_json": root_summary_copy,
        "protocol_files": protocol_files,
        "comparison_files": comparison_files,
        "poster_assets": {
            arm_id: artifacts.get("poster_png")
            for arm_id, artifacts in arm_artifacts.items()
            if artifacts.get("poster_png")
        },
        "support_docs": {
            "abstract_draft": str(abstract_path),
            "abstract_filled": str(abstract_filled_path),
            "poster_outline": str(poster_outline_path),
            "jury_qa": str(jury_qa_path),
            "limitations": str(limitations_path),
        },
        "main_figure": figure_outputs,
    }
    figure_manifest_path = bundle_root / "EUCYS_FIGURE_MANIFEST.json"
    figure_manifest_path.write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")

    reproducibility_payload = {
        "bundle_kind": "eucys_reproducibility_bundle",
        "study_root": str(root),
        "protocol_files": protocol_files,
        "root_summary_json": root_summary_copy,
        "arm_summaries": {
            arm_id: artifacts.get("summary_json")
            for arm_id, artifacts in arm_artifacts.items()
        },
        "comparison_files": comparison_files,
        "matrix_row_count": len(_read_json(protocol_dir / "study_design.json").get("matrix_rows", []))
        if study_design_json.is_file()
        else None,
        "checklist": [
            "Protocol bundle copied into EUCYS results folder",
            "Per-arm summaries copied into the bundle",
            "Comparisons copied into the bundle",
            "Support docs for abstract, poster, jury QA, and limitations generated",
            "Main poster figure generated from the clean certified arm",
            "Results table frozen as CSV for poster and abstract work",
        ],
    }
    reproducibility_path = bundle_root / "EUCYS_REPRODUCIBILITY_BUNDLE.json"
    reproducibility_path.write_text(json.dumps(reproducibility_payload, indent=2), encoding="utf-8")

    return {
        "bundle_root": str(bundle_root),
        "summary_markdown": str(summary_path),
        "results_table_csv": str(table_path),
        "figure_manifest_json": str(figure_manifest_path),
        "reproducibility_bundle_json": str(reproducibility_path),
        "limitations_markdown": str(limitations_path),
        "abstract_markdown": str(abstract_path),
        "abstract_filled_markdown": str(abstract_filled_path),
        "poster_outline_markdown": str(poster_outline_path),
        "jury_qa_markdown": str(jury_qa_path),
        "main_figure_png": figure_outputs["main_figure_png"],
        "main_figure_csv": figure_outputs["main_figure_csv"],
    }
