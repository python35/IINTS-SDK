"""Assemble ~/Desktop/EUCYS_JURY from material that already exists.

Rules this script follows, because the folder is meant to survive a jury that
can look things up:

* Nothing is invented here. Every file is either copied from the repository,
  copied from a review artefact produced earlier in this project, or computed
  from a seed-fixed run of the shipped SDK.
* Every copied file is recorded in MANIFEST.csv with its sha256 and its source
  path, so any number in the folder can be traced back to where it came from.
* Every file carries a provenance class (see the legend in 00_START_HERE.md).
  Material whose origin could not be established is not copied; it is listed in
  PROVENANCE_NOTES.md as excluded, with the reason.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import shutil

REPO = pathlib.Path("/Users/runebobbaers/Documents/IINTS-AF-SSD/IINTS-SDK-main")
DEST = pathlib.Path.home() / "Desktop" / "EUCYS_JURY"
FRESH = pathlib.Path("jury_build/fresh")
ARTIFACTS = json.loads(pathlib.Path("jury_build/artifact_paths.json").read_text())

FOLDERS = [
    "01_ARCHITECTURE",
    "02_PUMP_MECHANICS",
    "03_PHYSIOLOGICAL_MODEL",
    "04_AI_ALGORITHMS",
    "05_EXPERIMENTS",
    "06_RESULTS",
    "07_VALIDATION",
    "08_FAILURE_CASES",
    "09_LITERATURE",
    "10_SOURCE_CODE",
]

# (folder, source, provenance, note)
#   repo:<path>      file inside the SDK repository
#   art:<filename>   review artefact produced earlier in this project
#   fresh:<path>     computed or measured during this build
#   build:<path>     script from jury_build/ that produced the fresh files
SPEC = [
    # ---------------------------------------------------------------- 01
    ("01_ARCHITECTURE", "repo:EUCYS_POSTER_ASSETS/architecture_flowchart.png", "DIAGRAM",
     "System architecture, drawn diagram - carries no measured values."),
    ("01_ARCHITECTURE", "repo:EUCYS_POSTER_ASSETS/architecture_flowchart.pdf", "DIAGRAM",
     "Vector version of the same diagram, for printing."),
    ("01_ARCHITECTURE", "repo:EUCYS_POSTER_ASSETS/step_sequence.png", "DIAGRAM",
     "One simulation step, from sensor input to pump command."),
    ("01_ARCHITECTURE", "repo:research/eucys_pack/diagrams/system-architecture.mmd", "DIAGRAM",
     "Source text of the architecture diagram (Mermaid)."),
    ("01_ARCHITECTURE", "repo:research/eucys_pack/diagrams/simulation-step.mmd", "DIAGRAM",
     "Source text of the step diagram."),
    ("01_ARCHITECTURE", "repo:research/eucys_pack/diagrams/evidence-lifecycle.mmd", "DIAGRAM",
     "How a run becomes a hash-sealed evidence bundle."),
    ("01_ARCHITECTURE", "repo:research/eucys_pack/diagrams/numeric-authority.mmd", "DIAGRAM",
     "Which component is allowed to produce a number."),
    ("01_ARCHITECTURE", "repo:docs/ARCHITECTURE_OVERVIEW.md", "DOCUMENT", "Architecture in prose."),
    ("01_ARCHITECTURE", "repo:research/eucys_pack/EUCYS_01_TECHNICAL_BRIEF.md", "DOCUMENT",
     "Technical brief written for the jury."),
    # ---------------------------------------------------------------- 02
    ("02_PUMP_MECHANICS", "repo:EUCYS_POSTER_ASSETS/fpga_flow.png", "DIAGRAM",
     "FPGA safety-core logic flow, drawn diagram."),
    ("02_PUMP_MECHANICS", "repo:EUCYS_POSTER_ASSETS/jetson_flow.png", "DIAGRAM",
     "Jetson deployment flow, drawn diagram."),
    ("02_PUMP_MECHANICS", "repo:docs/ARDUINO_UNO_Q.md", "DOCUMENT",
     "Bench-only microcontroller workflow. Bench hardware, never a patient."),
    ("02_PUMP_MECHANICS", "fresh:edge_benchmark.json", "MEASURED",
     "Throughput, step latency and memory measured on the laptop that built this folder."),
    # ---------------------------------------------------------------- 03
    ("03_PHYSIOLOGICAL_MODEL", "repo:research/eucys_pack/EUCYS_02_PHYSIOLOGY_AND_DATA_BRIEF.md", "DOCUMENT",
     "Physiology and data brief."),
    ("03_PHYSIOLOGICAL_MODEL", "repo:research/eucys_pack/EUCYS_06_JURY_PHYSIOLOGY_BRIEF.md", "DOCUMENT",
     "Short physiology brief for jury questions."),
    ("03_PHYSIOLOGICAL_MODEL", "repo:research/eucys_pack/EUCYS_05_PHYSIOLOGY_REFERENCE_BROCHURE.md", "DOCUMENT",
     "Longer physiology reference."),
    ("03_PHYSIOLOGICAL_MODEL", "art:SDK_SCIENTIFIC_BASIS.md", "DOCUMENT",
     "Every formula family in the SDK traced to its source, with the three evidence classes."),
    # ---------------------------------------------------------------- 04
    ("04_AI_ALGORITHMS", "repo:results/eucys_2026/protocol/algorithms.json", "DOCUMENT",
     "The controllers entered in the study, as the study engine recorded them."),
    # ---------------------------------------------------------------- 05
    ("05_EXPERIMENTS", "repo:results/eucys_2026/protocol/STUDY_PROTOCOL.md", "DOCUMENT",
     "Pre-registered study protocol."),
    ("05_EXPERIMENTS", "repo:results/eucys_2026/protocol/study_design.json", "DOCUMENT",
     "Machine-readable design: arms, profiles, scenarios, repetitions."),
    ("05_EXPERIMENTS", "repo:results/eucys_2026/protocol/study_matrix.csv", "DOCUMENT",
     "The full run matrix, one row per planned run."),
    ("05_EXPERIMENTS", "repo:results/eucys_2026/protocol/study_experiment.yaml", "DOCUMENT",
     "Experiment file that reproduces the study."),
    ("05_EXPERIMENTS", "repo:research/eucys_pack/eucys_statistics.json", "DERIVED",
     "Study statistics as computed by the SDK."),
    # ---------------------------------------------------------------- 06
    ("06_RESULTS", "repo:results/eucys_2026/EUCYS_RESULTS_TABLE.csv", "DERIVED",
     "Main results table of the three-arm study."),
    ("06_RESULTS", "repo:results/eucys_2026/EUCYS_MAIN_FIGURE.csv", "DERIVED",
     "The numbers behind the main figure, so the figure can be rebuilt."),
    ("06_RESULTS", "repo:research/eucys_pack/assets/EUCYS_MAIN_FIGURE.png", "DERIVED",
     "Main figure as rendered by the SDK."),
    ("06_RESULTS", "art:crossfold_evidence.json", "DERIVED",
     "Cross-fold evidence bundle: cluster-level intervals plus model, data and config hashes."),
    ("06_RESULTS", "art:arm_comparison_delta.json", "DERIVED",
     "Arm comparison bundle. States its own refusal to report intervals below its minimum subject count."),
    ("06_RESULTS", "art:kruisvalidatie_resultaat.md", "DOCUMENT", "Read-out of the cross-fold bundle."),
    ("06_RESULTS", "art:delta_armvergelijking_resultaat.md", "DOCUMENT", "Read-out of the arm comparison."),
    ("06_RESULTS", "art:corrected_benchmark_inference.csv", "DERIVED",
     "Benchmark inference after the clustering correction."),
    # ---------------------------------------------------------------- 07
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/comparison_report.md", "MEASURED",
     "Forecasting comparison on the public OhioT1DM test set: 35,925 rows, 10 models."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/comparison_report.json", "MEASURED",
     "Same comparison, machine-readable."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/horizon_metrics.csv", "MEASURED",
     "Forecast error per horizon step."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/hypo_detection_metrics.csv", "MEASURED",
     "Hypoglycaemia detection quality per model."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/physiological_violation_metrics.csv", "MEASURED",
     "How often each model predicts something physiologically impossible."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/figures/horizon_mae.png", "MEASURED",
     "Error growth with horizon, on real data."),
    ("07_VALIDATION", "repo:results/glucose_model_comparison_ohio_final/figures/physiology_violations.png", "MEASURED",
     "Violation rates, on real data."),
    ("07_VALIDATION", "repo:results/research_realism_sweep_20260603_01/REPORT.md", "DERIVED",
     "Realism sweep across presets and algorithms."),
    ("07_VALIDATION", "repo:results/research_realism_sweep_20260603_01/plots/realism_score_heatmap.png", "DERIVED",
     "Realism score per preset and algorithm."),
    ("07_VALIDATION", "repo:results/research_realism_sweep_20260603_01/plots/verdict_distribution.png", "DERIVED",
     "Distribution of pass/fail verdicts over the sweep."),
    ("07_VALIDATION", "repo:results/research_realism_sweep_20260603_01/summary/run_summary.csv", "DERIVED",
     "One row per sweep run."),
    ("07_VALIDATION", "repo:results/research_realism_sweep_20260603_01/manifest.json", "DERIVED",
     "Sweep manifest."),
    ("07_VALIDATION", "fresh:replay_check.json", "MEASURED",
     "Deterministic replay: three runs, same seed, compared by hash."),
    ("07_VALIDATION", "fresh:pytest_summary.txt", "MEASURED",
     "Test suite as run during this build."),
    ("07_VALIDATION", "art:science_audit_report.md", "DOCUMENT",
     "Independent science audit of the SDK, with its findings."),
    ("07_VALIDATION", "art:science_audit_findings.csv", "DOCUMENT", "The findings as a table."),
    ("07_VALIDATION", "art:metric_bug_impact.csv", "MEASURED",
     "What the risk-index defect changed on four real cohorts, recomputed."),
    ("07_VALIDATION", "art:sensor_threshold_check.csv", "MEASURED",
     "Measured rate-of-change distribution behind the sensor threshold."),
    ("07_VALIDATION", "art:SCIENCE_FIXES_PRE_EUCYS.md", "DOCUMENT",
     "What was corrected in the week before the contest, and how it was verified."),
    ("07_VALIDATION", "art:PAPER_CODE_REVIEW.md", "DOCUMENT",
     "The paper checked line by line against the implementation."),
    ("07_VALIDATION", "art:cohort_audit_report.md", "MEASURED", "Audit of the four real cohorts."),
    ("07_VALIDATION", "art:cohort_summary.csv", "MEASURED", "Per-cohort summary."),
    ("07_VALIDATION", "art:subject_summary.csv", "MEASURED", "Per-subject summary."),
    ("07_VALIDATION", "art:feature_availability.csv", "MEASURED", "Which fields each cohort actually provides."),
    # ---------------------------------------------------------------- 08
    ("08_FAILURE_CASES", "repo:results/eucys_2026/comparisons/clean_vs_supervisor_off.md", "DERIVED",
     "Safety supervisor switched off: the central ablation."),
    ("08_FAILURE_CASES", "repo:results/eucys_2026/comparisons/clean_vs_supervisor_off.json", "DERIVED",
     "Same ablation, machine-readable."),
    ("08_FAILURE_CASES", "repo:results/eucys_2026/comparisons/clean_vs_corrupted.md", "DERIVED",
     "Corrupted, uncertified input data versus clean input."),
    ("08_FAILURE_CASES", "repo:results/eucys_2026/comparisons/clean_vs_corrupted.json", "DERIVED",
     "Same comparison, machine-readable."),
    ("08_FAILURE_CASES", "repo:results/research_realism_sweep_20260603_01/summary/worst_40_runs.csv", "DERIVED",
     "The forty worst runs of the sweep, not hidden."),
    ("08_FAILURE_CASES", "repo:results/research_realism_sweep_20260603_01/summary/largest_jump_40_runs.csv", "DERIVED",
     "The forty largest glucose jumps, a realism failure mode."),
    ("08_FAILURE_CASES", "repo:results/research_realism_sweep_20260603_01/plots/max_step_jump_by_preset.png", "DERIVED",
     "Largest step jump per preset."),
    ("08_FAILURE_CASES", "repo:results/demo/evidence_bundle/runs/supervisor_override/RUN_CARD.json", "SIMULATED",
     "Run card of a supervisor-override run, with its hashes."),
    ("08_FAILURE_CASES", "repo:results/demo/evidence_bundle/runs/supervisor_override/artifacts/run_manifest.json", "SIMULATED",
     "Manifest of that run."),
    ("08_FAILURE_CASES", "repo:results/demo/evidence_bundle/runs/supervisor_override/artifacts/results.csv", "SIMULATED",
     "Full time series of that run."),
    # ---------------------------------------------------------------- 09
    ("09_LITERATURE", "art:SDK_SCIENTIFIC_BASIS.md", "DOCUMENT",
     "Formula-by-formula sourcing, including what is adapted and what is heuristic."),
    ("09_LITERATURE", "art:PUBLICATION_READINESS.md", "DOCUMENT",
     "Assessment of where this work can realistically be published, and what is missing."),
    ("09_LITERATURE", "fresh:formula_registry.csv", "DOCUMENT",
     "The SDK's own formula registry, exported: formula, reference, evidence class."),
    ("09_LITERATURE", "fresh:evidence_sources.txt", "DOCUMENT",
     "Evidence sources the SDK reports for grounding its simulations (iints sources)."),
    # ---------------------------------------------------------------- 10
    ("10_SOURCE_CODE", "repo:LICENSE", "DOCUMENT", "Apache-2.0."),
    ("10_SOURCE_CODE", "repo:CITATION.cff", "DOCUMENT", "Machine-readable citation."),
    ("10_SOURCE_CODE", "repo:pyproject.toml", "DOCUMENT", "Package metadata and dependencies."),
    ("10_SOURCE_CODE", "repo:README.md", "DOCUMENT", "Repository README."),
    # --- computed in this build: figures, their raw runs, and the scripts that made them
    ("03_PHYSIOLOGICAL_MODEL", "fresh:figures/physiology_full_day.png", "SIMULATED",
     "One simulated day: sensor trace the controller sees, plus the ODE fluxes inside the model. "
     "Flux units taken from the SDK compartment schema."),
    ("02_PUMP_MECHANICS", "fresh:figures/pump_delivery_and_iob.png", "SIMULATED",
     "Bolus events, basal rate and insulin-on-board from the same run, with subcutaneous depot "
     "transfer and plasma appearance fluxes."),
    ("04_AI_ALGORITHMS", "fresh:figures/controller_comparison.png", "SIMULATED",
     "Four shipped controllers on an identical day and seed. No controller wins on both axes."),
    ("08_FAILURE_CASES", "fresh:figures/hypo_failure_case.png", "SIMULATED",
     "Hypoglycaemia-prone night: nadir 59 mg/dL with the safety supervisor acting on 102 steps."),
    ("06_RESULTS", "fresh:runs/run_metrics.csv", "SIMULATED",
     "Glycaemic metrics for all five runs behind the figures, computed by the SDK's own metrics module."),
    ("06_RESULTS", "fresh:runs/baseline_t1d__clinical_baseline.csv", "SIMULATED", "Raw per-step trace, reference run."),
    ("06_RESULTS", "fresh:runs/baseline_t1d__mpc.csv", "SIMULATED", "Raw per-step trace, MPC."),
    ("06_RESULTS", "fresh:runs/baseline_t1d__pid.csv", "SIMULATED", "Raw per-step trace, PID."),
    ("06_RESULTS", "fresh:runs/baseline_t1d__fixed_basal_bolus.csv", "SIMULATED", "Raw per-step trace, fixed basal-bolus."),
    ("08_FAILURE_CASES", "fresh:runs/hypo_prone_night__clinical_baseline.csv", "SIMULATED", "Raw per-step trace of the hypoglycaemic run."),
    ("08_FAILURE_CASES", "fresh:runs/hypo_prone_night__clinical_baseline__safety_report.json", "SIMULATED",
     "Safety supervisor report for that run, written by the SDK."),
    ("09_LITERATURE", "fresh:formula_registry.md", "DOCUMENT",
     "Same registry as the CSV, readable form: every formula with its provenance class."),
    ("10_SOURCE_CODE", "build:compute_runs.py", "DERIVED", "Script that produced the five runs in 06 and 08."),
    ("10_SOURCE_CODE", "build:make_figures.py", "DERIVED", "Script that produced the four computed figures."),
    ("10_SOURCE_CODE", "fresh:code_stats.csv", "DERIVED", "Counted source and test files, lines, git history."),
    ("10_SOURCE_CODE", "fresh:git_history.txt", "DERIVED", "Commit history summary."),
]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(source: str) -> pathlib.Path | None:
    kind, _, rest = source.partition(":")
    if kind == "repo":
        return REPO / rest
    if kind == "art":
        path = ARTIFACTS.get(rest)
        return pathlib.Path(path) if path else None
    if kind == "build":
        return pathlib.Path("jury_build") / rest
    if kind == "fresh":
        return FRESH / rest
    raise ValueError(source)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for folder in FOLDERS:
        (DEST / folder).mkdir(exist_ok=True)

    rows = []
    missing = []
    for folder, source, provenance, note in SPEC:
        src = resolve(source)
        if src is None or not src.is_file():
            missing.append((folder, source))
            continue
        target = DEST / folder / src.name
        shutil.copy2(src, target)
        rows.append(
            {
                "folder": folder,
                "file": src.name,
                "provenance": provenance,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "source": source,
                "note": note,
            }
        )

    copy_top_level_docs(DEST, rows)

    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["folder", "file", "provenance", "bytes", "sha256", "source", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"copied {len(rows)} files into {DEST}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["provenance"]] = counts.get(row["provenance"], 0) + 1
    print("by provenance:", dict(sorted(counts.items())))
    if missing:
        print("MISSING (not copied):")
        for folder, source in missing:
            print(f"  {folder}: {source}")


TOP_LEVEL_DOCS = ["00_START_HERE.md", "PROVENANCE_NOTES.md"]


def copy_top_level_docs(dest, rows):
    """Place the two reading documents at the folder root and record them too."""
    for name in TOP_LEVEL_DOCS:
        src = pathlib.Path("jury_build/docs") / name
        if not src.exists():
            raise FileNotFoundError(f"missing reading document: {src}")
        target = dest / name
        shutil.copy2(src, target)
        rows.append({
            "folder": ".", "file": name, "provenance": "DOCUMENT",
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "source": f"build:docs/{name}",
            "note": "Reading guide written for this folder; every number in it is read from a file in this folder.",
        })


if __name__ == "__main__":
    main()
