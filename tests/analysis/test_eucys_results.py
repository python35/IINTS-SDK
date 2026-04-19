from __future__ import annotations

import json
from pathlib import Path

from iints.analysis.eucys_results import generate_eucys_results_bundle


def _write_summary(path: Path, *, run_count: int, tir: float, below_70: float, above_180: float, glucose: float, interventions: float) -> None:
    payload = {
        "run_count": run_count,
        "aggregate": {
            "mean_tir_70_180": tir,
            "mean_tir_below_70": below_70,
            "mean_tir_above_180": above_180,
            "mean_glucose": glucose,
            "mean_supervisor_interventions": interventions,
        },
        "safety_summary": {
            "severe_hypo_run_count": 1,
            "terminated_early_run_count": 0,
        },
        "by_algorithm": {
            "CandidateAlgo": {
                "aggregate": {
                    "mean_tir_70_180": tir,
                    "mean_tir_below_70": below_70,
                    "mean_tir_above_180": above_180,
                    "mean_glucose": glucose,
                    "mean_supervisor_interventions": interventions,
                }
            },
            "PID Controller": {
                "aggregate": {
                    "mean_tir_70_180": tir - 4.0,
                    "mean_tir_below_70": below_70 + 0.5,
                    "mean_tir_above_180": above_180 + 4.0,
                    "mean_glucose": glucose + 8.0,
                    "mean_supervisor_interventions": interventions + 1.0,
                }
            },
            "Standard Pump": {
                "aggregate": {
                    "mean_tir_70_180": tir - 2.0,
                    "mean_tir_below_70": below_70 + 0.2,
                    "mean_tir_above_180": above_180 + 2.0,
                    "mean_glucose": glucose + 4.0,
                    "mean_supervisor_interventions": interventions + 0.5,
                }
            },
            "Correction Bolus": {
                "aggregate": {
                    "mean_tir_70_180": tir - 6.0,
                    "mean_tir_below_70": below_70 + 1.0,
                    "mean_tir_above_180": above_180 + 6.0,
                    "mean_glucose": glucose + 10.0,
                    "mean_supervisor_interventions": interventions + 2.0,
                }
            },
        },
        "pairwise_baseline_deltas": {
            "candidate_algorithm": "CandidateAlgo",
            "baselines": {
                "PID Controller": {"mean_deltas": {"tir_70_180": 4.0}},
                "Standard Pump": {"mean_deltas": {"tir_70_180": 2.0}},
                "Correction Bolus": {"mean_deltas": {"tir_70_180": 6.0}},
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_generate_eucys_results_bundle_writes_packaged_outputs(tmp_path: Path) -> None:
    study_root = tmp_path / "eucys_study"
    protocol_dir = study_root / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "STUDY_PROTOCOL.md").write_text("# protocol", encoding="utf-8")
    (protocol_dir / "study_matrix.csv").write_text("arm,seed\nclean,1\n", encoding="utf-8")
    (protocol_dir / "algorithms.json").write_text("[]", encoding="utf-8")
    (protocol_dir / "study_design.json").write_text(
        json.dumps(
            {
                "matrix_rows": [1, 2, 3],
                "profiles": [{"id": 1}, {"id": 2}, {"id": 3}],
                "scenarios": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
                "algorithms": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
            }
        ),
        encoding="utf-8",
    )
    (study_root / "study_summary.json").write_text(json.dumps({"run_count": 6}), encoding="utf-8")

    for folder_name, tir in (
        ("study_clean", 85.0),
        ("study_corrupted", 72.0),
        ("study_supervisor_off", 80.0),
    ):
        arm_dir = study_root / folder_name
        arm_dir.mkdir(parents=True)
        _write_summary(
            arm_dir / "study_summary.json",
            run_count=2,
            tir=tir,
            below_70=2.0,
            above_180=18.0,
            glucose=145.0,
            interventions=3.0,
        )
        (arm_dir / "study_summary.md").write_text("# summary", encoding="utf-8")
        (arm_dir / "study_poster.png").write_text("png", encoding="utf-8")

    comparisons_dir = study_root / "comparisons"
    comparisons_dir.mkdir(parents=True)
    (comparisons_dir / "clean_vs_corrupted.json").write_text("{}", encoding="utf-8")
    (comparisons_dir / "clean_vs_corrupted.md").write_text("# compare", encoding="utf-8")

    outputs = generate_eucys_results_bundle(study_root)

    bundle_root = Path(outputs["bundle_root"])
    assert (bundle_root / "EUCYS_SUMMARY.md").is_file()
    assert (bundle_root / "EUCYS_RESULTS_TABLE.csv").is_file()
    assert (bundle_root / "EUCYS_FIGURE_MANIFEST.json").is_file()
    assert (bundle_root / "EUCYS_REPRODUCIBILITY_BUNDLE.json").is_file()
    assert (bundle_root / "EUCYS_ABSTRACT_DRAFT.md").is_file()
    assert (bundle_root / "EUCYS_ABSTRACT_FILLED.md").is_file()
    assert (bundle_root / "EUCYS_POSTER_OUTLINE.md").is_file()
    assert (bundle_root / "EUCYS_JURY_QA.md").is_file()
    assert (bundle_root / "EUCYS_LIMITATIONS.md").is_file()
    assert (bundle_root / "EUCYS_MAIN_FIGURE.png").is_file()
    assert (bundle_root / "EUCYS_MAIN_FIGURE.csv").is_file()
    assert (bundle_root / "protocol" / "STUDY_PROTOCOL.md").is_file()
    assert (bundle_root / "arms" / "clean_certified" / "study_summary.json").is_file()
    assert (bundle_root / "comparisons" / "clean_vs_corrupted.json").is_file()

    abstract_text = (bundle_root / "EUCYS_ABSTRACT_FILLED.md").read_text(encoding="utf-8")
    assert "CandidateAlgo" in abstract_text
    assert "**3** patient profiles" in abstract_text


def test_generate_eucys_results_bundle_sanitizes_csv_formula_cells(tmp_path: Path) -> None:
    study_root = tmp_path / "eucys_study"
    protocol_dir = study_root / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "STUDY_PROTOCOL.md").write_text("# protocol", encoding="utf-8")
    (protocol_dir / "study_matrix.csv").write_text("arm,seed\nclean,1\n", encoding="utf-8")
    (protocol_dir / "algorithms.json").write_text("[]", encoding="utf-8")
    (protocol_dir / "study_design.json").write_text(
        json.dumps(
            {
                "matrix_rows": [1],
                "profiles": [{"id": 1}],
                "scenarios": [{"id": 1}],
                "algorithms": [{"id": 1}],
            }
        ),
        encoding="utf-8",
    )
    (study_root / "study_summary.json").write_text(json.dumps({"run_count": 1}), encoding="utf-8")

    arm_dir = study_root / "study_clean"
    arm_dir.mkdir(parents=True)
    payload = {
        "run_count": 1,
        "aggregate": {
            "mean_tir_70_180": 85.0,
            "mean_tir_below_70": 2.0,
            "mean_tir_above_180": 18.0,
            "mean_glucose": 145.0,
            "mean_supervisor_interventions": 3.0,
        },
        "safety_summary": {
            "severe_hypo_run_count": 0,
            "terminated_early_run_count": 0,
        },
        "by_algorithm": {
            "=Candidate": {
                "aggregate": {
                    "mean_tir_70_180": 85.0,
                    "mean_tir_below_70": 2.0,
                    "mean_tir_above_180": 18.0,
                    "mean_glucose": 145.0,
                    "mean_supervisor_interventions": 3.0,
                }
            },
            "@Baseline": {
                "aggregate": {
                    "mean_tir_70_180": 80.0,
                    "mean_tir_below_70": 3.0,
                    "mean_tir_above_180": 20.0,
                    "mean_glucose": 150.0,
                    "mean_supervisor_interventions": 4.0,
                }
            },
        },
        "pairwise_baseline_deltas": {
            "candidate_algorithm": "=Candidate",
            "baselines": {
                "@Baseline": {"mean_deltas": {"tir_70_180": 5.0}},
            },
        },
    }
    (arm_dir / "study_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (arm_dir / "study_summary.md").write_text("# summary", encoding="utf-8")
    (arm_dir / "study_poster.png").write_text("png", encoding="utf-8")
    for extra_arm in ("study_corrupted", "study_supervisor_off"):
        extra_dir = study_root / extra_arm
        extra_dir.mkdir(parents=True)
        (extra_dir / "study_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        (extra_dir / "study_summary.md").write_text("# summary", encoding="utf-8")
        (extra_dir / "study_poster.png").write_text("png", encoding="utf-8")

    outputs = generate_eucys_results_bundle(study_root)
    bundle_root = Path(outputs["bundle_root"])
    figure_csv = (bundle_root / "EUCYS_MAIN_FIGURE.csv").read_text(encoding="utf-8")

    assert "'=Candidate" in figure_csv
    assert "'@Baseline" in figure_csv
