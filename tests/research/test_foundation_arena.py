from __future__ import annotations

import json
from pathlib import Path

import pytest

from iints.research.foundation_arena import (
    FOUNDATION_ARENA_SCHEMA,
    load_foundation_evaluation,
    run_foundation_model_arena,
)


def _write_evaluation(
    path: Path,
    *,
    model_name: str,
    mae: float,
    benchmark_id: str = "ohio-ppgr-v1/subjects-fold-0",
    group_disjoint: bool = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": FOUNDATION_ARENA_SCHEMA,
                "model": {
                    "name": model_name,
                    "architecture": "test encoder",
                    "latent_dimension": 128,
                    "implementation_kind": "local-test",
                    "checkpoint_sha256": "a" * 64,
                },
                "evaluation": {
                    "benchmark_id": benchmark_id,
                    "task": "ppgr-trajectory",
                    "cohort_id": "ohio-test",
                    "split_id": "subjects-fold-0",
                    "split_strategy": "subject-grouped",
                    "group_disjoint": group_disjoint,
                    "n_groups": 12,
                    "n_samples": 480,
                    "seed": 42,
                },
                "metrics": {
                    "trajectory_mae_mgdl": {
                        "value": mae,
                        "unit": "mg/dL",
                        "direction": "lower",
                    },
                    "hypo_auc": {
                        "value": 0.8 if model_name == "Model A" else 0.9,
                        "unit": "AUROC",
                        "direction": "higher",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_foundation_model_arena_uses_supplied_evidence(tmp_path: Path) -> None:
    first = _write_evaluation(tmp_path / "a.json", model_name="Model A", mae=18.0)
    second = _write_evaluation(tmp_path / "b.json", model_name="Model B", mae=14.0)

    out_dir = tmp_path / "arena"
    report = run_foundation_model_arena(out_dir, [first, second])

    assert report.total_models_evaluated == 2
    assert report.metric_leaders["trajectory_mae_mgdl"] == "Model B"
    assert report.metric_leaders["hypo_auc"] == "Model B"
    assert report.comparison_csv_path.is_file()
    assert report.summary_json_path.is_file()
    text = report.report_md_path.read_text(encoding="utf-8")
    assert "only values loaded from supplied evaluation artifacts" in text
    assert "14 mg/dL" in text
    loaded = load_foundation_evaluation(first)
    assert len(loaded.source_sha256) == 64


def test_foundation_arena_never_fabricates_from_trial_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires one or more --result"):
        run_foundation_model_arena(tmp_path, n_benchmark_trials=50)


def test_foundation_arena_rejects_incomparable_splits(tmp_path: Path) -> None:
    first = _write_evaluation(tmp_path / "a.json", model_name="Model A", mae=18.0)
    second = _write_evaluation(
        tmp_path / "b.json",
        model_name="Model B",
        mae=14.0,
        benchmark_id="ohio-ppgr-v1/subjects-fold-1",
    )
    with pytest.raises(ValueError, match="benchmark_id differs"):
        run_foundation_model_arena(tmp_path / "out", [first, second])


def test_foundation_arena_rejects_row_level_leakage(tmp_path: Path) -> None:
    artifact = _write_evaluation(
        tmp_path / "leaky.json",
        model_name="Leaky model",
        mae=10.0,
        group_disjoint=False,
    )
    with pytest.raises(ValueError, match="group-disjoint"):
        run_foundation_model_arena(tmp_path / "out", [artifact])
