from __future__ import annotations

import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.registry import get_dataset, list_dataset_ids
from iints.data.research_catalog import build_research_dataset_matrix, write_research_dataset_plan

runner = CliRunner()


def test_curated_research_datasets_are_registered() -> None:
    ids = set(list_dataset_ids())
    expected = {
        "hupa_ucm",
        "azt1d",
        "t1d_uom",
        "ohio_t1dm",
        "dclp3_idcl",
        "jaeb_loop",
        "t1dexi",
        "t1dexip",
        "d1namo",
        "openaps_data_commons",
        "metabonet",
        "glucose_ml",
    }
    assert expected.issubset(ids)
    assert get_dataset("dclp3_idcl")["access"] == "public-download"
    assert "CC BY-SA" in get_dataset("d1namo")["license"]


def test_research_dataset_matrix_flags_useful_tasks() -> None:
    rows = build_research_dataset_matrix(["hupa_ucm", "t1dexi", "dclp3_idcl", "glucose_ml"])
    by_id = {row["dataset_id"]: row for row in rows}

    assert by_id["hupa_ucm"]["glucose_forecasting"] is True
    assert by_id["hupa_ucm"]["multimodal_research"] is True
    assert by_id["t1dexi"]["exercise_research"] is True
    assert by_id["dclp3_idcl"]["closed_loop_benchmarking"] is True
    assert by_id["glucose_ml"]["external_validation"] is True


def test_write_research_dataset_plan_outputs_manifest(tmp_path: Path) -> None:
    manifest = write_research_dataset_plan(tmp_path, dataset_ids=["hupa_ucm", "azt1d", "dclp3_idcl"])

    assert manifest["dataset_count"] == 3
    for output_path in manifest["outputs"].values():
        assert Path(output_path).is_file()

    plan = (tmp_path / "DATASET_ACQUISITION_PLAN.md").read_text(encoding="utf-8")
    assert "hupa_ucm" in plan
    assert "dclp3_idcl" in plan
    assert "Research Boundary" in plan

    with (tmp_path / "research_dataset_matrix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["dataset_id"] for row in rows] == ["hupa_ucm", "azt1d", "dclp3_idcl"]

    snapshot = json.loads((tmp_path / "dataset_registry_snapshot.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in snapshot] == ["hupa_ucm", "azt1d", "dclp3_idcl"]


def test_data_research_plan_cli(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"
    result = runner.invoke(
        app,
        [
            "data",
            "research-plan",
            "--output-dir",
            str(output_dir),
            "--dataset",
            "hupa_ucm",
            "--dataset",
            "t1dexi",
        ],
    )

    assert result.exit_code == 0
    assert "IINTS Diabetes Research Dataset Plan" in result.stdout
    assert (output_dir / "DATASET_ACQUISITION_PLAN.md").is_file()
    assert (output_dir / "SOURCE_CITATIONS.bib").is_file()
