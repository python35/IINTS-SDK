from __future__ import annotations

from pathlib import Path

import pytest

from iints_desktop.engine import (
    DesktopRunResult,
    append_run_history,
    _optional_path,
    _safe_slug,
    get_desktop_environment,
    get_desktop_preset,
    list_desktop_presets,
    read_run_history,
)
from iints_desktop.local_ai import SYSTEM_PROMPT
from iints_desktop.molecules import MoleculeStructureError, list_molecule_assets, load_molecule_backbone
from iints_desktop.results import build_ai_result_context, load_results_preview


def test_desktop_run_result_is_ui_friendly(tmp_path: Path) -> None:
    result = DesktopRunResult(
        run_id="demo-run",
        workflow_title="Doctor safety discussion",
        preset_name="hypo_prone_night",
        seed=42,
        output_dir=tmp_path,
        results_csv=tmp_path / "results.csv",
        report_pdf=None,
        config_path=tmp_path / "config.json",
        summary="Run completed",
    )

    assert result.run_id == "demo-run"
    assert result.report_pdf is None
    assert "completed" in result.summary


def test_optional_path_normalizes_values(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"

    assert _optional_path(None) is None
    assert _optional_path("") is None
    assert _optional_path(path) == path.resolve()


def test_desktop_presets_are_curated_and_resolvable() -> None:
    presets = list_desktop_presets()

    assert len(presets) >= 3
    assert get_desktop_preset("doctor-safety").preset_name == "hypo_prone_night"
    assert all(preset.title for preset in presets)
    assert all(preset.description for preset in presets)
    assert all(preset.talk_track for preset in presets)


def test_safe_slug_keeps_output_folders_portable() -> None:
    assert _safe_slug("Doctor Safety / Hypo Night") == "doctor-safety-hypo-night"
    assert _safe_slug("!!!") == "iints-desktop-run"


def test_run_history_roundtrip(tmp_path: Path) -> None:
    result = DesktopRunResult(
        run_id="demo-run",
        workflow_title="Doctor safety discussion",
        preset_name="hypo_prone_night",
        seed=42,
        output_dir=tmp_path / "doctor-safety",
        results_csv=tmp_path / "doctor-safety" / "results.csv",
        report_pdf=tmp_path / "doctor-safety" / "report.pdf",
        config_path=None,
        summary="Run completed",
    )

    history_path = append_run_history(tmp_path, result)
    entries = read_run_history(tmp_path)

    assert history_path.exists()
    assert len(entries) == 1
    assert entries[0].run_id == "demo-run"
    assert entries[0].preset_name == "hypo_prone_night"
    assert entries[0].seed == 42


def test_desktop_environment_reports_version() -> None:
    environment = get_desktop_environment(qt_available=True)

    assert environment.sdk_version
    assert environment.qt_available is True


def test_results_preview_builds_metrics_graph_and_bounded_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units",
                "0,110,0,0.01",
                "5,145,20,0.03",
                "10,185,0,0.02",
                "15,65,0,0.00",
            ]
        ),
        encoding="utf-8",
    )

    preview = load_results_preview(csv_path, max_rows=2)

    assert preview.row_count == 4
    assert len(preview.rows) == 2
    assert preview.metrics["Mean glucose"] == "126.2 mg/dL"
    assert preview.metrics["Time below 70"] == "25.0%"
    assert preview.graph_path is not None
    assert preview.graph_path.exists()


def test_ai_result_context_is_summary_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("timestamp,glucose,carbs,insulin\n0,100,0,0\n5,130,12,0.1\n", encoding="utf-8")

    context = build_ai_result_context(csv_path)

    assert "Summary metrics" in context
    assert "Mean glucose" in context
    assert "Do not infer treatment decisions" in context


def test_local_ai_prompt_is_research_only() -> None:
    assert "Not a medical device" in SYSTEM_PROMPT
    assert "Do not provide diagnosis" in SYSTEM_PROMPT


def test_molecule_assets_are_bundled_for_desktop_deep_dive() -> None:
    molecules = list_molecule_assets()

    assert {molecule.uniprot_id for molecule in molecules} == {"P01308", "P01275"}
    assert all(molecule.image_path.exists() for molecule in molecules)
    assert all(molecule.structure_path.exists() for molecule in molecules)
    assert all("Connects to:" in molecule.sdk_link for molecule in molecules)


def test_molecule_assets_contain_renderable_backbone_coordinates() -> None:
    backbones = {
        molecule.key: load_molecule_backbone(molecule.structure_path)
        for molecule in list_molecule_assets()
    }

    assert len(backbones["insulin"].atoms) >= 100
    assert len(backbones["glucagon"].atoms) >= 170
    assert all(backbone.chain_count >= 1 for backbone in backbones.values())
    assert all(backbone.radius > 1.0 for backbone in backbones.values())


def test_invalid_mmcif_is_rejected_before_the_viewer_uses_it(tmp_path: Path) -> None:
    invalid_structure = tmp_path / "not-a-structure.cif"
    invalid_structure.write_text("<Error>download failed</Error>", encoding="utf-8")

    with pytest.raises(MoleculeStructureError, match="not a valid AlphaFold mmCIF"):
        load_molecule_backbone(invalid_structure)
