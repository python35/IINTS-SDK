from __future__ import annotations

import importlib.util
import os
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
from iints_desktop.local_ai import SYSTEM_PROMPT, LocalAIStartResult, resolve_ollama_executable
from iints_desktop.molecules import (
    MoleculeStructureError,
    list_molecule_assets,
    load_molecule_backbone,
    pae_html_path,
)
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



def test_local_ai_can_resolve_ollama_from_extra_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ollama = tmp_path / "ollama"
    fake_ollama.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr("iints_desktop.local_ai.shutil.which", lambda _name: None)

    assert resolve_ollama_executable(extra_candidates=[fake_ollama]) == fake_ollama


def test_local_ai_start_result_is_ui_friendly() -> None:
    result = LocalAIStartResult(
        available=True,
        message="Local AI ready",
        resolved_model="ministral-3:8b",
        started_process=True,
        pulled_model=False,
    )

    assert result.available is True
    assert result.started_process is True
    assert "ready" in result.message.lower()

def test_molecule_assets_are_bundled_for_desktop_deep_dive() -> None:
    molecules = list_molecule_assets()

    assert {molecule.uniprot_id for molecule in molecules} == {"P01308", "P01275", "P06213", "P14672", "P47871"}
    assert all(molecule.image_path.exists() for molecule in molecules)
    assert all(molecule.structure_path.exists() for molecule in molecules)
    assert all("Connects to:" in molecule.sdk_link for molecule in molecules)
    assert {molecule.pae_target for molecule in molecules} == {"insulin-mutation", "glucagon", "insulin-receptor", "glut4", "glucagon-receptor"}
    assert pae_html_path("glucagon").as_posix().endswith("results/structural/glucagon_pae.html")


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


def test_qt_app_source_forces_readable_light_palette() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "def _apply_application_palette" in source
    assert "QPalette.ColorRole.WindowText" in source
    assert "QWidget, QWidget#root" in source
    assert "QGroupBox QLabel" in source


def test_qt_app_biology_copy_is_neutral_workbench_text() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "Structural biology assets for research context" in source
    assert "Scientific Deep Dive" not in source
    assert "Viewer guide:" not in source


def test_qt_app_exposes_biology_evidence_actions() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "class BiologyWorker" in source
    assert "Biology evidence actions" in source
    assert "Render GTEx Expression" in source
    assert "Analyze Insulin PK" in source
    assert "Simulate ClinVar Mutation" in source
    assert "Render STRING Pathways" in source
    assert "gtex-expression" in source
    assert "clinvar-mutation" in source


def test_qt_app_exposes_desktop_update_panel() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "DESKTOP_RELEASE_URL" in source
    assert "desktop-beta-2026-06-27-3" in source
    assert "class UpdateWorker" in source
    assert "Open App Downloads" in source
    assert "Open Update Docs" in source
    assert "Copy Update Command" in source
    assert "Update Python SDK Package" in source
    assert "iints-sdk-python35[full,desktop-qt,mdmp]" in source


def test_desktop_docs_use_main_branch_and_direct_downloads() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    app_install = Path("docs/APP_INSTALL.md").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/desktop-beta.yml").read_text(encoding="utf-8")

    combined = "\n".join([readme, app_install])
    assert "desktop-app" not in combined
    assert "IINTS-AF-Desktop-Beta-windows-x64.exe" in combined
    assert "IINTS-AF-Desktop-Beta-macos.dmg" in combined
    assert "IINTS-AF-Desktop-Beta-linux-x64" in combined
    assert "IINTS-AF-Desktop-Beta-windows-x64.zip" not in workflow
    assert "IINTS-AF-Desktop-Beta-macos.zip" not in workflow


def test_desktop_packager_creates_direct_windows_and_linux_assets(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "package_desktop_bundle",
        Path("tools/desktop/package_desktop_bundle.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app_name = "IINTS-AF-Desktop-Beta"
    windows_bundle = tmp_path / f"{app_name}.exe"
    windows_bundle.write_bytes(b"fake-windows-exe")
    linux_bundle = tmp_path / app_name
    linux_bundle.write_bytes(b"fake-linux-exe")
    output_dir = tmp_path / "desktop-dist"

    windows_asset = module.package_release_asset(windows_bundle, app_name, "windows-x64", output_dir)
    linux_asset = module.package_release_asset(linux_bundle, app_name, "linux-x64", output_dir)

    assert windows_asset.name == "IINTS-AF-Desktop-Beta-windows-x64.exe"
    assert windows_asset.read_bytes() == b"fake-windows-exe"
    assert linux_asset.name == "IINTS-AF-Desktop-Beta-linux-x64"
    assert linux_asset.read_bytes() == b"fake-linux-exe"
    if os.name != "nt":
        assert linux_asset.stat().st_mode & 0o111
