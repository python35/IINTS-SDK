from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

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
from iints_desktop.local_ai import (
    RECOMMENDED_OLLAMA_MODELS,
    SYSTEM_PROMPT,
    LocalAIStartResult,
    format_ai_answer,
    resolve_ollama_executable,
)
from iints_desktop.mdmp import create_desktop_mdmp_certificate
from iints_desktop.molecules import (
    MoleculeStructureError,
    list_molecule_assets,
    load_molecule_backbone,
    pae_html_path,
)
from iints_desktop.results import build_ai_result_context, load_results_preview
from iints_desktop.update import format_shell_command
from iints.research.structure import TARGETS


def _load_pyproject() -> dict[str, Any]:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)

    optional_dependencies: dict[str, list[str]] = {}
    current_extra: str | None = None
    in_optional_deps = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "[project.optional-dependencies]":
            in_optional_deps = True
            continue
        if in_optional_deps and line.startswith("[") and line.endswith("]"):
            break
        if not in_optional_deps or not line or line.startswith("#"):
            continue
        if line.endswith("[") and "=" in line:
            current_extra = line.split("=", 1)[0].strip()
            optional_dependencies[current_extra] = []
            continue
        if current_extra is not None:
            if line.startswith('"'):
                optional_dependencies[current_extra].append(line.split('"', 2)[1])
            elif line == "]":
                current_extra = None

    return {"project": {"optional-dependencies": optional_dependencies}}


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


def test_desktop_mdmp_certificate_signs_loaded_results(tmp_path: Path) -> None:
    if importlib.util.find_spec("cryptography") is None:
        pytest.skip("cryptography is not installed")

    from mdmp_core.crypto import MDMPVerifier

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time_minutes,glucose_actual_mgdl,carb_intake_grams,delivered_insulin_units",
                "0,110,0,0.01",
                "5,145,20,0.03",
                "10,185,0,0.02",
            ]
        ),
        encoding="utf-8",
    )

    result = create_desktop_mdmp_certificate(csv_path, output_dir=tmp_path / "certs", quick=True, quick_rows=2)

    assert result.certificate_path.exists()
    assert result.report_path.exists()
    verification = MDMPVerifier(public_key_path=result.public_key_path).verify(
        json.loads(result.certificate_path.read_text(encoding="utf-8"))
    )
    assert verification["valid"] is True


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
    assert "Clinical Overview" in SYSTEM_PROMPT
    assert RECOMMENDED_OLLAMA_MODELS


def test_local_ai_answer_formatter_removes_markdown_noise() -> None:
    formatted = format_ai_answer("## Clinical Overview\n- **Glucose** is stable\n---\n* Next check")

    assert "##" not in formatted
    assert "**" not in formatted
    assert "---" not in formatted
    assert "• Glucose is stable" in formatted


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
    assert TARGETS["glucagon-receptor"] == "P47871"
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


def test_desktop_app_bundles_and_applies_iints_logo_icon() -> None:
    qt_source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    tk_source = Path("src/iints_desktop/app.py").read_text(encoding="utf-8")
    cocoa_source = Path("src/iints_desktop/cocoa_app.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    qt_build = Path("tools/desktop/build_qt_desktop_app.py").read_text(encoding="utf-8")
    tk_build = Path("tools/desktop/build_desktop_app.py").read_text(encoding="utf-8")

    assert Path("src/iints_desktop/assets/app_icon.png").is_file()
    assert Path("src/iints_desktop/assets/app_icon.ico").is_file()
    assert Path("src/iints_desktop/assets/app_icon.icns").is_file()
    assert '"assets/*.png"' in pyproject
    assert '"assets/*.ico"' in pyproject
    assert '"assets/*.icns"' in pyproject
    assert "QIcon" in qt_source
    assert "desktop_icon_path" in qt_source
    assert "setWindowIcon(icon)" in qt_source
    assert "iconphoto(True" in tk_source
    assert "setApplicationIconImage_" in cocoa_source
    assert "--icon" in qt_build
    assert "app_icon.icns" in qt_build
    assert "app_icon.ico" in qt_build
    assert "app_icon.png" in qt_build
    assert "--icon" in tk_build


def test_qt_app_terminal_state_exists_before_about_tab_builds() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "self.terminal_dock: QWidget | None = None" in source
    assert source.index("self.terminal_dock: QWidget | None = None") < source.index("self._build_ui()")
    assert "self.terminal_dock = QDockWidget" in source
    assert source.index("self.terminal_dock = QDockWidget") < source.index("self._build_about_tab(about_tab)")
    assert source.index("self.terminal_dock.setWidget(self.terminal_text)") < source.index("self._build_about_tab(about_tab)")


def test_qt_app_biology_copy_is_neutral_workbench_text() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "Structural biology assets for research context" in source
    assert "Scientific Deep Dive" not in source
    assert "Viewer guide:" not in source


def test_qt_app_exposes_biology_evidence_actions() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "class GenomicsWorker" in source
    assert "Advanced Research & Algorithm Stressors" in source
    assert "Run Multi-Scale Simulation" in source
    assert "Highlight Mutation in 3D" in source
    assert "Open Genomics Folder" in source
    assert "pLDDT is not pathogenicity or metabolic" in source
    assert "mathematically translate it into a metabolic stress factor" not in source
    assert "Academic evidence and standards catalog" in source
    assert "list_evidence_connectors" in source
    assert "Only fixed HTTPS evidence links may be opened" in source


def test_qt_app_exposes_academic_reproducibility_package() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    build_source = Path("tools/desktop/build_qt_desktop_app.py").read_text(encoding="utf-8")

    assert "class AcademicBundleWorker" in source
    assert "Create Academic Package" in source
    assert "Open RO-Crate Metadata" in source
    assert "build_academic_bundle" in source
    assert "not peer review, privacy approval, or clinical validation" in source
    assert '"iints.research.academic_bundle"' in build_source


def test_qt_batch_runner_consumes_one_queue_entry_per_run() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert source.count("config = self.batch_queue.pop(0)") == 1


def test_qt_app_exposes_mdmp_and_model_selector_actions() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "RECOMMENDED_OLLAMA_MODELS" in source
    assert "Refresh Models" in source
    assert "Create MDMP Certificate" in source
    assert "class MDMPCertifyWorker" in source


def test_qt_app_exposes_desktop_update_panel() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    update_source = Path("src/iints_desktop/update.py").read_text(encoding="utf-8")

    assert "DESKTOP_RELEASE_URL" in source
    assert "desktop-beta-latest" in update_source
    assert "Open App Downloads" in source
    assert "Open Update Docs" in source
    assert "Copy Update Command" in source
    assert "Update Python SDK Package" in source
    assert "Developer Settings" in source
    assert "build_python_sdk_update_args()" in source
    assert "iints-sdk-python35[desktop-all]" in update_source


def test_desktop_extra_installs_pyside6_automatically() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    desktop_deps = extras["desktop"]
    desktop_qt_deps = extras["desktop-qt"]
    desktop_all_deps = extras["desktop-all"]
    build_source = Path("tools/desktop/build_qt_desktop_app.py").read_text(encoding="utf-8")

    assert any(dep.startswith("PySide6") for dep in desktop_deps)
    assert any(dep.startswith("plotly") for dep in desktop_deps)
    assert any(dep.startswith("PySide6") for dep in desktop_qt_deps)
    assert any(dep.startswith("PySide6") for dep in desktop_all_deps)
    assert any(dep.startswith("FMPy") for dep in desktop_all_deps)
    assert any(dep.startswith("libroadrunner") for dep in desktop_all_deps)
    assert 'python -m pip install -U -e ".[desktop-all]"' in build_source
    assert 'python -m pip install -U -e ".[desktop-qt]"' not in build_source


def test_qt_app_keyboard_shortcut_uses_existing_workflow_handler() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "def _run_selected_workflow" in source
    assert "self.run_shortcut.activated.connect(self._run_selected_workflow)" in source
    assert "self.run_shortcut.activated.connect(self._run_workflow)" not in source


def test_qt_app_update_terminal_is_cross_platform() -> None:
    source = Path("src/iints_desktop/terminal_utils.py").read_text(encoding="utf-8")

    assert "open_terminal_and_run" in source
    assert "_escape_applescript_string" in source
    assert "format_shell_command" in source
    assert "osascript" in source
    assert "cmd.exe" in source
    assert "x-terminal-emulator" in source
    assert "gnome-terminal" in source
    assert "konsole" in source
    assert "xfce4-terminal" in source
    assert "xterm" in source


def test_update_command_uses_native_windows_quoting() -> None:
    args = ["C:\\Program Files\\Python\\python.exe", "-m", "pip", "install", "pkg[extra]"]

    command = format_shell_command(args, platform_name="windows")

    assert '"C:\\Program Files\\Python\\python.exe"' in command
    assert "'" not in command


def test_genomics_worker_labels_functional_scalar_as_scenario_assumption() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")

    assert "Scenario assumption:" in source
    assert "Impact: {int(data['scalar']*100)}% affinity" not in source


def test_qt_app_avoids_embedded_webengine_on_macos_and_logs_startup() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    build_source = Path("tools/desktop/build_qt_desktop_app.py").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/desktop-beta.yml").read_text(encoding="utf-8")

    assert 'sys.platform != "darwin"' in source
    assert "QT_MAC_WANTS_LAYER" in source
    assert "ENABLE_EMBEDDED_WEBENGINE" in source
    assert "Library\" / \"Logs\" / \"IINTS-AF Desktop\" / \"desktop.log\"" in source
    assert "_install_crash_logging()" in source
    assert "faulthandler.enable" in source
    assert 'if sys.platform != "darwin"' in build_source
    assert "PySide6" in build_source
    assert 'OPTIONAL_BUNDLED_MODULES = ("plotly.graph_objects", "roadrunner", "fmpy")' in build_source
    assert 'BINARY_BUNDLED_MODULES = ("fmpy", "roadrunner")' in build_source
    assert 'command.extend(["--hidden-import", module_name])' in build_source
    assert 'command.extend(["--collect-binaries", module_name])' in build_source
    assert "def add_fmpy_sundials_binaries" in build_source
    assert 'destination = f"fmpy/sundials/{platform_tuple}"' in build_source
    assert 'command.extend(["--add-binary", f"{binary}{os.pathsep}{destination}"])' in build_source
    assert "add_fmpy_sundials_binaries(command)" in build_source
    assert "ENTRYPOINTS" in build_source
    assert '"cocoa": REPO_ROOT / "src" / "iints_desktop" / "cocoa_app.py"' in build_source
    assert '"tk": REPO_ROOT / "src" / "iints_desktop" / "app.py"' in build_source
    assert 'if args.backend == "qt"' in build_source
    assert 'if args.backend == "cocoa"' in build_source
    assert "require_pkg_resources_runtime_hook_compatibility" not in build_source
    assert '"--exclude-module",\n        "pytest"' in build_source
    assert '"--exclude-module",\n        "pkg_resources"' in build_source
    assert "--osx-bundle-identifier" in build_source
    assert '--backend qt --onedir --name "${APP_NAME}"' in workflow
    assert ".[desktop-all]" in workflow
    assert "import PySide6, fmpy, plotly, roadrunner" in workflow
    assert '"setuptools>=83,<84" pytest' in workflow
    assert 'python -m pip install -U -e ".[desktop-all]"' in workflow
    assert "repair_fmpy_macos_dylibs.py" in workflow
    assert 'python -m pip install -U -e ".[full,desktop-qt,mdmp]"' not in workflow
    assert "Smoke test bundled app on macOS" in workflow
    assert "continue-on-error: true\n        shell: bash\n        env:" not in workflow.split("Smoke test bundled app on macOS", 1)[1].split("Best-effort bundled smoke on Windows", 1)[0]


def test_tk_desktop_app_has_packaged_smoke_mode() -> None:
    source = Path("src/iints_desktop/app.py").read_text(encoding="utf-8")

    assert '"--smoke" in sys.argv' in source
    assert "Tk desktop smoke OK" in source
    assert "raise SystemExit(main())" in source


def test_qt_desktop_full_smoke_verifies_bundled_research_engines() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/desktop-beta.yml").read_text(encoding="utf-8")

    assert "def _verify_full_desktop_runtime" in source
    assert 'import_module("fmpy.sundials")' in source
    assert "CVodeSolver" in source
    assert "--smoke-full" in source
    assert workflow.count("--smoke-full") == 3
    assert "Best-effort bundled smoke on Windows" not in workflow
    assert "linux-smoke.log" in workflow
    assert "linux-smoke-combined.log" in workflow
    assert "Linux desktop smoke failed" in workflow
    assert 'cat "$HOME/.local/state/iints-af-desktop/desktop.log"' in workflow
    assert "libxkbcommon-x11-0" in workflow
    assert '--backend qt --console --name "${APP_NAME}"' in workflow
    assert "https://download.pytorch.org/whl/cpu" in workflow
    assert "torch.version.cuda is None" in workflow


def test_cocoa_desktop_app_is_macos_packaging_backend() -> None:
    source = Path("src/iints_desktop/cocoa_app.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "Cocoa desktop smoke OK" in source
    assert source.count("@objc.python_method") >= 10
    assert "NSApplication.sharedApplication()" in source
    assert "Not a medical device" in source
    assert "run_demo_preset" in source
    assert "pyobjc-framework-Cocoa" in pyproject
    assert "iints-desktop-cocoa" in pyproject
    assert "setuptools>=83.0.0,<84.0.0" in pyproject


def test_desktop_docs_use_main_branch_and_direct_downloads() -> None:
    source = Path("src/iints_desktop/qt_app.py").read_text(encoding="utf-8")
    update_source = Path("src/iints_desktop/update.py").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    app_install = Path("docs/APP_INSTALL.md").read_text(encoding="utf-8")
    desktop_docs = Path("docs/DESKTOP_APP.md").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/desktop-beta.yml").read_text(encoding="utf-8")
    release_tag = "desktop-beta-latest"

    combined = "\n".join([readme, app_install, desktop_docs])
    assert "desktop-app" not in combined
    assert "IINTS-AF-Desktop-Beta-windows-x64.exe" in combined
    assert "IINTS-AF-Desktop-Beta-macos.dmg" in combined
    assert "IINTS-AF-Desktop-Beta-linux-x64" in combined
    assert release_tag in readme
    assert release_tag in app_install
    assert release_tag in desktop_docs
    assert release_tag in source or release_tag in update_source
    assert release_tag in workflow
    assert not re.search(r"desktop-beta-\d{4}-\d{2}-\d{2}-\d+", combined)
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


def test_desktop_packager_builds_and_verifies_dmg_in_local_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "package_desktop_bundle",
        Path("tools/desktop/package_desktop_bundle.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        if command[1] == "create":
            Path(command[-1]).write_bytes(b"verified-dmg")

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/hdiutil")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    bundle = tmp_path / "IINTS-AF-Desktop-Beta.app"
    bundle.mkdir()
    output = tmp_path / "external-volume" / "IINTS-AF-Desktop-Beta-macos.dmg"
    output.parent.mkdir()

    module._create_dmg(bundle, output, "IINTS-AF-Desktop-Beta")

    assert output.read_bytes() == b"verified-dmg"
    assert calls[0][1] == "create"
    assert calls[1][1] == "verify"
    assert Path(calls[0][-1]).parent != output.parent


def test_desktop_beta_workflow_documents_optional_signing() -> None:
    workflow = Path(".github/workflows/desktop-beta.yml").read_text(encoding="utf-8")
    signing_docs = Path("docs/DESKTOP_SIGNING.md").read_text(encoding="utf-8")

    assert "WINDOWS_SIGNING_PFX_BASE64" in workflow
    assert "signtool" in workflow
    assert "MACOS_CERTIFICATE_P12_BASE64" in workflow
    assert "codesign --deep --force --options runtime" in workflow
    assert "xcrun notarytool submit" in workflow
    assert "xcrun stapler staple" in workflow
    assert "iints-temporary-ci-keychain" not in workflow
    assert "${MACOS_KEYCHAIN_PASSWORD:-" not in workflow
    assert "WINDOWS_SIGNING_PFX_BASE64" in signing_docs
    assert "MACOS_CERTIFICATE_P12_BASE64" in signing_docs
    assert "APPLE_APP_SPECIFIC_PASSWORD" in signing_docs


def test_tauri_app_exposes_diagnostics_and_safe_open_actions() -> None:
    rust_source = Path("apps/iints-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    frontend = Path("apps/iints-tauri/frontend/index.html").read_text(encoding="utf-8")
    frontend_js = Path("apps/iints-tauri/frontend/main.js").read_text(encoding="utf-8")
    frontend_css = Path("apps/iints-tauri/frontend/styles.css").read_text(encoding="utf-8")
    readme = Path("apps/iints-tauri/README.md").read_text(encoding="utf-8")

    assert "async fn desktop_diagnostics" in rust_source
    assert "async fn open_path" in rust_source
    assert "SAFE_EXTENSIONS" in rust_source
    assert "Refusing to open unsupported file type" in rust_source
    assert "diagnostics-btn" in frontend
    assert "open-run-folder-btn" in frontend
    assert "open-certificate-btn" in frontend
    assert "desktop_diagnostics" in frontend_js
    assert "open_path" in frontend_js
    assert "diagnostics-grid" in frontend_css
    assert "allowlisted native opener" in readme


def test_tauri_app_has_academic_navigation_and_platform_icons() -> None:
    app_root = Path("apps/iints-tauri")
    frontend = (app_root / "frontend/index.html").read_text(encoding="utf-8")
    frontend_js = (app_root / "frontend/main.js").read_text(encoding="utf-8")
    frontend_css = (app_root / "frontend/styles.css").read_text(encoding="utf-8")
    config = json.loads((app_root / "src-tauri/tauri.conf.json").read_text(encoding="utf-8"))

    views = {"overview", "run", "results", "reproducibility", "ai", "research", "evidence", "settings"}
    assert all(f'data-view="{view}"' in frontend for view in views)
    assert all(f'data-view-panel="{view}"' in frontend for view in views)
    assert "VIEW_METADATA" in frontend_js
    assert "Promise.allSettled" in frontend_js
    assert ".app-sidebar" in frontend_css
    assert "prefers-reduced-motion" in frontend_css

    configured_icons = set(config["bundle"]["icon"])
    assert {"icons/icon.icns", "icons/icon.ico", "icons/icon.png"}.issubset(configured_icons)
    assert all((app_root / "src-tauri" / icon).is_file() for icon in configured_icons)
    assert (app_root / "src-tauri/icons/icon-source.png").is_file()
    assert (app_root / "frontend/app-mark.png").is_file()
    assert (app_root / "frontend/iints-logo.png").is_file()


def test_tauri_workbench_is_sober_responsive_and_documented() -> None:
    app_root = Path("apps/iints-tauri")
    frontend = (app_root / "frontend/index.html").read_text(encoding="utf-8")
    frontend_js = (app_root / "frontend/main.js").read_text(encoding="utf-8")
    frontend_css = (app_root / "frontend/styles.css").read_text(encoding="utf-8")
    tauri_config = json.loads((app_root / "src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    guide = Path("docs/RESEARCH_WORKBENCH_GUIDE.md").read_text(encoding="utf-8")

    assert "skip-link" in frontend
    assert "guide-btn" in frontend
    assert "results-status" in frontend
    assert "ai-model-options" in frontend
    assert "renderAiAnswer" in frontend_js
    assert "appendReadableText" in frontend_js
    assert "initializeNativeInteractionPolicy" in frontend_js
    assert '"contextmenu"' in frontend_js
    assert 'renderMode = "immediate"' in frontend_js
    assert 'behavior: "smooth"' not in frontend_js
    assert "section-number" not in frontend
    assert '<span class="brand-wordmark">IINTS-AF</span>' in frontend
    assert "workbench-mark.svg" not in frontend
    assert not (app_root / "frontend/workbench-mark.svg").exists()
    assert "settings-save-btn" in frontend
    assert "settings-guide-btn" in frontend
    for picker_id in [
        "settings-output-browse-btn",
        "output-browse-btn",
        "csv-browse-btn",
        "academic-run-browse-btn",
        "mechanistic-model-browse-btn",
        "copasi-model-browse-btn",
        "cellml-model-browse-btn",
        "fmi-model-browse-btn",
    ]:
        assert picker_id in frontend
    assert "nativeOpenDialog" in frontend_js
    assert "chooseLocalPath" in frontend_js
    assert "refreshActionAvailability" in frontend_js
    assert ".path-picker" in frontend_css
    capabilities = json.loads(
        (app_root / "src-tauri/capabilities/main.json").read_text(encoding="utf-8")
    )
    assert "dialog:allow-open" in capabilities["permissions"]
    assert not any(
        permission.startswith(("fs:", "shell:", "http:", "updater:", "process:"))
        for permission in capabilities["permissions"]
    )
    assert "SETTINGS_STORAGE_KEY" in frontend_js
    assert "isAllowedLocalAiHost" in frontend_js
    assert "desktop_app_info" in frontend_js
    assert "tauri-beta-latest" in frontend_js
    assert "fn desktop_app_info" in (app_root / "src-tauri/src/main.rs").read_text(encoding="utf-8")
    assert "code-panel-header" in frontend
    assert "diagnostic-state::before" in frontend_css
    assert "user-select: none" in frontend_css
    assert "user-select: text" in frontend_css
    assert "linear-gradient" not in frontend_css
    assert "box-shadow: var(--shadow)" not in frontend_css
    assert "@media (max-width: 1040px)" in frontend_css
    assert "@media (max-width: 520px)" in frontend_css
    assert "overflow-x: auto" in frontend_css
    assert all(window.get("devtools") is False for window in tauri_config["app"]["windows"])
    assert "documentation fixture" in guide

    screenshots = [
        "01-overview.png",
        "02-run-protocol.png",
        "03-results.png",
        "04-local-ai.png",
        "05-reproducibility.png",
        "06-research-tools.png",
        "07-settings.png",
    ]
    assert all((Path("docs/assets/workbench") / name).is_file() for name in screenshots)


def test_tauri_beta_workflow_builds_native_installers_with_stable_links() -> None:
    workflow_path = Path(".github/workflows/tauri-desktop-beta.yml")
    workflow = workflow_path.read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    docs = Path("docs/TAURI_DESKTOP.md").read_text(encoding="utf-8")
    packager = Path("tools/desktop/package_tauri_bundle.py").read_text(encoding="utf-8")

    assert "tauri-beta-latest" in workflow
    assert 'git tag -f tauri-beta-latest "$GITHUB_SHA"' in workflow
    assert "Refresh stable desktop prerelease" in workflow
    assert "npm run tauri -- build --bundles" in workflow
    assert '"$executable" --smoke' in workflow
    assert "cargo clippy" in workflow
    assert 'echo "APPLE_SIGNING_IDENTITY=$MACOS_SIGNING_IDENTITY" >> "$GITHUB_ENV"' in workflow
    assert "APPLE_SIGNING_IDENTITY: ${{ secrets.MACOS_SIGNING_IDENTITY }}" not in workflow
    assert 'echo "APPLE_SIGNING_IDENTITY=-" >> "$GITHUB_ENV"' in workflow
    assert "codesign --verify --deep --strict --verbose=4" in workflow
    assert "Verify sealed macOS app inside DMG" in workflow
    assert "package_tauri_bundle.py" in workflow
    assert "IINTS-AF-Research-Workbench-windows-x64-setup.exe" in packager
    assert "IINTS-AF-Research-Workbench-macos.dmg" in packager
    assert "IINTS-AF-Research-Workbench-linux-x64.AppImage" in packager
    assert "tauri-beta-latest" in readme
    assert "tauri-beta-latest" in docs


def test_tauri_app_exposes_sdk_update_actions_safely() -> None:
    rust_source = Path("apps/iints-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    bridge_source = Path("src/iints_desktop/tauri_bridge.py").read_text(encoding="utf-8")
    frontend = Path("apps/iints-tauri/frontend/index.html").read_text(encoding="utf-8")
    frontend_js = Path("apps/iints-tauri/frontend/main.js").read_text(encoding="utf-8")
    frontend_css = Path("apps/iints-tauri/frontend/styles.css").read_text(encoding="utf-8")
    readme = Path("apps/iints-tauri/README.md").read_text(encoding="utf-8")

    assert "async fn desktop_update_info" in rust_source
    assert "async fn open_sdk_update_terminal" in rust_source
    assert "build_sdk_update_command_parts" in rust_source
    assert "build_sdk_install_command_text" in rust_source
    assert "build_sdk_maintenance_command_text" in rust_source
    assert "managed_python_engine_path" in rust_source
    assert "Python 3.10-3.14" in rust_source
    assert "import iints_desktop.tauri_bridge" in rust_source
    assert "iints-sdk-python35[desktop-all]" in rust_source
    assert "github.com" in rust_source
    assert "python35.github.io" in rust_source
    assert "def _update_info" in bridge_source
    assert "update-info" in bridge_source
    assert "update-terminal-btn" in frontend
    assert "install-engine-btn" in frontend
    assert "Install or update Python SDK" in frontend
    assert "update-status" in frontend
    assert "desktop_update_info" in frontend_js
    assert "open_sdk_update_terminal" in frontend_js
    assert "~/.iints-af/python-engine" in frontend_js
    assert "update-panel" in frontend_css
    assert "fixed Rust-owned command" in readme


def test_tauri_app_exposes_biology_and_stressor_actions() -> None:
    rust_source = Path("apps/iints-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    bridge_source = Path("src/iints_desktop/tauri_bridge.py").read_text(encoding="utf-8")
    frontend = Path("apps/iints-tauri/frontend/index.html").read_text(encoding="utf-8")
    frontend_js = Path("apps/iints-tauri/frontend/main.js").read_text(encoding="utf-8")
    frontend_css = Path("apps/iints-tauri/frontend/styles.css").read_text(encoding="utf-8")
    readme = Path("apps/iints-tauri/README.md").read_text(encoding="utf-8")

    assert "async fn list_molecule_assets" in rust_source
    assert "async fn generate_molecule_pae" in rust_source
    assert "async fn reveal_path" in rust_source
    assert "async fn run_genomics_simulation" in rust_source
    assert "async fn run_tissue_stress" in rust_source
    assert '"cif", "mmcif"' in rust_source
    assert "def _molecules" in bridge_source
    assert "def _molecule_pae" in bridge_source
    assert "def _genomics_sim" in bridge_source
    assert "def _tissue_stress" in bridge_source
    assert "contextlib.redirect_stdout" in bridge_source
    assert "molecule-list" in frontend
    assert "molecule-viewer-canvas" in frontend
    assert "genomics-run-btn" in frontend
    assert "tissue-run-btn" in frontend
    assert "list_molecule_assets" in frontend_js
    assert "generate_molecule_pae" in frontend_js
    assert "openMoleculeViewer" in frontend_js
    assert "reveal_path" in frontend_js
    assert "run_genomics_simulation" in frontend_js
    assert "run_tissue_stress" in frontend_js
    assert "molecule-card" in frontend_css
    assert "molecule-viewer-panel" in frontend_css
    assert "genomics and tissue-specific resistance stressor plots" in readme


def test_tauri_app_exposes_evidence_connectors_safely() -> None:
    rust_source = Path("apps/iints-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    bridge_source = Path("src/iints_desktop/tauri_bridge.py").read_text(encoding="utf-8")
    frontend = Path("apps/iints-tauri/frontend/index.html").read_text(encoding="utf-8")
    frontend_js = Path("apps/iints-tauri/frontend/main.js").read_text(encoding="utf-8")
    frontend_css = Path("apps/iints-tauri/frontend/styles.css").read_text(encoding="utf-8")
    readme = Path("apps/iints-tauri/README.md").read_text(encoding="utf-8")

    assert "async fn list_evidence_connectors" in rust_source
    assert "async fn open_external_url" in rust_source
    assert "ALLOWED_EXTERNAL_HOSTS" in rust_source
    assert "Only HTTPS evidence links are allowed" in rust_source
    assert "alphafold.ebi.ac.uk" in rust_source
    assert "platform-docs.opentargets.org" in rust_source
    assert "www.researchobject.org" in rust_source
    assert "sed-ml.org" in rust_source
    assert "def _evidence_connectors" in bridge_source
    assert "list_evidence_connectors" in bridge_source
    assert "evidence-refresh-btn" in frontend
    assert "evidence-list" in frontend
    assert "list_evidence_connectors" in frontend_js
    assert "open_external_url" in frontend_js
    assert "evidence-card" in frontend_css
    assert "Evidence connectors" in readme
    assert "Rust HTTPS host allowlist" in readme


def test_tauri_app_exposes_academic_bundle_through_audited_bridge() -> None:
    rust_source = Path("apps/iints-tauri/src-tauri/src/main.rs").read_text(encoding="utf-8")
    bridge_source = Path("src/iints_desktop/tauri_bridge.py").read_text(encoding="utf-8")
    frontend = Path("apps/iints-tauri/frontend/index.html").read_text(encoding="utf-8")
    frontend_js = Path("apps/iints-tauri/frontend/main.js").read_text(encoding="utf-8")

    assert "async fn export_academic_bundle" in rust_source
    assert "export_academic_bundle," in rust_source
    assert "def _academic_bundle" in bridge_source
    assert 'subcommands.add_parser("academic-bundle")' in bridge_source
    assert "academic-export-btn" in frontend
    assert "does not upload data" in frontend
    assert 'call("export_academic_bundle"' in frontend_js
