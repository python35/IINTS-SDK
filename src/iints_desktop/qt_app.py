from __future__ import annotations

import os
import subprocess
import sys
import json
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox"
if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

import traceback
import faulthandler
from importlib import import_module, resources
from pathlib import Path
from typing import Any, cast

from iints.ai.backends.ollama import DEFAULT_MINISTRAL_MODEL

from iints_desktop.engine import (
    DEFAULT_DESKTOP_PRESET_KEY,
    DesktopPreset,
    DesktopRunHistoryEntry,
    DesktopRunResult,
    get_desktop_environment,
    get_desktop_preset,
    list_desktop_presets,
    read_run_history,
    run_demo_preset,
    run_custom_preset,
)
from iints_desktop.evidence_connectors import EvidenceConnector, list_evidence_connectors
from iints_desktop.local_ai import (
    RECOMMENDED_OLLAMA_MODELS,
    ask_local_ai,
    check_local_ai,
    list_local_ai_models,
    start_local_ai_stack,
)
from iints_desktop.molecules import MoleculeAsset, list_molecule_assets, pae_html_path
from iints_desktop.results import ResultPreview, load_results_preview
from iints_desktop.fetcher import fetch_alphafold_structure
from iints_desktop.render_3dmol import generate_3dmol_html
from iints_desktop.update import (
    DESKTOP_RELEASE_URL,
    UPDATE_DOCS_URL,
    build_python_sdk_update_args,
    build_python_sdk_update_command,
)

PYTHON_SDK_UPDATE_COMMAND = build_python_sdk_update_command()
ENABLE_EMBEDDED_WEBENGINE = os.environ.get("QT_QPA_PLATFORM") != "offscreen"
_CRASH_LOG_HANDLE: Any | None = None
_QWEBENGINE_VIEW: Any = None



def _desktop_log_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "IINTS-AF Desktop" / "desktop.log"
    if sys.platform.startswith("win"):
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "IINTS-AF Desktop" / "desktop.log"
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "iints-af-desktop" / "desktop.log"


def _write_startup_log(message: str) -> None:
    try:
        log_path = _desktop_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


def _install_crash_logging() -> None:
    global _CRASH_LOG_HANDLE
    try:
        log_path = _desktop_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _CRASH_LOG_HANDLE = log_path.open("a", encoding="utf-8")
        _CRASH_LOG_HANDLE.write("\n--- IINTS-AF Desktop startup ---\n")
        _CRASH_LOG_HANDLE.write(f"platform={sys.platform} executable={sys.executable}\n")
        _CRASH_LOG_HANDLE.flush()
        faulthandler.enable(file=_CRASH_LOG_HANDLE)
    except Exception:
        pass


_install_crash_logging()

try:  # pragma: no cover - optional GUI dependency
    from PySide6.QtCore import Qt, QObject, QSettings, QThread, QUrl, Signal, Slot  # type: ignore[import-not-found]
    from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPalette, QPixmap, QShortcut, QTextCursor  # type: ignore[import-not-found]
    from PySide6.QtWidgets import (  # type: ignore[import-not-found]
        QApplication,
        QCheckBox,
        QComboBox,
        QDockWidget,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QScrollArea,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QListWidget,
        QTextEdit,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QGridLayout,
        QHeaderView,
        QSizePolicy,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional GUI dependency
    _PYSIDE_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:  # pragma: no cover - optional GUI dependency
    _PYSIDE_IMPORT_ERROR = None
    if ENABLE_EMBEDDED_WEBENGINE:
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore[import-not-found]

            _QWEBENGINE_VIEW = QWebEngineView
        except ModuleNotFoundError:
            _QWEBENGINE_VIEW = None
    else:
        _QWEBENGINE_VIEW = None


def desktop_icon_path() -> Path | None:
    """Return the bundled desktop icon path when available."""

    try:
        icon = resources.files("iints_desktop").joinpath("assets").joinpath("app_icon.png")
        if icon.is_file():
            return Path(str(icon))
    except Exception:
        return None
    return None


if _PYSIDE_IMPORT_ERROR is None:
    from iints_desktop.molecule_viewer import MolecularChainViewer

    class EmittingStream(QObject):
        textWritten = Signal(str)

        def write(self, text: str) -> None:
            self.textWritten.emit(text)

        def flush(self) -> None:
            pass

    class RunWorker(QObject):
        """Background SDK run worker so the Qt UI stays responsive."""

        finished = Signal(object)
        failed = Signal(str)
        log = Signal(str)
        telemetry = Signal(int, int, float)

        def __init__(self, *, output_dir: str, desktop_preset_key: str | None = None, seed: int, custom_preset: dict[str, Any] | None = None) -> None:
            super().__init__()
            self.output_dir = output_dir
            self.desktop_preset_key = desktop_preset_key
            self.seed = seed
            self.custom_preset = custom_preset

        @Slot()
        def run(self) -> None:
            try:
                def step_cb(step: int, total: int, glucose: float) -> None:
                    self.telemetry.emit(step, total, glucose)

                self.log.emit("Calling the IINTS-AF SDK engine...\n")
                if self.custom_preset is not None:
                    result = run_custom_preset(
                        output_dir=self.output_dir,
                        custom_preset=self.custom_preset,
                        seed=self.seed,
                        step_callback=step_cb,
                    )
                else:
                    assert self.desktop_preset_key is not None
                    result = run_demo_preset(
                        output_dir=self.output_dir,
                        desktop_preset_key=self.desktop_preset_key,
                        seed=self.seed,
                        step_callback=step_cb,
                    )
                self.finished.emit(result)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class AIWorker(QObject):
        """Background local-AI worker so Ollama calls do not freeze the UI."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, question: str, model: str, host: str, result_csv: str | None) -> None:
            super().__init__()
            self.question = question
            self.model = model
            self.host = host
            self.result_csv = result_csv

        @Slot()
        def run(self) -> None:
            try:
                answer = ask_local_ai(
                    question=self.question,
                    model=self.model,
                    host=self.host or None,
                    result_csv=self.result_csv,
                )
                self.finished.emit(answer)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class LocalAIStartWorker(QObject):
        """Background worker that starts Ollama and prepares the local model."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, model: str, host: str) -> None:
            super().__init__()
            self.model = model
            self.host = host

        @Slot()
        def run(self) -> None:
            try:
                result = start_local_ai_stack(model=self.model, host=self.host or None)
                self.finished.emit(result)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class MDMPCertifyWorker(QObject):
        """Background worker that signs a loaded result CSV with local MDMP evidence."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, csv_path: str) -> None:
            super().__init__()
            self.csv_path = csv_path

        @Slot()
        def run(self) -> None:
            try:
                from iints_desktop.mdmp import create_desktop_mdmp_certificate

                result = create_desktop_mdmp_certificate(self.csv_path)
                self.finished.emit(result)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class AcademicBundleWorker(QObject):
        """Background worker for FAIR-oriented metadata and checksum export."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(
            self,
            *,
            run_dir: Path,
            creator_name: str,
            creator_orcid: str,
            license_id: str,
        ) -> None:
            super().__init__()
            self.run_dir = run_dir
            self.creator_name = creator_name
            self.creator_orcid = creator_orcid
            self.license_id = license_id

        @Slot()
        def run(self) -> None:
            try:
                from iints.research.academic_bundle import build_academic_bundle

                result = build_academic_bundle(
                    self.run_dir,
                    creator_name=self.creator_name or None,
                    creator_orcid=self.creator_orcid or None,
                    license_id=self.license_id,
                )
                self.finished.emit(result)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())





    class PAEWorker(QObject):
        """Background worker that renders an interactive AlphaFold PAE heatmap."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, target: str, output_dir: Path) -> None:
            super().__init__()
            self.target = target
            self.output_dir = output_dir

        @Slot()
        def run(self) -> None:
            try:
                from iints.research.structure import render_pae

                results = render_pae(self.target, output_dir=self.output_dir)
                self.finished.emit(results[0] if results else None)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())



    class AlphaFoldFetchWorker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, uniprot_id: str, output_dir: Path):
            super().__init__()
            self.uniprot_id = uniprot_id
            self.output_dir = output_dir

        @Slot()
        def run(self) -> None:
            try:
                cif_path = fetch_alphafold_structure(self.uniprot_id, self.output_dir)
                html_path = generate_3dmol_html(cif_path, self.output_dir)
                self.finished.emit((cif_path, html_path, self.uniprot_id))
            except Exception as exc:
                self.failed.emit(str(exc))

    class GenomicsWorker(QObject):
        """Background worker for running multi-scale structural-metabolic coupling simulations."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, gene: str, variant: str, out_dir: Path) -> None:
            super().__init__()
            self.gene = gene
            self.variant = variant
            self.out_dir = out_dir

        @Slot()
        def run(self) -> None:
            try:
                from iints.research.genomics_engine import GenomicsEngine
                html_path, data = GenomicsEngine.run_multi_scale_simulation(self.gene, self.variant, self.out_dir)

                msg = (
                    f"Illustrative comparison for {self.gene} {self.variant}. "
                    f"Scenario assumption: {int(data['scalar'] * 100)}% retained function. "
                    f"Description: {data['desc']}. Plot saved to: {html_path}"
                )
                self.finished.emit((msg, data))
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())

    class TissueStressorWorker(QObject):
        """Background worker for running comparative tissue-specific stress tests."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, muscle_scalar: float, liver_scalar: float, out_dir: Path) -> None:
            super().__init__()
            self.muscle_scalar = muscle_scalar
            self.liver_scalar = liver_scalar
            self.out_dir = out_dir

        @Slot()
        def run(self) -> None:
            try:
                from iints.research.tissue_stressor import TissueStressor
                html_path, data = TissueStressor.run_stress_test(self.muscle_scalar, self.liver_scalar, self.out_dir)
                msg = f"Simulated tissue-specific stress (Muscle {int(data['muscle']*100)}%, Liver {int(data['liver']*100)}%). Plot saved to {html_path}."
                self.finished.emit((msg, data))
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class IINTSQtDesktopApp(QMainWindow):
        """PySide/Qt desktop shell for a more polished native app experience."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("IINTS-AF SDK | Research Workbench")
            self._apply_app_icon()
            self.resize(1240, 820)
            self.setMinimumSize(760, 520)

            self.settings = QSettings("IINTS-AF", "Desktop")
            self.presets = list_desktop_presets()
            saved_workflow = str(self.settings.value("workflow_key", DEFAULT_DESKTOP_PRESET_KEY))
            try:
                self.default_preset = get_desktop_preset(saved_workflow)
            except ValueError:
                self.default_preset = get_desktop_preset(DEFAULT_DESKTOP_PRESET_KEY)
            self.last_result: DesktopRunResult | None = None
            self.loaded_result: ResultPreview | None = None
            self.last_mdmp_certificate_dir: Path | None = None
            self.last_academic_bundle: object | None = None
            self.history_entries: list[DesktopRunHistoryEntry] = []
            self.current_thread: QThread | None = None
            self.current_worker: RunWorker | None = None
            self.ai_thread: QThread | None = None
            self.ai_worker: AIWorker | None = None
            self.ai_start_thread: QThread | None = None
            self.ai_start_worker: LocalAIStartWorker | None = None
            self.mdmp_thread: QThread | None = None
            self.mdmp_worker: MDMPCertifyWorker | None = None
            self.academic_thread: QThread | None = None
            self.academic_worker: AcademicBundleWorker | None = None
            self.update_thread: QThread | None = None
            self.update_worker = None
            self.pae_thread: QThread | None = None
            self.pae_worker: PAEWorker | None = None
            self.biology_thread: QThread | None = None
            self.biology_worker: QObject | None = None
            # Compatibility guard for older UI paths that expected a docked
            # terminal widget before the About tab was built.
            self.terminal_dock: QWidget | None = None
            self.tabs: QTabWidget | None = None
            self.workspace_status: QLabel | None = None
            self.molecules = list_molecule_assets()
            self.molecule_viewer: MolecularChainViewer | None = None
            self.responsive_splitters: list[QSplitter] = []

            default_output = str(self.settings.value("output_dir", str(Path.home() / "IINTS-Desktop-Runs")))
            self.output_dir = QLineEdit(default_output)
            self.output_dir.setMinimumWidth(0)
            self.workflow_combo = QComboBox()
            self.description = QLabel()
            self.description.setWordWrap(True)
            self.seed = QSpinBox()
            self.seed.setRange(0, 999_999_999)
            self.seed.setValue(self._saved_seed())
            self.status = QLabel("Ready")
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.result_csv_path = QLineEdit()
            self.result_csv_path.setMinimumWidth(0)
            self.result_summary = QLabel("No results loaded yet.")
            self.result_summary.setWordWrap(True)
            self.result_table = QTableWidget(0, 0)
            from PySide6.QtWidgets import QStackedWidget
            self.result_graph_stack = QStackedWidget()
            self.result_graph_label = QLabel("Load a results CSV to view a glucose graph.")
            self.result_graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.result_graph_stack.addWidget(self.result_graph_label)
            self.result_graph_web: Any | None
            if _QWEBENGINE_VIEW:
                self.result_graph_web = _QWEBENGINE_VIEW()
                self.result_graph_stack.addWidget(self.result_graph_web)
            else:
                self.result_graph_web = None
            self.history_table = QTableWidget(0, 7)
            self.ai_model = QComboBox()
            self.ai_model.setEditable(True)
            self.batch_queue: list[dict[str, Any]] = []
            self.batch_running = False
            self.batch_list_widget = QListWidget()
            self.ai_model.addItems(list(RECOMMENDED_OLLAMA_MODELS))
            self.ai_model.setCurrentText(DEFAULT_MINISTRAL_MODEL)
            self.ai_host = QComboBox()
            self.ai_host.setEditable(True)
            self.ai_host.addItems(["http://127.0.0.1:11434", "https://api.huggingface.co/v1"])
            self.ai_host.setCurrentText("http://127.0.0.1:11434")
            self.ai_model.setMinimumWidth(0)
            self.ai_host.setMinimumWidth(0)
            self.ai_question = QTextEdit()
            self.ai_answer = QTextEdit()
            self.ai_answer.setReadOnly(True)
            self.ai_status = QLabel("Local AI not checked yet.")
            self.ai_context_label = QLabel("AI context: no result CSV loaded yet.")
            self.ai_context_label.setWordWrap(True)
            self.molecule_selector = QComboBox()
            self.molecule_title = QLabel()
            self.molecule_explanation = QLabel()
            self.molecule_structure_status = QLabel()
            self.molecule_pae_status = QLabel()
            self.biology_action_status = QLabel("No biology evidence action has run yet.")
            self.biology_action_status.setWordWrap(True)
            self.biology_action_output = QTextEdit()
            self.biology_action_output.setReadOnly(True)
            self.evidence_connectors = list_evidence_connectors()
            self.evidence_connector_selector = QComboBox()
            self.evidence_connector_details = QLabel()
            self.evidence_connector_details.setWordWrap(True)
            self.open_evidence_portal_button = QPushButton("Open Official Portal")
            self.open_evidence_docs_button = QPushButton("Open API / Specification")
            self.molecule_web_view: Any | None = None
            self.pae_web_view: Any | None = None

            self.run_button = QPushButton("Run Selected Workflow")
            self.run_button.setObjectName("primaryAction")
            self.run_button.clicked.connect(self._run_selected_workflow)
            self.queue_button = QPushButton("Add to Batch")
            self.queue_button.clicked.connect(lambda: self._add_to_batch("preset"))
            self.open_folder_button = QPushButton("Open Output Folder")
            self.open_report_button = QPushButton("Open PDF Report")
            self.open_csv_button = QPushButton("Open Results CSV")
            self.open_selected_folder_button = QPushButton("Open Selected Folder")
            self.open_selected_report_button = QPushButton("Open Selected PDF")
            self.open_selected_csv_button = QPushButton("Open Selected CSV")
            self.load_selected_history_csv_button = QPushButton("View Selected CSV")
            self.copy_summary_button = QPushButton("Copy Last Summary")
            self.save_log_button = QPushButton("Save Log")
            self.load_result_button = QPushButton("Load CSV")
            self.browse_result_button = QPushButton("Browse CSV...")
            self.load_last_result_button = QPushButton("Load Last Run CSV")
            self.open_loaded_csv_button = QPushButton("Open Loaded CSV")
            self.open_graph_button = QPushButton("Open Graph PNG")
            self.create_mdmp_cert_button = QPushButton("Create MDMP Certificate")
            self.open_mdmp_cert_folder_button = QPushButton("Open Certificate Folder")
            self.create_academic_bundle_button = QPushButton("Create Academic Package")
            self.open_academic_metadata_button = QPushButton("Open RO-Crate Metadata")
            self.open_academic_audit_button = QPushButton("Open Academic Audit")
            self.academic_creator = QLineEdit(str(self.settings.value("academic_creator", "")))
            self.academic_creator.setPlaceholderText("Optional researcher name")
            self.academic_orcid = QLineEdit(str(self.settings.value("academic_orcid", "")))
            self.academic_orcid.setPlaceholderText("https://orcid.org/0000-0000-0000-0000")
            self.academic_license = QLineEdit(str(self.settings.value("academic_license", "NOASSERTION")))
            self.academic_license.setPlaceholderText("For example CC-BY-4.0")
            self.academic_bundle_status = QLabel("No academic package generated yet.")
            self.academic_bundle_status.setWordWrap(True)
            self.export_workspace_button = QPushButton("Export Workspace (.zip)")
            self.start_ai_button = QPushButton("Start Local AI")
            self.check_ai_button = QPushButton("Check Ollama")
            self.refresh_ai_models_button = QPushButton("Refresh Models")
            self.ask_ai_button = QPushButton("Ask Local AI")
            self.copy_ai_answer_button = QPushButton("Copy AI Answer")
            self.quick_explain_button = QPushButton("Explain Run")
            self.quick_realism_button = QPushButton("Find Realism Issues")
            self.quick_doctor_button = QPushButton("Doctor Summary")
            self.reset_molecule_view_button = QPushButton("Reset 3D View")
            self.open_molecule_image_button = QPushButton("Open Reference PNG")
            self.open_molecule_structure_button = QPushButton("Open mmCIF")
            self.generate_pae_button = QPushButton("Generate PAE Heatmap")
            self.open_pae_button = QPushButton("Open PAE HTML")
            self.open_pae_folder_button = QPushButton("Open PAE Folder")
            self.genomics_variant_input = QLineEdit("V938M")
            self.genomics_variant_input.setPlaceholderText("e.g. V938M, R1174W, A1135E")
            self.run_genomics_sim_button = QPushButton("Run Multi-Scale Simulation")
            self.highlight_mutation_button = QPushButton("Highlight Mutation in 3D")
            self.open_structural_folder_button = QPushButton("Open Genomics Folder")

            # Tissue-specific resistance UI
            self.tissue_muscle_input = QSpinBox()
            self.tissue_muscle_input.setRange(0, 100)
            self.tissue_muscle_input.setValue(100)
            self.tissue_muscle_input.setSuffix("%")
            self.tissue_liver_input = QSpinBox()
            self.tissue_liver_input.setRange(0, 100)
            self.tissue_liver_input.setValue(100)
            self.tissue_liver_input.setSuffix("%")
            self.run_tissue_stress_button = QPushButton("Stress-Test Pump Algorithm")
            self.open_app_downloads_button = QPushButton("Open App Downloads")

            # Setup keyboard shortcuts
            self.run_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
            self.run_shortcut.activated.connect(self._run_selected_workflow)
            self.open_update_docs_button = QPushButton("Open Update Docs")
            self.copy_update_command_button = QPushButton("Copy Update Command")
            self.run_package_update_button = QPushButton("Update Python SDK Package")
            self.update_status = QLabel("No update action has run yet.")
            self.update_status.setWordWrap(True)
            self.update_log = QTextEdit()
            self.update_log.setReadOnly(True)

            self._build_ui()
            self._apply_style()
            self._set_loaded_result_actions(False)
            self._on_workflow_changed()
            self._on_molecule_changed()

        def _apply_app_icon(self) -> None:
            icon_path = desktop_icon_path()
            if icon_path is None:
                return
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                self.setWindowIcon(icon)
                app = cast(QApplication | None, QApplication.instance())
                if app is not None:
                    app.setWindowIcon(icon)

        def _build_ui(self) -> None:
            self._build_menu_bar()
            self._build_tool_bar()
            central = QWidget()
            central.setObjectName("root")
            root = QVBoxLayout(central)
            root.setContentsMargins(8, 7, 8, 7)
            root.setSpacing(6)
            self.setCentralWidget(central)

            header = QWidget()
            header.setObjectName("workbenchHeader")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(10, 8, 10, 8)
            header_layout.setSpacing(10)
            title = QLabel("IINTS-AF SDK")
            title.setObjectName("appTitle")
            title_font = QFont()
            title_font.setPointSize(17)
            title_font.setBold(True)
            title.setFont(title_font)
            subtitle = QLabel("Research Workbench · simulation · data review · local AI")
            subtitle.setWordWrap(True)
            subtitle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            header_text = QVBoxLayout()
            header_text.setContentsMargins(0, 0, 0, 0)
            header_text.setSpacing(1)
            header_text.addWidget(title)
            header_text.addWidget(subtitle)
            header_layout.addLayout(header_text, stretch=1)
            research_badge = QLabel("RESEARCH ONLY · NOT FOR CLINICAL USE")
            research_badge.setObjectName("researchBadge")
            research_badge.setWordWrap(True)
            research_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            research_badge.setMinimumWidth(0)
            header_layout.addWidget(research_badge)
            root.addWidget(header)

            tabs = QTabWidget()
            tabs.setDocumentMode(True)
            self.tabs = tabs
            tabs.setUsesScrollButtons(True)
            tabs.setElideMode(Qt.TextElideMode.ElideRight)
            root.addWidget(tabs, stretch=1)

            run_tab = QWidget()
            results_tab = QWidget()
            ai_tab = QWidget()
            history_tab = QWidget()
            molecules_tab = QWidget()
            builder_tab = QWidget()
            about_tab = QWidget()
            tabs.addTab(run_tab, "Simulation")
            tabs.addTab(results_tab, "Results")
            tabs.addTab(ai_tab, "AI Review")
            tabs.addTab(history_tab, "Run Archive")
            tabs.addTab(molecules_tab, "Biology")
            tabs.addTab(builder_tab, "Scenario Builder")
            batch_tab = QWidget()
            tabs.addTab(batch_tab, "Batch Queue")
            tabs.addTab(about_tab, "Methods")
            # Integrated Terminal Dock
            self.terminal_dock = QDockWidget("Integrated Terminal Output", self)
            self.terminal_dock.setObjectName("TerminalDockWidget")
            self.terminal_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
            self.terminal_text = QPlainTextEdit()
            self.terminal_text.setReadOnly(True)
            self.terminal_text.setFont(QFont("Courier", 10))
            self.terminal_dock.setWidget(self.terminal_text)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.terminal_dock)
            self.terminal_dock.hide() # Hidden by default

            # Redirect stdout/stderr
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            self.stream_redirector = EmittingStream()
            self.stream_redirector.textWritten.connect(self._append_terminal_text)
            sys.stdout = self.stream_redirector
            sys.stderr = self.stream_redirector
            self._build_run_tab(run_tab)
            self._build_results_tab(results_tab)
            self._build_ai_tab(ai_tab)
            self._build_history_tab(history_tab)
            self._build_molecules_tab(molecules_tab)
            self._build_builder_tab(builder_tab)
            self._build_batch_tab(batch_tab)
            self._build_about_tab(about_tab)

            status_bar = QStatusBar(self)
            status_bar.setSizeGripEnabled(False)
            status_bar.addWidget(self.status, 1)

            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedWidth(150)
            self.progress_bar.hide()
            status_bar.addPermanentWidget(self.progress_bar)

            self.live_telemetry_label = QLabel()
            self.live_telemetry_label.setStyleSheet("color: #4facfe; font-weight: bold;")
            self.live_telemetry_label.hide()
            status_bar.addPermanentWidget(self.live_telemetry_label)

            workspace_label = QLabel(f"Workspace: {Path(self.output_dir.text()).expanduser()}")
            workspace_label.setObjectName("workspaceStatus")
            status_bar.addPermanentWidget(workspace_label)
            self.workspace_status = workspace_label
            self.setStatusBar(status_bar)

        @Slot(str)
        def _append_terminal_text(self, text: str) -> None:
            self.terminal_text.moveCursor(QTextCursor.MoveOperation.End)
            self.terminal_text.insertPlainText(text)

        def closeEvent(self, event: Any) -> None:
            if sys.stdout is self.stream_redirector:
                sys.stdout = self._original_stdout
            if sys.stderr is self.stream_redirector:
                sys.stderr = self._original_stderr
            super().closeEvent(event)


        def _build_menu_bar(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            open_csv_action = QAction("Open results CSV...", self)
            open_csv_action.triggered.connect(self._browse_result_csv)
            file_menu.addAction(open_csv_action)
            output_action = QAction("Open output folder", self)
            output_action.triggered.connect(self._open_output_folder)
            file_menu.addAction(output_action)
            file_menu.addSeparator()
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.close)
            file_menu.addAction(quit_action)

            workflow_menu = self.menuBar().addMenu("&Workflow")
            run_action = QAction("Run selected simulation", self)
            run_action.triggered.connect(self._run_selected_workflow)
            workflow_menu.addAction(run_action)
            results_action = QAction("View results", self)
            results_action.triggered.connect(lambda: self._select_tab(1))
            workflow_menu.addAction(results_action)
            archive_action = QAction("Refresh run archive", self)
            archive_action.triggered.connect(self._refresh_history)
            workflow_menu.addAction(archive_action)

            view_menu = self.menuBar().addMenu("&View")
            ai_action = QAction("AI review", self)
            ai_action.triggered.connect(lambda: self._select_tab(2))
            view_menu.addAction(ai_action)
            biology_action = QAction("Biology viewer", self)
            biology_action.triggered.connect(lambda: self._select_tab(4))
            view_menu.addAction(biology_action)

        def _build_tool_bar(self) -> None:
            toolbar = QToolBar("Core actions", self)
            toolbar.setObjectName("coreToolbar")
            toolbar.setMovable(False)
            toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.addToolBar(toolbar)

            run_action = QAction("Run simulation", self)
            run_action.triggered.connect(self._run_selected_workflow)
            toolbar.addAction(run_action)
            load_action = QAction("Load CSV", self)
            load_action.triggered.connect(self._browse_result_csv)
            toolbar.addAction(load_action)
            results_action = QAction("Results", self)
            results_action.triggered.connect(lambda: self._select_tab(1))
            toolbar.addAction(results_action)
            ai_action = QAction("AI review", self)
            ai_action.triggered.connect(lambda: self._select_tab(2))
            toolbar.addAction(ai_action)
            toolbar.addSeparator()
            output_action = QAction("Output folder", self)
            output_action.triggered.connect(self._open_output_folder)
            toolbar.addAction(output_action)

        def _select_tab(self, index: int) -> None:
            if self.tabs is not None:
                self.tabs.setCurrentIndex(index)

        def _scroll_tab_layout(self, parent: QWidget) -> QVBoxLayout:
            outer = QVBoxLayout(parent)
            outer.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            outer.addWidget(scroll)
            content = QWidget()
            content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            scroll.setWidget(content)
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(7)
            return layout

        def _button_grid(self, buttons: list[QPushButton], *, columns: int = 3) -> QGridLayout:
            grid = QGridLayout()
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(6)
            for index, button in enumerate(buttons):
                button.setMinimumWidth(0)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                grid.addWidget(button, index // columns, index % columns)
            for column in range(columns):
                grid.setColumnStretch(column, 1)
            return grid

        def _register_responsive_splitter(self, splitter: QSplitter) -> None:
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(8)
            self.responsive_splitters.append(splitter)
            self._apply_responsive_layout()

        def _apply_responsive_layout(self) -> None:
            compact = self.width() < 900
            orientation = Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
            for splitter in getattr(self, "responsive_splitters", []):
                if splitter.orientation() != orientation:
                    splitter.setOrientation(orientation)
                    splitter.setSizes([1, 1] if compact else [720, 460])

        def resizeEvent(self, event: object) -> None:  # pragma: no cover - GUI sizing hook
            super().resizeEvent(event)  # type: ignore[arg-type]
            self._apply_responsive_layout()

        def _build_run_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)

            environment = get_desktop_environment(qt_available=True)
            environment_strip = QLabel(
                f"SDK {environment.sdk_version}   |   Local Python simulation engine   |   "
                "Outputs: CSV, PDF report, audit record"
            )
            environment_strip.setObjectName("infoStrip")
            layout.addWidget(environment_strip)

            workflow_box = QGroupBox("Simulation protocol")
            workflow_layout = QVBoxLayout(workflow_box)
            workflow_row = QHBoxLayout()
            workflow_layout.addLayout(workflow_row)

            for preset in self.presets:
                self.workflow_combo.addItem(f"{preset.title} ({preset.preset_name})", preset.key)
            default_index = self._workflow_index(self.default_preset.key)
            self.workflow_combo.setCurrentIndex(default_index)
            self.workflow_combo.currentIndexChanged.connect(self._on_workflow_changed)
            workflow_row.addWidget(self.workflow_combo, stretch=1)

            workflow_row.addWidget(QLabel("Seed:"))
            workflow_row.addWidget(self.seed)
            workflow_layout.addWidget(self.description)
            layout.addWidget(workflow_box)

            body_splitter = QSplitter(Qt.Orientation.Horizontal)
            self._register_responsive_splitter(body_splitter)
            layout.addWidget(body_splitter, stretch=1)

            controls_panel = QWidget()
            controls_layout = QVBoxLayout(controls_panel)
            controls_layout.setContentsMargins(0, 0, 0, 0)
            controls_layout.setSpacing(7)

            output_box = QGroupBox("Output workspace")
            output_layout = QHBoxLayout(output_box)
            choose_button = QPushButton("Choose...")
            choose_button.clicked.connect(self._choose_output_dir)
            output_layout.addWidget(self.output_dir, stretch=1)
            output_layout.addWidget(choose_button)
            controls_layout.addWidget(output_box)

            action_box = QGroupBox("Actions")
            action_layout = QVBoxLayout(action_box)
            self.run_button.clicked.connect(self._run_selected_workflow)
            self.run_button.setObjectName("primaryAction")
            self.open_folder_button.clicked.connect(self._open_output_folder)
            self.open_report_button.clicked.connect(self._open_report)
            self.open_csv_button.clicked.connect(self._open_results_csv)
            self.copy_summary_button.clicked.connect(self._copy_last_summary)
            self.save_log_button.clicked.connect(self._save_log)
            clear_button = QPushButton("Clear Log")
            clear_button.clicked.connect(self.log.clear)

            action_layout.addLayout(
                self._button_grid(
                    [
                        self.run_button,
                        self.queue_button,
                        self.open_folder_button,
                        self.open_report_button,
                        self.open_csv_button,
                        self.copy_summary_button,
                        self.save_log_button,
                        clear_button,
                    ],
                    columns=3,
                )
            )
            self._set_result_buttons_enabled(False)
            controls_layout.addWidget(action_box)
            controls_layout.addStretch(1)
            body_splitter.addWidget(controls_panel)

            log_box = QGroupBox("Execution log")
            log_layout = QVBoxLayout(log_box)
            log_layout.addWidget(self.log)
            body_splitter.addWidget(log_box)
            body_splitter.setSizes([420, 760])
            self._write_log(
                "Workbench ready. Select a protocol and run it.\n\n"
                "The desktop application calls the same SDK engine as the CLI; it does not "
                "introduce a second simulation or safety implementation.\n"
            )

        def _build_results_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)

            csv_box = QGroupBox("Data source")
            csv_layout = QVBoxLayout(csv_box)
            csv_row = QHBoxLayout()
            self.browse_result_button.clicked.connect(self._browse_result_csv)
            self.load_result_button.clicked.connect(self._load_selected_result_csv)
            self.load_last_result_button.clicked.connect(self._load_last_result_csv)
            self.open_loaded_csv_button.clicked.connect(self._open_loaded_result_csv)
            self.open_graph_button.clicked.connect(self._open_loaded_graph)
            self.create_mdmp_cert_button.clicked.connect(self._create_mdmp_certificate)
            self.open_mdmp_cert_folder_button.clicked.connect(self._open_loaded_certificate_folder)
            self.create_academic_bundle_button.clicked.connect(self._create_academic_bundle)
            self.open_academic_metadata_button.clicked.connect(self._open_academic_metadata)
            self.open_academic_audit_button.clicked.connect(self._open_academic_audit)
            self.export_workspace_button.clicked.connect(self._export_workspace)
            csv_row.addWidget(self.result_csv_path, stretch=1)
            csv_row.addWidget(self.browse_result_button)
            csv_layout.addLayout(csv_row)
            csv_layout.addLayout(
                self._button_grid(
                    [
                        self.load_result_button,
                        self.load_last_result_button,
                        self.open_loaded_csv_button,
                        self.open_graph_button,
                        self.create_mdmp_cert_button,
                        self.open_mdmp_cert_folder_button,
                        self.export_workspace_button,
                    ],
                    columns=3,
                )
            )
            layout.addWidget(csv_box)

            academic_box = QGroupBox("Reproducibility package")
            academic_layout = QVBoxLayout(academic_box)
            academic_help = QLabel(
                "Create RO-Crate 1.2 metadata, SHA-256 checksums, an evidence-source snapshot, "
                "and a reproducibility audit beside the loaded run. This does not upload data and "
                "is not peer review, privacy approval, or clinical validation."
            )
            academic_help.setWordWrap(True)
            academic_layout.addWidget(academic_help)
            academic_form = QFormLayout()
            academic_form.addRow("Creator:", self.academic_creator)
            academic_form.addRow("ORCID:", self.academic_orcid)
            academic_form.addRow("Run-artifact licence:", self.academic_license)
            academic_layout.addLayout(academic_form)
            academic_layout.addLayout(
                self._button_grid(
                    [
                        self.create_academic_bundle_button,
                        self.open_academic_metadata_button,
                        self.open_academic_audit_button,
                    ],
                    columns=3,
                )
            )
            self.academic_bundle_status.setObjectName("academicBundleStatus")
            academic_layout.addWidget(self.academic_bundle_status)
            layout.addWidget(academic_box)

            workspace = QSplitter(Qt.Orientation.Horizontal)
            self._register_responsive_splitter(workspace)
            layout.addWidget(workspace, stretch=1)

            graph_box = QGroupBox("Glucose trajectory")
            graph_layout = QVBoxLayout(graph_box)
            self.result_graph_stack.setMinimumHeight(220)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.result_graph_stack)
            graph_layout.addWidget(scroll)
            workspace.addWidget(graph_box)

            table_box = QGroupBox("Metrics and data preview")
            table_layout = QVBoxLayout(table_box)
            self.result_summary.setObjectName("metricSummary")
            table_layout.addWidget(self.result_summary)
            self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table_layout.addWidget(self.result_table)
            workspace.addWidget(table_box)
            workspace.setSizes([720, 460])

        def _build_ai_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)

            disclaimer = QLabel(
                "Local AI review uses the loaded result summary. Click Start Local AI to start Ollama and prepare the model when Ollama is installed locally. The AI can critique a run, but it has no authority over dosing, diagnosis, or treatment."
            )
            disclaimer.setObjectName("infoStrip")
            disclaimer.setWordWrap(True)
            layout.addWidget(disclaimer)

            config_box = QGroupBox("Local model connection")
            config_layout = QGridLayout(config_box)
            config_layout.addWidget(QLabel("Model:"), 0, 0)
            config_layout.addWidget(self.ai_model, 0, 1)
            config_layout.addWidget(QLabel("Host:"), 1, 0)
            config_layout.addWidget(self.ai_host, 1, 1)
            self.start_ai_button.clicked.connect(self._start_local_ai)
            self.check_ai_button.clicked.connect(self._check_ai_status)
            self.refresh_ai_models_button.clicked.connect(self._refresh_ai_models)
            config_layout.addWidget(self.start_ai_button, 0, 2)
            config_layout.addWidget(self.check_ai_button, 1, 2)
            config_layout.addWidget(self.refresh_ai_models_button, 2, 2)
            config_layout.addWidget(self.ai_status, 0, 3)
            config_layout.addWidget(self.ai_context_label, 1, 3)
            config_layout.setColumnStretch(1, 2)
            config_layout.setColumnStretch(3, 3)
            layout.addWidget(config_box)

            workspace = QSplitter(Qt.Orientation.Horizontal)
            self._register_responsive_splitter(workspace)
            layout.addWidget(workspace, stretch=1)

            question_box = QGroupBox("Research question")
            question_layout = QVBoxLayout(question_box)
            self.ai_question.setPlaceholderText(
                "Example: Explain the loaded glucose run in plain English and point out realism limitations."
            )
            self.ai_question.setMinimumHeight(120)
            question_layout.addWidget(self.ai_question)
            self.quick_explain_button.clicked.connect(
                lambda: self._set_ai_question(
                    "Explain the loaded glucose run in plain English. "
                    "Mention what happened, what is uncertain, and what should be checked next."
                )
            )
            self.quick_realism_button.clicked.connect(
                lambda: self._set_ai_question(
                    "Critically review the loaded run for simulation realism. "
                    "Look for impossible glucose jumps, too-smooth curves, weak meal timing, or unclear safety events."
                )
            )
            self.quick_doctor_button.clicked.connect(
                lambda: self._set_ai_question(
                    "Write a short doctor-facing summary of the loaded run. "
                    "Keep it research-only and include three feedback questions for a clinician."
                )
            )
            question_layout.addLayout(
                self._button_grid(
                    [self.quick_explain_button, self.quick_realism_button, self.quick_doctor_button],
                    columns=3,
                )
            )
            self.ask_ai_button.clicked.connect(self._ask_ai)
            self.copy_ai_answer_button.clicked.connect(self._copy_ai_answer)
            question_layout.addLayout(self._button_grid([self.ask_ai_button, self.copy_ai_answer_button], columns=2))
            workspace.addWidget(question_box)

            answer_box = QGroupBox("Model response")
            answer_layout = QVBoxLayout(answer_box)
            self.ai_answer.setMinimumHeight(220)
            answer_layout.addWidget(self.ai_answer)
            workspace.addWidget(answer_box)
            workspace.setSizes([480, 700])

        def _build_history_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)
            layout.setSpacing(12)

            intro = QLabel(
                "Recent desktop runs from the selected output folder. "
                "This helps keep demo outputs manageable as the SDK generates more data."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            refresh_button = QPushButton("Refresh History")
            refresh_button.clicked.connect(self._refresh_history)
            open_base_button = QPushButton("Open Base Output Folder")
            open_base_button.clicked.connect(lambda: self._open_path(Path(self.output_dir.text())))
            self.open_selected_folder_button.clicked.connect(self._open_selected_history_folder)
            self.open_selected_report_button.clicked.connect(self._open_selected_history_report)
            self.open_selected_csv_button.clicked.connect(self._open_selected_history_csv)
            self.load_selected_history_csv_button.clicked.connect(self._load_selected_history_csv)
            self.history_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
            self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

            compare_button = QPushButton("Compare Selected Runs")
            compare_button.setStyleSheet("background-color: #2b1154; color: #4facfe; font-weight: bold; border: 1px solid #4facfe;")
            compare_button.clicked.connect(self._compare_selected_runs)

            layout.addLayout(
                self._button_grid(
                    [
                        refresh_button,
                        open_base_button,
                        compare_button,
                        self.load_selected_history_csv_button,
                        self.open_selected_folder_button,
                        self.open_selected_report_button,
                        self.open_selected_csv_button,
                    ],
                    columns=3,
                )
            )

            self.history_table.setHorizontalHeaderLabels(["Time", "Workflow", "Preset", "Seed", "Run ID", "PDF", "CSV"])
            self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.history_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
            self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.history_table.itemSelectionChanged.connect(self._update_history_action_buttons)
            header = self.history_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
            layout.addWidget(self.history_table, stretch=1)
            self._refresh_history()

        def _build_molecules_tab(self, parent: QWidget) -> None:
            outer_layout = QVBoxLayout(parent)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            outer_layout.addWidget(scroll)
            content = QWidget()
            scroll.setWidget(content)

            layout = QVBoxLayout(content)
            layout.setSpacing(12)

            intro = QLabel(
                "Structural biology assets for research context. Explore bundled AlphaFold protein backbones, "
                "generate PAE heatmaps, and open supporting files without leaving the app. These assets are "
                "documentation/evidence only and never feed treatment or dosing logic."
            )
            intro.setObjectName("deepDiveIntro")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            selector_box = QGroupBox("Structure")
            selector_layout = QGridLayout(selector_box)
            for molecule in self.molecules:
                self.molecule_selector.addItem(f"{molecule.title} (UniProt {molecule.uniprot_id})", molecule.key)
            self.molecule_selector.currentIndexChanged.connect(self._on_molecule_changed)
            self.molecule_selector.setMinimumWidth(0)
            selector_layout.addWidget(QLabel("Structure:"), 0, 0)
            selector_layout.addWidget(self.molecule_selector, 0, 1)
            viewer_label = QLabel("Interactive C-alpha backbone · pLDDT colours")
            viewer_label.setWordWrap(True)
            viewer_label.setToolTip("Colours use AlphaFold pLDDT confidence values.")
            selector_layout.addWidget(viewer_label, 1, 1)
            selector_layout.setColumnStretch(1, 1)

            self.custom_uniprot_input = QLineEdit()
            self.custom_uniprot_input.setPlaceholderText("UniProt ID (e.g. Q13131)")
            self.fetch_uniprot_button = QPushButton("Fetch Live")
            self.fetch_uniprot_button.clicked.connect(self._fetch_custom_uniprot)
            fetch_layout = QHBoxLayout()
            fetch_layout.addWidget(self.custom_uniprot_input)
            fetch_layout.addWidget(self.fetch_uniprot_button)
            selector_layout.addWidget(QLabel("Fetch custom:"), 2, 0)
            selector_layout.addLayout(fetch_layout, 2, 1)

            layout.addWidget(selector_box)

            viewer_row = QSplitter(Qt.Orientation.Horizontal)
            self._register_responsive_splitter(viewer_row)
            layout.addWidget(viewer_row, stretch=1)

            viewer_box = QGroupBox("3D chain viewer")
            viewer_layout = QVBoxLayout(viewer_box)
            self.molecule_viewer = MolecularChainViewer()
            self.molecule_viewer.setMinimumHeight(260)
            viewer_layout.addWidget(self.molecule_viewer, stretch=1)

            if _QWEBENGINE_VIEW is not None and os.environ.get("QT_QPA_PLATFORM") != "offscreen" and sys.platform != "darwin":
                self.molecule_web_view = _QWEBENGINE_VIEW()
                self.molecule_web_view.setMinimumHeight(260)
                viewer_layout.addWidget(self.molecule_web_view, stretch=1)
                self.molecule_viewer.hide() # fallback hidden
            else:
                self.molecule_web_view = None

            self.reset_molecule_view_button.clicked.connect(self._reset_molecule_view)
            self.open_molecule_image_button.clicked.connect(self._open_selected_molecule_image)
            self.open_molecule_structure_button.clicked.connect(self._open_selected_molecule_structure)

            self.open_3dmol_browser_button = QPushButton("Open 3D Viewer in Browser")
            self.open_3dmol_browser_button.clicked.connect(self._open_3dmol_in_browser)
            if sys.platform != "darwin":
                self.open_3dmol_browser_button.hide() # hide if embedded web view works


            viewer_layout.addWidget(self.open_3dmol_browser_button)
            viewer_layout.addLayout(
                self._button_grid(

                    [
                        self.reset_molecule_view_button,
                        self.open_molecule_image_button,
                        self.open_molecule_structure_button,
                    ],
                    columns=3,
                )
            )
            viewer_row.addWidget(viewer_box)

            context_box = QGroupBox("Research context")
            context_layout = QVBoxLayout(context_box)
            self.molecule_title.setObjectName("moleculeTitle")
            self.molecule_title.setWordWrap(True)
            context_layout.addWidget(self.molecule_title)
            self.molecule_explanation.setWordWrap(True)
            context_layout.addWidget(self.molecule_explanation)
            self.molecule_structure_status.setObjectName("moleculeStatus")
            self.molecule_structure_status.setWordWrap(True)
            context_layout.addWidget(self.molecule_structure_status)
            pae_box = QGroupBox("Predicted aligned error (PAE)")
            pae_layout = QVBoxLayout(pae_box)
            pae_help = QLabel(
                "Generate an interactive AlphaFold PAE heatmap. Dark green means lower predicted "
                "relative-position error; white means higher uncertainty between residue positions."
            )
            pae_help.setWordWrap(True)
            pae_layout.addWidget(pae_help)
            self.molecule_pae_status.setObjectName("moleculePAEStatus")
            self.molecule_pae_status.setWordWrap(True)
            pae_layout.addWidget(self.molecule_pae_status)
            self.generate_pae_button.clicked.connect(self._generate_pae_heatmap)
            self.open_pae_button.clicked.connect(self._open_selected_pae_html)
            self.open_pae_folder_button.clicked.connect(self._open_pae_folder)
            pae_layout.addLayout(
                self._button_grid(
                    [self.generate_pae_button, self.open_pae_button, self.open_pae_folder_button],
                    columns=3,
                )
            )
            if _QWEBENGINE_VIEW is not None and os.environ.get("QT_QPA_PLATFORM") != "offscreen" and sys.platform != "darwin":
                self.pae_web_view = _QWEBENGINE_VIEW()
                self.pae_web_view.setMinimumHeight(260)
                pae_layout.addWidget(self.pae_web_view, stretch=1)
            else:
                embedded_note = QLabel(
                    "Embedded interactive preview is unavailable in this environment; "
                    "the HTML opens in the system browser instead."
                )
                embedded_note.setWordWrap(True)
                pae_layout.addWidget(embedded_note)
            context_layout.addWidget(pae_box)

            evidence_box = QGroupBox("Advanced Research & Algorithm Stressors")
            evidence_layout = QVBoxLayout(evidence_box)

            # Genomics Panel
            genomics_label = QLabel("<b>1. AlphaFold Structural Genomics</b>")
            evidence_layout.addWidget(genomics_label)

            evidence_help = QLabel(
                "Enter a UniProt ID and variant (e.g. INSR V938M or P06213 V938M). The engine will "
                "inspect residue-level AlphaFold pLDDT as structural-confidence evidence and run a separate, "
                "explicitly labelled functional-scalar scenario. pLDDT is not pathogenicity or metabolic "
                "severity and never calibrates physiology automatically."
            )
            evidence_help.setWordWrap(True)
            evidence_layout.addWidget(evidence_help)

            input_layout = QHBoxLayout()
            input_label = QLabel("Gene & Variant:")
            input_layout.addWidget(input_label)
            input_layout.addWidget(self.genomics_variant_input)
            evidence_layout.addLayout(input_layout)

            self.run_genomics_sim_button.clicked.connect(self._run_genomics_simulation)
            self.highlight_mutation_button.clicked.connect(self._highlight_mutation)
            self.open_structural_folder_button.clicked.connect(self._open_structural_folder)

            evidence_layout.addLayout(
                self._button_grid(
                    [
                        self.run_genomics_sim_button,
                        self.highlight_mutation_button,
                        self.open_structural_folder_button,
                    ],
                    columns=3,
                )
            )

            # Tissue Panel
            tissue_label = QLabel("<b>2. Tissue-Specific Resistance Test (GTEx)</b>")
            tissue_label.setStyleSheet("margin-top: 10px;")
            evidence_layout.addWidget(tissue_label)

            tissue_help = QLabel(
                "Stress-test pump algorithms by isolating insulin resistance to specific organs "
                "(e.g., Hepatic vs Peripheral resistance), informed by GTEx expression profiles."
            )
            tissue_help.setWordWrap(True)
            evidence_layout.addWidget(tissue_help)

            tissue_input_layout = QHBoxLayout()
            tissue_input_layout.addWidget(QLabel("Muscle (Peripheral) Sensitivity:"))
            tissue_input_layout.addWidget(self.tissue_muscle_input)
            tissue_input_layout.addWidget(QLabel("Liver (Hepatic) Sensitivity:"))
            tissue_input_layout.addWidget(self.tissue_liver_input)
            evidence_layout.addLayout(tissue_input_layout)

            self.run_tissue_stress_button.clicked.connect(self._run_tissue_stress_simulation)
            evidence_layout.addWidget(self.run_tissue_stress_button)

            # Shared Status box
            self.biology_action_status.setObjectName("biologyActionStatus")
            evidence_layout.addWidget(self.biology_action_status)
            self.biology_action_output.setMinimumHeight(95)
            evidence_layout.addWidget(self.biology_action_output)
            context_layout.addWidget(evidence_box)

            connector_box = QGroupBox("Academic evidence and standards catalog")
            connector_layout = QVBoxLayout(connector_box)
            connector_intro = QLabel(
                "Curated resources are labelled as integrated, partial, planned, or portal-only. "
                "Opening a portal does not import evidence or validate a model."
            )
            connector_intro.setWordWrap(True)
            connector_layout.addWidget(connector_intro)
            for connector in self.evidence_connectors:
                connector_label = f"{connector.title} · {connector.integration_level}"
                self.evidence_connector_selector.addItem(connector_label, connector.key)
            self.evidence_connector_selector.currentIndexChanged.connect(
                self._on_evidence_connector_changed
            )
            connector_layout.addWidget(self.evidence_connector_selector)
            connector_layout.addWidget(self.evidence_connector_details)
            self.open_evidence_portal_button.clicked.connect(
                lambda: self._open_selected_evidence_url("primary_url")
            )
            self.open_evidence_docs_button.clicked.connect(
                lambda: self._open_selected_evidence_url("docs_url")
            )
            connector_layout.addLayout(
                self._button_grid(
                    [self.open_evidence_portal_button, self.open_evidence_docs_button],
                    columns=2,
                )
            )
            context_layout.addWidget(connector_box)
            self._on_evidence_connector_changed()

            usage_hint = QLabel(
                "Controls: drag to rotate, mouse wheel to zoom, double-click to reset."
            )
            usage_hint.setObjectName("subtleHint")
            usage_hint.setWordWrap(True)
            context_layout.addWidget(usage_hint)
            viewer_row.addWidget(context_box)
            viewer_row.setSizes([760, 420])

        def _build_builder_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)

            config_box = QGroupBox("Scenario Configuration")
            config_layout = QFormLayout(config_box)
            self.custom_duration = QSpinBox()
            self.custom_duration.setRange(60, 10080)
            self.custom_duration.setValue(1440)
            self.custom_duration.setToolTip("Total simulation time in minutes (e.g., 1440 for 24 hours).\nLonger durations require more compute time.")
            config_layout.addRow("Duration (mins):", self.custom_duration)

            self.custom_timestep = QSpinBox()
            self.custom_timestep.setRange(1, 60)
            self.custom_timestep.setValue(5)
            self.custom_timestep.setToolTip("Integration time step in minutes (dt).\nSmaller steps increase ODE numerical stability but slow down the simulation.")
            config_layout.addRow("Time Step (mins):", self.custom_timestep)

            self.custom_seed = QSpinBox()
            self.custom_seed.setRange(0, 99999)
            self.custom_seed.setValue(42)
            self.custom_seed.setToolTip("Random seed for generating physiological noise (sensor error, basal variations).\nKeep this constant to ensure perfectly reproducible runs.")
            config_layout.addRow("Random Seed:", self.custom_seed)

            self.custom_basal = QDoubleSpinBox()
            self.custom_basal.setRange(0.0, 3.0)
            self.custom_basal.setSingleStep(0.1)
            self.custom_basal.setValue(0.8)
            self.custom_basal.setToolTip("Basal Rate [U/h]:\nThe continuous background insulin infusion rate modeled via subcutaneous absorption kinetics.")
            config_layout.addRow("Basal Rate (U/h):", self.custom_basal)

            self.custom_isf = QDoubleSpinBox()
            self.custom_isf.setRange(10.0, 150.0)
            self.custom_isf.setValue(45.0)
            self.custom_isf.setToolTip("Insulin Sensitivity Factor (ISF) [mg/dL/U]:\nDefines the physiological drop in blood glucose concentration per unit of fast-acting insulin.")
            config_layout.addRow("Insulin Sensitivity (ISF):", self.custom_isf)

            self.custom_cr = QDoubleSpinBox()
            self.custom_cr.setRange(3.0, 30.0)
            self.custom_cr.setValue(10.0)
            self.custom_cr.setToolTip("Carbohydrate Ratio [g/U]:\nThe grams of carbohydrates that are covered by 1 Unit of insulin during a meal bolus calculation.")
            config_layout.addRow("Carb Ratio (g/U):", self.custom_cr)

            # --- STEM CELL RESEARCH ---
            stem_label = QLabel("<b>Stem Cell Graft Research</b>")
            config_layout.addRow("", stem_label)

            self.custom_engraftment = QDoubleSpinBox()
            self.custom_engraftment.setRange(0.0, 200.0)
            self.custom_engraftment.setValue(0.0)
            self.custom_engraftment.setToolTip("Stem Cell Engraftment (%):\n0% = Standard T1D, 100% = Healthy Beta Cell Mass.")
            config_layout.addRow("Engraftment (%):", self.custom_engraftment)

            self.custom_subq_fraction = QDoubleSpinBox()
            self.custom_subq_fraction.setRange(0.0, 1.0)
            self.custom_subq_fraction.setSingleStep(0.1)
            self.custom_subq_fraction.setValue(0.0)
            self.custom_subq_fraction.setToolTip("Subcutaneous Fraction (0.0 - 1.0):\n0.0 = Portal Vein Injection (Fast Kinetics)\n1.0 = SubQ Encapsulation (Delayed Kinetics via S1 compartment).")
            config_layout.addRow("SubQ Fraction:", self.custom_subq_fraction)

            self.custom_immune_decay = QDoubleSpinBox()
            self.custom_immune_decay.setRange(0.0, 0.1)
            self.custom_immune_decay.setDecimals(5)
            self.custom_immune_decay.setSingleStep(0.0001)
            self.custom_immune_decay.setValue(0.0)
            self.custom_immune_decay.setToolTip("Auto-immune Rejection Rate (1/min):\nSimulates graft death over time. Set to 0 for perfect immunosuppression.")
            config_layout.addRow("Immune Rejection (1/min):", self.custom_immune_decay)

            layout.addWidget(config_box)

            meals_box = QGroupBox("Meal Events")
            meals_layout = QVBoxLayout(meals_box)
            self.meals_table = QTableWidget(0, 2)
            self.meals_table.setHorizontalHeaderLabels(["Time (min)", "Carbs (g)"])
            meals_layout.addWidget(self.meals_table)

            meals_buttons = QHBoxLayout()
            add_meal_btn = QPushButton("Add Meal")
            add_meal_btn.clicked.connect(self._add_custom_meal)
            meals_buttons.addWidget(add_meal_btn)

            remove_meal_btn = QPushButton("Remove Selected")
            remove_meal_btn.clicked.connect(self._remove_custom_meal)
            meals_buttons.addWidget(remove_meal_btn)
            meals_layout.addLayout(meals_buttons)
            layout.addWidget(meals_box)

            actions_box = QGroupBox("Actions")
            actions_layout = QHBoxLayout(actions_box)
            run_custom_btn = QPushButton("Run Custom Scenario")
            run_custom_btn.setObjectName("primaryAction")
            run_custom_btn.clicked.connect(self._run_custom_scenario)
            actions_layout.addWidget(run_custom_btn)

            queue_custom_btn = QPushButton("Add Custom to Batch")
            queue_custom_btn.clicked.connect(lambda: self._add_to_batch("custom"))
            actions_layout.addWidget(queue_custom_btn)

            save_custom_btn = QPushButton("Save JSON")
            save_custom_btn.clicked.connect(self._save_custom_json)
            actions_layout.addWidget(save_custom_btn)

            load_custom_btn = QPushButton("Load JSON")
            load_custom_btn.clicked.connect(self._load_custom_json)
            actions_layout.addWidget(load_custom_btn)
            layout.addWidget(actions_box)
            layout.addStretch(1)

        def _build_batch_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)
            intro = QLabel("Queue multiple scenarios to run them sequentially. This is useful for large parameter sweeps or overnight runs.")
            intro.setWordWrap(True)
            layout.addWidget(intro)

            queue_box = QGroupBox("Queued Simulations")
            queue_layout = QVBoxLayout(queue_box)
            queue_layout.addWidget(self.batch_list_widget)

            btn_layout = QHBoxLayout()
            self.run_batch_btn = QPushButton("Run Batch Queue")
            self.run_batch_btn.setObjectName("primaryAction")
            self.run_batch_btn.clicked.connect(self._run_next_in_batch)
            btn_layout.addWidget(self.run_batch_btn)

            clear_batch_btn = QPushButton("Clear Queue")
            clear_batch_btn.clicked.connect(self._clear_batch)
            btn_layout.addWidget(clear_batch_btn)

            queue_layout.addLayout(btn_layout)
            layout.addWidget(queue_box)
            layout.addStretch(1)

        def _add_to_batch(self, config_type: str) -> None:
            if config_type == "preset":
                preset = self._selected_preset()
                config = {"type": "preset", "key": preset.key, "seed": int(self.seed.value())}
                title = f"{preset.title} (Seed: {config['seed']})"
            else:
                config = {"type": "custom", "payload": self._get_custom_preset_dict(), "seed": int(self.custom_seed.value())}
                title = f"Custom Scenario (Seed: {config['seed']})"
            self.batch_queue.append(config)
            self.batch_list_widget.addItem(title)
            self.status.setText(f"Added to batch: {title}")

        def _clear_batch(self) -> None:
            self.batch_queue.clear()
            self.batch_list_widget.clear()
            self.batch_running = False

        def _run_next_in_batch(self) -> None:
            if not self.batch_queue:
                self.batch_running = False
                QMessageBox.information(self, "Batch Complete", "All queued simulations have finished.")
                return

            self.batch_running = True
            config = self.batch_queue.pop(0)
            self.batch_list_widget.takeItem(0)

            if config["type"] == "preset":
                self.workflow_combo.setCurrentIndex(self._workflow_index(config["key"]))
                self.seed.setValue(config["seed"])
                self._run_selected_workflow()
            else:
                self._run_custom_payload(config["payload"], seed=int(config["seed"]))

        def _add_custom_meal(self) -> None:
            row = self.meals_table.rowCount()
            self.meals_table.insertRow(row)
            self.meals_table.setItem(row, 0, QTableWidgetItem("120"))
            self.meals_table.setItem(row, 1, QTableWidgetItem("60"))

        def _remove_custom_meal(self) -> None:
            selected_rows = {item.row() for item in self.meals_table.selectedItems()}
            for row in sorted(selected_rows, reverse=True):
                self.meals_table.removeRow(row)

        def _get_custom_preset_dict(self) -> dict[str, Any]:
            meals = []
            for row in range(self.meals_table.rowCount()):
                try:
                    time_item = self.meals_table.item(row, 0)
                    carbs_item = self.meals_table.item(row, 1)
                    if time_item and carbs_item:
                        time_min = int(time_item.text())
                        carbs_g = float(carbs_item.text())
                        meals.append(
                            {
                                "start_time": time_min,
                                "event_type": "meal",
                                "value": carbs_g,
                            }
                        )
                except ValueError:
                    pass

            return {
                "name": "custom_ui",
                "duration_minutes": self.custom_duration.value(),
                "time_step_minutes": self.custom_timestep.value(),
                "patient_config": {
                    "basal_insulin_rate": self.custom_basal.value(),
                    "insulin_sensitivity": self.custom_isf.value(),
                    "carb_factor": self.custom_cr.value(),
                    "stem_cell_engraftment_percent": self.custom_engraftment.value(),
                    "stem_cell_subq_fraction": self.custom_subq_fraction.value(),
                    "immune_rejection_rate": self.custom_immune_decay.value(),
                },
                "scenario": {
                    "scenario_name": "Desktop custom scenario",
                    "scenario_version": "1.0",
                    "description": "Scenario created with the IINTS-AF desktop workbench.",
                    "stress_events": meals,
                },
            }

        def _run_custom_scenario(self) -> None:
            self._run_custom_payload(
                self._get_custom_preset_dict(),
                seed=int(self.custom_seed.value()),
            )

        def _run_custom_payload(self, payload: dict[str, Any], *, seed: int) -> None:
            output_dir = self.output_dir.text().strip()
            if not output_dir:
                self.status.setText("Output workspace not selected.")
                return
            if self.current_thread is not None:
                self.status.setText("A simulation is already running.")
                return

            self._set_running_state(True)
            self.status.setText("Running Custom Scenario...")
            self.progress_bar.show()
            self._write_log("\nStarting custom workflow from Scenario Builder.\n")
            if self.tabs is not None:
                self.tabs.setCurrentIndex(0)

            thread = QThread(self)
            worker = RunWorker(
                output_dir=output_dir,
                seed=seed,
                custom_preset=payload,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_success)
            worker.failed.connect(self._handle_error)
            worker.log.connect(self._write_log)
            worker.finished.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            worker.telemetry.connect(self._update_telemetry)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_worker_refs)
            self.current_thread = thread
            self.current_worker = worker
            thread.start()

        def _save_custom_json(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Custom Scenario", "", "JSON Files (*.json)")
            if path:
                with open(path, "w") as f:
                    json.dump(self._get_custom_preset_dict(), f, indent=2)
                self.status.setText(f"Saved custom scenario to {path}")

        def _load_custom_json(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Load Custom Scenario", "", "JSON Files (*.json)")
            if path:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.custom_duration.setValue(data.get("duration_minutes", 1440))
                    self.custom_timestep.setValue(data.get("time_step_minutes", 5))
                    patient = data.get("patient_config", {})
                    self.custom_basal.setValue(
                        patient.get("basal_insulin_rate", patient.get("base_basal_rate", 0.8))
                    )
                    self.custom_isf.setValue(
                        patient.get("insulin_sensitivity", patient.get("insulin_sensitivity_factor", 45.0))
                    )
                    self.custom_cr.setValue(patient.get("carb_factor", patient.get("carb_ratio", 10.0)))
                    self.custom_engraftment.setValue(patient.get("stem_cell_engraftment_percent", 0.0))
                    self.custom_subq_fraction.setValue(patient.get("stem_cell_subq_fraction", 0.0))
                    self.custom_immune_decay.setValue(patient.get("immune_rejection_rate", 0.0))

                    self.meals_table.setRowCount(0)
                    scenario = data.get("scenario", {})
                    meals = [
                        event
                        for event in scenario.get("stress_events", [])
                        if event.get("event_type") == "meal"
                    ]
                    if not meals:
                        meals = scenario.get("meals", [])
                    for meal in meals:
                        self._add_custom_meal()
                        row = self.meals_table.rowCount() - 1
                        time_value = meal.get("start_time", meal.get("time_minutes", 0))
                        carb_value = meal.get("value", meal.get("carbohydrates_g", 0))
                        time_item = self.meals_table.item(row, 0)
                        carb_item = self.meals_table.item(row, 1)
                        if time_item is not None:
                            time_item.setText(str(time_value))
                        if carb_item is not None:
                            carb_item.setText(str(carb_value))
                    self.status.setText(f"Loaded custom scenario from {path}")
                except Exception as e:
                    self.status.setText(f"Failed to load JSON: {e}")


        def _build_about_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)
            intro = QLabel(
                "IINTS-AF Desktop is a native research workbench for running SDK simulations, "
                "reviewing generated results, asking local AI questions, and opening biology evidence artifacts. "
                "The Python SDK remains the single source of truth for formulas, reports, and validation."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            danger_box = QGroupBox("Danger Zone")
            danger_layout = QVBoxLayout(danger_box)
            purge_button = QPushButton("Self-Destruct SDK")
            purge_button.setStyleSheet("QPushButton { color: red; }")
            purge_button.clicked.connect(self._purge_sdk_data)
            danger_layout.addWidget(purge_button)
            layout.addWidget(danger_box)
            layout.addStretch(1)


            update_box = QGroupBox("Updates")
            update_layout = QVBoxLayout(update_box)
            update_help = QLabel(
                "Use this panel to get the newest desktop app build or update a Python-based SDK install. "
                "Packaged .exe/.dmg builds open the download page; Python installs can run the pip update command."
            )
            update_help.setWordWrap(True)
            update_layout.addWidget(update_help)
            self.update_status.setObjectName("updateStatus")
            update_layout.addWidget(self.update_status)
            self.open_app_downloads_button.clicked.connect(self._open_app_downloads)
            self.open_update_docs_button.clicked.connect(self._open_update_docs)
            self.copy_update_command_button.clicked.connect(self._copy_update_command)
            self.run_package_update_button.clicked.connect(self._run_package_update)
            if getattr(sys, "frozen", False):
                self.run_package_update_button.setEnabled(False)
                self.update_status.setText(
                    "Packaged app mode: download the newest .exe/.dmg/Linux build to update the app."
                )
            update_layout.addLayout(
                self._button_grid(
                    [
                        self.open_app_downloads_button,
                        self.open_update_docs_button,
                        self.copy_update_command_button,
                        self.run_package_update_button,
                    ],
                    columns=2,
                )
            )
            self.update_log.setMinimumHeight(110)
            update_layout.addWidget(self.update_log)
            layout.addWidget(update_box)

            dev_box = QGroupBox("Developer Settings")
            dev_layout = QVBoxLayout(dev_box)
            self.show_terminal_checkbox = QCheckBox("Show Integrated Terminal")
            terminal_dock = self.terminal_dock
            if terminal_dock is not None:
                self.show_terminal_checkbox.toggled.connect(terminal_dock.setVisible)
            dev_layout.addWidget(self.show_terminal_checkbox)
            layout.addWidget(dev_box)

            for preset in self.presets:
                label = QLabel(f"<b>{preset.title}</b><br>{preset.description}")
                label.setWordWrap(True)
                layout.addWidget(label)
            layout.addStretch(1)

        def _purge_sdk_data(self) -> None:
            reply = QMessageBox.warning(
                self,
                "Self-Destruct SDK",
                "WARNING: This will initiate a self-destruct sequence. All SDK code, generated data, configuration, and shortcuts across the OS will be permanently deleted! Are you completely sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                import sys
                import subprocess
                from pathlib import Path

                uninstall_script = Path(__file__).resolve().parent.parent.parent.parent / "uninstall_app.py"
                if not uninstall_script.exists():
                    QMessageBox.critical(self, "Error", f"Could not find uninstall script at {uninstall_script}")
                    return

                subprocess.Popen([sys.executable, str(uninstall_script), "--self-destruct"])
                QApplication.quit()

        def _apply_style(self) -> None:
            self.setStyleSheet(
                """
                * {
                    color: #d4d4d4;
                    selection-background-color: #264f78;
                    selection-color: #ffffff;
                }
                QMainWindow, QWidget, QWidget#root {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-size: 13px;
                }
                QLabel {
                    color: #d4d4d4;
                    background: transparent;
                }
                QFrame, QSplitter, QAbstractScrollArea, QScrollArea {
                    background-color: #252526;
                    color: #d4d4d4;
                    border-color: #3c3c3c;
                }
                QScrollArea > QWidget > QWidget {
                    background-color: #252526;
                    color: #d4d4d4;
                }
                QMenuBar, QToolBar, QStatusBar {
                    background: #333333;
                    color: #cccccc;
                    border-color: #3c3c3c;
                }
                QMenuBar {
                    border-bottom: 1px solid #3c3c3c;
                }
                QMenuBar::item {
                    padding: 4px 9px;
                }
                QMenuBar::item:selected, QMenu::item:selected {
                    background: #094771;
                }
                QMenu {
                    background: #252526;
                    color: #cccccc;
                    border: 1px solid #454545;
                }
                QToolBar {
                    spacing: 4px;
                    padding: 3px 5px;
                    border-bottom: 1px solid #3c3c3c;
                }
                QToolButton {
                    background: #333333;
                    color: #cccccc;
                    border: 1px solid transparent;
                    border-radius: 2px;
                    padding: 5px 8px;
                }
                QToolButton:hover {
                    background: #444444;
                    border-color: #555555;
                }
                QWidget#workbenchHeader {
                    background: #252526;
                    border: 1px solid #3c3c3c;
                }
                QLabel#appTitle {
                    color: #4facfe;
                }
                QLabel#researchBadge {
                    background: #4d4d00;
                    color: #ffff80;
                    border: 1px solid #666600;
                    padding: 3px 7px;
                    font-weight: 700;
                }
                QTabWidget::pane {
                    border: 1px solid #3c3c3c;
                    background: #252526;
                    padding: 4px;
                }
                QTabBar::tab {
                    background: #2d2d2d;
                    color: #999999;
                    border: 1px solid #3c3c3c;
                    border-bottom: none;
                    border-radius: 0;
                    padding: 7px 12px;
                    margin-right: 2px;
                    font-weight: 600;
                }
                QTabBar::tab:selected {
                    background: #1e1e1e;
                    color: #4facfe;
                    border-top: 3px solid #007acc;
                    padding-top: 5px;
                }
                QLabel#infoStrip, QLabel#deepDiveIntro {
                    background: #1e3a5f;
                    color: #e0f2fe;
                    border: 1px solid #2a5286;
                    padding: 6px 8px;
                }
                QLabel#metricSummary {
                    background: #252526;
                    border: 1px solid #3c3c3c;
                    color: #cccccc;
                    padding: 7px;
                }
                QLabel#moleculeTitle {
                    color: #4facfe;
                    font-size: 16px;
                    font-weight: 700;
                }
                QLabel#moleculeStatus {
                    background: #1b4332;
                    color: #95d5b2;
                    border: 1px solid #2d6a4f;
                    padding: 6px 8px;
                }
                QLabel#moleculePAEStatus, QLabel#biologyActionStatus, QLabel#updateStatus {
                    background: #1f3a2c;
                    color: #d8f3dc;
                    border: 1px solid #40916c;
                    padding: 6px 8px;
                }
                QGroupBox {
                    background: #252526;
                    color: #d4d4d4;
                    border: 1px solid #3c3c3c;
                    border-radius: 2px;
                    margin-top: 9px;
                    padding: 10px 8px 8px 8px;
                    font-weight: 650;
                }
                QGroupBox QLabel {
                    color: #d4d4d4;
                    background: transparent;
                    font-weight: 400;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px;
                    color: #4facfe;
                }
                QPushButton {
                    background: #3c3c3c;
                    color: #cccccc;
                    border: 1px solid #555555;
                    border-radius: 2px;
                    padding: 5px 9px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #505050;
                    border-color: #666666;
                }
                QPushButton#primaryAction {
                    background: #007acc;
                    color: #ffffff;
                    border-color: #005c99;
                    font-weight: 700;
                }
                QPushButton:disabled {
                    background: #2d2d2d;
                    color: #666666;
                    border-color: #444444;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QTableWidget, QTableView {
                    background: #1e1e1e;
                    color: #d4d4d4;
                    border: 1px solid #3c3c3c;
                    border-radius: 1px;
                    padding: 5px;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus, QTableWidget:focus {
                    border: 1px solid #007acc;
                }
                QComboBox QAbstractItemView {
                    background: #1e1e1e;
                    color: #d4d4d4;
                    selection-background-color: #264f78;
                    selection-color: #ffffff;
                }
                QPlainTextEdit:disabled, QTextEdit:disabled, QLineEdit:disabled {
                    color: #888888;
                    background: #2d2d2d;
                }
                QTableWidget::item:selected {
                    background: #264f78;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: #333333;
                    color: #cccccc;
                    border: 1px solid #444444;
                    padding: 4px;
                    font-weight: 650;
                }
                QPlainTextEdit, QTextEdit {
                    font-family: Menlo, Consolas, monospace;
                }
                QLabel#subtleHint {
                    color: #999999;
                    background: #2d2d2d;
                    border: 1px solid #444444;
                    padding: 5px 7px;
                }
                QStatusBar {
                    color: #cccccc;
                    border-top: 1px solid #3c3c3c;
                }
                QLabel#workspaceStatus {
                    color: #999999;
                    padding-left: 10px;
                }
                """
            )

        def _workflow_index(self, key: str) -> int:
            for index, preset in enumerate(self.presets):
                if preset.key == key:
                    return index
            return 0

        def _selected_preset(self) -> DesktopPreset:
            key = str(self.workflow_combo.currentData())
            return get_desktop_preset(key)

        def _on_workflow_changed(self) -> None:
            preset = self._selected_preset()
            self.description.setText(
                f"Audience: {preset.audience}\n"
                f"SDK preset: {preset.preset_name}\n"
                f"{preset.description}\n"
                f"Output: {preset.expected_output}"
            )
            self.settings.setValue("workflow_key", preset.key)

        def _choose_output_dir(self) -> None:
            chosen = QFileDialog.getExistingDirectory(
                self,
                "Choose output folder",
                self.output_dir.text() or str(Path.home()),
            )
            if chosen:
                self.output_dir.setText(chosen)
                self.settings.setValue("output_dir", chosen)
                if self.workspace_status is not None:
                    self.workspace_status.setText(f"Workspace: {Path(chosen).expanduser()}")
                self._refresh_history()

        def _run_selected_workflow(self) -> None:
            preset = self._selected_preset()
            seed = int(self.seed.value())
            self.settings.setValue("output_dir", self.output_dir.text())
            self.settings.setValue("workflow_key", preset.key)
            self.settings.setValue("seed", seed)
            self._set_running_state(True)
            self.status.setText(f"Running {preset.title}...")
            self.progress_bar.show()
            self.run_button.setEnabled(False)
            self._write_log(
                f"\nStarting workflow: {preset.title}\n"
                f"SDK preset: {preset.preset_name}\n"
                f"Seed: {seed}\n"
                "Generating simulation outputs, reports, and audit artifacts...\n"
            )

            thread = QThread(self)
            worker = RunWorker(
                output_dir=self.output_dir.text(),
                desktop_preset_key=preset.key,
                seed=seed,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.log.connect(self._write_log)
            worker.finished.connect(self._handle_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            worker.telemetry.connect(self._update_telemetry)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_worker_refs)
            self.current_thread = thread
            self.current_worker = worker
            thread.start()

        @Slot(object)
        def _handle_success(self, result: object) -> None:
            if not isinstance(result, DesktopRunResult):
                self._handle_error("Unexpected run result returned by SDK worker.")
                return
            self.last_result = result
            self.status.setText("Run completed")
            self._set_running_state(False)
            self.run_button.setEnabled(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.hide()
            self.live_telemetry_label.hide()
            self.open_folder_button.setEnabled(True)
            self.open_report_button.setEnabled(bool(result.report_pdf and result.report_pdf.exists()))
            self.open_csv_button.setEnabled(bool(result.results_csv and result.results_csv.exists()))
            self.copy_summary_button.setEnabled(True)
            self._write_log("\n" + result.summary + "\n")
            self._refresh_history()
            if result.results_csv:
                self.result_csv_path.setText(str(result.results_csv))
                self._load_result_csv(result.results_csv)
                if self.tabs is not None and not self.batch_running:
                    self.tabs.setCurrentIndex(1)

            if self.batch_running:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(500, self._run_next_in_batch)

        @Slot(str)
        def _handle_error(self, details: str) -> None:
            self.status.setText("Run failed")
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.hide()
            self.live_telemetry_label.hide()
            self._set_running_state(False)
            if self.batch_running:
                self.batch_running = False
                self.status.setText("Batch halted due to error")
            self._write_log(f"\nERROR:\n{details}\n")
            QMessageBox.critical(self, "IINTS-AF Desktop", details)

        @Slot(int, int, float)
        def _update_telemetry(self, step: int, total: int, glucose: float) -> None:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(step)
            self.progress_bar.setTextVisible(True)
            self.live_telemetry_label.setText(f" Live Glucose: {glucose:.1f} mg/dL ")
            self.live_telemetry_label.show()

        @Slot()
        def _clear_worker_refs(self) -> None:
            self.current_thread = None
            self.current_worker = None

        def _set_running_state(self, is_running: bool) -> None:
            self.run_button.setEnabled(not is_running)
            if is_running:
                self._set_result_buttons_enabled(False)

        def _set_result_buttons_enabled(self, enabled: bool) -> None:
            self.open_folder_button.setEnabled(enabled)
            self.open_report_button.setEnabled(enabled)
            self.open_csv_button.setEnabled(enabled)
            self.copy_summary_button.setEnabled(enabled)

        def _set_loaded_result_actions(self, enabled: bool) -> None:
            self.open_loaded_csv_button.setEnabled(enabled)
            self.open_graph_button.setEnabled(enabled and bool(self.loaded_result and self.loaded_result.graph_path))
            self.create_mdmp_cert_button.setEnabled(enabled)
            self.open_mdmp_cert_folder_button.setEnabled(enabled)
            self.create_academic_bundle_button.setEnabled(enabled)
            self.open_academic_metadata_button.setEnabled(enabled and self.last_academic_bundle is not None)
            self.open_academic_audit_button.setEnabled(enabled and self.last_academic_bundle is not None)
            self.export_workspace_button.setEnabled(enabled)

        def _open_output_folder(self) -> None:
            path = self.last_result.output_dir if self.last_result else Path(self.output_dir.text())
            self._open_path(path)

        def _open_report(self) -> None:
            if self.last_result and self.last_result.report_pdf:
                self._open_path(self.last_result.report_pdf)

        def _open_results_csv(self) -> None:
            if self.last_result and self.last_result.results_csv:
                self._open_path(self.last_result.results_csv)

        def _browse_result_csv(self) -> None:
            chosen, _ = QFileDialog.getOpenFileName(
                self,
                "Choose results CSV",
                self.output_dir.text() or str(Path.home()),
                "CSV files (*.csv);;All files (*)",
            )
            if chosen:
                self.result_csv_path.setText(chosen)
                self._load_result_csv(Path(chosen))

        def _load_selected_result_csv(self) -> None:
            if not self.result_csv_path.text().strip():
                QMessageBox.information(self, "IINTS-AF Desktop", "Choose a results CSV first.")
                return
            self._load_result_csv(Path(self.result_csv_path.text()))

        def _load_last_result_csv(self) -> None:
            if not self.last_result or not self.last_result.results_csv:
                QMessageBox.information(self, "IINTS-AF Desktop", "No last run CSV is available yet.")
                return
            self.result_csv_path.setText(str(self.last_result.results_csv))
            self._load_result_csv(self.last_result.results_csv)

        def _open_loaded_result_csv(self) -> None:
            if self.loaded_result:
                self._open_path(self.loaded_result.csv_path)

        def _open_loaded_graph(self) -> None:
            if self.loaded_result and self.loaded_result.graph_path:
                self._open_path(self.loaded_result.graph_path)

        def _open_loaded_certificate_folder(self) -> None:
            if self.last_mdmp_certificate_dir and self.last_mdmp_certificate_dir.exists():
                self._open_path(self.last_mdmp_certificate_dir)
                return
            if self.loaded_result:
                self._open_path(self.loaded_result.csv_path.parent / "mdmp_certificates")

        def _export_workspace(self) -> None:
            if not self.loaded_result:
                return
            from PySide6.QtWidgets import QFileDialog
            import shutil

            target_dir = self.loaded_result.csv_path.parent
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Workspace as ZIP",
                f"{target_dir.name}_export.zip",
                "ZIP Archives (*.zip)"
            )
            if save_path:
                if save_path.endswith(".zip"):
                    save_path = save_path[:-4]  # make_archive appends .zip automatically
                try:
                    shutil.make_archive(save_path, 'zip', str(target_dir))
                    QMessageBox.information(self, "Export Successful", f"Workspace exported to:\n{save_path}.zip")
                except Exception as e:
                    QMessageBox.critical(self, "Export Failed", f"Failed to export workspace:\n{e}")

        def _set_mdmp_certifying_state(self, is_running: bool) -> None:
            self.create_mdmp_cert_button.setEnabled(not is_running and self.loaded_result is not None)
            self.open_mdmp_cert_folder_button.setEnabled(not is_running and self.loaded_result is not None)

        def _create_mdmp_certificate(self) -> None:
            if not self.loaded_result:
                QMessageBox.information(self, "IINTS-AF Desktop", "Load a results CSV first.")
                return
            self._set_mdmp_certifying_state(True)
            self.status.setText("Creating signed MDMP certificate")
            thread = QThread(self)
            worker = MDMPCertifyWorker(csv_path=str(self.loaded_result.csv_path))
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_mdmp_certificate_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_mdmp_certificate_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_mdmp_certificate_refs)
            self.mdmp_thread = thread
            self.mdmp_worker = worker
            thread.start()

        @Slot(object)
        def _handle_mdmp_certificate_success(self, result: object) -> None:
            self._set_mdmp_certifying_state(False)
            cert_path = Path(str(getattr(result, "certificate_path", "")))
            report_path = Path(str(getattr(result, "report_path", "")))
            public_key_path = Path(str(getattr(result, "public_key_path", "")))
            self.last_mdmp_certificate_dir = cert_path.parent if cert_path.name else None
            grade = str(getattr(result, "grade", "unknown"))
            score = float(getattr(result, "compliance_score", 0.0))
            row_count = int(getattr(result, "row_count", 0))
            message = (
                f"Signed MDMP certificate created.\n\n"
                f"Grade: {grade}\n"
                f"Compliance: {score:.2f}%\n"
                f"Rows scanned: {row_count}\n\n"
                f"Certificate: {cert_path}\n"
                f"Report: {report_path}\n"
                f"Verifier public key: {public_key_path}\n\n"
                "The certificate is Ed25519-signed locally. Share the public key if an external reviewer "
                "needs to verify the certificate."
            )
            self.ai_answer.setPlainText(message)
            self.status.setText("MDMP certificate ready")

        @Slot(str)
        def _handle_mdmp_certificate_error(self, details: str) -> None:
            self._set_mdmp_certifying_state(False)
            self.ai_answer.setPlainText(details)
            self.status.setText("MDMP certification failed")

        @Slot()
        def _clear_mdmp_certificate_refs(self) -> None:
            self.mdmp_thread = None
            self.mdmp_worker = None

        def _set_academic_export_state(self, is_running: bool) -> None:
            has_result = self.loaded_result is not None
            has_bundle = self.last_academic_bundle is not None
            self.create_academic_bundle_button.setEnabled(not is_running and has_result)
            self.open_academic_metadata_button.setEnabled(not is_running and has_bundle)
            self.open_academic_audit_button.setEnabled(not is_running and has_bundle)

        def _create_academic_bundle(self) -> None:
            if not self.loaded_result:
                QMessageBox.information(self, "IINTS-AF Desktop", "Load a results CSV first.")
                return
            creator_name = self.academic_creator.text().strip()
            creator_orcid = self.academic_orcid.text().strip()
            license_id = self.academic_license.text().strip() or "NOASSERTION"
            self.settings.setValue("academic_creator", creator_name)
            self.settings.setValue("academic_orcid", creator_orcid)
            self.settings.setValue("academic_license", license_id)
            self._set_academic_export_state(True)
            self.academic_bundle_status.setText("Hashing run artifacts and building academic metadata...")
            self.status.setText("Creating academic reproducibility package")
            thread = QThread(self)
            worker = AcademicBundleWorker(
                run_dir=self.loaded_result.csv_path.parent,
                creator_name=creator_name,
                creator_orcid=creator_orcid,
                license_id=license_id,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_academic_bundle_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_academic_bundle_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_academic_bundle_refs)
            self.academic_thread = thread
            self.academic_worker = worker
            thread.start()

        @Slot(object)
        def _handle_academic_bundle_success(self, result: object) -> None:
            self.last_academic_bundle = result
            self._set_academic_export_state(False)
            status = str(getattr(result, "readiness_status", "unknown"))
            score = float(getattr(result, "readiness_score_pct", 0.0))
            artifacts = int(getattr(result, "artifact_count", 0))
            sources = int(getattr(result, "source_count", 0))
            crate_path = Path(str(getattr(result, "ro_crate_metadata", "")))
            audit_path = Path(str(getattr(result, "audit_json", "")))
            message = (
                f"Readiness: {status} · audit score: {score:.2f}% · "
                f"{artifacts} artifacts · {sources} sources\n"
                f"RO-Crate: {crate_path}\nAudit: {audit_path}\n"
                "Review failed checks and inspect privacy before sharing."
            )
            self.academic_bundle_status.setText(message)
            self._write_log(f"Academic reproducibility package created.\n{message}\n")
            self.status.setText("Academic package ready for review")

        @Slot(str)
        def _handle_academic_bundle_error(self, details: str) -> None:
            self._set_academic_export_state(False)
            self.academic_bundle_status.setText("Academic package failed. See the execution log for details.")
            self._write_log(details)
            self.status.setText("Academic package export failed")

        @Slot()
        def _clear_academic_bundle_refs(self) -> None:
            self.academic_thread = None
            self.academic_worker = None

        def _open_academic_metadata(self) -> None:
            path = getattr(self.last_academic_bundle, "ro_crate_metadata", None)
            if path:
                self._open_path(Path(str(path)))

        def _open_academic_audit(self) -> None:
            path = getattr(self.last_academic_bundle, "audit_json", None)
            if path:
                self._open_path(Path(str(path)))

        def _load_result_csv(self, path: Path) -> None:
            try:
                preview = load_results_preview(path)
            except Exception as exc:
                QMessageBox.critical(self, "IINTS-AF Desktop", str(exc))
                return
            self.loaded_result = preview
            self.last_academic_bundle = None
            self.academic_bundle_status.setText("No academic package generated for this loaded result yet.")
            self.result_csv_path.setText(str(preview.csv_path))
            self._render_result_preview(preview)
            self.status.setText(f"Loaded results: {preview.csv_path.name}")

        def _render_result_preview(self, preview: ResultPreview) -> None:
            metrics = "\n".join(
                f"{key}: {value}" for key, value in preview.metrics.items() if key != "Rows"
            )
            self.result_summary.setText(
                f"Loaded: {preview.csv_path.name}\nRows: {preview.row_count}\n{metrics}"
            )
            self.result_table.setColumnCount(len(preview.columns))
            self.result_table.setHorizontalHeaderLabels(preview.columns)
            self.result_table.setRowCount(len(preview.rows))
            for row_index, row in enumerate(preview.rows):
                for column_index, value in enumerate(row):
                    self.result_table.setItem(row_index, column_index, QTableWidgetItem(value))
            self.result_table.resizeColumnsToContents()
            html_path = preview.csv_path.with_name(preview.csv_path.name.replace(".csv", ".html"))
            if html_path.exists() and self.result_graph_web:
                from PySide6.QtCore import QUrl
                self.result_graph_web.load(QUrl.fromLocalFile(str(html_path.absolute())))
                self.result_graph_stack.setCurrentWidget(self.result_graph_web)
            elif preview.graph_path and preview.graph_path.exists():
                pixmap = QPixmap(str(preview.graph_path))
                self.result_graph_label.clear()
                self.result_graph_label.setPixmap(
                    pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
                )
                self.result_graph_stack.setCurrentWidget(self.result_graph_label)
            else:
                self.result_graph_label.clear()
                self.result_graph_label.setText("No glucose column was found, so no graph could be generated.")
                self.result_graph_stack.setCurrentWidget(self.result_graph_label)
            self.ai_context_label.setText(f"AI context: loaded summary from {preview.csv_path.name}")
            self._set_loaded_result_actions(True)

        def _copy_last_summary(self) -> None:
            if self.last_result:
                QApplication.clipboard().setText(self.last_result.summary)
                self.status.setText("Run summary copied")

        def _set_ai_starting_state(self, is_starting: bool) -> None:
            self.start_ai_button.setEnabled(not is_starting)
            self.check_ai_button.setEnabled(not is_starting)
            self.refresh_ai_models_button.setEnabled(not is_starting)
            self.ask_ai_button.setEnabled(not is_starting)

        def _selected_ai_model(self) -> str:
            return self.ai_model.currentText().strip() or DEFAULT_MINISTRAL_MODEL

        def _start_local_ai(self) -> None:
            self._set_ai_starting_state(True)
            self.ai_status.setText("Starting local AI. First model download can take a while...")
            self.status.setText("Starting local AI")
            thread = QThread(self)
            worker = LocalAIStartWorker(
                model=self._selected_ai_model(),
                host=self.ai_host.currentText().strip(),
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_ai_start_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_ai_start_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_ai_start_refs)
            self.ai_start_thread = thread
            self.ai_start_worker = worker
            thread.start()

        @Slot(object)
        def _handle_ai_start_success(self, result: object) -> None:
            self._set_ai_starting_state(False)
            available = bool(getattr(result, "available", False))
            message = getattr(result, "message", str(result))
            self.ai_status.setText(message)
            self.status.setText("Local AI ready" if available else "Local AI not ready")

        @Slot(str)
        def _handle_ai_start_error(self, details: str) -> None:
            self._set_ai_starting_state(False)
            self.ai_status.setText("Local AI failed to start. See response panel for details.")
            self.ai_answer.setPlainText(details)
            self.status.setText("Local AI failed")

        @Slot()
        def _clear_ai_start_refs(self) -> None:
            self.ai_start_thread = None
            self.ai_start_worker = None

        def _check_ai_status(self) -> None:
            status = check_local_ai(
                model=self._selected_ai_model(),
                host=self.ai_host.currentText().strip() or None,
            )
            self.ai_status.setText(status.message)
            self.status.setText("Local AI ready" if status.available else "Local AI not ready")

        def _refresh_ai_models(self) -> None:
            selected = self._selected_ai_model()
            models = list_local_ai_models(host=self.ai_host.currentText().strip() or None)
            self.ai_model.clear()
            self.ai_model.addItems(models)
            self.ai_model.setCurrentText(selected if selected in models else models[0])
            self.ai_status.setText(f"Model list refreshed ({len(models)} choices).")

        def _set_ai_question(self, text: str) -> None:
            self.ai_question.setPlainText(text)
            self.status.setText("AI question template loaded")

        def _ask_ai(self) -> None:
            question = self.ai_question.toPlainText().strip()
            if not question:
                QMessageBox.information(self, "IINTS-AF Desktop", "Write a question first.")
                return
            self.ask_ai_button.setEnabled(False)
            self.ai_answer.setPlainText("Thinking locally with Ollama...\n")
            result_csv = str(self.loaded_result.csv_path) if self.loaded_result else None
            thread = QThread(self)
            worker = AIWorker(
                question=question,
                model=self._selected_ai_model(),
                host=self.ai_host.currentText().strip(),
                result_csv=result_csv,
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_ai_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_ai_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_ai_worker_refs)
            self.ai_thread = thread
            self.ai_worker = worker
            thread.start()

        @Slot(object)
        def _handle_ai_success(self, answer: object) -> None:
            self.ask_ai_button.setEnabled(True)
            text = getattr(answer, "answer", str(answer))
            model = getattr(answer, "model", self._selected_ai_model())
            self.ai_answer.setPlainText(f"Model: {model}\n\n{text}")
            self.status.setText("Local AI answer ready")

        @Slot(str)
        def _handle_ai_error(self, details: str) -> None:
            self.ask_ai_button.setEnabled(True)
            self.ai_answer.setPlainText(details)
            self.status.setText("Local AI failed")

        @Slot()
        def _clear_ai_worker_refs(self) -> None:
            self.ai_thread = None
            self.ai_worker = None

        def _copy_ai_answer(self) -> None:
            QApplication.clipboard().setText(self.ai_answer.toPlainText())
            self.status.setText("AI answer copied")


        def _fetch_custom_uniprot(self) -> None:
            uniprot_id = self.custom_uniprot_input.text().strip().upper()
            if not uniprot_id:
                QMessageBox.information(self, "IINTS-AF Desktop", "Please enter a UniProt ID.")
                return

            self.fetch_uniprot_button.setEnabled(False)
            self.molecule_structure_status.setText(f"Fetching {uniprot_id} from AlphaFold...")

            thread = QThread(self)
            worker = AlphaFoldFetchWorker(uniprot_id, self._structural_output_dir())
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_fetch_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_fetch_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            self.af_fetch_thread = thread
            self.af_fetch_worker = worker
            thread.start()

        @Slot(object)
        def _handle_fetch_success(self, result: object) -> None:
            self.fetch_uniprot_button.setEnabled(True)
            cif_path, _html_path, uniprot_id = cast(tuple[Path, Path, str], result)

            # Create a dynamic MoleculeAsset
            from iints_desktop.molecules import MoleculeAsset
            new_mol = MoleculeAsset(
                key=uniprot_id.lower(),
                title=f"Custom: {uniprot_id}",
                uniprot_id=uniprot_id,
                image_path=Path("nonexistent.png"),
                structure_path=cif_path,
                explanation="Dynamically fetched AlphaFold structure.",
                sdk_link="Custom research target.",
                pae_target=uniprot_id.lower(),
                pae_note="Dynamic PAE heatmap target."
            )
            self.molecules.append(new_mol)
            self.molecule_selector.addItem(f"{new_mol.title} (UniProt {new_mol.uniprot_id})", new_mol.key)
            self.molecule_selector.setCurrentIndex(self.molecule_selector.count() - 1)
            self.molecule_structure_status.setText(f"Successfully loaded {uniprot_id}.")

        @Slot(str)
        def _handle_fetch_error(self, details: str) -> None:
            self.fetch_uniprot_button.setEnabled(True)
            self.molecule_structure_status.setText(f"Fetch failed: {details}")
            self.molecule_structure_status.setStyleSheet("background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa;")

        def _selected_molecule(self) -> MoleculeAsset:
            key = str(self.molecule_selector.currentData())
            for molecule in self.molecules:
                if molecule.key == key:
                    return molecule
            return self.molecules[0]

        def _selected_evidence_connector(self) -> EvidenceConnector | None:
            key = str(self.evidence_connector_selector.currentData() or "")
            for connector in self.evidence_connectors:
                if connector.key == key:
                    return connector
            return self.evidence_connectors[0] if self.evidence_connectors else None

        def _on_evidence_connector_changed(self) -> None:
            connector = self._selected_evidence_connector()
            if connector is None:
                self.evidence_connector_details.setText("No evidence connector is available.")
                self.open_evidence_portal_button.setEnabled(False)
                self.open_evidence_docs_button.setEnabled(False)
                return
            local_artifact = "yes" if connector.writes_local_evidence else "no"
            self.evidence_connector_details.setText(
                f"Category: {connector.category}\n"
                f"Maturity: {connector.integration_level} — {connector.integration_status}\n"
                f"Access: {connector.access_mode}\n"
                f"Writes local evidence: {local_artifact}\n\n"
                f"{connector.why_it_matters}\n\n"
                f"Workbench use: {connector.app_use}\n"
                f"Provenance note: {connector.provenance_note}"
            )
            self.open_evidence_portal_button.setEnabled(bool(connector.primary_url))
            self.open_evidence_docs_button.setEnabled(bool(connector.docs_url))

        def _open_selected_evidence_url(self, field_name: str) -> None:
            connector = self._selected_evidence_connector()
            if connector is None:
                return
            url = str(getattr(connector, field_name, "")).strip()
            if not url.startswith("https://"):
                QMessageBox.warning(self, "IINTS-AF Desktop", "Only fixed HTTPS evidence links may be opened.")
                return
            QDesktopServices.openUrl(QUrl(url))

        def _on_molecule_changed(self) -> None:
            if not self.molecules or self.molecule_viewer is None:
                return
            viewer = self.molecule_viewer
            molecule = self._selected_molecule()
            self.molecule_title.setText(
                f"<b>{molecule.title}</b><br>UniProt {molecule.uniprot_id} / AlphaFold structure"
            )
            self.molecule_explanation.setText(
                f"{molecule.explanation}<br><br>"
                f"<b>{molecule.sdk_link}</b>"
            )
            viewer.set_structure(
                molecule.structure_path,
                display_name=f"{molecule.title} / UniProt {molecule.uniprot_id}",
            )
            if viewer.error:
                self.molecule_structure_status.setText(
                    f"3D structure could not be loaded: {viewer.error}"
                )
                self.molecule_structure_status.setStyleSheet(
                    "background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa;"
                )
            else:
                structure = viewer.structure
                residue_count = len(structure.atoms) if structure else 0
                chain_count = structure.chain_count if structure else 0
                self.molecule_structure_status.setText(
                    f"Local 3D model loaded: {residue_count} C-alpha residues across {chain_count} chain(s). "
                    "Colours show AlphaFold pLDDT confidence, not a clinical score."
                )
                self.molecule_structure_status.setStyleSheet("")

            # 3Dmol.js rendering path
            web_view = self.molecule_web_view
            if web_view is not None:
                from iints_desktop.render_3dmol import generate_3dmol_html
                try:
                    out_dir = self._structural_output_dir()
                    html_path = generate_3dmol_html(molecule.structure_path, out_dir)
                    web_view.setUrl(QUrl.fromLocalFile(str(html_path.absolute())))
                    viewer.hide()
                    web_view.show()
                except Exception as e:
                    print(f"3Dmol.js render failed: {e}")
                    web_view.hide()
                    viewer.show()


            self._update_pae_controls()

        def _reset_molecule_view(self) -> None:
            if self.molecule_viewer is not None:
                self.molecule_viewer.reset_camera()
                self.status.setText("3D molecular view reset")

        def _open_selected_molecule_image(self) -> None:
            self._open_path(self._selected_molecule().image_path)


        def _open_3dmol_in_browser(self) -> None:
            molecule = self._selected_molecule()
            if not molecule: return
            from iints_desktop.render_3dmol import generate_3dmol_html
            try:
                from pathlib import Path
                out_dir = self._structural_output_dir()
                html_path = generate_3dmol_html(molecule.structure_path, out_dir)
                import webbrowser
                webbrowser.open(html_path.absolute().as_uri())
            except Exception as e:
                print(f"3Dmol.js render failed: {e}")

        def _open_selected_molecule_structure(self) -> None:

            self._open_path(self._selected_molecule().structure_path)

        def _selected_pae_target(self) -> str | None:
            return self._selected_molecule().pae_target

        def _selected_pae_html_path(self) -> Path | None:
            target = self._selected_pae_target()
            return pae_html_path(target, self._structural_output_dir()) if target else None

        def _update_pae_controls(self) -> None:
            molecule = self._selected_molecule()
            target = molecule.pae_target
            is_busy = self.pae_thread is not None
            if not target:
                self.molecule_pae_status.setText("No PAE target is configured for this molecule.")
                self.generate_pae_button.setEnabled(False)
                self.open_pae_button.setEnabled(False)
                self.open_pae_folder_button.setEnabled(False)
                return

            html_path = pae_html_path(target, self._structural_output_dir())
            state = "available" if html_path.exists() else "not generated yet"
            self.molecule_pae_status.setText(
                f"Target: {target} / output: {html_path}\n"
                f"Status: {state}. {molecule.pae_note}"
            )
            if html_path.exists() and self.pae_web_view is not None:
                self._load_pae_html_in_app(html_path)
            elif self.pae_web_view is not None:
                self._set_empty_pae_preview(target)
            self.generate_pae_button.setEnabled(not is_busy)
            self.open_pae_button.setEnabled(html_path.exists() and not is_busy)
            self.open_pae_folder_button.setEnabled(html_path.parent.exists() and not is_busy)

        def _generate_pae_heatmap(self) -> None:
            target = self._selected_pae_target()
            if not target:
                QMessageBox.information(self, "IINTS-AF Desktop", "No PAE target is configured.")
                return
            self.molecule_pae_status.setText(
                f"Generating interactive AlphaFold PAE heatmap for {target}. "
                "The first run needs internet access."
            )
            self.generate_pae_button.setEnabled(False)
            self.open_pae_button.setEnabled(False)
            self.open_pae_folder_button.setEnabled(False)
            self.status.setText("Generating PAE heatmap")

            thread = QThread(self)
            worker = PAEWorker(target=target, output_dir=self._structural_output_dir())
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_pae_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_pae_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_pae_refs)
            self.pae_thread = thread
            self.pae_worker = worker
            thread.start()

        @Slot(object)
        def _handle_pae_success(self, result: object) -> None:
            html_path_value = getattr(result, "html_path", None)
            html_path = Path(str(html_path_value)) if html_path_value else self._selected_pae_html_path()
            if html_path is not None and html_path.exists():
                self.molecule_pae_status.setText(f"PAE heatmap ready: {html_path}")
                self.status.setText("PAE heatmap ready")
                self._display_pae_html(html_path)
            else:
                self.molecule_pae_status.setText("PAE heatmap generation finished, but no HTML file was found.")
                self.status.setText("PAE heatmap missing")
            self._update_pae_controls()

        @Slot(str)
        def _handle_pae_error(self, details: str) -> None:
            self.molecule_pae_status.setText(
                "PAE heatmap failed. Check internet access and install Plotly with the research or desktop extras."
            )
            self._write_log(f"\nPAE ERROR:\n{details}\n")
            self.status.setText("PAE heatmap failed")
            self._update_pae_controls()

        @Slot()
        def _clear_pae_refs(self) -> None:
            self.pae_thread = None
            self.pae_worker = None
            self._update_pae_controls()

        def _open_selected_pae_html(self) -> None:
            html_path = self._selected_pae_html_path()
            if html_path and html_path.exists():
                self._display_pae_html(html_path)
            else:
                QMessageBox.information(
                    self,
                    "IINTS-AF Desktop",
                    "Generate the PAE heatmap first.",
                )

        def _open_pae_folder(self) -> None:
            html_path = self._selected_pae_html_path()
            folder = html_path.parent if html_path else self._structural_output_dir()
            folder.mkdir(parents=True, exist_ok=True)
            self._open_path(folder)

        def _display_pae_html(self, html_path: Path) -> None:
            if self.pae_web_view is not None:
                self._load_pae_html_in_app(html_path)
                self.status.setText(f"PAE loaded in app: {html_path.name}")
            else:
                self._open_path(html_path)

        def _load_pae_html_in_app(self, html_path: Path) -> None:
            if self.pae_web_view is not None:
                self.pae_web_view.setUrl(QUrl.fromLocalFile(str(html_path.resolve())))

        def _set_empty_pae_preview(self, target: str) -> None:
            if self.pae_web_view is not None and hasattr(self.pae_web_view, "setHtml"):
                self.pae_web_view.setHtml(
                    "<html><body style='font-family: sans-serif; color: #2f3b44; "
                    "background: #f8faf8; padding: 18px;'>"
                    f"<h3>PAE heatmap: {target}</h3>"
                    "<p>Click <b>Generate PAE Heatmap</b> to fetch the AlphaFold PAE JSON "
                    "and load the interactive matrix here.</p>"
                    "</body></html>"
                )

        def _run_genomics_simulation(self) -> None:
            if self.biology_thread is not None:
                QMessageBox.information(self, "IINTS-AF Desktop", "A simulation is already running.")
                return
            full_input = self.genomics_variant_input.text().strip()
            if not full_input:
                QMessageBox.information(self, "IINTS-AF Desktop", "Please enter a variant (e.g. INSR V938M).")
                return

            parts = full_input.split(maxsplit=1)
            if len(parts) == 2:
                gene, variant = parts[0], parts[1]
            else:
                gene, variant = "INSR", full_input

            self._set_biology_action_state(True)
            self.biology_action_status.setText(f"Running AlphaFold Simulation for {gene} {variant}...")
            self.biology_action_output.setPlainText("Working...\n")
            self.status.setText("Running genomics simulation")
            self.progress_bar.show()

            out_dir = self._structural_output_dir()
            thread = QThread(self)
            worker = GenomicsWorker(gene=gene, variant=variant, out_dir=out_dir)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_genomics_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_biology_action_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_biology_refs)
            self.biology_thread = thread
            self.biology_worker = worker
            thread.start()

        def _run_tissue_stress_simulation(self) -> None:
            if self.biology_thread is not None:
                QMessageBox.information(self, "IINTS-AF Desktop", "A simulation is already running.")
                return

            muscle_val = self.tissue_muscle_input.value() / 100.0
            liver_val = self.tissue_liver_input.value() / 100.0

            self._set_biology_action_state(True)
            self.biology_action_status.setText(f"Running Tissue Stress Simulation (Muscle {muscle_val}, Liver {liver_val})...")
            self.biology_action_output.setPlainText("Working...\n")
            self.status.setText("Running tissue stress simulation")
            self.progress_bar.show()

            out_dir = self._structural_output_dir()
            thread = QThread(self)
            worker = TissueStressorWorker(muscle_scalar=muscle_val, liver_scalar=liver_val, out_dir=out_dir)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_tissue_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_biology_action_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_biology_refs)
            self.biology_thread = thread
            self.biology_worker = worker
            thread.start()

        @Slot(object)
        def _handle_tissue_success(self, result: object) -> None:
            msg, data = cast(tuple[str, dict[str, Any]], result)
            self.biology_action_status.setText(msg)
            self.biology_action_output.setPlainText(f"Tissue Impact Data:\n{data}")
            self.status.setText("Tissue simulation complete")
            self._set_biology_action_state(False)

            # Automatically open the generated plot in the browser
            plot_path = Path(data["html_path"])
            if plot_path.exists():
                self._open_path(plot_path)

        @Slot()
        def _highlight_mutation(self) -> None:
            variant = self.genomics_variant_input.text().strip()
            if not variant:
                return

            # Very simple regex match to find the number in the variant
            import re
            match = re.search(r'\d+', variant)
            if match and self.molecule_web_view is not None:
                residue = int(match.group())
                # Generate Javascript to highlight the specific residue in red
                js = f"if(typeof glviewer !== 'undefined') {{ glviewer.setStyle({{resi: {residue}}}, {{sphere: {{color: 'red'}}}}); glviewer.render(); }}"
                self.molecule_web_view.page().runJavaScript(js)
                self.biology_action_status.setText(f"Highlighted residue {residue} in 3D viewer.")

        @Slot(object)
        def _handle_genomics_success(self, result: object) -> None:
            msg, data = cast(tuple[str, dict[str, Any]], result)
            self.biology_action_status.setText(msg)
            self.biology_action_output.setPlainText(f"Mutation Impact Data:\n{data}")
            self.status.setText("Genomics simulation complete")
            self._set_biology_action_state(False)

            # Automatically open the generated plot in the browser
            html_path_text = str(data.get("html_path", ""))
            plot_path = Path(html_path_text) if html_path_text else Path()
            if not html_path_text:
                full_input = self.genomics_variant_input.text().strip().upper()
                parts = full_input.split(maxsplit=1)
                gene, variant = (parts[0], parts[1]) if len(parts) == 2 else ("INSR", full_input)
                plot_path = self._structural_output_dir() / f"multiscale_{gene}_{variant}.html"
            if plot_path.exists():
                self._open_path(plot_path)

        @Slot(str)
        def _handle_biology_action_error(self, details: str) -> None:
            self.biology_action_status.setText("Biology evidence action failed. See details below.")
            self.biology_action_output.setPlainText(details)
            self._write_log(f"\nBIOLOGY ACTION ERROR:\n{details}\n")
            self.status.setText("Biology evidence action failed")
            self._set_biology_action_state(False)

        @Slot()
        def _clear_biology_refs(self) -> None:
            self.biology_thread = None
            self.biology_worker = None
            self._set_biology_action_state(False)

        def _set_biology_action_state(self, is_running: bool) -> None:
            for button in (
                self.run_genomics_sim_button,
                self.highlight_mutation_button,
                self.open_structural_folder_button,
                self.run_tissue_stress_button,
            ):
                button.setEnabled(not is_running)
            self.genomics_variant_input.setEnabled(not is_running)
            self.tissue_muscle_input.setEnabled(not is_running)
            self.tissue_liver_input.setEnabled(not is_running)

        def _open_structural_folder(self) -> None:
            folder = self._structural_output_dir()
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(
                    self,
                    "Could not open structural output folder",
                    str(exc),
                )
                return
            self._open_path(folder)

        def _structural_output_dir(self) -> Path:
            base_text = self.output_dir.text().strip()
            base = Path(base_text).expanduser() if base_text else Path.home() / "IINTS-Desktop-Runs"
            return base.resolve() / "structural"

        def _open_app_downloads(self) -> None:
            QDesktopServices.openUrl(QUrl(DESKTOP_RELEASE_URL))
            self.status.setText("Opened app downloads")

        def _open_update_docs(self) -> None:
            QDesktopServices.openUrl(QUrl(UPDATE_DOCS_URL))
            self.status.setText("Opened update docs")

        def _copy_update_command(self) -> None:
            QApplication.clipboard().setText(PYTHON_SDK_UPDATE_COMMAND)
            self.update_status.setText(f"Copied: {PYTHON_SDK_UPDATE_COMMAND}")
            self.status.setText("Update command copied")

        def _run_package_update(self) -> None:
            if getattr(sys, "frozen", False):
                QMessageBox.information(
                    self,
                    "IINTS-AF Desktop",
                    "Packaged app builds update by downloading the newest app build. "
                    "Use 'Open App Downloads'.",
                )
                return
            if QMessageBox.question(
                self,
                "Update Python SDK package",
                "This will open a terminal and run the pip update command. Continue?",
            ) != QMessageBox.StandardButton.Yes:
                return

            from iints_desktop.terminal_utils import open_terminal_and_run

            success = open_terminal_and_run(build_python_sdk_update_args())
            if success:
                self.update_status.setText("Update command launched in external terminal.")
                self.update_log.setPlainText(PYTHON_SDK_UPDATE_COMMAND + "\n\nSee external terminal for progress.")
                self.status.setText("Update command launched")
            else:
                self.update_status.setText("Failed to open external terminal.")
                self.update_log.setPlainText("Could not launch your system's terminal automatically. Please copy the command and run it manually.")
                self.status.setText("Terminal launch failed")



        def _save_log(self) -> None:
            default_path = Path(self.output_dir.text()).expanduser() / "iints-desktop-log.txt"
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Save desktop log",
                str(default_path),
                "Text files (*.txt);;All files (*)",
            )
            if not chosen:
                return
            target = Path(chosen).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.log.toPlainText(), encoding="utf-8")
            self.status.setText(f"Log saved: {chosen}")

        def _open_path(self, path: Path) -> None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

        def _write_log(self, text: str) -> None:
            self.log.appendPlainText(text.rstrip("\n"))

        def _refresh_history(self) -> None:
            self.history_entries = read_run_history(self.output_dir.text(), limit=50)
            self.history_table.setRowCount(len(self.history_entries))
            for row, entry in enumerate(self.history_entries):
                report_state = "exists" if entry.report_pdf and Path(entry.report_pdf).exists() else "missing"
                csv_state = "exists" if entry.results_csv and Path(entry.results_csv).exists() else "missing"
                values = [
                    entry.timestamp_utc,
                    entry.workflow_title,
                    entry.preset_name,
                    "" if entry.seed is None else str(entry.seed),
                    entry.run_id,
                    report_state,
                    csv_state,
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, row)
                    self.history_table.setItem(row, column, item)
            self._update_history_action_buttons()

        def _selected_history_entry(self) -> DesktopRunHistoryEntry | None:
            selected = self.history_table.selectedItems()
            if not selected:
                return None
            row = selected[0].row()
            if row < 0 or row >= len(self.history_entries):
                return None
            return self.history_entries[row]

        def _saved_seed(self) -> int:
            value = self.settings.value("seed", 42)
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return 42

        def _update_history_action_buttons(self) -> None:
            entry = self._selected_history_entry()
            folder_exists = bool(entry and Path(entry.output_dir).exists())
            report_exists = bool(entry and entry.report_pdf and Path(entry.report_pdf).exists())
            csv_exists = bool(entry and entry.results_csv and Path(entry.results_csv).exists())
            self.load_selected_history_csv_button.setEnabled(csv_exists)
            self.open_selected_folder_button.setEnabled(folder_exists)
            self.open_selected_report_button.setEnabled(report_exists)
            self.open_selected_csv_button.setEnabled(csv_exists)

        def _open_selected_history_folder(self) -> None:
            entry = self._selected_history_entry()
            if entry:
                self._open_path(Path(entry.output_dir))

        def _open_selected_history_report(self) -> None:
            entry = self._selected_history_entry()
            if entry and entry.report_pdf:
                self._open_path(Path(entry.report_pdf))

        def _open_selected_history_csv(self) -> None:
            entry = self._selected_history_entry()
            if entry and entry.results_csv:
                self._open_path(Path(entry.results_csv))

        def _load_selected_history_csv(self) -> None:
            entry = self._selected_history_entry()
            if entry and entry.results_csv:
                self.result_csv_path.setText(entry.results_csv)
                self._load_result_csv(Path(entry.results_csv))
                if self.tabs is not None:
                    self.tabs.setCurrentIndex(1)

        def _compare_selected_runs(self) -> None:
            selected_ranges = self.history_table.selectedRanges()
            if not selected_ranges:
                self.status.setText("Select at least 2 runs to compare.")
                return

            rows = set()
            for r in selected_ranges:
                for i in range(r.topRow(), r.bottomRow() + 1):
                    rows.add(i)

            if len(rows) < 2:
                self.status.setText("Select at least 2 runs to compare.")
                return

            import pandas as pd
            import plotly.graph_objects as go

            fig = go.Figure()
            valid_runs = 0

            for row in sorted(rows):
                if row >= len(self.history_entries):
                    continue
                entry = self.history_entries[row]
                if not entry.results_csv or not Path(entry.results_csv).exists():
                    continue

                try:
                    df = pd.read_csv(entry.results_csv)
                    time_column = "time_minutes" if "time_minutes" in df.columns else "time"
                    glucose_column = (
                        "glucose_actual_mgdl"
                        if "glucose_actual_mgdl" in df.columns
                        else "glucose"
                    )
                    if time_column in df.columns and glucose_column in df.columns:
                        name = f"{entry.preset_name} (Seed: {entry.seed})"
                        fig.add_trace(
                            go.Scatter(
                                x=df[time_column],
                                y=df[glucose_column],
                                mode="lines",
                                name=name,
                            )
                        )
                        valid_runs += 1
                except Exception as e:
                    self._write_log(f"Failed to load CSV for {entry.preset_name}: {e}\n")

            if valid_runs < 2:
                self.status.setText("Not enough valid results.csv files selected.")
                return

            fig.update_layout(
                title="Simulation Comparison (Glucose Over Time)",
                xaxis_title="Time (minutes)",
                yaxis_title="Glucose (mg/dL)",
                template="plotly_dark",
            )

            html_path = Path(self.output_dir.text()).expanduser() / ".cache" / "comparison_graph.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path))

            if self.result_graph_web is not None:
                from PySide6.QtCore import QUrl
                self.result_graph_web.load(QUrl.fromLocalFile(str(html_path)))
                self.result_csv_path.setText("Comparing multiple runs...")
                if self.tabs is not None:
                    self.tabs.setCurrentIndex(1)
                self.status.setText("Comparison loaded.")
            else:
                self.status.setText("QWebEngineView not available. Graph generated but cannot be shown.")



def _apply_application_palette(app: QApplication) -> None:
    """Force a readable light palette even when the OS is in dark mode."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#e9edf1"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#202a35"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f1f4f6"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#202a35"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#202a35"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#f2f4f6"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1f2d3a"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#cfe2f0"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#102436"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6c7a86"))
    app.setPalette(palette)


def _verify_full_desktop_runtime() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("plotly", "roadrunner", "fmpy", "torch"):
        module = import_module(module_name)
        versions[module_name] = str(getattr(module, "__version__", "unknown"))
    sundials = import_module("fmpy.sundials")
    if getattr(sundials, "CVodeSolver", None) is None:
        raise RuntimeError("FMPy SUNDIALS CVodeSolver is unavailable in the packaged runtime.")
    versions["fmpy.sundials"] = "available"
    return versions


def main() -> int:
    if _PYSIDE_IMPORT_ERROR is not None:
        message = (
            "PySide6 is not installed. Install it with: "
            'python -m pip install -U "iints-sdk-python35[desktop-all]" '
            'or, from a source checkout, python -m pip install -U -e ".[desktop-all]"'
        )
        _write_startup_log(message)
        raise RuntimeError(message) from _PYSIDE_IMPORT_ERROR

    try:
        app = QApplication(sys.argv)
        _apply_application_palette(app)
        window = IINTSQtDesktopApp()
        if "--smoke" in sys.argv or "--smoke-full" in sys.argv:
            window.resize(760, 520)
            app.processEvents()
            window.resize(1240, 820)
            app.processEvents()
            full_runtime = (
                _verify_full_desktop_runtime() if "--smoke-full" in sys.argv else None
            )
            print(
                "Qt desktop smoke OK:",
                window.windowTitle(),
                f"workflows={window.workflow_combo.count()}",
                f"history_rows={window.history_table.rowCount()}",
                f"min={window.minimumWidth()}x{window.minimumHeight()}",
                file=sys.__stdout__,
            )
            if full_runtime is not None:
                print(
                    "Full desktop runtime OK:",
                    ", ".join(f"{name}={version}" for name, version in full_runtime.items()),
                    file=sys.__stdout__,
                )
            window.close()
            app.quit()
            return 0
        window.show()
        return app.exec()
    except Exception:
        details = traceback.format_exc()
        _write_startup_log(details)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
