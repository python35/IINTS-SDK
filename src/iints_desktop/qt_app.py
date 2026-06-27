from __future__ import annotations

import os
import subprocess
import sys
import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox"

import traceback
from pathlib import Path

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
)
from iints_desktop.local_ai import ask_local_ai, check_local_ai, start_local_ai_stack
from iints_desktop.molecules import MoleculeAsset, list_molecule_assets, pae_html_path
from iints_desktop.results import ResultPreview, load_results_preview
from iints_desktop.fetcher import fetch_alphafold_structure
from iints_desktop.render_3dmol import generate_3dmol_html

DESKTOP_RELEASE_URL = "https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-2026-06-27-4"
UPDATE_DOCS_URL = "https://python35.github.io/IINTS-SDK/APP_INSTALL/"
PYTHON_SDK_UPDATE_COMMAND = 'python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"'

try:  # pragma: no cover - optional GUI dependency
    from PySide6.QtCore import Qt, QObject, QSettings, QThread, QUrl, Signal, Slot  # type: ignore[import-not-found]
    from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QPalette, QPixmap  # type: ignore[import-not-found]
    from PySide6.QtWidgets import (  # type: ignore[import-not-found]
        QApplication,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QScrollArea,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
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
    _QWEBENGINE_VIEW = None
else:  # pragma: no cover - optional GUI dependency
    _PYSIDE_IMPORT_ERROR = None
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView as _QWEBENGINE_VIEW  # type: ignore[import-not-found,no-redef]
    except ModuleNotFoundError:
        _QWEBENGINE_VIEW = None


if _PYSIDE_IMPORT_ERROR is None:
    from iints_desktop.molecule_viewer import MolecularChainViewer

    class RunWorker(QObject):
        """Background SDK run worker so the Qt UI stays responsive."""

        finished = Signal(object)
        failed = Signal(str)
        log = Signal(str)

        def __init__(self, *, output_dir: str, desktop_preset_key: str, seed: int) -> None:
            super().__init__()
            self.output_dir = output_dir
            self.desktop_preset_key = desktop_preset_key
            self.seed = seed

        @Slot()
        def run(self) -> None:
            try:
                self.log.emit("Calling the IINTS-AF SDK engine...\n")
                result = run_demo_preset(
                    output_dir=self.output_dir,
                    desktop_preset_key=self.desktop_preset_key,
                    seed=self.seed,
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


    class UpdateWorker(QObject):
        """Background worker for Python-package SDK updates from source installs."""

        finished = Signal(object)
        failed = Signal(str)

        @Slot()
        def run(self) -> None:
            try:
                if getattr(sys, "frozen", False):
                    self.finished.emit(
                        "Packaged app builds cannot self-update safely yet. "
                        "Open the app downloads page and install the newest .exe/.dmg build."
                    )
                    return
                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "iints-sdk-python35[full,desktop-qt,mdmp]",
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = completed.stdout or ""
                if completed.returncode != 0:
                    raise RuntimeError(output or f"pip exited with code {completed.returncode}")
                self.finished.emit(output or "SDK package update completed.")
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class PAEWorker(QObject):
        """Background worker that renders an interactive AlphaFold PAE heatmap."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, target: str) -> None:
            super().__init__()
            self.target = target

        @Slot()
        def run(self) -> None:
            try:
                from iints.research.structure import render_pae

                results = render_pae(self.target)
                self.finished.emit(results[0] if results else None)
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    
    class AlphaFoldFetchWorker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, uniprot_id: str):
            super().__init__()
            self.uniprot_id = uniprot_id

        @Slot()
        def run(self) -> None:
            try:
                from pathlib import Path
                out_dir = Path("results") / "structural"
                cif_path = fetch_alphafold_structure(self.uniprot_id, out_dir)
                html_path = generate_3dmol_html(cif_path, out_dir)
                self.finished.emit((cif_path, html_path, self.uniprot_id))
            except Exception as exc:
                self.failed.emit(str(exc))

    class BiologyWorker(QObject):
        """Background worker for optional public biomedical evidence helpers."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, *, action: str, value: str) -> None:
            super().__init__()
            self.action = action
            self.value = value

        @Slot()
        def run(self) -> None:
            try:
                if self.action == "gtex-expression":
                    from iints.research.anatomy import render_expression

                    result = render_expression(self.value)
                    if result is None:
                        self.finished.emit("No GTEx expression artifact was generated.")
                    else:
                        self.finished.emit(
                            f"GTEx expression ready for {result.official_gene}: {result.html_path}"
                        )
                elif self.action == "insulin-pk":
                    from iints.research.pharmacology import analyze_insulin

                    molecule, profile = analyze_insulin(self.value)
                    molecule_name = molecule.preferred_name if molecule is not None else "no ChEMBL match"
                    self.finished.emit(
                        f"Insulin PK mapping ready for {self.value}: {profile.label}, "
                        f"t_max,I={profile.tmax_minutes} min ({molecule_name})."
                    )
                elif self.action == "clinvar-mutation":
                    from iints.research.genetics import simulate_mutation

                    variants = simulate_mutation(self.value)
                    self.finished.emit(
                        f"ClinVar mutation stressor complete for {self.value}: "
                        f"{len(variants)} public variant summaries shown."
                    )
                elif self.action == "string-pathways":
                    from iints.research.physiology import render_pathways

                    results = render_pathways(self.value)
                    paths = ", ".join(str(result.png_path) for result in results)
                    self.finished.emit(f"STRING pathway render complete: {paths}")
                else:
                    raise ValueError(f"Unknown biology evidence action: {self.action}")
            except Exception:  # pragma: no cover - GUI error path
                self.failed.emit(traceback.format_exc())


    class IINTSQtDesktopApp(QMainWindow):
        """PySide/Qt desktop shell for a more polished native app experience."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("IINTS-AF SDK | Research Workbench")
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
            self.history_entries: list[DesktopRunHistoryEntry] = []
            self.current_thread: QThread | None = None
            self.current_worker: RunWorker | None = None
            self.ai_thread: QThread | None = None
            self.ai_worker: AIWorker | None = None
            self.ai_start_thread: QThread | None = None
            self.ai_start_worker: LocalAIStartWorker | None = None
            self.update_thread: QThread | None = None
            self.update_worker: UpdateWorker | None = None
            self.pae_thread: QThread | None = None
            self.pae_worker: PAEWorker | None = None
            self.biology_thread: QThread | None = None
            self.biology_worker: BiologyWorker | None = None
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
            self.result_graph = QLabel("Load a results CSV to view a glucose graph.")
            self.result_graph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.history_table = QTableWidget(0, 7)
            self.ai_model = QLineEdit(DEFAULT_MINISTRAL_MODEL)
            self.ai_host = QLineEdit("http://127.0.0.1:11434")
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
            self.molecule_reference_render = QLabel()
            self.molecule_structure_status = QLabel()
            self.molecule_pae_status = QLabel()
            self.biology_action_status = QLabel("No biology evidence action has run yet.")
            self.biology_action_status.setWordWrap(True)
            self.biology_action_output = QTextEdit()
            self.biology_action_output.setReadOnly(True)
            self.pae_web_view: QWidget | None = None

            self.run_button = QPushButton("Run Selected Workflow")
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
            self.start_ai_button = QPushButton("Start Local AI")
            self.check_ai_button = QPushButton("Check Ollama")
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
            self.render_gtex_button = QPushButton("Render GTEx Expression")
            self.analyze_insulin_button = QPushButton("Analyze Insulin PK")
            self.simulate_mutation_button = QPushButton("Simulate ClinVar Mutation")
            self.render_pathways_button = QPushButton("Render STRING Pathways")
            self.open_structural_folder_button = QPushButton("Open Biology Output Folder")
            self.open_app_downloads_button = QPushButton("Open App Downloads")
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
            about_tab = QWidget()
            tabs.addTab(run_tab, "Simulation")
            tabs.addTab(results_tab, "Results")
            tabs.addTab(ai_tab, "AI Review")
            tabs.addTab(history_tab, "Run Archive")
            tabs.addTab(molecules_tab, "Biology")
            tabs.addTab(about_tab, "Methods")
            self._build_run_tab(run_tab)
            self._build_results_tab(results_tab)
            self._build_ai_tab(ai_tab)
            self._build_history_tab(history_tab)
            self._build_molecules_tab(molecules_tab)
            self._build_about_tab(about_tab)

            status_bar = QStatusBar(self)
            status_bar.setSizeGripEnabled(False)
            status_bar.addWidget(self.status, 1)
            workspace_label = QLabel(f"Workspace: {Path(self.output_dir.text()).expanduser()}")
            workspace_label.setObjectName("workspaceStatus")
            status_bar.addPermanentWidget(workspace_label)
            self.workspace_status = workspace_label
            self.setStatusBar(status_bar)

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
                self.workflow_combo.addItem(preset.title, preset.key)
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
                    ],
                    columns=4,
                )
            )
            layout.addWidget(csv_box)

            workspace = QSplitter(Qt.Orientation.Horizontal)
            self._register_responsive_splitter(workspace)
            layout.addWidget(workspace, stretch=1)

            graph_box = QGroupBox("Glucose trajectory")
            graph_layout = QVBoxLayout(graph_box)
            self.result_graph.setMinimumHeight(220)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.result_graph)
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
            config_layout.addWidget(self.start_ai_button, 0, 2)
            config_layout.addWidget(self.check_ai_button, 1, 2)
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
            layout.addLayout(
                self._button_grid(
                    [
                        refresh_button,
                        open_base_button,
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
            self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
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

            if _QWEBENGINE_VIEW is not None and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                self.molecule_web_view = _QWEBENGINE_VIEW()
                self.molecule_web_view.setMinimumHeight(260)
                viewer_layout.addWidget(self.molecule_web_view, stretch=1)
                self.molecule_viewer.hide() # fallback hidden
            else:
                self.molecule_web_view = None

            self.reset_molecule_view_button.clicked.connect(self._reset_molecule_view)
            self.open_molecule_image_button.clicked.connect(self._open_selected_molecule_image)
            self.open_molecule_structure_button.clicked.connect(self._open_selected_molecule_structure)
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
            if _QWEBENGINE_VIEW is not None and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
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

            evidence_box = QGroupBox("Biology evidence actions")
            evidence_layout = QVBoxLayout(evidence_box)
            evidence_help = QLabel(
                "Run optional public-database helpers from the app. These actions fetch research context "
                "from GTEx, ChEMBL, ClinVar, or STRING and write normal SDK artifacts under results/structural."
            )
            evidence_help.setWordWrap(True)
            evidence_layout.addWidget(evidence_help)
            self.biology_action_status.setObjectName("biologyActionStatus")
            evidence_layout.addWidget(self.biology_action_status)
            self.render_gtex_button.clicked.connect(lambda: self._run_biology_action("gtex-expression", "GLUT4"))
            self.analyze_insulin_button.clicked.connect(lambda: self._run_biology_action("insulin-pk", "lispro"))
            self.simulate_mutation_button.clicked.connect(lambda: self._run_biology_action("clinvar-mutation", "INSR"))
            self.render_pathways_button.clicked.connect(lambda: self._run_biology_action("string-pathways", "all"))
            self.open_structural_folder_button.clicked.connect(self._open_structural_folder)
            evidence_layout.addLayout(
                self._button_grid(
                    [
                        self.render_gtex_button,
                        self.analyze_insulin_button,
                        self.simulate_mutation_button,
                        self.render_pathways_button,
                        self.open_structural_folder_button,
                    ],
                    columns=2,
                )
            )
            self.biology_action_output.setMinimumHeight(95)
            evidence_layout.addWidget(self.biology_action_output)
            context_layout.addWidget(evidence_box)

            self.molecule_reference_render.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.molecule_reference_render.setMinimumHeight(200)
            context_layout.addWidget(self.molecule_reference_render, stretch=1)
            usage_hint = QLabel(
                "Controls: drag to rotate, mouse wheel to zoom, double-click to reset."
            )
            usage_hint.setObjectName("subtleHint")
            usage_hint.setWordWrap(True)
            context_layout.addWidget(usage_hint)
            viewer_row.addWidget(context_box)
            viewer_row.setSizes([760, 420])


        def _build_about_tab(self, parent: QWidget) -> None:
            layout = self._scroll_tab_layout(parent)
            intro = QLabel(
                "IINTS-AF Desktop is a native research workbench for running SDK simulations, "
                "reviewing generated results, asking local AI questions, and opening biology evidence artifacts. "
                "The Python SDK remains the single source of truth for formulas, reports, and validation."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

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

            for preset in self.presets:
                label = QLabel(f"<b>{preset.title}</b><br>{preset.description}")
                label.setWordWrap(True)
                layout.addWidget(label)
            layout.addStretch(1)

        def _apply_style(self) -> None:
            self.setStyleSheet(
                """
                * {
                    color: #202a35;
                    selection-background-color: #cfe2f0;
                    selection-color: #102436;
                }
                QMainWindow, QWidget, QWidget#root {
                    background-color: #e9edf1;
                    color: #202a35;
                    font-size: 13px;
                }
                QLabel {
                    color: #202a35;
                    background: transparent;
                }
                QFrame, QSplitter, QAbstractScrollArea, QScrollArea {
                    background-color: #f7f8fa;
                    color: #202a35;
                    border-color: #b9c3cc;
                }
                QScrollArea > QWidget > QWidget {
                    background-color: #f7f8fa;
                    color: #202a35;
                }
                QMenuBar, QToolBar, QStatusBar {
                    background: #f6f7f8;
                    color: #202a35;
                    border-color: #b9c3cc;
                }
                QMenuBar {
                    border-bottom: 1px solid #b9c3cc;
                }
                QMenuBar::item {
                    padding: 4px 9px;
                }
                QMenuBar::item:selected, QMenu::item:selected {
                    background: #dbe7f1;
                }
                QMenu {
                    background: #ffffff;
                    color: #202a35;
                    border: 1px solid #aebac5;
                }
                QToolBar {
                    spacing: 4px;
                    padding: 3px 5px;
                    border-bottom: 1px solid #b9c3cc;
                }
                QToolButton {
                    background: #f6f7f8;
                    color: #1f2d3a;
                    border: 1px solid transparent;
                    border-radius: 2px;
                    padding: 5px 8px;
                }
                QToolButton:hover {
                    background: #e1e9ef;
                    border-color: #9dacb9;
                }
                QWidget#workbenchHeader {
                    background: #ffffff;
                    border: 1px solid #b9c3cc;
                }
                QLabel#appTitle {
                    color: #173b5c;
                }
                QLabel#researchBadge {
                    background: #fff3cd;
                    color: #714b00;
                    border: 1px solid #d6b765;
                    padding: 3px 7px;
                    font-weight: 700;
                }
                QTabWidget::pane {
                    border: 1px solid #aebac5;
                    background: #f7f8fa;
                    padding: 4px;
                }
                QTabBar::tab {
                    background: #dfe5ea;
                    color: #34414d;
                    border: 1px solid #aebac5;
                    border-bottom: none;
                    border-radius: 0;
                    padding: 7px 12px;
                    margin-right: 2px;
                    font-weight: 600;
                }
                QTabBar::tab:selected {
                    background: #f7f8fa;
                    color: #173b5c;
                    border-top: 3px solid #2b618d;
                    padding-top: 5px;
                }
                QLabel#infoStrip, QLabel#deepDiveIntro {
                    background: #edf4f8;
                    color: #1f4b6e;
                    border: 1px solid #a9c1d4;
                    padding: 6px 8px;
                }
                QLabel#metricSummary {
                    background: #f1f4f6;
                    border: 1px solid #c7d0d8;
                    color: #273746;
                    padding: 7px;
                }
                QLabel#moleculeTitle {
                    color: #173b5c;
                    font-size: 16px;
                    font-weight: 700;
                }
                QLabel#moleculeStatus {
                    background: #eef5ee;
                    color: #285a32;
                    border: 1px solid #a9c5ad;
                    padding: 6px 8px;
                }
                QLabel#moleculePAEStatus, QLabel#biologyActionStatus, QLabel#updateStatus {
                    background: #f4f7f4;
                    color: #2f4830;
                    border: 1px solid #bdcdbd;
                    padding: 6px 8px;
                }
                QGroupBox {
                    background: #ffffff;
                    color: #202a35;
                    border: 1px solid #b9c3cc;
                    border-radius: 2px;
                    margin-top: 9px;
                    padding: 10px 8px 8px 8px;
                    font-weight: 650;
                }
                QGroupBox QLabel {
                    color: #202a35;
                    background: transparent;
                    font-weight: 400;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 8px;
                    padding: 0 4px;
                    color: #173b5c;
                }
                QPushButton {
                    background: #f2f4f6;
                    color: #1f2d3a;
                    border: 1px solid #aebac5;
                    border-radius: 2px;
                    padding: 5px 9px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #dfe9f1;
                    border-color: #7d94a7;
                }
                QPushButton#primaryAction {
                    background: #245a82;
                    color: #ffffff;
                    border-color: #173b5c;
                    font-weight: 700;
                }
                QPushButton:disabled {
                    background: #e4e8eb;
                    color: #7d8993;
                    border-color: #c6cdd3;
                }
                QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit, QTableWidget, QTableView {
                    background: #ffffff;
                    color: #202a35;
                    border: 1px solid #aebac5;
                    border-radius: 1px;
                    padding: 5px;
                }
                QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus, QTableWidget:focus {
                    border: 1px solid #2b618d;
                }
                QComboBox QAbstractItemView {
                    background: #ffffff;
                    color: #202a35;
                    selection-background-color: #d8e7f2;
                    selection-color: #202a35;
                }
                QPlainTextEdit:disabled, QTextEdit:disabled, QLineEdit:disabled {
                    color: #52616f;
                    background: #eef1f3;
                }
                QTableWidget::item:selected {
                    background: #d8e7f2;
                    color: #1f2d3a;
                }
                QHeaderView::section {
                    background: #dfe5ea;
                    color: #243746;
                    border: 1px solid #b9c3cc;
                    padding: 4px;
                    font-weight: 650;
                }
                QPlainTextEdit, QTextEdit {
                    font-family: Menlo, Consolas, monospace;
                }
                QLabel#subtleHint {
                    color: #5c6b78;
                    background: #f6f7f8;
                    border: 1px solid #d5dde4;
                    padding: 5px 7px;
                }
                QStatusBar {
                    color: #314251;
                    border-top: 1px solid #b9c3cc;
                }
                QLabel#workspaceStatus {
                    color: #667684;
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
            self.open_folder_button.setEnabled(True)
            self.open_report_button.setEnabled(bool(result.report_pdf and result.report_pdf.exists()))
            self.open_csv_button.setEnabled(bool(result.results_csv and result.results_csv.exists()))
            self.copy_summary_button.setEnabled(True)
            self._write_log("\n" + result.summary + "\n")
            self._refresh_history()
            if result.results_csv:
                self.result_csv_path.setText(str(result.results_csv))
                self._load_result_csv(result.results_csv)
                if self.tabs is not None:
                    self.tabs.setCurrentIndex(1)

        @Slot(str)
        def _handle_error(self, details: str) -> None:
            self.status.setText("Run failed")
            self._set_running_state(False)
            self._write_log(f"\nERROR:\n{details}\n")
            QMessageBox.critical(self, "IINTS-AF Desktop", details)

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

        def _load_result_csv(self, path: Path) -> None:
            try:
                preview = load_results_preview(path)
            except Exception as exc:
                QMessageBox.critical(self, "IINTS-AF Desktop", str(exc))
                return
            self.loaded_result = preview
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
            if preview.graph_path and preview.graph_path.exists():
                pixmap = QPixmap(str(preview.graph_path))
                self.result_graph.clear()
                self.result_graph.setPixmap(
                    pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
                )
            else:
                self.result_graph.clear()
                self.result_graph.setText("No glucose column was found, so no graph could be generated.")
            self.ai_context_label.setText(f"AI context: loaded summary from {preview.csv_path.name}")
            self._set_loaded_result_actions(True)

        def _copy_last_summary(self) -> None:
            if self.last_result:
                QApplication.clipboard().setText(self.last_result.summary)
                self.status.setText("Run summary copied")

        def _set_ai_starting_state(self, is_starting: bool) -> None:
            self.start_ai_button.setEnabled(not is_starting)
            self.check_ai_button.setEnabled(not is_starting)
            self.ask_ai_button.setEnabled(not is_starting)

        def _start_local_ai(self) -> None:
            self._set_ai_starting_state(True)
            self.ai_status.setText("Starting local AI. First model download can take a while...")
            self.status.setText("Starting local AI")
            thread = QThread(self)
            worker = LocalAIStartWorker(
                model=self.ai_model.text().strip() or DEFAULT_MINISTRAL_MODEL,
                host=self.ai_host.text().strip(),
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
                model=self.ai_model.text().strip() or DEFAULT_MINISTRAL_MODEL,
                host=self.ai_host.text().strip() or None,
            )
            self.ai_status.setText(status.message)
            self.status.setText("Local AI ready" if status.available else "Local AI not ready")

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
                model=self.ai_model.text().strip() or DEFAULT_MINISTRAL_MODEL,
                host=self.ai_host.text().strip(),
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
            model = getattr(answer, "model", self.ai_model.text())
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
            worker = AlphaFoldFetchWorker(uniprot_id)
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
            cif_path, html_path, uniprot_id = result
            
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

        def _on_molecule_changed(self) -> None:
            if not self.molecules or self.molecule_viewer is None:
                return
            molecule = self._selected_molecule()
            self.molecule_title.setText(
                f"<b>{molecule.title}</b><br>UniProt {molecule.uniprot_id} / AlphaFold structure"
            )
            self.molecule_explanation.setText(
                f"{molecule.explanation}<br><br>"
                f"<b>{molecule.sdk_link}</b>"
            )
            self.molecule_viewer.set_structure(
                molecule.structure_path,
                display_name=f"{molecule.title} / UniProt {molecule.uniprot_id}",
            )
            if self.molecule_viewer.error:
                self.molecule_structure_status.setText(
                    f"3D structure could not be loaded: {self.molecule_viewer.error}"
                )
                self.molecule_structure_status.setStyleSheet(
                    "background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa;"
                )
            else:
                structure = self.molecule_viewer.structure
                residue_count = len(structure.atoms) if structure else 0
                chain_count = structure.chain_count if structure else 0
                self.molecule_structure_status.setText(
                    f"Local 3D model loaded: {residue_count} C-alpha residues across {chain_count} chain(s). "
                    "Colours show AlphaFold pLDDT confidence, not a clinical score."
                )
                self.molecule_structure_status.setStyleSheet("")

            # 3Dmol.js rendering path
            if getattr(self, "molecule_web_view", None) is not None:
                from iints_desktop.render_3dmol import generate_3dmol_html
                try:
                    out_dir = Path("results") / "structural"
                    html_path = generate_3dmol_html(molecule.structure_path, out_dir)
                    self.molecule_web_view.setUrl(QUrl.fromLocalFile(str(html_path.absolute())))
                    self.molecule_viewer.hide()
                    self.molecule_web_view.show()
                except Exception as e:
                    print(f"3Dmol.js render failed: {e}")
                    self.molecule_web_view.hide()
                    self.molecule_viewer.show()
            

            if molecule.image_path.exists():
                pixmap = QPixmap(str(molecule.image_path))
                self.molecule_reference_render.setPixmap(
                    pixmap.scaled(
                        330,
                        270,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.molecule_reference_render.setToolTip("Static PyMOL reference render")
            else:
                self.molecule_reference_render.setText("Reference render is unavailable.")
            self._update_pae_controls()

        def _reset_molecule_view(self) -> None:
            if self.molecule_viewer is not None:
                self.molecule_viewer.reset_camera()
                self.status.setText("3D molecular view reset")

        def _open_selected_molecule_image(self) -> None:
            self._open_path(self._selected_molecule().image_path)

        def _open_selected_molecule_structure(self) -> None:
            self._open_path(self._selected_molecule().structure_path)

        def _selected_pae_target(self) -> str | None:
            return self._selected_molecule().pae_target

        def _selected_pae_html_path(self) -> Path | None:
            target = self._selected_pae_target()
            return pae_html_path(target) if target else None

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

            html_path = pae_html_path(target)
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
            worker = PAEWorker(target=target)
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
            folder = html_path.parent if html_path else Path("results") / "structural"
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

        def _run_biology_action(self, action: str, value: str) -> None:
            if self.biology_thread is not None:
                QMessageBox.information(self, "IINTS-AF Desktop", "A biology evidence action is already running.")
                return
            self._set_biology_action_state(True)
            self.biology_action_status.setText(
                f"Running {action} ({value}). The first run may need internet access."
            )
            self.biology_action_output.setPlainText("Working...\n")
            self.status.setText("Running biology evidence action")

            thread = QThread(self)
            worker = BiologyWorker(action=action, value=value)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_biology_action_success)
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
        def _handle_biology_action_success(self, result: object) -> None:
            text = str(result)
            self.biology_action_status.setText(text)
            self.biology_action_output.setPlainText(text)
            self.status.setText("Biology evidence action complete")
            self._set_biology_action_state(False)

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
                self.render_gtex_button,
                self.analyze_insulin_button,
                self.simulate_mutation_button,
                self.render_pathways_button,
                self.open_structural_folder_button,
            ):
                button.setEnabled(not is_running)

        def _open_structural_folder(self) -> None:
            folder = Path("results") / "structural"
            folder.mkdir(parents=True, exist_ok=True)
            self._open_path(folder)

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
            if self.update_thread is not None:
                QMessageBox.information(self, "IINTS-AF Desktop", "An update action is already running.")
                return
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
                "Run pip to update iints-sdk-python35 in this Python environment?",
            ) != QMessageBox.StandardButton.Yes:
                return
            self.run_package_update_button.setEnabled(False)
            self.update_status.setText("Updating Python SDK package...")
            self.update_log.setPlainText(PYTHON_SDK_UPDATE_COMMAND + "\n\nWorking...")
            self.status.setText("Updating SDK package")

            thread = QThread(self)
            worker = UpdateWorker()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_update_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_update_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._clear_update_refs)
            self.update_thread = thread
            self.update_worker = worker
            thread.start()

        @Slot(object)
        def _handle_update_success(self, result: object) -> None:
            text = str(result)
            self.update_status.setText("SDK package update finished. Restart the app to use updated code.")
            self.update_log.setPlainText(text)
            self.run_package_update_button.setEnabled(True)
            self.status.setText("SDK update finished")

        @Slot(str)
        def _handle_update_error(self, details: str) -> None:
            self.update_status.setText("SDK package update failed. See details below.")
            self.update_log.setPlainText(details)
            self.run_package_update_button.setEnabled(True)
            self.status.setText("SDK update failed")

        @Slot()
        def _clear_update_refs(self) -> None:
            self.update_thread = None
            self.update_worker = None

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


def main() -> int:
    if _PYSIDE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PySide6 is not installed. Install it with: "
            'python -m pip install -U -e ".[full,desktop-qt,mdmp]"'
        ) from _PYSIDE_IMPORT_ERROR

    app = QApplication(sys.argv)
    _apply_application_palette(app)
    window = IINTSQtDesktopApp()
    if "--smoke" in sys.argv:
        window.resize(760, 520)
        app.processEvents()
        window.resize(1240, 820)
        app.processEvents()
        print(
            "Qt desktop smoke OK:",
            window.windowTitle(),
            f"workflows={window.workflow_combo.count()}",
            f"history_rows={window.history_table.rowCount()}",
            f"min={window.minimumWidth()}x{window.minimumHeight()}",
        )
        window.close()
        app.quit()
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
