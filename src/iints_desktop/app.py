from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from iints_desktop.engine import (
    DEFAULT_DESKTOP_PRESET_KEY,
    DesktopPreset,
    DesktopRunResult,
    get_desktop_preset,
    list_desktop_presets,
    run_demo_preset,
)


class IINTSDesktopApp:
    """Native desktop shell around the IINTS-AF SDK engine."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IINTS-AF Desktop")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)

        self.presets = list_desktop_presets()
        self.preset_by_title = {preset.title: preset for preset in self.presets}
        default_preset = get_desktop_preset(DEFAULT_DESKTOP_PRESET_KEY)

        self.output_dir = tk.StringVar(value=str(Path.home() / "IINTS-Desktop-Runs"))
        self.selected_workflow = tk.StringVar(value=default_preset.title)
        self.workflow_description = tk.StringVar(value=self._format_description(default_preset))
        self.seed = tk.StringVar(value="42")
        self.status = tk.StringVar(value="Ready")
        self.last_result: DesktopRunResult | None = None
        self.messages: queue.Queue[DesktopRunResult | Exception] = queue.Queue()

        self._build_ui()
        self.root.after(200, self._poll_messages)

    def _build_ui(self) -> None:
        self._configure_style()

        shell = ttk.Frame(self.root, padding=18)
        shell.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(shell)
        header.pack(fill=tk.X)

        title = ttk.Label(header, text="IINTS-AF Desktop", style="Title.TLabel")
        title.pack(anchor=tk.W)
        subtitle = ttk.Label(
            header,
            text=(
                "Friendly native launcher for research-only diabetes simulation, "
                "reports, and demo workflows."
            ),
            style="Subtitle.TLabel",
        )
        subtitle.pack(anchor=tk.W, pady=(4, 0))

        disclaimer = ttk.Label(
            shell,
            text="Research only. Not a medical device. Not for diagnosis, dosing, or treatment decisions.",
            style="Warning.TLabel",
            padding=(10, 8),
        )
        disclaimer.pack(fill=tk.X, pady=(14, 10))

        notebook = ttk.Notebook(shell)
        notebook.pack(fill=tk.BOTH, expand=True)

        run_tab = ttk.Frame(notebook, padding=14)
        about_tab = ttk.Frame(notebook, padding=14)
        notebook.add(run_tab, text="Run")
        notebook.add(about_tab, text="About")

        self._build_run_tab(run_tab)
        self._build_about_tab(about_tab)

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 24, "bold"))
        style.configure("Subtitle.TLabel", foreground="#334155")
        style.configure("Section.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("Warning.TLabel", background="#fff7ed", foreground="#9a3412")
        style.configure("Status.TLabel", foreground="#0f766e")

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        workflow_box = ttk.LabelFrame(parent, text="1. Choose a workflow", padding=12)
        workflow_box.pack(fill=tk.X)

        workflow_row = ttk.Frame(workflow_box)
        workflow_row.pack(fill=tk.X)
        workflow_combo = ttk.Combobox(
            workflow_row,
            textvariable=self.selected_workflow,
            values=[preset.title for preset in self.presets],
            state="readonly",
        )
        workflow_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        workflow_combo.bind("<<ComboboxSelected>>", self._on_workflow_selected)

        ttk.Label(workflow_row, text="Seed:").pack(side=tk.LEFT, padx=(12, 4))
        seed_entry = ttk.Entry(workflow_row, textvariable=self.seed, width=8)
        seed_entry.pack(side=tk.LEFT)

        description = ttk.Label(
            workflow_box,
            textvariable=self.workflow_description,
            wraplength=880,
            justify=tk.LEFT,
            foreground="#475569",
        )
        description.pack(anchor=tk.W, fill=tk.X, pady=(10, 0))

        output_box = ttk.LabelFrame(parent, text="2. Output folder", padding=12)
        output_box.pack(fill=tk.X, pady=(12, 0))
        output_row = ttk.Frame(output_box)
        output_row.pack(fill=tk.X)
        folder_entry = ttk.Entry(output_row, textvariable=self.output_dir)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_row, text="Choose...", command=self._choose_output_dir).pack(
            side=tk.RIGHT,
            padx=(8, 0),
        )

        action_box = ttk.LabelFrame(parent, text="3. Run and inspect", padding=12)
        action_box.pack(fill=tk.X, pady=(12, 0))
        button_row = ttk.Frame(action_box)
        button_row.pack(fill=tk.X)

        self.run_button = ttk.Button(
            button_row,
            text="Run Selected Workflow",
            command=self._run_demo,
        )
        self.run_button.pack(side=tk.LEFT)
        self.open_folder_button = ttk.Button(
            button_row,
            text="Open Output Folder",
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_folder_button.pack(side=tk.LEFT, padx=(8, 0))
        self.open_report_button = ttk.Button(
            button_row,
            text="Open PDF Report",
            command=self._open_report,
            state=tk.DISABLED,
        )
        self.open_report_button.pack(side=tk.LEFT, padx=(8, 0))
        self.open_csv_button = ttk.Button(
            button_row,
            text="Open Results CSV",
            command=self._open_results_csv,
            state=tk.DISABLED,
        )
        self.open_csv_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)

        ttk.Label(action_box, textvariable=self.status, style="Status.TLabel").pack(
            anchor=tk.W,
            pady=(8, 0),
        )

        log_box = ttk.LabelFrame(parent, text="Run log", padding=8)
        log_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log = tk.Text(log_box, wrap=tk.WORD, height=16)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_box, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scrollbar.set)
        self._write_log(
            "Welcome. Pick a workflow and click 'Run Selected Workflow'.\n\n"
            "The app calls the same public SDK engine as the CLI, so every run "
            "still produces normal reproducible SDK artifacts.\n"
        )

    def _build_about_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Why this is a desktop app", style="Section.TLabel").pack(anchor=tk.W)
        text = (
            "The scientific engine stays in Python because the SDK models, reports, "
            "data importers, and validation tools already live there. The desktop "
            "app is deliberately thin: it chooses workflows, starts runs, and opens "
            "the generated evidence.\n\n"
            "This first version uses Tkinter because it ships with Python on many "
            "systems and packages cleanly with PyInstaller. A future polished app "
            "could move to PySide6/Qt or Tauri while keeping the Python SDK as the "
            "single source of scientific truth."
        )
        ttk.Label(parent, text=text, wraplength=880, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 16))

        ttk.Label(parent, text="Current workflows", style="Section.TLabel").pack(anchor=tk.W)
        for preset in self.presets:
            line = f"{preset.title}: {preset.description}"
            ttk.Label(parent, text=line, wraplength=880, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

    def _selected_preset(self) -> DesktopPreset:
        title = self.selected_workflow.get()
        return self.preset_by_title.get(title, self.presets[0])

    def _format_description(self, preset: DesktopPreset) -> str:
        return (
            f"Audience: {preset.audience}\n"
            f"SDK preset: {preset.preset_name}\n"
            f"{preset.description}\n"
            f"Output: {preset.expected_output}"
        )

    def _on_workflow_selected(self, _event: object | None = None) -> None:
        self.workflow_description.set(self._format_description(self._selected_preset()))

    def _choose_output_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.output_dir.get() or str(Path.home()))
        if chosen:
            self.output_dir.set(chosen)

    def _run_demo(self) -> None:
        preset = self._selected_preset()
        try:
            seed = int(self.seed.get())
        except ValueError:
            messagebox.showerror("IINTS-AF Desktop", "Seed must be an integer.")
            return

        self._set_running_state(is_running=True)
        self.status.set(f"Running {preset.title}...")
        self._write_log(
            f"\nStarting workflow: {preset.title}\n"
            f"SDK preset: {preset.preset_name}\n"
            f"Seed: {seed}\n"
            "Generating simulation outputs, reports, and audit artifacts...\n"
        )

        thread = threading.Thread(
            target=self._run_demo_worker,
            args=(preset.key, seed),
            daemon=True,
        )
        thread.start()

    def _run_demo_worker(self, preset_key: str, seed: int) -> None:
        try:
            result = run_demo_preset(
                output_dir=self.output_dir.get(),
                desktop_preset_key=preset_key,
                seed=seed,
            )
            self.messages.put(result)
        except Exception as exc:  # pragma: no cover - GUI error path
            self.messages.put(exc)

    def _poll_messages(self) -> None:
        try:
            while True:
                item = self.messages.get_nowait()
                if isinstance(item, DesktopRunResult):
                    self._handle_success(item)
                else:
                    self._handle_error(item)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_messages)

    def _handle_success(self, result: DesktopRunResult) -> None:
        self.last_result = result
        self.status.set("Run completed")
        self._set_running_state(is_running=False)
        self.open_folder_button.configure(state=tk.NORMAL)
        if result.report_pdf and result.report_pdf.exists():
            self.open_report_button.configure(state=tk.NORMAL)
        if result.results_csv and result.results_csv.exists():
            self.open_csv_button.configure(state=tk.NORMAL)
        self._write_log("\n" + result.summary + "\n")

    def _handle_error(self, exc: Exception) -> None:
        self.status.set("Run failed")
        self._set_running_state(is_running=False)
        self._write_log(f"\nERROR: {exc}\n")
        messagebox.showerror("IINTS-AF Desktop", str(exc))

    def _set_running_state(self, *, is_running: bool) -> None:
        state = tk.DISABLED if is_running else tk.NORMAL
        self.run_button.configure(state=state)
        if is_running:
            self.open_folder_button.configure(state=tk.DISABLED)
            self.open_report_button.configure(state=tk.DISABLED)
            self.open_csv_button.configure(state=tk.DISABLED)

    def _open_output_folder(self) -> None:
        path = self.last_result.output_dir if self.last_result else Path(self.output_dir.get())
        self._open_path(path)

    def _open_report(self) -> None:
        if self.last_result and self.last_result.report_pdf:
            self._open_path(self.last_result.report_pdf)

    def _open_results_csv(self) -> None:
        if self.last_result and self.last_result.results_csv:
            self._open_path(self.last_result.results_csv)

    def _clear_log(self) -> None:
        self.log.delete("1.0", tk.END)

    @staticmethod
    def _open_path(path: Path) -> None:
        webbrowser.open(path.resolve().as_uri())

    def _write_log(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)


def main() -> None:
    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - depends on host GUI support
        raise RuntimeError("IINTS-AF Desktop requires a graphical desktop session with Tk support.") from exc
    IINTSDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
