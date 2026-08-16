from __future__ import annotations

import queue
import sys
import threading
from importlib import resources
from pathlib import Path
from typing import Any

from iints_desktop.engine import (
    DEFAULT_DESKTOP_PRESET_KEY,
    DesktopPreset,
    DesktopRunResult,
    get_desktop_preset,
    list_desktop_presets,
    run_demo_preset,
)


def _smoke() -> int:
    presets = list_desktop_presets()
    default = get_desktop_preset(DEFAULT_DESKTOP_PRESET_KEY)
    print(f"Cocoa desktop smoke OK: {len(presets)} workflows loaded; default={default.key}")
    return 0


def main() -> int:
    if "--smoke" in sys.argv:
        return _smoke()

    try:
        from Cocoa import (  # type: ignore[import-not-found]
            NSAlert,
            NSApplication,
            NSApplicationActivationPolicyRegular,
            NSBackingStoreBuffered,
            NSButton,
            NSClosableWindowMask,
            NSTextField,
            NSTextView,
            NSMakeRect,
            NSMiniaturizableWindowMask,
            NSModalResponseOK,
            NSImage,
            NSOpenPanel,
            NSPopUpButton,
            NSResizableWindowMask,
            NSRunningApplication,
            NSScrollView,
            NSTitledWindowMask,
            NSTimer,
            NSView,
            NSWindow,
            NSWorkspace,
            NSObject,
            NSURL,
        )
        import objc  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - macOS dependency path
        raise RuntimeError(
            "The macOS desktop app requires the bundled Cocoa runtime. "
            "Use the GitHub DMG or install the desktop-macos extra."
        ) from exc

    class IINTSCocoaDelegate(NSObject):  # type: ignore[misc,valid-type]
        window: Any
        presets: list[DesktopPreset]
        messages: queue.Queue[DesktopRunResult | Exception]
        last_result: DesktopRunResult | None

        def applicationDidFinishLaunching_(self, _notification: Any) -> None:
            self._apply_app_icon()
            self.presets = list_desktop_presets()
            self.messages = queue.Queue()
            self.last_result = None
            self._build_window()
            self.window.makeKeyAndOrderFront_(None)
            NSRunningApplication.currentApplication().activateWithOptions_(1 << 1)
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                0.2, self, "pollMessages:", None, True
            )

        def applicationShouldTerminateAfterLastWindowClosed_(self, _sender: Any) -> bool:
            return True

        @objc.python_method
        def _apply_app_icon(self) -> None:
            try:
                icon_ref = resources.files("iints_desktop").joinpath("assets").joinpath("app_icon.png")
                if icon_ref.is_file():
                    image = NSImage.alloc().initWithContentsOfFile_(str(icon_ref))
                    if image is not None:
                        NSApplication.sharedApplication().setApplicationIconImage_(image)
            except Exception:
                pass

        @objc.python_method
        def _label(self, parent: Any, text: str, frame: tuple[float, float, float, float], *, size: float = 13.0, bold: bool = False) -> Any:
            field = NSTextField.alloc().initWithFrame_(NSMakeRect(*frame))
            field.setStringValue_(text)
            field.setBezeled_(False)
            field.setDrawsBackground_(False)
            field.setEditable_(False)
            field.setSelectable_(False)
            if bold:
                from Cocoa import NSFont  # type: ignore[import-not-found]

                field.setFont_(NSFont.boldSystemFontOfSize_(size))
            else:
                from Cocoa import NSFont  # type: ignore[import-not-found]

                field.setFont_(NSFont.systemFontOfSize_(size))
            parent.addSubview_(field)
            return field

        @objc.python_method
        def _button(self, parent: Any, text: str, frame: tuple[float, float, float, float], action: str) -> Any:
            button = NSButton.alloc().initWithFrame_(NSMakeRect(*frame))
            button.setTitle_(text)
            button.setBezelStyle_(1)
            button.setTarget_(self)
            button.setAction_(action)
            parent.addSubview_(button)
            return button

        @objc.python_method
        def _build_window(self) -> None:
            style = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask
            self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, 980, 720), style, NSBackingStoreBuffered, False
            )
            self.window.center()
            self.window.setTitle_("IINTS-AF Desktop Beta")

            content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 980, 720))
            self.window.setContentView_(content)

            self._label(content, "IINTS-AF Desktop", (28, 672, 420, 30), size=24, bold=True)
            self._label(
                content,
                "Native macOS beta shell for research-only diabetes simulation workflows.",
                (30, 646, 720, 20),
                size=13,
            )
            self._label(
                content,
                "Research only. Not a medical device. Not for diagnosis, dosing, treatment decisions, or real-time patient care.",
                (30, 610, 900, 24),
                size=13,
                bold=True,
            )

            self._label(content, "Workflow", (30, 560, 120, 22), bold=True)
            self.workflow_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(150, 556, 520, 28), False)
            for preset in self.presets:
                self.workflow_popup.addItemWithTitle_(preset.title)
            default = get_desktop_preset(DEFAULT_DESKTOP_PRESET_KEY)
            for index, preset in enumerate(self.presets):
                if preset.key == default.key:
                    self.workflow_popup.selectItemAtIndex_(index)
                    break
            self.workflow_popup.setTarget_(self)
            self.workflow_popup.setAction_("workflowChanged:")
            content.addSubview_(self.workflow_popup)

            self._label(content, "Seed", (700, 560, 50, 22), bold=True)
            self.seed_field = NSTextField.alloc().initWithFrame_(NSMakeRect(750, 556, 70, 28))
            self.seed_field.setStringValue_("42")
            content.addSubview_(self.seed_field)

            self.description_label = self._label(content, "", (30, 505, 900, 44), size=12)
            self.workflowChanged_(None)

            self._label(content, "Output folder", (30, 462, 130, 22), bold=True)
            self.output_field = NSTextField.alloc().initWithFrame_(NSMakeRect(150, 458, 650, 28))
            self.output_field.setStringValue_(str(Path.home() / "IINTS-Desktop-Runs"))
            content.addSubview_(self.output_field)
            self._button(content, "Choose...", (815, 456, 110, 32), "chooseOutputFolder:")

            self.run_button = self._button(content, "Run Workflow", (30, 405, 150, 36), "runWorkflow:")
            self.open_folder_button = self._button(content, "Open Output", (195, 405, 130, 36), "openOutputFolder:")
            self.open_report_button = self._button(content, "Open Report", (340, 405, 130, 36), "openReport:")
            self.open_csv_button = self._button(content, "Open CSV", (485, 405, 110, 36), "openCSV:")
            self.open_folder_button.setEnabled_(False)
            self.open_report_button.setEnabled_(False)
            self.open_csv_button.setEnabled_(False)

            self.status_label = self._label(content, "Ready", (620, 412, 300, 20), size=13, bold=True)

            self._label(content, "Run log", (30, 365, 200, 22), bold=True)
            scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(30, 30, 900, 325))
            scroll.setHasVerticalScroller_(True)
            self.log_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 900, 325))
            self.log_view.setEditable_(False)
            self.log_view.setString_(
                "Welcome. Choose a workflow and click Run Workflow.\n\n"
                "This beta app calls the same SDK engine as the command line and writes normal reproducible output folders.\n"
            )
            scroll.setDocumentView_(self.log_view)
            content.addSubview_(scroll)

        @objc.python_method
        def selectedPreset(self) -> DesktopPreset:
            index = int(self.workflow_popup.indexOfSelectedItem())
            if index < 0 or index >= len(self.presets):
                return self.presets[0]
            return self.presets[index]

        def workflowChanged_(self, _sender: Any) -> None:
            preset = self.selectedPreset()
            self.description_label.setStringValue_(
                f"Audience: {preset.audience} | SDK preset: {preset.preset_name}\n"
                f"{preset.description} Output: {preset.expected_output}"
            )

        def chooseOutputFolder_(self, _sender: Any) -> None:
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(False)
            panel.setCanChooseDirectories_(True)
            panel.setAllowsMultipleSelection_(False)
            if panel.runModal() == NSModalResponseOK:
                url = panel.URL()
                if url is not None:
                    self.output_field.setStringValue_(url.path())

        def runWorkflow_(self, _sender: Any) -> None:
            preset = self.selectedPreset()
            try:
                seed = int(str(self.seed_field.stringValue()))
            except ValueError:
                self._show_error("Seed must be an integer.")
                return
            self._set_running(True)
            self.status_label.setStringValue_(f"Running {preset.title}...")
            self._write_log(
                f"\nStarting workflow: {preset.title}\n"
                f"SDK preset: {preset.preset_name}\n"
                f"Seed: {seed}\n"
                "Generating simulation outputs, reports, and audit artifacts...\n"
            )
            thread = threading.Thread(target=self._run_worker, args=(preset.key, seed), daemon=True)
            thread.start()

        @objc.python_method
        def _run_worker(self, preset_key: str, seed: int) -> None:
            try:
                result = run_demo_preset(
                    output_dir=str(self.output_field.stringValue()),
                    desktop_preset_key=preset_key,
                    seed=seed,
                )
                self.messages.put(result)
            except Exception as exc:  # pragma: no cover - GUI error path
                self.messages.put(exc)

        def pollMessages_(self, _timer: Any) -> None:
            try:
                while True:
                    item = self.messages.get_nowait()
                    if isinstance(item, DesktopRunResult):
                        self._handle_success(item)
                    else:
                        self._handle_error(item)
            except queue.Empty:
                return

        @objc.python_method
        def _handle_success(self, result: DesktopRunResult) -> None:
            self.last_result = result
            self._set_running(False)
            self.status_label.setStringValue_("Run completed")
            self.open_folder_button.setEnabled_(True)
            self.open_report_button.setEnabled_(bool(result.report_pdf and result.report_pdf.exists()))
            self.open_csv_button.setEnabled_(bool(result.results_csv and result.results_csv.exists()))
            self._write_log("\n" + result.summary + "\n")

        @objc.python_method
        def _handle_error(self, exc: Exception) -> None:
            self._set_running(False)
            self.status_label.setStringValue_("Run failed")
            self._write_log(f"\nERROR: {exc}\n")
            self._show_error(str(exc))

        @objc.python_method
        def _set_running(self, is_running: bool) -> None:
            self.run_button.setEnabled_(not is_running)
            if is_running:
                self.open_folder_button.setEnabled_(False)
                self.open_report_button.setEnabled_(False)
                self.open_csv_button.setEnabled_(False)

        def openOutputFolder_(self, _sender: Any) -> None:
            path = self.last_result.output_dir if self.last_result else Path(str(self.output_field.stringValue()))
            self._open_path(path)

        def openReport_(self, _sender: Any) -> None:
            if self.last_result and self.last_result.report_pdf:
                self._open_path(self.last_result.report_pdf)

        def openCSV_(self, _sender: Any) -> None:
            if self.last_result and self.last_result.results_csv:
                self._open_path(self.last_result.results_csv)

        @objc.python_method
        def _open_path(self, path: Path) -> None:
            url = NSURL.fileURLWithPath_(str(path.resolve()))
            NSWorkspace.sharedWorkspace().openURL_(url)

        @objc.python_method
        def _show_error(self, message: str) -> None:
            alert = NSAlert.alloc().init()
            alert.setMessageText_("IINTS-AF Desktop")
            alert.setInformativeText_(message)
            alert.runModal()

        @objc.python_method
        def _write_log(self, text: str) -> None:
            current = str(self.log_view.string())
            self.log_view.setString_(current + text)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = IINTSCocoaDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()
    _ = objc  # keep PyInstaller aware that PyObjC is intentionally used
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
