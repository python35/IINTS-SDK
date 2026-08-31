# Desktop App Installation

The IINTS-AF Research Workbench is the recommended desktop interface for the Python research SDK. It uses a native Rust/Tauri shell and keeps simulation, validation, reporting, and research calculations inside the Python SDK.

!!! warning "Research scope"
    IINTS-AF is research and education software. It is not a medical device and must not be used for diagnosis, dosing, treatment decisions, or real-time patient care.

## Download The Current Beta

The stable links below always point to the newest tested Tauri beta:

| Platform | Installer | Start |
| --- | --- | --- |
| Windows | [Download `.exe`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-windows-x64-setup.exe) | run the setup file, then open **IINTS-AF Research Workbench** |
| macOS | [Download `.dmg`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-macos.dmg) | open the DMG, drag the app to Applications, then open it |
| Linux | [Download `.AppImage`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-linux-x64.AppImage) | mark it executable, then run it |

Matching `.sha256` files are published beside each installer. See [Desktop App Signing](DESKTOP_SIGNING.md) for signature and notarization details.

## First Launch

The native app needs the IINTS-AF Python engine for scientific calculations. The app now handles this setup through a visible, fixed maintenance command:

1. Open the workbench.
2. If **Python bridge unavailable** appears, select **Install Python engine** on Overview, or open **Settings** and select **Install or update Python SDK**.
3. Keep the terminal open while the installation completes.
4. Return to the app and select **Refresh versions**.
5. Run **Run diagnostics** before the first experiment.

The app creates a private environment at `~/.iints-af/python-engine`. It does not modify an unrelated project virtual environment. On later runs, the same action updates or repairs that private engine.

The bootstrap requires Python `3.10` through `3.14`. On macOS, the app checks common Homebrew, MacPorts, and Python.org framework locations before the limited Finder application `PATH`. Advanced users can override discovery with `IINTS_PYTHON`.

### Manual Fallback

If Python is not installed, install a current release from [python.org](https://www.python.org/downloads/), reopen the app, and use **Install or update Python SDK** again.

Advanced users may prepare the engine manually:

=== "macOS or Linux"

    ```bash
    python3 -m venv "$HOME/.iints-af/python-engine"
    "$HOME/.iints-af/python-engine/bin/python" -m pip install --upgrade pip "iints-sdk-python35[tauri-engine]"
    ```

=== "Windows PowerShell"

    ```powershell
    py -3 -m venv "$HOME\.iints-af\python-engine"
    & "$HOME\.iints-af\python-engine\Scripts\python.exe" -m pip install --upgrade pip "iints-sdk-python35[tauri-engine]"
    ```

## macOS Security Messages

The CI pipeline verifies the complete app bundle inside the finished DMG. Public downloads still require Apple Developer ID signing and notarization to avoid all Gatekeeper warnings.

The stable beta channel publishes only Developer ID signed and notarized macOS installers. If macOS reports that an official download is damaged, stop, download the newest DMG again, and verify its SHA-256 value; do not bypass a failed integrity or signature check and do not disable Gatekeeper system-wide.

## Linux AppImage

If the file is not executable:

```bash
chmod +x IINTS-AF-Research-Workbench-linux-x64.AppImage
./IINTS-AF-Research-Workbench-linux-x64.AppImage
```

The private Python engine still needs a system Python with `venv` support. On Debian or Ubuntu, a missing `venv` module is normally supplied by the distribution's `python3-venv` package.

### Omarchy Linux

Omarchy users can install the AppImage, verified checksum, private Python engine, launcher, icon, and application-menu entry in one flow:

```bash
omarchy update
curl -fsSLO https://raw.githubusercontent.com/python35/IINTS-SDK/main/tools/install/install_omarchy.sh
less install_omarchy.sh
bash install_omarchy.sh --profile desktop
```

The script uses `omarchy pkg add` and Mise rather than direct `pacman` commands. See [Install On Omarchy Linux](OMARCHY_INSTALL.md) for the dry run, exact changes, profile selection, and troubleshooting.

## Optional External Tools

The SDK package installs supported Python libraries. Large or independently licensed external applications remain explicit:

- Ollama and local model files
- COPASI
- OpenCOR
- trusted FMUs or external model files

The workbench detects these tools and explains which feature needs them. It does not silently download executables, model weights, or native-code FMUs.

## Python And Classic Qt App

Researchers who prefer a Python-managed desktop environment can still install the classic Qt interface:

```bash
python -m pip install --upgrade "iints-sdk-python35[desktop-all]"
iints-desktop
```

The classic Qt source remains available for compatibility testing, but it is no longer published as a separate desktop product. New users should install the Rust/Tauri workbench above.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Python bridge unavailable | select **Install Python engine**, wait for the terminal, then select **Refresh versions** |
| Python 3.10-3.14 was not found | install Python from python.org and repeat the maintenance action |
| `No module named iints_desktop` | update or repair the private engine from Settings; the selected Python does not contain the SDK bridge |
| protocol list is empty | refresh versions, run diagnostics, then refresh protocols |
| optional research engine unavailable | install only the external engine required by that lab |
| app update available | close the app and install the newest file from the stable beta release |

For the complete illustrated workflow, continue with the [Research Workbench User Guide](RESEARCH_WORKBENCH_GUIDE.md).
