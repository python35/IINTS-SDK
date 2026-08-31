# Installation

IINTS-AF supports Python 3.10 through 3.14 on Windows, macOS, and Linux. Use a virtual environment so the SDK and its dependencies remain isolated from the system Python.

Check [System Requirements](SYSTEM_REQUIREMENTS.md) before selecting a research, local-AI, desktop, or large-study profile. Omarchy users should follow the dedicated [Omarchy Linux installation](OMARCHY_INSTALL.md), which uses Omarchy package tools and Mise.

## Choose An Install

| Need | Install | Recommended for |
| --- | --- | --- |
| Simulation, reports, and MDMP | `iints-sdk-python35[full,mdmp]` | most users |
| AI model training and interactive research plots | add `[research]` | research workstations and Jetson |
| Serial hardware interfaces | add `[edge]` | Raspberry Pi, Pico, UNO Q, FPGA bridges |
| Python/PySide desktop development | add `[desktop]` | desktop contributors |
| Complete desktop/research workbench | `[desktop-all]` | app users who want every supported Python-side engine |
| Latest source code | editable Git clone | contributors only |

The packaged desktop application has its own installer. See [Install The Desktop App](APP_INSTALL.md).

## Standard Installation

### macOS And Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "iints-sdk-python35[full,mdmp]"
```

### Omarchy Linux

Use the supported Omarchy installer rather than modifying its system Python:

```bash
omarchy update
curl -fsSLO https://raw.githubusercontent.com/python35/IINTS-SDK/main/tools/install/install_omarchy.sh
less install_omarchy.sh
bash install_omarchy.sh --profile standard
```

Use `--profile desktop` for the native workbench and complete Python engine. The [Omarchy guide](OMARCHY_INSTALL.md) documents profiles, integrity verification, updates, and every path changed by the script.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "iints-sdk-python35[full,mdmp]"
```

Verify the installation:

```bash
iints --version
iints doctor --smoke-run
```

## Optional Profiles

Install only what the workflow needs.

### Research And Glucose AI

```bash
python -m pip install "iints-sdk-python35[full,mdmp,research]"
```

This adds packages for model training, ONNX work, Parquet/HDF5 data, and interactive Plotly outputs. It does not download private diabetes datasets or pretrained models.

### Edge Hardware

```bash
python -m pip install "iints-sdk-python35[edge,mdmp]"
iints edge doctor
```

Continue with the [Hardware Hub](HARDWARE.md) and the guide for the selected device.

### Local Ollama AI

The Python install does not install or start Ollama automatically. The desktop app can guide the setup; CLI users should follow [AI Assistant](AI_ASSISTANT.md).

Verify the backend only after Ollama and a model are available:

```bash
iints ai local-check
```

## Install From Source

Use this route only when changing the SDK itself.

```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,full,mdmp]"
```

Run the quick contributor checks:

```bash
tools/dev/sdk_check.sh quick
```

Important folder rule:

- run repository scripts from the repository root
- run scaffolded project commands from that project directory
- do not run `pip install -e .` from your home folder or another directory without `pyproject.toml`

## Tauri Desktop Development

The native Tauri shell requires Node.js and Rust/Cargo in addition to Python.

On macOS with Homebrew:

```bash
brew install node rust
```

Then:

```bash
cd apps/iints-tauri
npm install
npm run check
cargo check --manifest-path src-tauri/Cargo.toml
```

See [Tauri Desktop Shell](TAURI_DESKTOP.md) for architecture, permissions, and packaging.

## Update An Existing Install

```bash
iints update --dry-run
iints update
iints doctor --smoke-run
```

The desktop app exposes the same update flow. Read [Update The SDK](UPDATING.md) before changing an installation used for a reproducible study.

## Remove The SDK

Preview destructive cleanup first:

```bash
iints delete --dry-run
```

Only confirm deletion after reviewing every path. Research data and run bundles may be irreplaceable; archive them separately before removing software or local state.

## Common Problems

### `iints: command not found`

Confirm the virtual environment is active and inspect:

```bash
python -m pip show iints-sdk-python35
python -m pip --version
```

Use `python -m iints.cli.cli` only for debugging source imports; normal users should use the installed `iints` command.

### Wrong Python Version

```bash
python --version
```

Use Python 3.10 through 3.14. Recreate the virtual environment after changing Python versions.

### Missing Optional Feature

Install the relevant extra in the active environment. For example:

```bash
python -m pip install --upgrade "iints-sdk-python35[research]"
```

### Desktop App Does Not Start

Do not install PySide6 separately into a packaged `.app`, `.dmg`, `.exe`, or Linux bundle. Runtime dependencies are included in the artifact. Follow [Desktop App Installation](APP_INSTALL.md) and collect the launch log if it still fails.

## Next

1. Run the [First Run](QUICKSTART.md).
2. Read [Core Concepts](CORE_CONCEPTS.md).
3. Use [Troubleshooting](TROUBLESHOOTING.md) for platform-specific failures.
