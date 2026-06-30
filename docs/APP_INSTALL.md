# Desktop App Install

IINTS-AF has two ways to use the same SDK engine:

- the normal `iints` command-line SDK
- the native desktop app through `iints-desktop`

The desktop app is meant for users who want to run demos, load results, view graphs, ask local AI questions, and open reports without memorizing many terminal commands.

It is still research software. It is not a medical device and must not be used for treatment decisions.

## Download Beta App

The current desktop beta release is available on GitHub:

[Open the desktop beta release](https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-latest)

The links below use the stable `desktop-beta-latest` release tag, so the README and docs keep pointing to the newest desktop beta after each refreshed app build.

| Platform | Download | Start |
| --- | --- | --- |
| Windows | [IINTS-AF-Desktop-Beta-windows-x64.exe](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-windows-x64.exe) | download and open the `.exe` |
| macOS | [IINTS-AF-Desktop-Beta-macos.dmg](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-macos.dmg) | open the `.dmg`, then open the app |
| Linux | [IINTS-AF-Desktop-Beta-linux-x64](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-linux-x64) | mark executable if needed, then run it |

The macOS DMG uses the small native Cocoa shell for now. That keeps the beta app opening reliably on unsigned/downloaded macOS builds while the richer Qt bundle is being hardened. The Qt app is still available through the Python install below.

These beta builds are unsigned unless release signing secrets are configured. Windows or macOS may show a security warning until the project has code-signing certificates. See [Desktop App Signing](DESKTOP_SIGNING.md).

The app also includes an update panel that links back to the latest app release and can copy/update the Python SDK package command for Python-based installs.

## Install From PyPI

This is the easiest path for most users:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop,mdmp]"
iints-desktop
```

If the Qt app cannot start, the launcher can fall back to the simpler Tkinter app:

```bash
iints-desktop-tk
```

To force the Qt app:

```bash
iints-desktop-qt
```

## Install From GitHub Main

Use this when you want the newest SDK code:

```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[full,desktop,mdmp]"
iints-desktop
```

On Windows PowerShell:

```powershell
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U -e ".[full,desktop,mdmp]"
iints-desktop
```

## What The App Can Do

- Run curated SDK demo workflows.
- Load generated `results.csv` files.
- Preview glucose graphs and metrics.
- Open PDF reports and output folders.
- Ask local Ollama/Mistral questions about a loaded run.
- View bundled AlphaFold insulin/glucagon structures.
- Generate local PAE heatmap HTML files when Plotly and internet access are available.
- Run biology evidence actions for GTEx expression, ChEMBL insulin context, ClinVar mutation stressors, and STRING pathway images.

## What The App Does Not Do

- It does not replace the SDK engine.
- It does not contain separate medical formulas.
- It does not make treatment decisions.
- It does not silently install Ollama itself.
- It does not connect to a real patient or real pump.

## Local AI Setup

Install Ollama once from the official Ollama website, then open the app and click `Start Local AI`.

The app can start the local Ollama server and prepare the selected model when the Ollama binary is available on your system.

## Troubleshooting

If the app command is missing, reinstall with the desktop extra:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop,mdmp]"
```

If the Qt app fails because PySide6 is missing, reinstall the SDK with the desktop extra. The desktop extra installs PySide6 automatically:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop,mdmp]"
```

If you want to verify the SDK first:

```bash
iints doctor --smoke-run
```

## Related Docs

- [Desktop App](DESKTOP_APP.md)
- [Quickstart](QUICKSTART.md)
- [Command Cheatsheet](CLI_CHEATSHEET.md)
- [Updating The SDK](UPDATING.md)
