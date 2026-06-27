# Desktop App Install

IINTS-AF has two ways to use the same SDK engine:

- the normal `iints` command-line SDK
- the native desktop app through `iints-desktop`

The desktop app is meant for users who want to run demos, load results, view graphs, ask local AI questions, and open reports without memorizing many terminal commands.

It is still research software. It is not a medical device and must not be used for treatment decisions.

## Download Beta App

The current desktop beta release is available on GitHub:

[Open the desktop beta release](https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-2026-06-27-2)

| Platform | Download | Start |
| --- | --- | --- |
| Windows | [IINTS-AF-Desktop-Beta-windows-x64.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-2/IINTS-AF-Desktop-Beta-windows-x64.zip) | extract the zip, then open `IINTS-AF-Desktop-Beta.exe` |
| macOS | [IINTS-AF-Desktop-Beta-macos.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-2/IINTS-AF-Desktop-Beta-macos.zip) | extract the zip, then open the `.app` bundle |
| Linux | [IINTS-AF-Desktop-Beta-linux-x64.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-2/IINTS-AF-Desktop-Beta-linux-x64.zip) | extract the zip, then run `./IINTS-AF-Desktop-Beta` |

These beta builds are unsigned. Windows or macOS may show a security warning until the project has code-signing certificates.

## Install From PyPI

This is the easiest path for most users:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"
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
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

On Windows PowerShell:

```powershell
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

## App-Focused Branch

The `main` branch contains the complete SDK. The `desktop-app` branch exists for users who mainly want the native app and app installation notes.

```bash
git clone --branch desktop-app https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

Use the app branch when:

- you mainly want the native desktop workbench
- you are testing beta app packaging
- you want app-first documentation in the repository root

Use `main` when:

- you are developing the SDK itself
- you need all research, CLI, hardware, and CI docs together
- you want the default development branch

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
python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"
```

If the Qt app fails because PySide6 is missing:

```bash
python -m pip install -U PySide6
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
