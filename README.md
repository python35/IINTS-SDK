# IINTS-AF Desktop App

This branch is the app-focused entry point for IINTS-AF.

IINTS-AF Desktop is a native research workbench for diabetes-technology simulation. It lets you run SDK demo workflows, load generated results, preview glucose graphs, open reports, ask local AI questions, and inspect biology evidence artifacts without needing to remember every CLI command.

The app uses the same IINTS-AF SDK engine underneath. It does not duplicate formulas, safety logic, reports, or simulation code.

IINTS-AF is not a medical device. It must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

## Download The Beta App

Current beta release:

[Desktop Beta Release: desktop-beta-2026-06-27-1](https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-2026-06-27-1)

| Platform | Download | How to start |
| --- | --- | --- |
| Windows | [IINTS-AF-Desktop-Beta-windows-x64.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-1/IINTS-AF-Desktop-Beta-windows-x64.zip) | extract the zip, then open `IINTS-AF-Desktop-Beta.exe` |
| macOS | [IINTS-AF-Desktop-Beta-macos.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-1/IINTS-AF-Desktop-Beta-macos.zip) | extract the zip, then open the `.app` bundle |
| Linux | [IINTS-AF-Desktop-Beta-linux-x64.zip](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-1/IINTS-AF-Desktop-Beta-linux-x64.zip) | extract the zip, then run `./IINTS-AF-Desktop-Beta` |

These beta builds are unsigned. Windows or macOS may show a security warning until the project has code-signing certificates.

## What The App Can Do

- Run curated digital-patient demo workflows.
- Load `results.csv` files.
- Preview glucose graphs and summary metrics.
- Open generated PDF, AGP, CSV, and output folders.
- Ask local Ollama/Mistral questions about a loaded run.
- View bundled AlphaFold insulin/glucagon structures.
- Generate PAE heatmaps.
- Run biology evidence actions for GTEx, ChEMBL, ClinVar, and STRING.

## Install From Python Instead

If you prefer installing from PyPI:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"
iints-desktop
```

If you want the app branch source code:

```bash
git clone --branch desktop-app https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

On Windows PowerShell:

```powershell
git clone --branch desktop-app https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -U -e ".[full,desktop-qt,mdmp]"
iints-desktop
```

## App Commands

```bash
iints-desktop
```

Start the best available desktop app. It prefers the PySide6/Qt app and can fall back to the simpler Tkinter app.

```bash
iints-desktop-qt
```

Force the PySide6/Qt app.

```bash
iints-desktop-tk
```

Force the lightweight Tkinter app.

## First Run

1. Start the app with `iints-desktop`.
2. Choose an output folder.
3. Pick a workflow such as `Doctor safety discussion`, `EUCYS experiment run`, or `Booth / public demo`.
4. Click `Run Selected Workflow`.
5. Open the `Results` tab to inspect the glucose graph, metrics, and CSV preview.

## Local AI Mode

The app can use local Ollama/Mistral models for research-only explanation of loaded results.

Install Ollama once, then click `Start Local AI` in the app. The app can start the local Ollama server and prepare the selected model when the Ollama binary is available.

The AI mode is only a review assistant. It does not dose insulin, diagnose, or make treatment decisions.

## Branches

- `main`: full SDK, docs, CLI, hardware, research, and app source.
- `desktop-app`: app-focused landing branch with desktop download/install instructions.

## Documentation

- Website: [iints.org](https://iints.org)
- Full docs: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)
- Desktop App Install: [docs/APP_INSTALL.md](docs/APP_INSTALL.md)
- Desktop App Details: [docs/DESKTOP_APP.md](docs/DESKTOP_APP.md)

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.
