# IINTS-AF Desktop App Branch

This branch is the app-focused entry point for IINTS-AF.

Use this branch if you mainly want the native desktop workbench instead of starting with the full command-line SDK documentation.

The desktop app uses the same IINTS-AF SDK engine underneath. It does not duplicate formulas, safety logic, reports, or simulation code.

IINTS-AF is research and education software. It is not a medical device and must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

## What The App Is

The IINTS-AF Desktop App is a native research workbench for diabetes-technology simulation.

It helps you:

- run curated digital-patient demo workflows
- open generated SDK results folders
- load `results.csv` files
- preview glucose graphs and metrics
- open generated PDF/AGP/report artifacts
- ask local Ollama/Mistral questions about a loaded run
- inspect bundled structural-biology context such as insulin/glucagon AlphaFold assets

## Install From This Branch

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

## Install From PyPI

If you do not need the branch source code, install the app directly:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"
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

1. Start the app:

```bash
iints-desktop
```

2. Choose an output folder.
3. Pick a workflow such as `Doctor safety discussion`, `EUCYS experiment run`, or `Booth / public demo`.
4. Click `Run Selected Workflow`.
5. Open the `Results` tab to inspect the glucose graph, metrics, and CSV preview.

## Local AI Mode

The app can use local Ollama/Mistral models for research-only explanation of loaded results.

Install Ollama once, then in the app click `Start Local AI`.

The app can start the local Ollama server and prepare the selected model when the Ollama binary is available.

The AI mode is only a review assistant. It does not dose insulin, diagnose, or make treatment decisions.

## What This Branch Contains

This branch still includes the SDK source code because the app calls the SDK directly.

Important paths:

| Path | Purpose |
| --- | --- |
| `src/iints_desktop/` | native desktop app source |
| `src/iints/` | SDK engine used by the app |
| `docs/APP_INSTALL.md` | app-first installation guide |
| `docs/DESKTOP_APP.md` | desktop app architecture and features |
| `tools/desktop/` | app packaging helpers |
| `.github/workflows/desktop-beta.yml` | beta desktop build workflow |

## Full SDK Branch

For full SDK development, CLI workflows, hardware docs, research tooling, and release maintenance, use `main`:

```bash
git clone https://github.com/python35/IINTS-SDK.git
```

## Documentation

- [Desktop App Install](docs/APP_INSTALL.md)
- [Desktop App Details](docs/DESKTOP_APP.md)
- [Full Documentation Site](https://python35.github.io/IINTS-SDK/)

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.

Built as research software by Rune Bobbaers.
