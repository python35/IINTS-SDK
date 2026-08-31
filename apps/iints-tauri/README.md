# IINTS-AF Research Workbench

Native Tauri desktop application for the IINTS-AF research SDK. The workbench provides a
structured interface for running simulation protocols, reviewing outputs, creating reproducibility
packages, and opening cross-scale research tools.

The application has three explicit architectural boundaries:

- **Rust/Tauri** owns native process, URL, file-opening, and application-security boundaries.
- **Python SDK** remains the only scientific computation and report-generation engine.
- **Static frontend** presents inputs and outputs; it does not reimplement physiological equations.

Communication uses the narrow, audited `python -m iints_desktop.tauri_bridge` command surface.
Local AI output is advisory and cannot replace deterministic SDK results.

Research only. Not a medical device.

User documentation: [Research Workbench User Guide](../../docs/RESEARCH_WORKBENCH_GUIDE.md)

## Run Locally

From the repository root:

```bash
python -m pip install -U -e ".[tauri-engine]"
cd apps/iints-tauri
npm install
npm run check
npm run tauri dev
```

If Tauri cannot find the Python SDK:

```bash
export IINTS_PYTHON=/absolute/path/to/python
npm run tauri dev
```

Packaged builds provide a safer first-launch route: **Install Python engine** creates a private
`~/.iints-af/python-engine` environment through a fixed Rust-owned command. macOS discovery also
checks common Homebrew, MacPorts, and Python.org framework locations because Finder applications
do not inherit the user's full shell `PATH`.

## Build

```bash
npm run build
```

`npm run build` compiles the native executable without creating an installer. Use
`npm run bundle` when you explicitly need platform installer artifacts such as a DMG.

## Brand Assets

The native `.icns`, `.ico`, and PNG bundle icons are generated from the official
`img/iints_logo.png` artwork. Regenerate every desktop size after changing that source:

```bash
python scripts/build-brand-icons.py
```

The complete logo is used for the native application icon. Inside the workbench, the sidebar uses
the text-only `IINTS-AF` name so the application identity stays legible at every window size.

## Current Scope

- Navigate between compact Overview, Runs, Results, Reproducibility, Local AI, Research Labs,
  Evidence, and Settings workspaces.
- Save non-sensitive local defaults for output location, deterministic seed, Ollama model/host,
  and startup diagnostics.
- Select output folders, completed runs, result CSVs, SBML, COPASI, CellML, and FMU files through
  native macOS, Windows, and Linux dialogs while retaining editable paths for advanced workflows.
- Load SDK version/status.
- Check SDK and app releases independently with semantic comparison, bounded network access, and
  explicit cache/offline states; bootstrap or repair a private Python engine; open the newest
  versioned Tauri app release and documentation; and launch a fixed maintenance terminal.
- Run a readiness diagnostics check for Python, optional SDK modules, MDMP, Plotly/Matplotlib, and Ollama.
- List curated desktop workflows.
- Run a workflow as a monitored background job with progress and safe-boundary cancellation.
- Preview generated `results.csv` files.
- Show persisted run history for the chosen output folder.
- Open generated folders/reports/certificates through an allowlisted native opener.
- Create local MDMP certificates for CSV outputs.
- Create a FAIR-oriented academic package with RO-Crate metadata, checksums, a source snapshot,
  and a reproducibility audit without uploading data.
- Start/check local Ollama and list installed models.
- Ask a local Ollama model to interpret the loaded result CSV.
- Browse bundled AlphaFold molecule assets in a local interactive C-alpha viewer, reveal their
  mmCIF source files, open static renders, and generate reviewable PAE heatmaps on demand.
- Inspect local SBML safely and optionally run an isolated independent time course through libRoadRunner.
- Run research-only genomics and tissue-specific resistance stressor plots.
- Browse maturity-labelled evidence connectors for AlphaFold, Ensembl VEP/AlphaMissense, Open Targets,
  Reactome, RCSB PDB, UniProt, Human Protein Atlas, GTEx, ChEMBL, ClinPGx/PharmGKB,
  BioModels, STRING DB, ClinVar, RO-Crate, FAIR4RS, SED-ML, SBML, PubMed,
  ClinicalTrials.gov, and Zenodo.

## Optional Mechanistic Engine

SBML structural inspection is included with the normal SDK. Independent equation-model execution uses the optional libRoadRunner backend:

```bash
python -m pip install -U -e ".[mechanistic]"
```

The Mechanistic Reference Model Lab records model hash, source, licence, engine version, selected variables, model-time bounds, and generated artifacts. It does not infer unit conversions and never calibrates IINTS patient parameters automatically.

## Evidence connectors

The evidence connector panel lists official biology, pharmacology, variant, pathway, and model
provenance resources that are useful while interpreting SDK experiments. These portals are opened
outside the app in the system browser through an audited Rust HTTPS host allowlist; the app does not
embed remote research websites or use them as treatment logic.

Cards explicitly distinguish integrated, partially integrated, planned, and portal-only resources.
The workbench does not present a portal link as an implemented scientific pipeline.

## Security Notes

- Public macOS releases require a Developer ID signature and successful Apple notarization.
- Public Windows releases require an Authenticode signature with timestamp verification.
- Unsigned/ad-hoc builds are allowed only as explicitly labelled, non-release CI artifacts.
- No shell plugin.
- No arbitrary command execution from the frontend.
- No broad filesystem plugin.
- Native selectors use only Tauri's user-mediated open-dialog permission; they do not grant the
  frontend general filesystem read or write access.
- The SDK maintenance terminal is a fixed Rust-owned command, not user-provided shell text. It may
  create `~/.iints-af/python-engine` and installs only the allowlisted `iints-sdk-python35[tauri-engine]`
  package specification.
- Native file opening is limited to existing folders and safe evidence/report file types.
- Structural evidence opening is limited to bundled/report artifacts such as PNG, HTML, and mmCIF.
- Official evidence resources open only in the system browser through a Rust HTTPS host allowlist.
- No embedded remote web content.
- CSP is configured in `src-tauri/tauri.conf.json`.
- The Rust command layer only exposes a small audited allowlist.
