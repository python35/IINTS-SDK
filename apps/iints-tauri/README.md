# IINTS-AF Tauri Desktop Prototype

Experimental next-generation desktop shell for IINTS-AF.

This app is intentionally small:

- Tauri/Rust owns the native desktop boundary.
- The frontend is static HTML/CSS/JS.
- The Python SDK remains the scientific engine.
- Communication happens through `python -m iints_desktop.tauri_bridge`.

Research only. Not a medical device.

## Run Locally

From the repository root:

```bash
python -m pip install -U -e ".[full,mdmp,research]"
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

## Build

```bash
npm run build
```

`npm run build` compiles the native executable without creating an installer. Use
`npm run bundle` when you explicitly need platform installer artifacts such as a DMG.

## Current Scope

- Load SDK version/status.
- Show SDK/app update information, copy the fixed SDK update command, open app downloads/docs,
  and launch a fixed update terminal.
- Run a readiness diagnostics check for Python, optional SDK modules, MDMP, Plotly/Matplotlib, and Ollama.
- List curated desktop workflows.
- Run a workflow through the normal Python SDK engine.
- Preview generated `results.csv` files.
- Show persisted run history for the chosen output folder.
- Open generated folders/reports/certificates through an allowlisted native opener.
- Create local MDMP certificates for CSV outputs.
- Start/check local Ollama and list installed models.
- Ask a local Ollama model to interpret the loaded result CSV.
- Browse bundled AlphaFold molecule assets and open mmCIF/PAE evidence files.
- Run research-only genomics and tissue-specific resistance stressor plots.
- Browse official evidence connectors for AlphaFold, Ensembl VEP/AlphaMissense, Open Targets,
  Reactome, RCSB PDB, UniProt, Human Protein Atlas, GTEx, ChEMBL, ClinPGx/PharmGKB,
  BioModels, STRING DB, and ClinVar.

## Evidence connectors

The evidence connector panel lists official biology, pharmacology, variant, pathway, and model
provenance resources that are useful while interpreting SDK experiments. These portals are opened
outside the app in the system browser through an audited Rust HTTPS host allowlist; the app does not
embed remote research websites or use them as treatment logic.

## Security Notes

- No shell plugin.
- No arbitrary command execution from the frontend.
- No broad filesystem plugin.
- The SDK update terminal is a fixed Rust-owned command, not user-provided shell text.
- Native file opening is limited to existing folders and safe evidence/report file types.
- Structural evidence opening is limited to bundled/report artifacts such as PNG, HTML, and mmCIF.
- Official evidence resources open only in the system browser through a Rust HTTPS host allowlist.
- No embedded remote web content.
- CSP is configured in `src-tauri/tauri.conf.json`.
- The Rust command layer only exposes a small audited allowlist.
