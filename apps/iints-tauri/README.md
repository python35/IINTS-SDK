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
python ../../tools/desktop/build_tauri_desktop_app.py
```

## Current Scope

- Load SDK version/status.
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

## Security Notes

- No shell plugin.
- No arbitrary command execution from the frontend.
- No broad filesystem plugin.
- Native file opening is limited to existing folders and safe evidence/report file types.
- Structural evidence opening is limited to bundled/report artifacts such as PNG, HTML, and mmCIF.
- No remote web content.
- CSP is configured in `src-tauri/tauri.conf.json`.
- The Rust command layer only exposes a small audited allowlist.
