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
- List curated desktop workflows.
- Run a workflow through the normal Python SDK engine.
- Preview generated `results.csv` files.
- Check local Ollama model readiness.

## Security Notes

- No shell plugin.
- No arbitrary command execution from the frontend.
- No broad filesystem plugin.
- No remote web content.
- CSP is configured in `src-tauri/tauri.conf.json`.
- The Rust command layer only exposes a small audited allowlist.
