# IINTS-AF Tauri Desktop Shell

This is the next-generation desktop architecture for IINTS-AF.

The goal is **not** to rewrite the scientific SDK. The Python SDK remains the single source of truth for simulation, formulas, reports, MDMP certification, and run-quality review. Tauri adds a smaller, more secure native shell around that engine.

Research only. Not a medical device. Not for diagnosis, dosing, treatment decisions, or real-time patient care.

## Why Tauri + Rust + Python?

The current PySide app is useful, but it places a lot of UI, process, update, and security logic in one large Python process. Tauri gives a cleaner split:

| Layer | Responsibility |
| --- | --- |
| Frontend | Workbench UI, protocol selection, run history, results preview, local AI panel |
| Rust/Tauri | Native app boundary, async command allowlist, packaging, signing/updating later |
| Python SDK | Diabetes simulation, reports, local AI review, MDMP, scientific logic |

This keeps the medical/research math in Python where the SDK already lives, while moving desktop authority and packaging into Rust.

## Current Scaffold

Files:

- `apps/iints-tauri/frontend/index.html`
- `apps/iints-tauri/frontend/styles.css`
- `apps/iints-tauri/frontend/main.js`
- `apps/iints-tauri/src-tauri/src/main.rs`
- `apps/iints-tauri/src-tauri/tauri.conf.json`
- `apps/iints-tauri/src-tauri/Cargo.toml`
- `src/iints_desktop/tauri_bridge.py`

The bridge supports these fixed commands:

- `status`
- `workflows`
- `run`
- `preview`
- `history`
- `mdmp-certify`
- `ai-check`
- `ai-models`
- `ai-start`
- `ai-ask`

The Rust shell does **not** expose arbitrary shell execution. It calls:

```bash
python -m iints_desktop.tauri_bridge <fixed-command>
```

The Python executable is resolved in this order:

1. `IINTS_PYTHON`
2. `python3` / `python`
3. Windows fallback: `py -3`

## Development

Install the Python SDK in your development environment:

```bash
python -m pip install -U -e ".[full,mdmp,research]"
```

Install Tauri prerequisites for your OS, then from the repo:

```bash
cd apps/iints-tauri
npm install
npm run check
npm run tauri dev
```

If the app cannot find the SDK:

```bash
export IINTS_PYTHON=/absolute/path/to/python
npm run tauri dev
```

## Build

```bash
cd apps/iints-tauri
npm install
npm run build
```

`npm run build` compiles the native executable without packaging an installer. This is the
recommended development and CI check because it works without Finder/DMG tooling. To create
platform installer artifacts, run:

```bash
npm run bundle
```

Generated binaries and bundles are created under `apps/iints-tauri/src-tauri/target/`.

## Security Direction

This scaffold intentionally avoids:

- shell command plugins
- broad filesystem plugins
- remote web content
- automatic self-update logic
- arbitrary Python execution from the UI

Future hardening checklist:

- Add signed Tauri updater after release signing is stable.
- Use a dedicated app data directory for outputs.
- Add a file-picker plugin with strict scopes instead of free path text fields.
- Stream run progress through Tauri events instead of waiting for a long command.
- Add per-command input validation in Rust before calling Python.
- Sign and notarize macOS builds; sign Windows builds with timestamping.
- Keep the Python bridge command list small and audited.

## Migration Plan

1. Keep PySide as the rich beta app while Tauri matures.
2. Use Tauri first for the core workflow: run preset, preview CSV, certify MDMP, ask local AI.
3. Stream run progress through Tauri events so long simulations feel live.
4. Move biology/AlphaFold viewers into Tauri only after file/plugin scopes are locked down.
5. Make Tauri the default downloadable app once CI builds and signing are stable.
