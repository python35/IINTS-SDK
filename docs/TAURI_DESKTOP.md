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

## Download

The stable beta tag always points to the current platform installers:

| Platform | Download |
| --- | --- |
| Windows | [`.exe` installer](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-windows-x64-setup.exe) |
| macOS | [`.dmg`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-macos.dmg) |
| Linux | [`.AppImage`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-linux-x64.AppImage) |

The native shell delegates scientific operations to the Python SDK. On first launch, select
**Install Python engine** if Overview reports that the bridge is unavailable. The app opens a
visible terminal, creates `~/.iints-af/python-engine`, and installs the supported SDK package.
Return to the app and select **Refresh versions** when the terminal completes.

This bootstrap needs Python `3.10` through `3.14`. The app checks common macOS Python.org,
Homebrew, and MacPorts locations in addition to the application `PATH`. Advanced users can still
set `IINTS_PYTHON` explicitly. See [Desktop App Installation](APP_INSTALL.md) for manual recovery.

COPASI, OpenCOR, Ollama, and model files remain optional external research tools. They are detected explicitly rather than installed silently.

The **Settings** workspace keeps maintenance separate from experiments. It stores only local,
non-sensitive defaults, compares the native app and Python SDK against separate maintained release
channels, opens a fixed SDK install/update command in a visible terminal, and links to the newest
versioned desktop installer release. Version checks use semantic comparison, bounded network access,
and an explicit cache/offline state rather than silently assuming that an unknown version is current.
It does not store credentials or silently replace the running executable.

### macOS integrity and Gatekeeper

Every macOS beta is signed as one complete app bundle and then checked from inside the generated DMG with Apple's strict `codesign` verifier. When no Apple Developer ID is configured, CI uses an ad-hoc signature. This fixes the invalid partial-bundle state that macOS reports as **"the app is damaged"**.

Ad-hoc signing is not Apple notarization. Until a Developer ID certificate and notary credentials are configured, macOS may still show the normal **unidentified developer** warning. Use Finder's **Open** context-menu action for a beta you downloaded from this official repository; do not disable Gatekeeper system-wide.

## Workbench Components

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
2. the private app engine at `~/.iints-af/python-engine`
3. common absolute Python locations on macOS
4. `python3` / `python`
5. Windows fallback: `py -3`

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

The workbench intentionally avoids:

- shell command plugins
- broad filesystem plugins
- remote web content
- silent executable self-update logic
- arbitrary Python execution from the UI

The current Settings panel opens the signed stable release for explicit app installation. Its
Python-engine bootstrap is a fixed, visible maintenance command rather than arbitrary frontend
shell input. Hardening roadmap:

- Add signed Tauri updater after release signing is stable.
- Use a dedicated app data directory for outputs.
- Keep native file/folder selection limited to user-mediated open dialogs; do not grant broad
  frontend filesystem permissions.
- Stream run progress through Tauri events instead of waiting for a long command.
- Add per-command input validation in Rust before calling Python.
- Sign and notarize macOS builds; sign Windows builds with timestamping.
- Keep the Python bridge command list small and audited.

## Release Validation

The cross-platform beta workflow runs frontend static checks, Rust formatting, unit tests, strict Clippy checks, a native executable smoke test, installer packaging, SHA-256 generation, and strict verification of the app mounted from the finished DMG. Developer ID signing and macOS notarization run when the repository certificate secrets are configured; otherwise CI applies and verifies a complete ad-hoc bundle signature.

The release workflow is `.github/workflows/tauri-desktop-beta.yml`, and its stable release tag is `tauri-beta-latest`.

## Migration Plan

1. Keep PySide as the rich beta app while Tauri matures.
2. Use Tauri first for the core workflow: run preset, preview CSV, certify MDMP, ask local AI.
3. Stream run progress through Tauri events so long simulations feel live.
4. Move biology/AlphaFold viewers into Tauri only after file/plugin scopes are locked down.
5. Make Tauri the default bundled app once the Python scientific engine can be distributed as an audited sidecar and signing is stable.
