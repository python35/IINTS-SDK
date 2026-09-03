# IINTS-AF Tauri Desktop Shell

This is the maintained desktop architecture for IINTS-AF.

The goal is **not** to rewrite the scientific SDK. The Python SDK remains the single source of truth for simulation, formulas, reports, MDMP certification, and run-quality review. Tauri adds a smaller, more secure native shell around that engine.

Research only. Not a medical device. Not for diagnosis, dosing, treatment decisions, or real-time patient care.

## Why Tauri + Rust + Python?

The legacy PySide interface placed UI, process, update, and security logic in one large Python process. The maintained Tauri workbench uses a clearer split:

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

A Developer ID certificate signs and notarizes the macOS DMG when the required secrets are configured (see [Desktop Signing](DESKTOP_SIGNING.md)). Without them, CI falls back to an ad-hoc signature and skips notarization for every build, including public releases through the stable beta tag -- Gatekeeper will warn about, or require an explicit override to run, a build signed this way.

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

- Use a dedicated app data directory for outputs.
- Keep native file/folder selection limited to user-mediated open dialogs; do not grant broad
  frontend filesystem permissions.
- Keep background-job progress, cancellation, and error state covered by bridge-contract tests.
- Add per-command input validation in Rust before calling Python.
- Sign with real certificates once available (see docs/DESKTOP_SIGNING.md); until then, public installers ship ad-hoc-signed/unsigned/unnotarized rather than failing the release.
- Keep the Python bridge command list small and audited.

## Release Validation

The cross-platform beta workflow runs frontend dependency audit/static checks, Rust dependency audit, formatting, unit tests, strict Clippy checks, a native executable smoke test, installer packaging, SHA-256 generation, and platform signature verification. Public uploads fall back to an ad-hoc macOS signature, an unsigned Windows installer, and skipped notarization when the corresponding certificates are absent, rather than failing the release (see [Desktop Signing](DESKTOP_SIGNING.md)).

The release workflow is `.github/workflows/tauri-desktop-beta.yml`, and its stable release tag is `tauri-beta-latest`.

## Legacy Interface

The PySide source remains available for compatibility and historical reproducibility, but it is no longer a separately published desktop product. New UI, packaging, signing, update, accessibility, and security work belongs in the Rust/Tauri workbench. The CLI remains the preferred route for batch studies and automation.
