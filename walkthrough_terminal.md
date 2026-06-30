# IINTS-AF Desktop Terminal Walkthrough

This walkthrough documents the desktop app terminal/update flow.

## Goal

The desktop app should make SDK maintenance visible and less intimidating. Instead of asking users to copy long commands into a terminal, the app exposes two routes:

- a real operating-system terminal for transparent update installs
- an integrated app terminal for support/debug output inside the Methods tab

The app does not hide installation output. Users can see warnings, pip progress, and failures directly.

## System Terminal Update

The `Open Update Terminal` button runs the Python SDK update command in a native terminal window:

```bash
python -m pip install -U "iints-sdk-python35[full,desktop-qt,mdmp]"
```

The launcher is platform-aware:

- macOS: opens Terminal through `osascript`
- Windows: opens `cmd.exe`
- Linux: tries `x-terminal-emulator`, `gnome-terminal`, `konsole`, `xfce4-terminal`, then `xterm`

This is meant for visible maintenance. It is especially useful during demos or remote support because the user can see exactly what the installer is doing.

## Integrated App Terminal

The Methods tab contains `Developer Settings / Integrated App Terminal`.

It can:

- run the SDK update command inside the app
- run a version check
- capture stdout and stderr line-by-line
- keep the output available for copying into issues or support messages

The integrated terminal is not a full shell. It is a safe command runner for curated SDK maintenance actions.

## Safety Boundary

The update terminal only updates the Python SDK package environment. Replacing a packaged `.exe`, `.dmg`, or Linux app bundle still depends on the downloadable GitHub release assets.

This separation keeps beta builds simple and avoids pretending the app is a fully signed silent self-updater before the project has platform signing/notarization infrastructure.
