# Install On Omarchy Linux

Omarchy is an Arch Linux environment built around Hyprland and its own package, update, and development-runtime workflows. IINTS-AF therefore uses Omarchy's tools instead of asking users to manage a separate system Python or run raw `pacman -Syu` commands.

!!! warning "Research scope"
    IINTS-AF is research and education software. It is not a medical device and must not be used for diagnosis, dosing, treatment, or real-time patient care.

## What You Get

The installer can prepare:

- a fixed Python 3.11 runtime managed by Mise
- an isolated SDK engine at `~/.iints-af/python-engine`
- the `iints` command under `~/.local/bin`
- optionally, the verified Linux x64 Research Workbench AppImage
- an Omarchy application-menu entry for the workbench

Python 3.11 is the default because it is the reference version used by the desktop release workflow and is inside the SDK-supported Python 3.10-3.14 range.

## Before Installing

Update Omarchy through its supported update path:

```bash
omarchy update
```

Do not substitute a direct full-system `pacman -Syu` or `yay -Syu` update. Omarchy's update command also handles snapshots, migrations, and configuration changes described in the [official update guide](https://omarchy.org/manual/updates/).

The current desktop AppImage is x86_64-only. Check the machine:

```bash
uname -m
```

Use the CLI-only profile on another architecture.

## Recommended Install

Download and inspect the installer before executing it:

```bash
curl -fsSLO https://raw.githubusercontent.com/python35/IINTS-SDK/main/tools/install/install_omarchy.sh
less install_omarchy.sh
bash install_omarchy.sh --profile desktop
```

Preview every planned action without changing the machine:

```bash
bash install_omarchy.sh --profile desktop --dry-run
```

After installation, open **IINTS-AF Research Workbench** from the Omarchy application launcher or run:

```bash
iints-workbench
```

Verify the scientific engine:

```bash
iints --version
iints doctor --smoke-run --suggest
```

## Choose A Profile

| Goal | Command | Installed SDK extras |
| --- | --- | --- |
| CLI, simulations, reports, MDMP | `bash install_omarchy.sh` | `[full,mdmp]` |
| AI/data research | `bash install_omarchy.sh --profile research` | `[full,mdmp,research]` |
| Complete app and maintained engine | `bash install_omarchy.sh --profile desktop` | `[tauri-engine]` plus AppImage |
| Standard engine plus app | `bash install_omarchy.sh --desktop` | `[full,mdmp]` plus AppImage |

For a reproducible study, pin an exact release:

```bash
bash install_omarchy.sh --profile research --version 1.5.34
```

You may select another supported Python release with `--python-version`, but Python 3.11 is the tested desktop reference:

```bash
bash install_omarchy.sh --python-version 3.13
```

## What The Installer Changes

The script is designed to be repeatable and explicit. It:

1. requests `mise` and `curl` through `omarchy pkg add`
2. requests `fuse2` only when the AppImage is selected
3. uses `mise use --global python@3.11`
4. creates or updates `~/.iints-af/python-engine`
5. links the CLI as `~/.local/bin/iints`
6. downloads the workbench and its matching `.sha256` file when requested
7. verifies the AppImage before installation
8. installs the launcher, icon, and desktop entry below `~/.local`
9. runs `iints doctor --smoke-run --suggest`

It does not run `pacman` or `yay` directly, replace an unrelated virtual environment, install Ollama models, or download research datasets.

If an existing private engine uses a different Python minor version, it is retained as a timestamped `python-engine.previous.*` backup before replacement.

## Updating

Rerun the installer to update or repair the selected profile:

```bash
bash install_omarchy.sh --profile desktop
```

For the Python SDK alone, the normal version-aware flow also works:

```bash
iints version --refresh
iints update --dry-run
iints update
iints doctor --smoke-run
```

Pin the SDK version used by a published protocol rather than automatically moving that environment to the newest release.

## Troubleshooting

### Command Not Found

Omarchy normally exposes `~/.local/bin`. If it does not:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Add that line to the shell configuration only after confirming the directory is not already present.

### AppImage Does Not Start

Confirm architecture and integrity:

```bash
uname -m
ls -l ~/.local/opt/iints-af/IINTS-AF-Research-Workbench-linux-x64.AppImage
```

The generated launcher uses FUSE when available and falls back to AppImage extraction when `/dev/fuse` is unavailable.

### Python Engine Is Unavailable

Check Mise and the private environment:

```bash
mise current python
~/.iints-af/python-engine/bin/python --version
~/.iints-af/python-engine/bin/iints doctor --full --suggest
```

### Optional Research Integration Is Missing

Use `--profile research` or `--profile desktop`. External applications such as Ollama, COPASI, OpenCOR, and user-supplied FMUs remain separate and explicit.

## Official References

- [Omarchy manual](https://omarchy.org/manual/)
- [Omarchy development tools and Mise](https://omarchy.org/manual/development-tools/)
- [Omarchy updates](https://omarchy.org/manual/updates/)
- [Omarchy package installation guide](https://github.com/basecamp/omarchy/blob/quattro/manual/29-other-packages.md)
- [Tauri Linux prerequisites](https://v2.tauri.app/start/prerequisites/)
- [Tauri AppImage distribution](https://v2.tauri.app/distribute/appimage/)
- [Arch Linux `fuse2` package](https://archlinux.org/packages/extra/x86_64/fuse2/)

Last reviewed: 2026-08-22.
