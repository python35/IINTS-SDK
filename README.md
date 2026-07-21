# IINTS-AF SDK

[![EUCYS 2026](https://img.shields.io/badge/EUCYS-2026%20Selected-gold?style=flat)](https://www.uni-kiel.de/en/eucys2026)
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat)](https://python35.github.io/IINTS-SDK/)

IINTS-AF is an open-source research SDK for diabetes technology simulation.

The SDK lets researchers, students, and developers test diabetes-algorithm ideas in a virtual environment. It can simulate digital patients, meals, insulin delivery, CGM/sensor behavior, safety checks, glucose-prediction experiments, and generated reports.

The purpose of IINTS-AF is to make algorithm behavior easier to inspect and discuss before anything is connected to a real person.

IINTS-AF is not a medical device. It must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

## Links

- Website: [iints.org](https://iints.org)
- Documentation: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)
- Desktop app downloads: [latest beta](https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-latest)

## Desktop App

IINTS-AF also has a native Qt desktop app for running demos, reviewing results, certifying data, and using the research workbench without memorising terminal commands. The packaged Windows, macOS, and Linux betas include their Python runtime and supported Python-side research engines.

Current beta downloads:

| Platform | Download |
| --- | --- |
| Windows | [`.exe`](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-windows-x64.exe) |
| macOS | [`.dmg`](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-macos.dmg) |
| Linux | [executable](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-latest/IINTS-AF-Desktop-Beta-linux-x64) |

Python install, including the PySide6 desktop runtime:

```bash
python -m pip install -U "iints-sdk-python35[desktop-all]"
iints-desktop
```

Experimental next-generation shell: a Tauri + Rust desktop prototype lives in `apps/iints-tauri`. It keeps the Python SDK as the scientific engine while moving the native app boundary into Rust. See `docs/TAURI_DESKTOP.md`.

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.
