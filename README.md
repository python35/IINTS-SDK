# IINTS-AF SDK

[![EUCYS 2026](https://img.shields.io/badge/EUCYS-2026%20Selected-gold?style=flat)](https://www.uni-kiel.de/en/eucys2026)
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Docs](https://img.shields.io/badge/docs-IINTS--AF-0a66c2?style=flat)](https://python35.github.io/IINTS-SDK/)

IINTS-AF is an open-source research SDK for diabetes-technology simulation. It provides virtual-patient scenarios, algorithm experiments, deterministic safety checks, data-assurance tools, glucose-forecasting research, and reproducible reports.

Its purpose is to make diabetes-technology behavior easier to inspect, test, and discuss before anything is connected to a real person.

IINTS-AF is not a medical device. It must not be used for diagnosis, insulin dosing, treatment decisions, or real-time patient care.

## Start

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge]"
iints demo
iints menu
iints version --refresh
```

## Desktop App

The optional Rust/Tauri research workbench offers guided runs, result inspection, local AI review, and data-assurance tools.

| Platform | Latest beta |
| --- | --- |
| Windows | [`.exe` installer](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-windows-x64-setup.exe) |
| macOS | [`.dmg`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-macos.dmg) |
| Linux | [`.AppImage`](https://github.com/python35/IINTS-SDK/releases/download/tauri-beta-latest/IINTS-AF-Research-Workbench-linux-x64.AppImage) |

## Links

- Website: [iints.org](https://iints.org)
- Documentation: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)
- Source: [github.com/python35/IINTS-SDK](https://github.com/python35/IINTS-SDK)
- Desktop guide: [Tauri research workbench](https://python35.github.io/IINTS-SDK/TAURI_DESKTOP/)

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.
