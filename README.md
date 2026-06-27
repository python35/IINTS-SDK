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
- Desktop app downloads: [latest beta](https://github.com/python35/IINTS-SDK/releases/tag/desktop-beta-2026-06-27-4)

## Desktop App

IINTS-AF also has a native desktop app for running demos, opening results, asking local AI questions, and viewing biology evidence without using many terminal commands.

Current beta downloads:

| Platform | Download |
| --- | --- |
| Windows | [`.exe`](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-5/IINTS-AF-Desktop-Beta-windows-x64.exe) |
| macOS | [`.dmg`](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-5/IINTS-AF-Desktop-Beta-macos.dmg) |
| Linux | [executable](https://github.com/python35/IINTS-SDK/releases/download/desktop-beta-2026-06-27-5/IINTS-AF-Desktop-Beta-linux-x64) |

### 🛡️ Waarom zie ik een beveiligingswaarschuwing? (Security Warnings)
Omdat de IINTS-AF Desktop App gratis en open-source academische software is, is deze niet cryptografisch ondertekend met dure commerciële ontwikkelaarscertificaten van Apple of Microsoft. Hierdoor geven besturingssystemen uit voorzorg een standaard waarschuwing (Gatekeeper op Mac, SmartScreen op Windows). **Dit is volkomen normaal voor open-source tools.**

* **macOS ("Apple cannot check it for malicious software")**:
  Sinds recente macOS-updates blokkeert Apple soms de "Rechtermuisknop -> Open" optie. Je hebt twee simpele oplossingen:
  1. **Makkelijkst**: Ga naar je Mac's *Systeeminstellingen* -> *Privacy en beveiliging* (Privacy & Security). Scroll naar beneden, en je ziet een melding over IINTS-AF met een knop **"Open Anyway"** (Toch openen).
  2. **Via Terminal**: Open de "Terminal" app en wis de quarantaine-vlag met dit commando: 
     `xattr -cr /Applications/IINTS-AF-Desktop-Beta.app` (pas het pad aan als hij ergens anders staat).
* **Windows ("Windows protected your PC")**:
  Klik op **More info** (Meer informatie) en vervolgens op **Run anyway** (Toch uitvoeren).

The app is strictly research-only and runs entirely locally using the same SDK engine as the command-line tools. No data is sent externally.

## License

Apache-2.0 licensed, with legacy MIT notices where applicable.
