---
tags:
  - iints/user
  - iints/journey
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
role: new user
---
# User Journey - First 30 Minutes

> [!tip] Who this is for
> A new user who wants a clear path through the SDK without reading the whole repository first.

## Goal

Get from intention to a reproducible IINTS output bundle with the fewest confusing detours.

## Steps

- [ ] Create a clean environment and install the SDK.
- [ ] Run `iints doctor` to confirm the installation.
- [ ] Run `iints demo` for a zero-config proof that the CLI works.
- [ ] Run `iints quickstart --output-dir iints_quickstart` to create a self-contained project.
- [ ] Open the generated results and compare them with the source library.

## Commands

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp]"
iints doctor
iints demo
iints quickstart --output-dir iints_quickstart
```

## Open Next

- [[Command Cookbook - User Edition]]
- [[Troubleshooting From User Perspective]]
- [[Source Library Index]]
