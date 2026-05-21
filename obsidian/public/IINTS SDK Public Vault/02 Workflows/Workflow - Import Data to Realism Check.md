---
tags:
  - iints/workflow
  - iints/data
  - iints/user
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
---
# Workflow - Import Data to Realism Check

## Goal

Take a real or public CGM/pump export, standardize it, and evaluate whether it looks plausible and usable for research.

## Steps

- [ ] Identify dataset/source and access terms in [[Dataset Source Library]].
- [ ] Import data into IINTS standard schema.
- [ ] Run realism/quality checks.
- [ ] Save source citation and hash if available.
- [ ] Use [[How To Use Sources In A Report]] for wording.

## Commands

```bash
iints data list
iints data info azt1d
iints data cite azt1d
iints import-data raw.csv --output-dir data/imported_run
iints data realism-check data/imported_run/standard_cgm.csv
```

## Done Means

- [ ] Standard CSV exists.
- [ ] Scenario JSON exists if generated.
- [ ] Quality or realism report exists.
- [ ] Dataset source/citation is recorded.
