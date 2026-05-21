---
tags:
  - iints/user
  - iints/journey
cssclasses:
  - iints-dashboard
status: active
updated: 2026-05-21
role: data user
---
# User Journey - Data Quality Researcher

> [!tip] Who this is for
> A data user who wants a clear path through the SDK without reading the whole repository first.

## Goal

Get from intention to a reproducible IINTS output bundle with the fewest confusing detours.

## Steps

- [ ] List official datasets and understand access/licensing.
- [ ] Import a data file into the IINTS schema.
- [ ] Run realism and MDMP checks.
- [ ] Export source/citation information with the study bundle.
- [ ] Use results only as research evidence, not treatment proof.

## Commands

```bash
iints data list
iints data info ohio_t1dm
iints import-data raw.csv --output-dir data/imported_run
iints data realism-check data/imported_run/standard_cgm.csv
```

## Open Next

- [[Dataset Source Library]]
- [[Realism Reference Envelopes]]
- [[Data Quality and MDMP]]
