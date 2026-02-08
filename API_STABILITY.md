# API Stability & Deprecation Policy

## Versioning
This project follows **Semantic Versioning (SemVer)**:
- **MAJOR**: breaking API changes.
- **MINOR**: backward‑compatible new features.
- **PATCH**: backward‑compatible bug fixes.

## Stability Promise
We treat the following as **public API** and stable within a MAJOR version:
- `iints.run_simulation`, `iints.run_full`
- `iints.Simulator`, `iints.PatientModel`, `iints.PatientProfile`
- `iints.generate_report`, `iints.generate_quickstart_report`, `iints.generate_demo_report`
- `iints.import_cgm_csv`, `iints.import_cgm_dataframe`, `iints.scenario_from_csv`
- CLI commands documented in `README.md` and `TECHNICAL_README.md`

## Deprecation Process
When we need to change or remove a public API:
1. **Deprecation Notice** in release notes and docstrings.
2. **Grace Period** of at least one MINOR release.
3. **Removal** only in the next MAJOR release.

## Experimental APIs
Anything not listed above is considered **internal** and may change without notice.

## Backward Compatibility
We aim to keep:
- Existing CLI flags functional
- Config files forward‑compatible when possible

If a breaking change is unavoidable, we provide a migration note.
