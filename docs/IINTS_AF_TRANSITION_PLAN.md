# IINTS-AF Transition Plan (Updated for v0.1.15)

This document tracks the original transition plan and the current implementation status.
It reflects the SDK state as of **v0.1.15**.

---

## Current Status Summary (v0.1.15)

| Area | Status | Notes |
|------|--------|-------|
| Data Bridge | ✅ Complete | Universal parser, column mapper, data quality checks, dataset registry, bundled demo dataset, CLI data list/info/fetch, Nightscout import, Tidepool stub |
| Algorithm Engine | ✅ Complete | Base algorithm + metadata, battle runner, baseline comparison, standard metrics |
| Safety Layer | ✅ Complete | SafetyConfig, IndependentSupervisor, InputValidator, termination guard |
| Parallel Runs | ✅ Complete | `iints run-parallel` batch runner |
| Emulation | 🟡 Partial | Medtronic/Tandem/Omnipod emulators implemented, parameters are best‑effort |
| UX / Reporting | 🟡 Partial | PDF reports, audit trail, cockpit plots; no live dashboard |

---

## What Exists Now (Canonical Paths)

### Data & Import
- Universal parser: `src/iints/data/universal_parser.py`
- Column mapper: `src/iints/data/column_mapper.py`
- Quality checks: `src/iints/data/quality_checker.py`
- Dataset registry: `src/iints/data/datasets.json`
- Bundled demo data: `src/iints/data/demo/demo_cgm.csv`
- Nightscout import: `src/iints/data/nightscout.py`
- Tidepool stub: `src/iints/data/tidepool.py`

### Algorithm Engine & Baselines
- Base algorithm: `src/iints/api/base_algorithm.py`
- Battle runner: `src/iints/core/algorithms/battle_runner.py`
- PID controller: `src/iints/core/algorithms/pid_controller.py`
- Baseline comparison: `src/iints/analysis/baseline_comparison.py`
- Metrics: `src/iints/analysis/clinical_metrics.py`

### Safety
- Safety config: `src/iints/core/safety/config.py`
- Supervisor: `src/iints/core/supervisor.py`
- Input validator: `src/iints/core/safety/input_validator.py`
- Termination guard: `src/iints/core/simulator.py`

### Emulation
- Base emulator: `src/iints/emulation/legacy_base.py`
- Medtronic 780G: `src/iints/emulation/medtronic_780g.py`
- Tandem Control‑IQ: `src/iints/emulation/tandem_controliq.py`
- Omnipod 5: `src/iints/emulation/omnipod_5.py`

### CLI Entry Points
- Quickstart project: `iints quickstart`
- Preset run: `iints presets run`
- Parallel batch: `iints run-parallel`
- Scenario generator: `iints scenarios generate` / `iints scenarios wizard`
- Data registry: `iints data list/info/fetch`
- Nightscout: `iints import-nightscout`
- Demo import: `iints import-demo`

---

## Remaining Gaps (Post‑v0.1.15)

1. **Emulator parameter verification**
   - Cross‑check pump parameters against latest public docs and user guides.

2. **Coverage & regression stability**
   - Add golden tests for PDF output and baseline summaries.

3. **Public docs site**
   - Auto‑render notebooks + manual into a hosted docs site.

4. **Tidepool integration**
   - Expand the stub into a full auth + data pull workflow.

---

## Suggested Next Milestones

- **v0.1.16**: Emulator parameter audit + added regression tests
- **v0.1.17**: Docs site pipeline + coverage badge wired to CI
- **v0.1.18**: Tidepool auth flow + data pull example
