# Desktop App Usability Audit

## Scope

This audit records an end-user review of the IINTS-AF Research Workbench. It is a software usability and integration review, not a physiological, clinical, cybersecurity-certification, or regulatory assessment.

| Item | Audited value |
| --- | --- |
| Review date | 4-5 September 2026 |
| Interactive platform | macOS |
| Installed app exercised | `0.2.12` |
| Python SDK reported by installed app | `1.5.33` |
| Source candidate reviewed after testing | `0.2.13` |
| Desktop architecture | Tauri/Rust shell with a versioned Python SDK bridge |
| Test data | locally generated research-only demo artifacts |

The installed application was used to find defects. Fixes were then applied to the source candidate and checked with automated frontend, Python, Rust, and documentation tests. Windows and Linux were not interactively operated during this review.

## Method

The review followed the path a new researcher is likely to take:

1. Inspect environment readiness and versions.
2. Select and execute a curated protocol.
3. Load and inspect the generated CSV.
4. inspect diagram and illustrated compartment views.
5. Generate an MDMP data-contract certificate.
6. Open the reproducibility-package workspace.
7. Check and use a local Ollama model.
8. Inspect Foundation AI and scientific-portfolio controls.
9. Open the research tools, including the local AlphaFold viewer.
10. Search and filter evidence connectors.
11. Validate settings and resize the application window.

The review deliberately checked whether labels described what the software actually did. A successful button click was not treated as scientific validation.

## Verification Results

| Check | Result |
| --- | --- |
| Full Python suite | `1233 passed, 2 skipped` |
| Desktop and bridge regression subset | `67 passed` |
| Full SDK mypy gate | `238 source files`, no errors |
| Strict safety-core mypy gates | passed |
| Frontend static, compartment, and digital-twin checks | passed |
| Rust unit tests | `9 passed` |
| Rust formatting and clippy with warnings denied | passed |
| MkDocs strict build | passed |
| Local packaged macOS executable `--smoke` | passed |

The host system volume had approximately 250 MiB free during the review. An initial full-suite run therefore failed inside pytest temporary-file capture with `ENOSPC`. Re-running with `TMPDIR`, caches, and pytest scratch space on the external workspace volume completed successfully with the result above. This was an environment-capacity failure, not a test assertion failure, but the desktop diagnostics should eventually surface low temporary-space capacity before long jobs begin.

## Test Matrix

| Area | Result | Evidence or limitation |
| --- | --- | --- |
| App startup | Pass | installed macOS app opened and remained responsive |
| Python bridge status | Pass | SDK and Python engine were discovered |
| Curated protocol list | Pass | cards loaded through the Python bridge |
| Booth protocol execution | Pass | local CSV, report, and audit artifacts were generated |
| Result CSV preview | Pass | metrics, graph, and bounded table rendered |
| Compartment diagram | Pass | states and fluxes loaded from the run schema |
| Illustrated digital twin | Pass with source fix | installed build used a fixed label/range; source now derives both from the loaded run |
| MDMP contract check | Pass with source clarification | certificate was generated; UI now distinguishes a contract grade from clinical validity |
| Reproducibility form | Pass with source fix | controls worked; source now clears stale prior-run status when context changes |
| Ollama connection | Pass | local service and `ministral-3:8b` were detected |
| AI response completion | Pass with source fix | `ministral-3:8b` completed against a loaded CSV; a generated dosing-adjustment suggestion exposed a guard gap, after which deterministic line filtering and ordered-list normalization were added |
| Foundation UI | Source defect fixed | bridge responses were incorrectly read through a second `data` envelope |
| Scientific Portfolio | Source defect fixed | it previously reused the Foundation view; it now has its own bounded dossier panel |
| Local AlphaFold 3D viewer | Pass | open, rotate, zoom, auto-rotate, reset, and close were exercised |
| Evidence search/filter | Pass | text search and integration-status filter updated the visible card count |
| Settings validation | Pass | non-local Ollama host was rejected |
| Compact window layout | Source defect fixed | navigation now wraps and the research-only scope remains visible |
| SBML/COPASI/CellML/FMI execution | Not end-to-end tested | no reviewed fixture models were bundled for these external engines |
| Windows native interaction | Not tested | automated build/smoke checks remain required |
| Linux native interaction | Not tested | automated build/smoke checks remain required |

## Corrections In The Source Candidate

### Privacy and status clarity

- The global header now reports the SDK and Python versions without exposing the full local executable path or account name.
- The full Python path remains available in Settings, where it is useful for diagnostics.
- Ollama diagnostics distinguish command discovery from local service/model readiness.
- Common macOS application locations are checked even when Finder supplies a limited `PATH`.

### Run and result workflow

- Result preview is shown as soon as the CSV is available; loading the richer compartment timeline continues separately.
- Starting or selecting another result clears run-derived MDMP and reproducibility state.
- The public protocol is named **Booth meal-response demo** and explicitly states that target-range excursions are part of the stress scenario.
- Compartment labels are derived from the loaded schema rather than hard-coding Hovorka.
- Illustrated playback uses the actual first and last timestamps instead of assuming a 1,440-minute run.
- Playback stops at the true end and restarts from the true beginning.

### Scientific boundaries

- The MDMP action is named **Check MDMP data contract**.
- The result labels say **Contract grade** and **Data-contract compliance score**.
- The interface states that MDMP does not establish physiology, clinical quality, or suitability for care.
- The Scientific Portfolio is a separate workspace that indexes existing evidence and reports missing inputs instead of inventing figures.
- Foundation AI controls are grouped by pretraining versus checkpoint-backed inspection.
- Local-AI review no longer relies on prompt obedience alone: unsupported quantities and treatment-adjustment suggestions are deterministically hidden, reported in metadata, and covered by regression tests.

### Layout and navigation

- Foundation AI and Scientific Portfolio use the full workspace width.
- At compact widths, navigation becomes a wrapping grid instead of a clipped horizontal strip.
- The research-only scope statement remains visible in compact layouts.
- File and folder inputs retain native selectors and remain manually editable for exact reproducibility paths.

### Regression protection

Static checks now verify:

- every referenced frontend element exists;
- every invoked Tauri command is registered and permitted;
- Foundation and Scientific Portfolio have distinct view mappings;
- the digital-twin label and time range are data-driven;
- the frontend does not incorrectly read an already-unwrapped bridge response through `result.data`.

## Remaining Improvements

### Priority 1 — release confidence

1. Add one small, licence-compatible SBML fixture, one CellML fixture, and a safe test FMU or mock so each external-engine workflow can be tested end to end in CI.
2. Add a packaged-app smoke that loads a fixture CSV, switches both compartment views, and confirms the real time range and model label.
3. Add a bounded Ollama mock server test that verifies readable headings, lists, warnings, cancellation, timeout, and malformed-response handling.
4. Perform the same keyboard, native-dialog, compact-layout, and updater walkthrough on Windows and Linux release artifacts.
5. Add generator version, SDK version, UTC generation time, source-artifact provenance, and content hashes to the Scientific Portfolio manifest.

### Priority 2 — long-running work

1. Give Foundation pretraining, evidence generation, and local-AI analysis explicit progress, cancellation, elapsed time, and durable logs.
2. Prevent simultaneous resource-heavy jobs and explain which process currently owns the workbench.
3. Preserve interrupted training state only when the training backend can verify checkpoint integrity.

### Priority 3 — result management

1. Add a persistent local result index with run ID, protocol, seed, SDK version, model, duration, artifact completeness, and tags.
2. Support comparison only after checking compatible units, time step, cohort definition, and protocol metadata.
3. Add archive/reveal/remove-from-index actions without deleting source artifacts silently.
4. Keep patient-derived datasets outside the index unless the user explicitly registers them and acknowledges privacy requirements.

### Priority 4 — accessibility and research ergonomics

1. Complete a keyboard-only and screen-reader audit on all platforms.
2. Add visible focus-state and contrast tests to CI.
3. Add jump navigation within the long Research Tools workspace.
4. Restore focus to the invoking control after closing the molecule viewer or a native dialog.
5. Expose chart values in an accessible data table for every visual result.

### Priority 5 — scientific presentation

1. Maintain a small set of versioned golden scenarios with expected qualitative behaviour and numerical envelopes.
2. Present stress scenarios as stress scenarios, not as examples of good control.
3. Show sample count and whether metrics use the full file or a preview subset beside every summary.
4. Keep external structure, variant, expression, binding, and pathway evidence separate from calibrated physiology.

## Recommended Release Gate

A desktop beta should be published only when all of the following pass for the release commit:

```text
frontend static and scientific-view checks
Python bridge and desktop integration tests
Rust format, unit tests, and clippy
strict documentation build
packaged-app startup smoke on macOS, Windows, and Linux
signed update-manifest validation
manual macOS/Windows/Linux first-run checklist for a milestone release
```

Record any skipped item in the release notes. A green build proves that the checked software paths behaved as specified; it does not prove physiological or clinical validity.

## Follow-up: automatic run handover and physiology map (2026-09-06)

This check used the local macOS Tauri package with the repository Python engine
(SDK 1.5.36, Python 3.13.9). It did not verify an updated managed Python installation
or a downloaded, signed release.

### Changes verified

- A protocol run creates a unique directory and opens its results automatically.
- Full-file metrics, the glucose figure, the data-contract check and reproducibility
  package are prepared in sequence. Optional failures retain the simulation output.
- Reopening the unchanged CSV restores the saved report links without rerunning it.
  A CSV hash prevents attaching an old preparation record to changed data.
- Run history clears on a directory change and ignores obsolete responses.
- Primary metrics and files remain visible; technical metadata is collapsible.
- Local AI remains opt-in. An actual Ollama review ran on the synthetic output;
  the numerical guard rejected four unsupported lines. This verifies the guard
  path, not the scientific quality of the model's interpretation.
- Private MDMP signing keys stay outside the result directory. A local signature
  is not an independent endorsement or clinical certification.
- The Physiology map replaces the decorative torso. It uses recorded model fields,
  units and sample times, identifies source/sink terms, and distinguishes missing
  values from zero. The Bergman endogenous-production and uptake fields now map
  correctly. A glucagon depot cannot be selected as the insulin depot.

### Numerical and software checks

The native booth meal-response run produced 37 samples over 180 minutes. At the
last sample, glucose was 274.463 mg/dL, endogenous production 0.2352 mg/dL/min,
meal appearance 2.26363 mg/dL/min and glucose disappearance 1.60665 mg/dL/min.
These are a stress-scenario output, not a claim of good glucose control.
A repeated seed-42 run matched all 84 non-latency CSV columns; the three measured
execution-latency columns are expected to differ.

The full Python suite passed 1,241 tests, with two skips. After the final bridge
changes, 75 targeted desktop tests and the edited-module mypy check passed.
Frontend checks cover missing channels, signed arrows, model-specific bindings,
recorded-time selection and stale workflow responses. The Mac app was built and
opened for native UI checks. Screenshots are included in the workbench guide.

### Still required before a release

- Windows and Linux packaged-app smoke and first-run checks.
- macOS signing/notarization and clean-machine managed-engine installation checks.
- Complete keyboard/screen-reader testing and broad narrow-window coverage.
- Live verification of optional external tools and their failure/cancellation paths.
- Independent model calibration and scientific validation; these UI and software
  checks do not establish clinical accuracy.

Final local checks also passed: Rust formatting and nine unit tests, the frontend
check suite, and the strict MkDocs build. The final Physiology map screenshot
shows the recorded plasma insulin value (28.14 mU/L at 180 minutes) separately
from glucose. A smaller native window was inspected and then restored.

The final build produced the local `.app` and it launched successfully. Updater
archive signing then failed because `TAURI_SIGNING_PRIVATE_KEY` was not available.
No signed updater or new public release was published during this review.
