# Understand A Run

Do not judge a run from one graph alone. Review the time series, configuration, warnings, and provenance together.

This page explains the common output bundle. Some commands add or omit files, so treat the run folder as the source of truth.

## Common Files

| File or folder | Purpose | First question to ask |
| --- | --- | --- |
| `results.csv` | Time-series states, events, actions, and measurements | Are timestamps ordered and values finite? |
| `report.pdf` or `clinical_report.pdf` | Human-readable summary and figures | Does the report match the CSV and stated duration? |
| `run_metadata.json` | Inputs, versions, and execution settings | Which patient, scenario, algorithm, seed, and SDK version were used? |
| `run_manifest.json` | File inventory and integrity information | Are expected artifacts present and unchanged? |
| `audit/` | Safety decisions or detailed trace artifacts | When and why did the safety layer intervene? |
| `validation_report.json` | Profile or contract checks | Which checks passed, warned, or failed? |
| `sources_manifest.json` | Evidence and source references | Which assumptions or external references support the run? |

## Read The CSV First

Start with structure, not performance:

1. Confirm timestamps are monotonic and use the expected interval.
2. Check for missing, infinite, or duplicated values.
3. Identify actual glucose separately from sensor glucose.
4. Locate meal, insulin, glucagon, exercise, stress, and device-event columns that apply.
5. Locate candidate actions separately from delivered actions.
6. Locate safety interventions and termination indicators.

Column names vary by workflow. Never silently substitute one glucose or insulin column for another.

## Then Check The Experiment Settings

Record at least:

- SDK version and source revision, if available
- patient model and patient configuration
- scenario and event schedule
- candidate algorithm and configuration
- safety profile
- duration and time step
- random seed
- sensor and device settings
- early-termination status

If one of these is unknown, the run may still be useful for exploration, but it is not fully reproducible.

## Interpret Metrics Carefully

Common glucose metrics include mean glucose, coefficient of variation, time in range, time below range, and time above range. Their meaning depends on:

- the glucose column used
- sampling interval
- missing-data handling
- report period
- threshold definitions
- whether the run ended early

For an AGP-style summary, a single simulated day is still a one-day profile. It should not be described as a clinical multi-day AGP dataset.

## Review Safety Events

A safety intervention means configured logic changed or blocked candidate behavior. It does not automatically mean the final behavior was safe.

For every intervention, ask:

1. What observation triggered it?
2. Which rule and threshold were active?
3. What did the candidate propose?
4. What was delivered instead?
5. Did the run continue or terminate?

Unexpectedly many interventions can indicate an aggressive candidate, unsuitable patient/scenario pairing, or overly restrictive settings. Investigate rather than optimising the count away.

## Basic Quality Checks

Run the profile appropriate to your workflow. For example:

```bash
iints validate-run \
  --results-csv results/first_run/results.csv \
  --profile screening \
  --output-json results/first_run/validation_report.json
```

For a dataset contract:

```bash
iints data certify \
  contracts/clinical_mdmp_contract.yaml \
  results/first_run/results.csv \
  --output-json results/first_run/certification.json
```

Use `iints validation-profiles` and `iints data certify --help` when the example profile or contract does not match your project.

## Red Flags

Stop and investigate when you see:

- impossible glucose values or abrupt changes inconsistent with the configured time step
- a perfectly flat trace without a documented reason
- meals or delivered insulin missing from the output
- a report period that differs from the CSV
- early termination hidden by summary metrics
- different seeds or patient settings in an algorithm comparison
- AI commentary that contradicts deterministic metrics
- a high realism score without an independent reference or documented calibration set

## Optional AI Review

After deterministic checks pass, a local model may help summarise patterns:

```bash
iints ai report results/first_run
```

Use the AI output as a review note. Verify every number against the run artifacts. The local model must not replace parsing, metric calculation, validation, or safety logic.

## Continue

- [Complete First Workflow](GETTING_STARTED.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Certification Quickstart](MDMP_QUICKSTART.md)
- [Study Analysis](STUDY_ANALYSIS.md)
