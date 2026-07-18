# Workflow Hub

Every IINTS-AF workflow should make four things clear: **input, command, output, and review rule**.

Choose the workflow that matches your question. Do not enable every feature simply because it is available.

## Simulation

**Question:** How does a candidate algorithm behave for a controlled virtual patient and scenario?

**Start:** [Complete First Workflow](GETTING_STARTED.md)

**Output:** time series, report, metadata, manifest, and optional audit/validation files.

**Review rule:** confirm the experiment settings and raw time series before interpreting summary metrics.

## Algorithm Or Scenario Study

**Question:** Does one controlled change alter outcomes across seeds, patients, or scenarios?

**Start:** [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)

**Output:** protocol, study runs, aggregate metrics, comparisons, figures, and exclusions.

**Review rule:** predefine the comparison and preserve failed runs.

## Data Certification

**Question:** Does a CSV satisfy an explicit schema and quality contract?

**Start:** [Certification Quickstart](MDMP_QUICKSTART.md)

**Output:** certification JSON, grade, validation details, optional dashboard, and integrity artifacts.

**Review rule:** certification reports checks performed; it does not establish clinical fitness or remove bias.

## Local AI Review

**Question:** Can a local model explain or challenge a completed, validated run?

**Start:** [AI Assistant](AI_ASSISTANT.md)

**Output:** review notes or a structured explanation linked to prepared evidence.

**Review rule:** verify every number and causal claim against deterministic artifacts.

## Glucose Forecast Research

**Question:** How accurately and plausibly can a model forecast future glucose at defined horizons?

**Start:** [Glucose Forecast Model](GLUCOSE_MODEL.md)

**Output:** checkpoint, resolved configuration, split manifest, horizon metrics, hypo-detection metrics, violation metrics, and model card.

**Review rule:** prevent subject leakage and report horizon-specific performance, uncertainty, and physiological violations.

## Results Management

**Question:** How can many completed runs be indexed and compared without losing provenance?

**Start:** [Study Analysis](STUDY_ANALYSIS.md)

**Output:** run index, aggregate tables, study summaries, and evidence-ready exports.

**Review rule:** keep immutable source run bundles and derive summaries from them.

## Reports And Public Evidence

**Question:** How can a completed experiment be communicated without hiding assumptions?

**Start:** [Research Evidence Bundle](EVIDENCE_BUNDLE.md)

**Output:** copied manifests, run cards, model/data context, figures, and a public file index.

**Review rule:** include limitations, intended use, sources, and failed or excluded results.

## Desktop Workflow

**Question:** Can the same SDK operations be used through a graphical workbench?

**Start:** [Desktop App](DESKTOP_APP.md)

**Output:** normal SDK run bundles. The app is a shell; it does not define separate scientific behavior.

**Review rule:** keep the output folder and command-equivalent settings available for reproducibility.

## Hardware Workflow

**Question:** How does SDK logic behave on a bench device or edge computer?

**Start:** [Hardware Hub](HARDWARE.md)

**Output:** device diagnostics, transport logs, timing measurements, comparison artifacts, or bench reports.

**Review rule:** no real patient connection or insulin delivery; document hardware, firmware, transport, and timing.

## Quick Command Map

```bash
iints demo quick --output-dir results/demo
iints run --dry-run --preset baseline_t1d
iints validation-profiles
iints data certify --help
iints ai local-check
iints run-study --help
iints evidence build --help
iints edge doctor
```

For exact arguments, use [Command Cheatsheet](CLI_CHEATSHEET.md) or the command's `--help` output.
