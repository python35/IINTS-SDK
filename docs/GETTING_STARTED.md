# Complete First Workflow

This tutorial moves from a bundled demo to a controlled project with explicit patient, scenario, algorithm, and data-quality artifacts.

Complete [First Run](QUICKSTART.md) before starting.

## What You Will Produce

```text
iints_quickstart/
├── algorithms/
├── scenarios/
├── patients/
├── contracts/
├── data/
├── audit/
└── results/
```

The project keeps inputs separate from generated evidence. Do not edit a completed run folder to change an experiment; change the source configuration and create a new run.

## 1. Create The Project

From the folder where you keep research projects:

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

Review these files before running anything:

- `algorithms/example_algorithm.py`
- `patients/stable_patient.yaml`
- `scenarios/clinic_safe_baseline.json`
- `contracts/clinical_mdmp_contract.yaml`

## 2. Validate The Plan

Use a dry run first:

```bash
iints run \
  --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --seed 42 \
  --dry-run
```

Check the patient, scenario, duration, time step, seed, output location, and algorithm source printed by the CLI.

## 3. Execute The Simulation

```bash
iints run \
  --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --seed 42 \
  --output-dir results/baseline_seed_42
```

Do not close the terminal until the run completes or records an explicit termination reason.

## 4. Inspect Before Scoring

Follow [Understand A Run](RUN_OUTPUTS.md). At minimum:

1. verify the CSV timestamps and duration
2. check meals, insulin, and safety events
3. review metadata and the seed
4. compare report metrics with the CSV
5. record warnings or early termination

## 5. Validate The Run

List available validation profiles:

```bash
iints validation-profiles
```

Then validate the run with the profile appropriate to your experiment. For the scaffolded baseline:

```bash
iints validate-run \
  --results-csv results/baseline_seed_42/results.csv \
  --profile research_default \
  --output-json results/baseline_seed_42/validation_report.json
```

Validation is evidence about configured checks. It is not a clinical approval.

## 6. Certify The Output Data

```bash
iints data certify \
  contracts/clinical_mdmp_contract.yaml \
  results/baseline_seed_42/results.csv \
  --output-json results/baseline_seed_42/certification.json
```

Optional visual summary:

```bash
iints data certify-visualizer \
  results/baseline_seed_42/certification.json \
  --output-html results/baseline_seed_42/certification_dashboard.html
```

Read [Certification Quickstart](MDMP_QUICKSTART.md) to understand grades, contracts, and limitations.

## 7. Add Optional AI Review

Only after deterministic validation:

```bash
iints ai report results/baseline_seed_42
```

This requires a configured local AI backend. AI output is commentary, not an authoritative metric or dosing decision. See [AI Assistant](AI_ASSISTANT.md).

## 8. Make A Controlled Comparison

Change one intended factor at a time. For example, use a different algorithm while keeping patient, scenario, duration, time step, and seed fixed.

Store each run in a separate folder. Never overwrite the baseline.

For multi-run studies, continue with [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) rather than scripting ad hoc comparisons.

## Completion Checklist

- [ ] The environment and SDK version are known.
- [ ] Patient, scenario, algorithm, duration, time step, and seed are explicit.
- [ ] The run completed or recorded why it stopped.
- [ ] CSV, report, metadata, and manifest agree.
- [ ] Validation and certification artifacts are preserved.
- [ ] AI output, if used, was checked against deterministic evidence.
- [ ] Limitations and failed runs are documented.

## Continue By Goal

| Goal | Next page |
| --- | --- |
| compare algorithms or patient groups | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) |
| aggregate completed runs | [Study Analysis](STUDY_ANALYSIS.md) |
| work with imported diabetes data | [Certification Guide](MDMP_FULL_GUIDE.md) |
| train a glucose predictor | [Glucose Forecast Model](GLUCOSE_MODEL.md) |
| use Raspberry Pi, Jetson, UNO Q, Pico, or FPGA | [Hardware Hub](HARDWARE.md) |
