# Learning Path

This learning path is the recommended route for someone who did not build IINTS-AF and wants to understand it independently.

You do not need diabetes-device experience to begin. Complete the modules in order; each module has a practical checkpoint before you continue.

## Module 1: Understand The Scope

**Goal:** know what IINTS-AF is, and what it is not.

Read:

1. [Plain-Language Overview](PLAIN_LANGUAGE_GUIDE.md)
2. [Project Boundaries](PROJECT_BOUNDARIES.md)
3. [Core Concepts](CORE_CONCEPTS.md)

Checkpoint:

- You can explain the difference between a virtual patient and a real patient.
- You know that candidate algorithms are experimental.
- You know that local AI may explain results but does not calculate authoritative physiology or dosing.

## Module 2: Install And Verify

**Goal:** create an isolated Python environment and verify the SDK.

Read [Installation](INSTALLATION.md), then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "iints-sdk-python35[full,mdmp]"
iints doctor --smoke-run
```

Checkpoint:

- `iints --version` prints a version.
- `iints doctor --smoke-run` completes without a blocking error.
- You know which virtual environment contains the SDK.

## Module 3: Run One Experiment

**Goal:** create a deterministic run bundle.

Follow [First Run](QUICKSTART.md):

```bash
iints demo quick --output-dir results/first_run
```

Checkpoint:

- `results/first_run/results.csv` exists.
- A report and manifest are present in the run folder.
- Repeating the same command with the same seed gives reproducible results.

## Module 4: Read The Evidence

**Goal:** understand what the run produced before changing an algorithm.

Read [Understand A Run](RUN_OUTPUTS.md). Inspect:

1. the time-series CSV
2. the report
3. the run metadata
4. the manifest and audit information

Checkpoint:

- You can identify glucose, carbohydrate, insulin, and safety-event columns.
- You can distinguish a plotted trajectory from proof of clinical realism.
- You can find the seed, SDK version, and run settings.

## Module 5: Build A Complete Workflow

**Goal:** move from a demo to an explicit project with controlled inputs.

Follow [Complete First Workflow](GETTING_STARTED.md). You will:

1. scaffold a project
2. select an algorithm, patient, and scenario
3. run a simulation
4. validate the output
5. review the evidence bundle

Checkpoint:

- The patient, scenario, algorithm, duration, time step, and seed are recorded.
- The output can be traced back to those inputs.
- Any warnings are documented rather than hidden.

## Module 6: Choose A Specialisation

Continue only with the route you need:

| Route | Start with | Main question |
| --- | --- | --- |
| Simulation studies | [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) | How do I compare algorithms reproducibly? |
| Data quality | [Certification Quickstart](MDMP_QUICKSTART.md) | Is this dataset sufficiently documented and valid for the intended analysis? |
| Local AI | [AI Assistant](AI_ASSISTANT.md) | How can a local model explain already-computed evidence safely? |
| Glucose forecasting | [Glucose Forecast Model](GLUCOSE_MODEL.md) | How do I train and evaluate a research predictor without data leakage? |
| Physiology | [Physiology Reference](PHYSIOLOGY_REFERENCE.md) | Which equations, parameters, assumptions, and limitations are implemented? |
| Hardware | [Hardware Hub](HARDWARE.md) | How do I run a bench-only hardware experiment? |
| SDK development | [Developer Portal](DEVELOPER_PORTAL.md) | How do I change the code without breaking contracts or reproducibility? |

## Research Habits To Keep

- Fix the protocol before looking at the final result.
- Preserve seeds, versions, configuration files, and manifests.
- Use subject-level splits for patient datasets whenever possible.
- Compare against simple baselines, not only against your previous model.
- Report failed runs and excluded data.
- Treat simulation realism, predictive accuracy, and clinical validity as separate claims.
- Keep private or licensed datasets outside Git.

The [Scientific Workflow](SCIENTIFIC_WORKFLOW.md) expands these rules into a full study process.
