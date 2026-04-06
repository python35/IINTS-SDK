# Booth Demo Guide

The SDK now includes a fair-ready demo flow that creates:

- three simulation runs
- a jury-friendly poster
- a written talk track
- optional AI-ready artifacts for the safety case

## What It Shows

The booth demo tells one clean story in three panels:

1. `Normal Run`
   A moderate day where the controller keeps glucose in range.
2. `Meal Stress Test`
   A harder day with larger meals and exercise.
3. `Supervisor Override`
   A deliberately unsafe AI request that gets blocked by the safety supervisor.

That makes it easy to explain the SDK in public:

- it simulates
- it stresses
- it protects

## Fastest Way To Run It

From the repository root, for the cleanest live on-stage flow:

```bash
./scripts/run_live_stage_demo.sh
```

That script is backed by `examples/demos/07_live_stage_demo.py`, which is the best file to show first because the top of the file clearly exposes the patient profile, output directory, duration, and deterministic seed.

It also visibly demonstrates these SDK features in code:

- `run_full(...)`
- `generate_results_poster(...)`
- `prepare_ai_ready_artifacts(...)`

You can also run the full booth bundle script directly:

```bash
./scripts/run_booth_demo.sh
```

Or with the CLI:

```bash
iints demo-booth
```

Or from source:

```bash
PYTHONPATH=src python3 examples/demos/06_booth_demo.py
```

## Main Outputs

By default the outputs go to `results/booth_demo/`.

Key files:

- `results/booth_demo/booth_demo_poster.png`
- `results/booth_demo/booth_demo_poster.json`
- `results/booth_demo/JURY_TALK_TRACK.md`
- `results/booth_demo/BEURS_LIVE_DEMO_SCRIPT.txt`
- `results/booth_demo/run_commands.md`
- `results/booth_demo/demo_summary.json`

For the larger public story, you can also build:

```bash
iints demo-expo --output-dir results/expo_demo
```

That adds:

- `results/expo_demo/study_summary.json`
- `results/expo_demo/study_summary.md`
- `results/expo_demo/evidence_table.csv`
- `results/expo_demo/evidence_table.md`
- `results/expo_demo/study_poster.png`

Each scenario also gets its own full run bundle with:

- `results.csv`
- `audit/`
- `baseline/`
- `clinical_report.pdf`
- `run_manifest.json`

## Best Live Flow

Use the smallest readable source file as your starting point:

```bash
examples/demos/07_live_stage_demo.py
```

What to point at first:

- `PATIENT_CONFIG` - show that a patient can be swapped
- `OUTPUT_DIR` - show that all artifacts land in one bundle
- `DURATION_MINUTES` and `TIME_STEP_MINUTES` - show simulation control
- `SEED` - show reproducibility

Then run:

```bash
./scripts/run_live_stage_demo.sh
```

That wrapper script resolves the repository root automatically, so you do not have to manually `cd` into the right place first if you launch it via its full path.

Then open:

- `results/booth_demo_live/booth_demo_poster.png`
- `results/booth_demo_live/JURY_TALK_TRACK.md`
- `results/booth_demo_live/BEURS_LIVE_DEMO_SCRIPT.txt`

If you are on a machine that only has the installed SDK and not the repository checkout, first export the same showable demo code with:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

If someone wants proof that the poster is real, open one of the scenario folders and show:

- `results.csv`
- `clinical_report.pdf`
- `run_manifest.json`

## Optional Local AI Step

If Ollama and a local Ministral model are ready, the booth demo also prepares the Supervisor Override run for AI explanation.

Recommended live check:

```bash
iints ai local-check --model ministral-3:3b
iints ai report results/booth_demo/03_supervisor_override --model ministral-3:3b
iints ai review results/booth_demo/03_supervisor_override --model ministral-3:3b
iints ai explain results/booth_demo/03_supervisor_override --model ministral-3:3b
```

## Jury Framing

Suggested one-line pitch:

> IINTS-AF is a safety-first SDK for testing insulin-delivery algorithms before they ever touch a real patient.

Suggested walkthrough:

1. Open `examples/demos/07_live_stage_demo.py` and point at `PATIENT_CONFIG`, `OUTPUT_DIR`, `DURATION_MINUTES`, and `SEED`.
2. Say that a different patient can be tested by swapping the patient profile name or pointing to another YAML config.
3. Run `./scripts/run_live_stage_demo.sh`.
4. Open the left panel first: normal control.
5. Move to the middle panel: stress handling under meals and exercise.
6. Finish on the right panel: the supervisor blocks unsafe insulin.
7. Point out that each panel comes from a full reproducible run bundle, not a hand-drawn mockup.
8. If time allows, show the local AI explanation on the safety case.

## Why This Is Good For A Fair

- one command to run
- one poster to show
- one markdown talk track to read from
- one safety story that non-technical people can understand

## Alternative: Live Raspberry Pi Demo

If you want a more "alive" booth setup, use the new persistent patient runtime instead of the three-panel booth bundle:

```bash
iints quickstart --project-name iints_pi_demo
cd iints_pi_demo
iints patient scenarios
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x
```

Then present:

- the dashboard at `http://127.0.0.1:8765/dashboard`
- `iints patient inject-meal --carbs 60 --workspace patient_runtime`
- `iints patient expo-reset --workspace patient_runtime`
- `iints patient export-service --workspace patient_runtime`
- optionally `iints patient review --workspace patient_runtime --model ministral-3:3b`

That path is documented in:

- [Digital Patient On Raspberry Pi](DIGITAL_PATIENT_PI.md)
