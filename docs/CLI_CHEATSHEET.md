# IINTS Command Cheatsheet

This is the fast command page for demos, research runs, data prep, local AI, Jetson endurance, and bench-only pump work.

!!! warning "Research-only boundary"
    IINTS-AF is for simulation, education, benchmarking, and bench-only research. Do not use these commands for real insulin dosing.

## First-Time Setup

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge]"
iints doctor --full --suggest
iints guide
```

After the SDK is installed, future updates are shorter:

```bash
iints update
iints update --source github --yes   # newest GitHub version
iints update --dry-run               # show the exact pip command first
```

If you are installing from the latest GitHub source instead of PyPI:

```bash
python -m pip install -U "iints-sdk-python35[full,mdmp,research,edge] @ git+https://github.com/python35/IINTS-SDK.git"
```

## Fast Demo

```bash
iints demo
```

Best live forms:

```bash
iints demo doctor --output-dir results/demo_doctor
iints demo eucys --output-dir results/demo_eucys
iints demo booth --output-dir results/demo_booth
iints demo --audience jury --output-dir results/live_demo
iints demo --audience clinical --output-dir results/live_demo
iints demo --dry-run --output-dir results/live_demo_rehearsal
```

Use `doctor` for clinical feedback, `eucys` for the science-fair experiment story, and `booth` for public digital-patient explanation.

Expected outputs:

```text
results/live_demo/
  PRESENTER_GUIDE.md
  DEMO_STORY.md
  DEMO_CUE_CARD.md
  DEMO_ARTIFACTS.md
  booth_demo_poster.png
  evidence_bundle/
```

## Quickstart Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

Run a safe starter simulation:

```bash
iints run \
  --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --time-step 5 \
  --output-dir results/one_day
```

Every normal run now writes reviewer-facing quality artifacts:

```text
results/one_day/
  results.csv
  report.pdf
  run_manifest.json
  realism_report.json
  realism_dashboard.html
  safety_visualizer.html
  safety_visualizer.json
```

Preflight before long runs:

```bash
iints run-doctor \
  --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --time-step 5
```

## Patient Profiles

```bash
iints profiles presets
iints profiles create --name stable_patient --preset stable-demo
iints profiles create --name endurance_patient --preset endurance
iints profiles create --name stress_patient --preset stress-test
```

## Study And Evidence

Run a benchmark study:

```bash
iints run-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/study
```

Analyze and make a poster:

```bash
iints analyze --study-dir results/study --output-dir results/study_summary
iints poster-study --study-dir results/study --output results/study_poster.png
```

Build public evidence from runs:

```bash
iints evidence build \
  --run normal=results/live_demo/results/01_normal_run \
  --run stress=results/live_demo/results/02_meal_stress \
  --output-dir results/live_demo/evidence_bundle
```

Generate an AGP-style glucose report:

```bash
iints report \
  --results-csv results/live_demo/results/01_normal_run/results.csv \
  --style agp \
  --png \
  --subject-name "live demo normal run" \
  --bundle-dir results/live_demo/agp_report
```

Inspect one run's safety supervisor behavior:

```bash
iints safety-visualize \
  --results-csv results/live_demo/results/01_normal_run/results.csv \
  --output-html results/live_demo/safety_visualizer.html \
  --output-json results/live_demo/safety_visualizer.json
```

## Real Data And Datasets

List and inspect datasets:

```bash
iints data list
iints data info hupa_ucm
iints data cite hupa_ucm
iints data research-plan --output-dir data_packs/research_dataset_plan
```

Prepare supported research datasets:

```bash
iints research prepare-hupa
iints research prepare-azt1d
iints research prepare-ohio
```

Import personal/exported CGM data:

```bash
iints import-data \
  --input-csv data/my_trace.csv \
  --output-dir results/imported_trace \
  --data-format generic
```

Import common platforms:

```bash
iints import-carelink --input-csv data/carelink.csv --output-dir results/carelink
iints import-nightscout --url https://example.herokuapp.com --output-dir results/nightscout
iints import-tidepool --output-dir results/tidepool
```

Realism and MDMP checks:

```bash
iints data realism-check results/imported_trace/cgm_standard.csv \
  --reference free_living_t1d \
  --output-html results/imported_trace/realism_dashboard.html

iints data certify \
  --input results/imported_trace/cgm_standard.csv \
  --output-dir results/imported_trace/certification

iints data eu-ai-pact-review \
  --payload results/imported_trace/certification/certification.json
```

## Local AI Research

Inspect local model choices and Mistral Serverless migration targets:

```bash
iints ai models
```

Blend prepared predictor datasets:

```bash
iints research blend-datasets \
  --source hupa=data_packs/public/hupa_ucm/processed/hupa_ucm_merged.csv \
  --source azt1d=data_packs/public/azt1d/processed/azt1d_merged.csv \
  --source ohio=data_packs/public/ohio_t1dm/processed/ohio_t1dm_merged.csv \
  --output data_packs/processed/predictor_blend.csv \
  --manifest data_packs/processed/predictor_blend_manifest.json
```

Train from completed runs:

```bash
iints research train-local-ai \
  --run day1=results/jetson_research_day \
  --output-dir results/local_ai_lab
```

Controller-specific flow:

```bash
iints research build-control-dataset \
  --run day1=results/jetson_research_day \
  --output data_packs/processed/controller_teacher_dataset.csv

iints research train-controller \
  --data data_packs/processed/controller_teacher_dataset.csv \
  --output models/controller_imitation.json

iints research evaluate-controller \
  --model models/controller_imitation.json \
  --model-kind linear \
  --output-dir results/controller_eval
```

## Jetson Endurance

Start a true 24-hour wall-clock research run:

```bash
iints jetson endurance start \
  --algo iints_quickstart/algorithms/example_algorithm.py \
  --duration 1d \
  --profile normal \
  --wall-clock \
  --output-dir results/jetson_research_day
```

Monitor it:

```bash
iints jetson endurance status --output-dir results/jetson_research_day
iints jetson endurance monitor --output-dir results/jetson_research_day --watch
```

Finalize research outputs:

```bash
iints jetson endurance finalize-research --output-dir results/jetson_research_day
iints jetson endurance export --output-dir results/jetson_research_day --output results/jetson_research_day.zip
```

## Raspberry Pi / Booth Runtime

```bash
iints edge doctor
iints edge quickstart --board raspberry_pi --output-dir iints_pi_demo
iints edge deploy --host raspberrypi.local --user rune --local-output-dir iints_pi_demo
```

Maker Faire style:

```bash
iints makerfaire up
iints makerfaire watchdog
iints makerfaire autostart
```

## Pico Pump Bench Workflow

Create a bench-only workspace:

```bash
iints pump init --output-dir pico_pump_lab
```

Compile, test, and upload a locked non-actuating bundle:

```bash
iints pump compile \
  --algorithm algorithms/pico_bench_algorithm.py \
  --output-dir bundles/pico_bench_bundle

iints pump bench-test \
  --bundle-dir bundles/pico_bench_bundle \
  --output-json bundles/pico_bench_bundle/bench_test_report.json

iints pump upload \
  --bundle-dir bundles/pico_bench_bundle \
  --drive /Volumes/CIRCUITPY \
  --i-understand-bench-only
```

## FPGA Safety-Core Workflow

Check and create the bench-only FPGA lab:

```bash
iints fpga doctor
iints fpga setup --output-dir iints_fpga_lab
```

Run the deterministic mock FPGA comparison:

```bash
iints fpga simulate \
  --events iints_fpga_lab/scenarios/night_hypo_risk.json \
  --output-dir results/fpga_mock_run
```

Review the outputs:

```bash
iints fpga compare --run-dir results/fpga_mock_run
iints fpga report --run-dir results/fpga_mock_run
```

Make a complete demo bundle:

```bash
iints fpga demo --output-dir results/fpga_demo
```

Open these after the demo:

```text
results/fpga_demo/events.csv
results/fpga_demo/results.json
results/fpga_demo/manifest.json
results/fpga_demo/report.md
results/fpga_demo/lab/FPGA_STORY.md
```

## Troubleshooting

```bash
iints --help
iints doctor --full --suggest
iints doctor --smoke-run
iints data list
iints validation-profiles
iints sources
```

If a command exists in the docs but not on your machine, update the SDK:

```bash
iints update --source github --yes
hash -r
iints --help
```

## What To Open After A Demo

```bash
open results/live_demo/PRESENTER_GUIDE.md
open results/live_demo/DEMO_CUE_CARD.md
open results/live_demo/booth_demo_poster.png
open results/live_demo/evidence_bundle/README.md
open results/live_demo/evidence_bundle/MODEL_CARD.md
```

On Linux, replace `open` with `xdg-open`.
