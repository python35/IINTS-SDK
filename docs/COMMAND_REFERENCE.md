# Command Reference

Use this page when you know the task and want the public command family quickly, without reading the full technical manual.

**Read before:** [Choose Your Path](USER_GUIDE_MAP.md) if you are not sure which command family you need.

**Need the shortest practical list?** Use the [Command Cheatsheet](CLI_CHEATSHEET.md).

**Read next:** [Technical Reference](TECHNICAL_README.md) for deeper integration details.

## Command Navigation

### `iints menu` (or `iints hub`)
Choose a maintained command domain and print reliable entry points without invoking a workflow automatically.

### `iints map` (or `iints overview`)
Print the maintained command domains. Typer-generated `--help` remains authoritative for options.

## Beginner-Friendly Entry Points

### `iints guide`
Run the short starter demo or open command navigation.

### `iints start`
Print a goal-based first-run plan, or run the safe starter action.

Common forms:

```bash
iints start
iints start --goal project --run
iints start --goal edge
iints start --goal data
```

### `iints onboard`
Show the one recommended path from machine check to first study bundle.

```bash
iints onboard
iints onboard --run-safe-steps
```

The safe mode runs `doctor`, a full demo, demo-data import, and a realism check,
then prints the two study commands you can run once you choose an algorithm.

### `iints demo`
The main live-demo button. It exports showable code, prints the presenter story,
runs the demo in audience-safe stage mode, writes a technical log off-screen,
and produces the cue card, artifact map, presenter guide, poster, and rerun
script. By default it also builds `evidence_bundle/` with a public README,
model card, run index, and copied proof artifacts.

Common forms:

```bash
iints demo
iints demo doctor
iints demo eucys
iints demo booth
iints demo --audience jury --output-dir results/live_demo
iints demo --audience clinical
iints demo --dry-run          # rehearsal/preflight only, not the live call
iints demo --no-evidence      # fast rehearsal without the public proof bundle
iints demo --technical        # show raw execution detail for debugging
iints demo --simulation-only --quick
iints demo --simulation-only --full
```

Story presets:

- `iints demo doctor` starts with a clinical safety discussion: virtual patient, normal day, meal stress, risky context, and supervisor decision.
- `iints demo eucys` frames the run as one experiment with a research question, hypothesis, three runs, and evidence bundle.
- `iints demo booth` frames the run as a public digital-patient story: scenario changes, algorithm suggestion, safety check, and visual proof.

### `iints demo-live`
Explicit alias for the same live presentation engine used by `iints demo`.

It exports showable Python code, prints an audience-aware opening talk track, runs the live demo, writes `PRESENTER_GUIDE.md`, `DEMO_STORY.md`, `DEMO_CUE_CARD.md`, `DEMO_ARTIFACTS.md`, and `RUN_LIVE_DEMO.sh`, then lists the poster plus proof artifacts to open next. Non-SDK story modes keep code as the proof layer instead of the first thing shown.

Common forms:

```bash
iints demo-live
iints demo-live --output-dir results/live_demo
iints demo-live --no-run
iints demo-live --prepare-ai
iints demo-live --audience clinical
iints demo-live --audience engineering
iints demo-live --story doctor
iints demo-live --story eucys
iints demo-live --story booth
```

### `iints quickstart`
Create a ready-to-run project folder.

```bash
iints quickstart --project-name iints_quickstart
```

The generated project is self-contained: it includes `patients/stable_patient.yaml`, a scenario file, and an editable starter algorithm so you can run locally without depending on packaged patient assets.

### `iints run --wizard`
Interactive custom run builder.

### `iints version`
Compare the active Python environment with the latest stable PyPI release.

```bash
iints version --refresh
iints version --offline --json
iints version --refresh --fail-if-outdated --fail-if-unknown
```

The report distinguishes installed distribution metadata from active source code and includes the Python executable, CLI path, package location, release source, and check time. Exit code `2` means a verified update is available; exit code `3` means release status could not be verified when `--fail-if-unknown` is active.

### `iints update`
Update the current Python environment to the newest SDK release.

Common forms:

```bash
iints update
iints update --dry-run
iints update --repair --force-reinstall --yes
iints update --no-cache-dir --yes
iints update --source github --github-ref stable --yes
iints update --extras full,mdmp,research,edge
```

The command prints the exact `python -m pip install -U ...` invocation before it changes anything. By default `--source auto` uses the stable PyPI release and never silently falls back to GitHub `main`. Use `--dry-run` during a live demo setup check, or `--check` for a status-only automation check.

### `iints delete`
Remove IINTS from the current machine/environment with a visible deletion plan.

Common forms:

```bash
iints delete --dry-run
iints delete --yes
iints delete --everything --dry-run
iints delete --everything --yes
iints delete --source-checkout --yes
iints delete --local-outputs --yes
iints delete --no-packages --path results/old_iints_run --yes
```

Default behavior removes the active Python SDK packages plus user-level IINTS config, plugin, and cache folders. `--everything` also includes known generated output folders in the current directory and a detected local `IINTS-SDK` source checkout. It still does **not** guess private datasets, external-drive research archives, or unrelated virtual environments.

## Core Simulation Commands

### `iints run`
Run one simulation.

Examples:

```bash
iints run --preset baseline_t1d
iints run --algo algorithms/example_algorithm.py --scenario scenarios/example_scenario.json
iints run --dry-run --preset baseline_t1d
```

### `iints run-full`
One-line run with full output bundle.

### `iints run-parallel`
Run a matrix of scenarios in parallel.

### `iints benchmark`
Compare algorithms across standard workloads.

## Extension Commands

### `iints plugin install`
Install a local algorithm plugin without editing SDK source code.

```bash
iints plugin install algorithms/my_algo.py
iints algorithms list
```

The SDK copies the file into the local plugin home and records it in
`~/.iints/plugins/registry.json`. For tests or portable environments, set
`IINTS_PLUGIN_HOME` to another folder.

### `iints plugin register`
Register extension files by kind.

```bash
iints plugin register algo algorithms/my_algo.py
iints plugin register patient-model patient_models/my_model.py --name "My Model"
iints plugin register data-source data_sources/my_importer.py
iints plugin register validator validators/my_check.py
```

Algorithm plugins become visible in `iints algorithms list`. Patient model,
data source, and validator plugins are registered for discovery/documentation
hooks so the SDK can grow without source-code edits.

### `iints plugin list`
Show local extension plugins.

```bash
iints plugin list
iints plugin list --kind algorithm
```

### `iints plugin uninstall`
Remove a local plugin registry entry.

```bash
iints plugin uninstall "My Algorithm"
iints plugin uninstall "My Algorithm" --remove-file
```

### `iints patientmodel list`
Show built-in and locally registered patient models.

```bash
iints patientmodel list
```

## Study / Research Commands

### `iints study-protocol`
Write the official study protocol bundle.

### `iints run-study`
Run the scientific benchmark matrix.

### `iints analyze`
Aggregate a study directory.

### `iints compare-study`
Compare two study outputs.

### `iints poster-study`
Generate poster-ready figures from a study.

### `iints research regenerative panels`
Inspect the bundled regenerative-islet protein panels and optionally export the
required evidence plan:

```bash
iints research regenerative panels \
  --panel beta_cell_identity_and_function \
  --output-json results/regenerative/panels.json
```

### `iints research regenerative compare`
Run a provenance-preserving descriptive protein comparison between normalized
SC-islet and primary-islet observations:

```bash
iints research regenerative compare \
  --dataset data/standardized_islet_proteomics.csv \
  --panel beta_cell_identity_and_function \
  --normalization-note "Joint normalization within one experiment" \
  --output-dir results/regenerative/protein_comparison
```

The command does not calculate treatment efficacy, approve a cell product, or
map protein abundance into graft-model parameters.

### `iints research regenerative import-proteomics`
Standardize MaxQuant `proteinGroups.txt`, DIA-NN `report.tsv`, or wide matrix files into the regenerative comparator contract:

```bash
iints research regenerative import-proteomics \
  --input-file data/maxquant/proteinGroups.txt \
  --sample-metadata data/sample_annotations.csv \
  --source-id PXD001539 \
  --format maxquant \
  --output-csv data/standardized_islet_proteomics.csv
```

## Data Commands

### `iints data list`
Show public data packs.

### `iints data fetch`
Fetch a pack into a local directory.

### `iints data research-plan`
Generate the curated diabetes dataset acquisition plan for local AI research.

```bash
iints data research-plan --output-dir data_packs/research_dataset_plan
```

Use `--dataset <id>` repeatedly to generate a focused plan for a subset.

### `iints data certify`
Run data certification.

### `iints data realism-check`
Judge whether a glucose trace looks physiologically plausible for research or demo use.
Supports:
- `--reference free_living_t1d`, `--reference azt1d`, or `--reference hupa_ucm`
- `--output-json results/realism_report.json`
- `--output-html results/realism_dashboard.html`
- `--min-realism-verdict needs_review`

### `iints data import-cgmacros`
Import and standardize the multimodal **CGMacros** dataset (*Nature Scientific Data*, 2025):
- Synchronizes Abbott FreeStyle Libre Pro and Dexcom G6 Pro sensors.
- Extracts discrete meals with exact macronutrient breakdown (carbs, protein, fat, fiber, calories).
- Merges participant clinical screening (`bio.csv`).

```bash
iints data import-cgmacros \
  --input-dir data/cgmacros_raw \
  --output-dir data/cgmacros_standardized
```

### `iints research ppgr-benchmark`
Benchmark 2-hour Postprandial Glycemic Response (PPGR) models (Google GlucoFM style):
- Compares Carb-Only baseline, Multi-Macronutrient biphasic regression, and Dual-Stream GlucoFM architectures.
- Computes trajectory MAE, RMSE, Pearson $r$, Peak Glucose MAE, and Time-to-Peak error.

```bash
iints research ppgr-benchmark \
  --meals-file data/cgmacros_standardized/cgmacros_meals.csv \
  --sensor dexcom \
  --output-dir results/ppgr_benchmark
```

### `iints research cgm-jepa-embed`
Extract a 96-dimensional latent representation from a 24h simulation or CGM trace using **CGM-JEPA** (*arXiv:2605.00933*):

```bash
iints research cgm-jepa-embed \
  --input results/baseline/results.csv \
  --output-dir results/cgm_jepa_embedding
```

### `iints research cgm-jepa-experiment`
Run a 100-simulation virtual patient parameter sweep (varying $S_I$) and test latent manifold alignment and sensor noise robustness ($\cos \theta$):

```bash
iints research cgm-jepa-experiment \
  --output-dir results/cgm_jepa_study \
  --n-simulations 100 \
  --sweep-param insulin_sensitivity
```

### `iints research cgm-jepa-confounder`
Run the **Physiological Confounder Benchmark** (50 paired cohorts) testing whether observational CGM foundation models confound divergent underlying biological states:

```bash
iints research cgm-jepa-confounder \
  --output-dir results/cgm_jepa_confounder \
  --num-pairs 50
```

## Medical Device Safety & FDA Verification Commands

### `iints safety fda-list`
List all verified medical device recall cases from the US FDA database registered in IINTS-AF:

```bash
iints safety fda-list
```

### `iints safety fda-benchmark`
Run the complete FDA Adverse Event Benchmark evaluating unmitigated automated controllers vs the IINTS-AF Dual-Guard Safety Supervisor across real device failures (Tandem, MiniMed, Omnipod, Dexcom):

```bash
iints safety fda-benchmark \
  --output-dir results/fda_safety_study
```

### `iints import-carelink`, `iints import-nightscout`, `iints import-tidepool`
Import real-world CGM sources.

## Edge / Booth Commands

### `iints edge doctor`
Preflight for Raspberry Pi or UNO Q.

### `iints edge quickstart`
Create the easiest Pi or UNO Q demo project and optionally start the Linux-side runtime.

### `iints edge setup`
Generate an edge project scaffold.

### `iints edge deploy`
Scaffold, upload, install, and start a Raspberry Pi edge project in one command.

### `iints edge offline-bundle`
Build a USB-friendly offline install tarball for Raspberry Pi or UNO Q setups.

### `iints edge study`
Run a reproducible multi-seed study directly on the current edge machine.

### `iints edge long-study`
Run a multi-day or multi-week YAML-driven study directly on the Pi, with rolling day profiles and export-friendly nested outputs.
Use `--resume` to continue from the next incomplete day after a reboot.

### `iints edge study-snapshot`
Create a `.tar.gz` snapshot of a long-study folder for crash recovery or USB backup.

### `iints edge study-export`
Package a long-study folder into a transfer-ready zip archive for another device.

### `iints edge remote-status`, `iints edge remote-reset`, `iints edge remote-stop`
Run common Raspberry Pi maintenance commands remotely over SSH.

## Jetson Endurance Commands

### `iints jetson doctor`
Check Jetson-like hardware probes, thermal zones, and NVIDIA tooling before a long headless run.

### `iints jetson endurance start`
Run a headless adversarial endurance study.

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --predictor models/lstm_predictor.pt \
  --duration 7d \
  --output-dir results/jetson_7day \
  --profile mixed_adversarial \
  --seed 42 \
  --checkpoint-interval 360 \
  --hardware-sample-interval 60
```

Add `--wall-clock` when the study horizon should consume real time instead of
finishing as fast as possible. A run such as `--duration 1d --wall-clock`
therefore lasts about 24 real hours and writes a training-ready `research/`
bundle next to the normal endurance outputs.

### `iints jetson endurance status`
Show progress, current glucose, TIR so far, interventions, critical events, the
latest checkpoint, resume count, and wall-clock ETA.

### `iints jetson endurance monitor`
Print the same status repeatedly with `--watch`.

### `iints jetson endurance stop`
Request a safe stop and optional report finalization.

### `iints jetson endurance export`
Package the complete endurance folder into a transfer-ready `.zip`.

### `iints jetson endurance finalize-research`
Train post-run local research models from one endurance bundle and write a held-out closed-loop evaluation report.

### `iints jetson endurance install-service`
Write a systemd service file with automatic `--resume` for multi-day Jetson runs.
Pass `--wall-clock` here too when the generated service should preserve real-time pacing.

## Local AI Research Commands

### `iints research blend-datasets`
Blend already prepared real datasets into one source-aware predictor dataset.

### `iints research prepare-ohio`
Prepare a local OhioT1DM XML folder into a gitignored processed dataset. Do not commit the raw `OhioT1DM-volledig/` folder to GitHub:

```bash
iints research prepare-ohio \
  --input-dir /path/to/OhioT1DM-volledig \
  --splits train \
  --output data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --report data_packs/public/ohio_t1dm_full/processed/ohio_train_quality_report.json
```

### `iints research glucose-model build-dataset`
Normalize one or more prepared glucose datasets into the dedicated `iints-glucose-forecast-v0` training contract:

```bash
iints research glucose-model build-dataset \
  --input data_packs/public/ohio_t1dm_full/processed/ohio_train.csv \
  --input results/realism_learning_10k/research/predictor_training.csv \
  --labels ohio_full,sim_10k \
  --profile long \
  --output-dir models/iints-glucose-forecast-v0/dataset
```

### `iints research glucose-model train`
Train the dedicated glucose-forecast model and optionally build a Hugging Face-ready export folder:

```bash
iints research glucose-model train \
  --data models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --config models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml \
  --output-dir models/iints-glucose-forecast-v0 \
  --epochs 220 \
  --comparison-dir results/glucose_model_comparison \
  --export-hf
```

### `iints research glucose-model compare`
Compare transparent baselines and trained MSE/Band/PINN checkpoints against physiology-aware gates:

```bash
iints research glucose-model compare \
  --data data_packs/public/ohio_t1dm_full/processed/ohio_test.csv \
  --config models/iints-glucose-forecast-v0/dataset/glucose_model_config.yaml \
  --model mse=models/glucose_mse/predictor.pt \
  --model pinn=models/iints-glucose-forecast-v0/predictor.pt \
  --mc-samples 30 \
  --output-dir results/glucose_model_comparison
```

### `iints research glucose-model export-hf`
Package `predictor.pt`, `training_report.json`, the model config, privacy/limitations notes, examples, comparison metrics, a research-only model card, and a redacted public dataset manifest for Hugging Face:

```bash
iints research glucose-model export-hf \
  --model-dir models/iints-glucose-forecast-v0 \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --comparison-dir results/glucose_model_comparison \
  --repo-id IINTS/iints-glucose-forecast-v0
```

### `iints research glucose-model jetson-train-hf`
Continue training an existing Hugging Face glucose model on Jetson with warm-start, candidate comparison, and local champion promotion:

```bash
iints research glucose-model jetson-train-hf \
  --repo-id IINTS/iints-glucose-forecast-v0 \
  --dataset models/iints-glucose-forecast-v0/dataset/glucose_training_dataset.csv \
  --dataset-manifest models/iints-glucose-forecast-v0/dataset/glucose_dataset_manifest.json \
  --work-dir models/jetson_hf_training \
  --max-trials 1 \
  --epochs 2 \
  --batch-size 64 \
  --upload-mode none
```

Use `--upload-mode pr` after review to upload a promoted champion as a Hugging Face pull request. The command uploads model artifacts and redacted metadata only, not raw private dataset rows.

### `iints research build-control-dataset`
Combine one or more run bundles into a supervised controller teacher dataset.

### `iints research train-controller`
Train the first auditable local controller baseline from safe-action labels.

### `iints research train-neural-controller`
Train the stronger PyTorch controller from the same supervised safe-action labels.

### `iints research evaluate-controller`
Compare a learned controller against the clinical baseline on held-out presets and seeds.

### `iints research local-ai-lab`
Combine completed Jetson/simulator runs into one local AI workspace:
predictor dataset, controller-teacher dataset, dataset card, local controller models, optional predictor training, and held-out controller evaluation.

### `iints research train-local-ai`
Friendly alias for the same local AI workspace command. Prefer this name in new demos and docs:

```bash
iints research train-local-ai \
  --run day1=results/jetson_research_day \
  --output-dir results/local_ai_lab
```

Full workflow: [Jetson Endurance Mode](JETSON_ENDURANCE.md).

## Results Management

### `iints results`
Index all run-level `results.csv` files and every generated artifact under a results root:

```bash
iints results --root results
```

This writes a compact management bundle:

- `run_index.csv`
- `artifact_inventory.csv`
- `RESULTS_INDEX.md`
- `result_manager_manifest.json`
- `results_index.xlsx` when spreadsheet export is available

Use `--include-raw` only when you want one combined long table for downstream local-AI/data analysis:

```bash
iints results --root results/research_realism_sweep_20260603_02 --include-raw
```

The same command is also available as `iints research results-index` for research workflows.

## Evidence Commands

### `iints run-doctor`
Preflight an algorithm, patient YAML, scenario JSON, duration, time step, and output folder before a long run:

```bash
iints run-doctor \
  --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --time-step 5
```

It catches missing files, validation errors, aggressive glucose drift, likely pre-meal hypoglycemia, and output path problems.

### `iints evidence build`
Build a public research evidence bundle from one or more completed runs:

```bash
iints evidence build \
  --run normal=results/live_demo/results/01_normal_run \
  --run stress=results/live_demo/results/02_meal_stress \
  --output-dir results/live_demo/evidence_bundle
```

Optional inputs:

- `--local-ai-dir results/local_ai_lab`
- `--pump-bundle-dir bundles/pico_bench_bundle`

### `iints report --style agp`
Generate an AGP-style research PDF from a dense simulation or CGM CSV:

```bash
iints report \
  --results-csv results/one_day/results.csv \
  --style agp \
  --png \
  --svg \
  --subject-name "stable demo run" \
  --bundle-dir results/one_day/agp_report
```

This writes `agp_report.pdf`, `agp_summary.json`, `agp_assets/agp_profile.png`, `agp_assets/agp_profile.svg`, `agp_assets/daily_profiles.png`, and `agp_assets/daily_profiles.svg`. When a run contains `explainable_events`, the AGP asset folder also includes `xai_events.txt` for human review and `xai_events.json` for downstream analysis. The layout includes glucose statistics, time-in-ranges, an AGP-style modal-day percentile plot, and daily glucose profiles.

### `iints safety-visualize`
Create a standalone HTML safety visualizer from a run CSV:

```bash
iints safety-visualize \
  --results-csv results/one_day/results.csv \
  --output-html results/one_day/safety_visualizer.html \
  --output-json results/one_day/safety_visualizer.json
```

Normal `iints run`, `iints presets run`, `run_full(...)`, and `run_simulation(...)` already create `realism_report.json`, `realism_dashboard.html`, `safety_visualizer.html`, and `safety_visualizer.json` inside the run bundle.

## Pico Pump Bench Commands

### `iints pump init`
Create a bench-only Pico pump lab workspace.

### `iints pump compile`
Package a simulated SDK algorithm with locked, non-actuating Pico firmware:

```bash
iints pump compile \
  --algorithm algorithms/pico_bench_algorithm.py \
  --output-dir bundles/pico_bench_bundle
```

### `iints pump bench-test`
Validate the bundle before upload:

```bash
iints pump bench-test \
  --bundle-dir bundles/pico_bench_bundle \
  --output-json bundles/pico_bench_bundle/bench_test_report.json
```

### `iints pump upload`
Copy the locked bundle to a writable Pico/CircuitPython-style drive after explicit bench-only confirmation.

## FPGA Safety-Core Commands

### `iints fpga start`
Run the easiest FPGA quickstart: create a lab scaffold and run the golden mock safety-core demo.

```bash
iints fpga start --output-dir results/fpga_start
```

### `iints fpga doctor`
Check whether FPGA mode can run locally. Mock transport works without hardware; serial transport is optional and requires `pyserial`.

### `iints fpga setup`
Create a bench-only FPGA lab workspace:

```bash
iints fpga setup --output-dir iints_fpga_lab
```

The workspace includes a safety contract, demo events, a golden `night_hypo_risk` scenario, JSON-lines protocol description, Verilog scaffold, Verilog smoke test, JSON-lines bridge stub, `FPGA_STORY.md`, and a mock-demo shell script.

### `iints fpga simulate`
Run a software-reference versus FPGA-style safety-core comparison:

```bash
iints fpga simulate \
  --events iints_fpga_lab/scenarios/night_hypo_risk.json \
  --output-dir results/fpga_mock_run
```

Use `--transport serial --port /dev/ttyUSB0` when a real FPGA bridge is available.

### `iints fpga export-events`
Convert an existing IINTS results CSV into FPGA event JSON:

```bash
iints fpga export-events \
  --results-csv results/my_run/results.csv \
  --output-events results/fpga_events.json
```

### `iints fpga replay`
Convert an existing IINTS results CSV and immediately run the FPGA comparison:

```bash
iints fpga replay \
  --results-csv results/my_run/results.csv \
  --output-dir results/fpga_replay
```

### `iints fpga compare`
Read `fpga_comparison.json` and fail if hardware-style output diverged from the SDK reference:

```bash
iints fpga compare --run-dir results/fpga_mock_run
```

### `iints fpga report`
Print the report location and summary:

```bash
iints fpga report --run-dir results/fpga_mock_run
```

### `iints fpga demo`
Create the lab scaffold and run a mock FPGA safety-core demo in one command:

```bash
iints fpga demo --output-dir results/fpga_demo
```

The demo bundle includes reviewer-friendly top-level files: `events.csv`, `results.json`, `manifest.json`, and `report.md`.

## Maker Faire Commands

### `iints makerfaire up`
Start the Pi booth flow.

### `iints makerfaire autostart`
Prepare booth autostart files.

### `iints makerfaire watchdog`
Recover the booth runtime if it stops.

## Diagnostics

### `iints doctor`
Basic and full environment checks.

```bash
iints doctor
iints doctor --full --suggest
iints doctor --smoke-run
```

`doctor` reports the installed SDK version, active Python executable, install path, and available command groups. This is the fastest way to catch an old Python interpreter that silently resolves only legacy SDK releases.

### `iints profiles`

```bash
iints profiles presets
iints profiles create --name stable_patient --preset stable-demo
iints profiles create --name endurance_patient --preset endurance
```

Starter presets are provided for `stable-demo`, `stress-test`, and `endurance`.

## MDMP Cryptography And Provenance

### `iints mdmp encrypt-data`
Encrypt a local dataset with ChaCha20-Poly1305 authenticated encryption. Text passphrases are processed with a random per-envelope scrypt salt:

```bash
iints mdmp encrypt-data --input data/patient_cgm.csv --output data/patient_cgm.enc
```

### `iints mdmp decrypt-data`
Decrypt and authenticate encrypted dataset files, ensuring tamper-resistance:

```bash
iints mdmp decrypt-data --input data/patient_cgm.enc --output data/restored.csv
```

### `iints mdmp sign`
Sign an MDMP dataset passport card with Ed25519. The SDK does not currently implement or claim an ML-DSA/post-quantum signature:

```bash
iints mdmp sign --card results/passport.json --key keys/mdmp_private_v1.pem
```

### `iints mdmp verify`
Verify an MDMP dataset passport card signature, expiry, and SHA-256 fingerprint:

```bash
iints mdmp verify results/passport.json --dataset data/patient_cgm.csv
```

### `iints mdmp validate`
Validate dataset against an MDMP contract schema and range rules:

```bash
iints mdmp validate contracts/cgm_contract.yaml data/patient_cgm.csv --min-mdmp-grade research_grade
```

## Foundation Models & Multi-Sensor Research Commands

### `iints research glucofm-embed`
Extract Google GlucoFM Dual-Stream 256-dimensional patient representation embeddings:

```bash
iints research glucofm-embed \
  --input-file data/patient_run.csv \
  --output-file results/glucofm/embedding.csv
```

### `iints research foundation-arena`
Run head-to-head Foundation Model Arena benchmark (Google GlucoFM vs CGM-JEPA vs GluFormer vs IINTS-AF Digital Twin):

```bash
iints research foundation-arena \
  --output-dir results/foundation_arena \
  --n-trials 50
```

### `iints data download-cgmacros`
Download or generate the complete 45-participant CGMacros multi-sensor open science cohort with macronutrient meal events:

```bash
iints data download-cgmacros \
  --output-dir data/cgmacros_cohort \
  --participants 45
```

## Full Details

For every option and advanced workflow, continue to:
- [Choose Your Path](USER_GUIDE_MAP.md)
- [CLI & Advanced Reference](TECHNICAL_README.md)
- [Scientific Workflow](SCIENTIFIC_WORKFLOW.md)
- [Study Analysis](STUDY_ANALYSIS.md)
- [CGM Foundation Models & Benchmarking](CGM_FOUNDATION_MODELS.md)
- [CGMacros Multi-Sensor Pipeline](CGMACROS_DUAL_STREAM.md)
- [OpenFDA Safety Benchmarks](OPENFDA_SAFETY_BENCHMARK.md)
