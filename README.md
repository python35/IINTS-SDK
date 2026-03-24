# IINTS-AF SDK
[![PyPI version](https://badge.fury.io/py/iints-sdk-python35.svg)](https://badge.fury.io/py/iints-sdk-python35)
[![Python Package CI](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml)
[![Site](https://img.shields.io/badge/site-IINTS--AF-0a66c2?style=flat&logo=firefox-browser&logoColor=white)](https://python35.github.io/IINTS-Site/index.html)

IINTS-AF is a safety-first SDK for insulin-algorithm research.
It lets you simulate, validate, and report results with reproducible artifacts.

Docs (GitHub Pages): [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)

## What You Can Do
- Run virtual patient simulations.
- Test algorithm safety gates (deterministic supervisor).
- Add optional AI glucose forecasting.
- Validate datasets before training/evaluation.
- Generate audit-ready CSV/JSON/PDF/HTML outputs.

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"
```

```bash
iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

If you want the clearest install/path rules first, read:
- `docs/INSTALLATION.md`

Short rule:
- installed `iints ...` commands can run from any folder
- `python -m pip install -e ".[mdmp]"` only works from the SDK repo root
- after `iints quickstart`, switch into the generated project folder

## CareLink Import

The SDK can now ingest Medtronic CareLink / MiniMed CSV exports and convert them into the standard IINTS schema:

```bash
iints import-carelink \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/imported_carelink
```

This writes:
- `cgm_standard.csv`
- `scenario.json`
- `carelink_summary.json`

It extracts glucose, carb, and insulin events from the CareLink event log and aligns them onto an IINTS-ready timeline.

If you want a reusable personal-data workspace in one command, build a CareLink workbench:

```bash
iints carelink-workbench \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/personal_carelink
```

This adds:
- `carelink_timeline.csv`
- `carelink_metrics.json`
- `carelink_dashboard.png`
- `carelink_poster.png`
- `carelink_dashboard.html`
- `ai/report_payload.json`
- `ai/trends_payload.json`
- `ai/anomalies_payload.json`
- `ai/step_riskiest.json`

That workbench is designed for three things:
- inspect your own data visually
- reuse the generated `scenario.json` inside IINTS experiments
- let the local AI assistant explain what the imported patterns mean

## AI Assistant (Ministral 3 Open-Weight via Ollama)

The SDK now includes a research-only AI assistant layer for explanations and run summaries.
It is gated by MDMP verification before any LLM call is allowed.

Use an active virtual environment for the full flow.

If you installed the released SDK from PyPI, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"
```

If you are developing from source instead, first move into the SDK repo root and then run:

```bash
cd /path/to/IINTS-SDK
python -m pip install -U -e ".[mdmp]"
```

Run the open local Mistral model locally with Ollama:

```bash
ollama pull ministral-3:8b
iints ai models
```

Recommended first-time setup:

```bash
ollama pull ministral-3:8b
iints ai local-check --model ministral-3:8b
```

`local-check` now performs a tiny generation smoke-test by default, so it verifies both model presence and real inference readiness.

Recommended flow:

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints ai prepare results/<run_id>
iints ai report results/<run_id>
```

For imported CareLink data, the matching flow is:

```bash
iints carelink-workbench \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/personal_carelink

iints ai report results/personal_carelink --model ministral-3:3b
iints ai trends results/personal_carelink --model ministral-3:3b
iints ai explain results/personal_carelink --model ministral-3:3b
```

Direct JSON mode still works if you already have your own payloads and signed MDMP artifact:

```bash
iints ai explain results/step.json \
  --mdmp-cert results/report.signed.mdmp
```

Notes:
- AI analysis is blocked if the MDMP artifact is invalid.
- Minimum required MDMP grade defaults to `research_grade`.
- The SDK now targets the open local `Ministral 3` Ollama model by default.
- Users can choose a larger or smaller local Mistral-family model with `--model ...`.
- Large JSON payloads are clipped automatically before prompt generation to keep local inference stable.
- `iints ai prepare <run_dir>` now creates AI-ready JSON payloads and, when MDMP is installed, a local development certificate plus keypair in `<run_dir>/ai/`.
- `iints carelink-workbench` now does the same kind of AI preparation for imported personal CareLink data and also generates a dashboard PNG/HTML pair.
- If Ollama closes the connection during generation, the SDK now surfaces an explicit recovery hint and points users toward `ministral-3:3b` for lower-memory systems.
- After `iints ai prepare`, you can point `iints ai explain|trends|anomalies|report` directly at the run directory.
- After `iints carelink-workbench`, you can point those same AI commands directly at the generated CareLink workspace directory.
- Output is research-only and not medical advice.

## Jury Poster / Demo Graphic

You can now generate a poster-style PNG directly from one to three real run bundles:

```bash
iints poster \
  --run-dir results/normal_run \
  --run-dir results/meal_stress \
  --run-dir results/supervisor_override \
  --label "Normal Run" \
  --label "Meal Stress Test" \
  --label "Supervisor Override" \
  --output-path results/posters/iints_results_poster.png
```

The poster shows:
- glucose curves with the target range highlighted
- meal events
- supervisor interventions
- panel summaries with TIR, hypo time, meal count, and intervention count

If you omit `--run-dir`, the CLI auto-discovers the latest run bundles under `./results`.

## Fair / Jury Demo

If you want one clean live demo for a booth, jury, or pitch session, use the built-in booth flow:

```bash
./scripts/run_booth_demo.sh
```

This generates:
- three run bundles (`Normal Run`, `Meal Stress Test`, `Supervisor Override`)
- a ready-to-show poster PNG
- a markdown jury talk track
- a plain-text live demo script for the stand
- optional AI-ready artifacts for the safety case

You can also run it through the CLI:

```bash
iints demo-booth --output-dir results/booth_demo
```

For a cleaner live explanation, show this source file first:

```bash
examples/demos/07_live_stage_demo.py
```

That file is deliberately small and readable, so you can point to:
- `PATIENT_CONFIG`
- `OUTPUT_DIR`
- `DURATION_MINUTES`
- `TIME_STEP_MINUTES`
- `SEED`

Then run:

```bash
./scripts/run_live_stage_demo.sh
```

That shell wrapper resolves the SDK repo root automatically, so it still works if you launch it from another working directory via its full path.

And open:
- `results/booth_demo_live/booth_demo_poster.png`
- `results/booth_demo_live/JURY_TALK_TRACK.md`
- `results/booth_demo_live/BEURS_LIVE_DEMO_SCRIPT.txt`

What makes this script good for a booth:
- it visibly calls `run_full(...)`
- it visibly calls `generate_results_poster(...)`
- it visibly calls `prepare_ai_ready_artifacts(...)`
- you can point to one patient setting and explain how the same pipeline reruns for another patient

If you installed the SDK on another machine and do not have the repository checkout there, export the same demo code with:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

## Updating The SDK

If another machine is missing newer commands like `iints ai ...` or `iints demo-booth`, upgrade inside the active virtual environment to the latest release:

```bash
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"
hash -r
python -c "import iints; print(iints.__version__)"
```

If that machine still behaves like an old install, run:

```bash
iints-sdk-doctor
```

Full guide:
- `docs/UPDATING.md`

If you specifically need to reproduce a known environment, you can pin an exact release number instead of using the unpinned upgrade command above.

Troubleshooting:
- If `iints ai ...` says `No such command 'ai'`, your environment usually still has a legacy `iints` package installed alongside `iints-sdk-python35`.
- Run `iints-sdk-doctor` first.
- If it reports a conflict, repair the environment with:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[mdmp]"
hash -r
```

## MDMP (Short)
MDMP is the data-quality protocol used by IINTS.

- `Contract`: defines expected columns, types, units, and bounds.
- `Validation`: checks a dataset against the contract.
- `Fingerprint + Grade`: writes deterministic hashes and a grade (`draft`, `research_grade`, `clinical_grade`).
- `Visualizer`: builds a single-file HTML report for audits.

Use the dedicated namespace:

```bash
iints mdmp template --output-path mdmp_contract.yaml
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv --output-json results/mdmp_report.json
iints mdmp visualizer results/mdmp_report.json --output-html results/mdmp_dashboard.html
```

Use standalone MDMP backend (optional):

```bash
export IINTS_MDMP_BACKEND=mdmp_core
```

Staleness / lineage checks (standalone MDMP CLI):

```bash
mdmp fingerprint-record data/my_cgm.csv --output-json results/fingerprint.json --expires-days 365
mdmp fingerprint-check results/fingerprint.json data/my_cgm.csv
mdmp lineage-card-refresh results/mdmp_model_card.yaml
mdmp registry init --registry registry/mdmp_registry.json
mdmp registry push --registry registry/mdmp_registry.json --report results/mdmp_report.json
```

## Dual Repo Workflow
- SDK repo: `python35/IINTS-SDK`
- MDMP repo: `python35/MDMP`

Local helper scripts:
- `tools/dev/dual_repo_status.sh`
- `tools/dev/dual_repo_commit_push.sh`

Full process: `docs/DUAL_REPO_WORKFLOW.md`

MDMP sync CI gate:
- `.github/workflows/mdmp-sync.yml`
- Uses private-repo checkout when `MDMP_REPO_TOKEN` is configured.
- Falls back to `mdmp-protocol` from PyPI when checkout is unavailable.
- Auto dependency updates for MDMP are handled via Dependabot (`.github/dependabot.yml`).

## Tools Layout

Repository helpers are now grouped by purpose:

- `scripts/`: simple user-facing shortcuts like test, lint, and demo entrypoints
- `tools/ci/`: CI gates and policy checks
- `tools/dev/`: maintainer workflows and multi-repo helpers
- `tools/docs/`: manual and documentation builders
- `tools/data/`: dataset import and conversion utilities
- `tools/analysis/`: plotting, diagnostics, and report helpers
- `tools/assets/`: branding and asset generation helpers

Reference: `tools/README.md`

## Typical Workflow
1. Prepare or import data.
2. Validate data with MDMP.
3. Run simulation or forecast evaluation.
4. Review report artifacts and metrics.

## Key Commands
```bash
iints run-full --algo algorithms/example_algorithm.py --scenario-path scenarios/clinic_safe_baseline.json --output-dir results/run_full
iints scorecard --algo algorithms/example_algorithm.py --profile research_default --output-dir results/scorecard
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
iints sources --output-json results/source_manifest.json
```

## Documentation
- Docs site: [python35.github.io/IINTS-SDK](https://python35.github.io/IINTS-SDK/)
- Plain guide: [docs/PLAIN_LANGUAGE_GUIDE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/PLAIN_LANGUAGE_GUIDE.md)
- Comprehensive guide: [docs/COMPREHENSIVE_GUIDE.md](https://github.com/python35/IINTS-SDK/blob/main/docs/COMPREHENSIVE_GUIDE.md)
- MDMP draft: [docs/MDMP.md](https://github.com/python35/IINTS-SDK/blob/main/docs/MDMP.md)
- Demos: [examples/demos](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)
- Notebooks: [examples/notebooks](https://github.com/python35/IINTS-SDK/tree/main/examples/notebooks)

## Safety Notice
For research use only. Not a medical device. No clinical dosing advice.
