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
pip install iints-sdk-python35
```

```bash
iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

## AI Assistant (Ministral 3 Open-Weight via Ollama)

The SDK now includes a research-only AI assistant layer for explanations and run summaries.
It is gated by MDMP verification before any LLM call is allowed.

Use an active virtual environment for the full flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[mdmp]"
```

Run the open local Mistral model locally with Ollama:

```bash
python -m pip install -e ".[mdmp]"
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
- If Ollama closes the connection during generation, the SDK now surfaces an explicit recovery hint and points users toward `ministral-3:3b` for lower-memory systems.
- After `iints ai prepare`, you can point `iints ai explain|trends|anomalies|report` directly at the run directory.
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

Troubleshooting:
- If `iints ai ...` says `No such command 'ai'`, your environment usually still has a legacy `iints` package installed alongside `iints-sdk-python35`.
- Run `iints-sdk-doctor` first.
- If it reports a conflict, repair the environment with:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[mdmp]==1.1.3"
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
