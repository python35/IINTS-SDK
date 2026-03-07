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
pip install iints-sdk-python35
```

```bash
iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
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

## Dual Repo Workflow
- SDK repo: `python35/IINTS-SDK`
- MDMP repo: `python35/MDMP`

Local helper scripts:
- `tools/local/dual_repo_status.sh`
- `tools/local/dual_repo_commit_push.sh`

Full process: `docs/DUAL_REPO_WORKFLOW.md`

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
