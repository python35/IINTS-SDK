# IINTS-AF SDK Docs

IINTS-AF is a safety-first SDK for insulin algorithm simulation and validation.

This documentation site is organized for both technical and non-technical readers.

## Start Here

- If you are new: read `Plain Language Guide`.
- If you need architecture and CLI details: read `Technical README`.
- If you are preparing research/audit artifacts: read `MDMP (Draft)` and `Public Overview`.

## Core Workflows

1. Create a project scaffold:
   - `iints quickstart --project-name iints_quickstart`
   - or `iints init --project-name iints_trial --template clinical-trial`
2. Run a simulation and validate:
   - `iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready`
3. Validate dataset contracts:
   - `iints data contract-run data_contract.yaml data/my_cgm.csv --output-json results/contract_data_report.json`
4. Build audit dashboard:
   - `iints data mdmp-visualizer results/contract_data_report.json --output-html results/mdmp_dashboard.html`
5. Generate synthetic-safe mirror data:
   - `iints data synthetic-mirror data/my_cgm.csv data_contract.yaml --output-csv data/synthetic_mirror.csv --output-json results/synthetic_mirror_report.json`

## Safety and Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.

