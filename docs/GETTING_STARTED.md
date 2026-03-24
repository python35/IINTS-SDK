# Getting Started

This page gives the fastest reliable path from install to a validated run.

If you mainly need help with folder choice and install mode, read [Installation And Paths](INSTALLATION.md) first.

## 1) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

All commands below assume this `.venv` is active.

## 2) Install

```bash
python -m pip install -U "iints-sdk-python35[mdmp]"
```

Optional extras:

```bash
pip install "iints-sdk-python35[research]"
pip install "iints-sdk-python35[nightscout]"
```

## 3) Verify Environment

```bash
iints doctor --smoke-run
```

If this fails, fix environment issues before running long experiments.

## 4) Create a Project

```bash
iints quickstart --project-name iints_quickstart
cd iints_quickstart
```

Generated structure includes:
- `algorithms/`
- `scenarios/`
- `results/`

Important:
- before `iints quickstart`, commands can be run from any folder
- after `iints quickstart`, move into the generated project folder
- repo scripts such as `./scripts/run_live_stage_demo.sh` belong to the SDK repository, not the quickstart project

## 5) Run a Baseline Simulation

```bash
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

## 6) Check Outputs

A typical run writes:
- `results.csv`: time-series simulation output.
- `clinical_report.pdf`: report for review.
- `audit/`: decision and safety trail.
- `run_manifest.json`: file hashes for reproducibility.
- `run_metadata.json`: run config and environment details.

## 7) Build a Study-Ready Bundle

```bash
iints study-ready --algo algorithms/example_algorithm.py --output-dir results/study_ready
```

Adds:
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`

## 8) Next Steps

- Import a Medtronic CareLink / MiniMed CSV export:

```bash
iints import-carelink \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/imported_carelink
```

- Or build the full personal-data workspace at once:

```bash
iints carelink-workbench \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/personal_carelink
```

- Optional AI assistant (Ministral 3 via Ollama):

```bash
iints ai models
ollama pull ministral-3:8b
iints ai local-check --model ministral-3:8b
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
iints ai prepare results/<run_id>
iints ai report results/<run_id>
```

- CareLink + local AI flow:

```bash
ollama pull ministral-3:3b
iints ai local-check --model ministral-3:3b
iints carelink-workbench \
  --input-csv "/path/to/CareLink export.csv" \
  --output-dir results/personal_carelink
iints ai report results/personal_carelink --model ministral-3:3b
```

`iints ai local-check` now runs a tiny smoke-test generation by default. If it reports that Ollama closed the generation connection, the most likely causes are a restarting daemon or insufficient memory; in that case, try `ministral-3:3b`.

- Jury/demo poster from real run bundles:

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

- One-command booth / jury demo:

```bash
./scripts/run_live_stage_demo.sh
```

This is the easiest script to show on a fair stand because `examples/demos/07_live_stage_demo.py` exposes the patient profile and runtime knobs right at the top of the file.

It also visibly shows these SDK features in one place:

- `run_full(...)`
- `generate_results_poster(...)`
- `prepare_ai_ready_artifacts(...)`

You can still use the full booth bundle command:

```bash
./scripts/run_booth_demo.sh
```

This creates `results/booth_demo/` with three scenario runs, a poster PNG, and `JURY_TALK_TRACK.md`.

For a fair or jury table, the cleanest live flow is:

1. open `examples/demos/07_live_stage_demo.py`
2. point to the patient profile and runtime constants
3. run `./scripts/run_live_stage_demo.sh`
4. open the generated poster and scenario folders under `results/booth_demo_live/`

That wrapper script resolves the repository root automatically, so it is more forgiving than running raw relative Python commands by hand.

If the machine only has the installed SDK and not the repository checkout, export the same demo code first:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

- Updating an existing install to the latest release:

```bash
source .venv/bin/activate
python -m pip install -U "iints-sdk-python35[mdmp]"
hash -r
python -c "import iints; print(iints.__version__)"
```

- Full update guide: [Updating The SDK](UPDATING.md)
- Full install/path guide: [Installation And Paths](INSTALLATION.md)
- Need a fixed environment for a demo or paper? Pin a specific version only when reproducibility matters.

- Data validation: [MDMP Quickstart](MDMP_QUICKSTART.md)
- Full command reference: [Technical README](TECHNICAL_README.md)
- End-to-end examples: [Demos](https://github.com/python35/IINTS-SDK/tree/main/examples/demos)

## Safety Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
