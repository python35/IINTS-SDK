# Installation And Paths

This page is the simplest answer to two common questions:

1. "How do I install IINTS correctly?"
2. "From which folder am I supposed to run this command?"

## The Short Rule

- `iints ...` commands can run from any working folder once the SDK is installed.
- `pip install -e ".[...]"` must be run from the SDK repository root, where `pyproject.toml` lives.
- `./scripts/run_live_stage_demo.sh` and `./scripts/run_booth_demo.sh` belong to the SDK repository and resolve the repo root automatically.
- After `iints quickstart`, switch into the generated project folder before running project commands.

## Option 1: Install The Released SDK

This is the best path for most users.

You can run these commands from any folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"
```

Then verify:

```bash
iints doctor --smoke-run
python -c "import iints; print(iints.__version__)"
```

## Option 2: Install From Source

Use this only if you are developing the SDK itself.

First go to the repository root. That is the folder containing:

- `pyproject.toml`
- `src/`
- `scripts/`
- `examples/`

Then run:

```bash
cd /path/to/IINTS-SDK
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[mdmp]"
```

If you see:

```text
ERROR: ... does not appear to be a Python project
```

you are almost certainly not inside the repository root.

## Optional: Add Local Ollama AI

If you want the local research AI features, there is one extra component:

- the SDK itself
- a local Ollama server
- a local open Mistral-family model such as `ministral-3:8b` or `ministral-3:3b`

The SDK talks to Ollama over HTTP.
By default it expects:

```text
http://127.0.0.1:11434
```

### Small Setup Sequence

1. Install Ollama

On macOS/Linux, the quickest path is:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama -v
```

On Windows, install Ollama from the official download page first, then open a new terminal.

2. Start Ollama

```bash
ollama serve
```

If Ollama is already running as a background service on your machine, you do not need to start it again.

3. Pull a local model

Balanced default:

```bash
ollama pull ministral-3:8b
```

Lighter fallback for smaller machines:

```bash
ollama pull ministral-3:3b
```

4. Link Ollama to IINTS and verify

If you use the default local endpoint, the SDK finds Ollama automatically.
If you want to make the link explicit, set:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
```

Then verify the connection:

```bash
iints ai local-check --model ministral-3:8b
```

You can also override the endpoint per command:

```bash
iints ai local-check \
  --model ministral-3:8b \
  --ollama-host http://127.0.0.1:11434
```

Important:

- `OLLAMA_HOST` is the normal way to point the SDK at a non-default local Ollama endpoint.
- Remote Ollama endpoints are blocked by default for safety. Only enable them intentionally.
- If `ministral-3:8b` is unstable on your hardware, try `ministral-3:3b` first.
- Full AI usage guide: `docs/AI_ASSISTANT.md`

## Folder Map

There are three important places to keep straight:

### 1. SDK repository root

Example:

```text
/path/to/IINTS-SDK
```

Use this for:

- `python -m pip install -e ".[mdmp]"`
- `./scripts/run_live_stage_demo.sh`
- `./scripts/run_booth_demo.sh`
- opening `examples/demos/07_live_stage_demo.py`

### 2. Generated quickstart project

Created by:

```bash
iints quickstart --project-name iints_quickstart
```

Example:

```text
/path/to/where/you/running/iints_quickstart
```

Use this for:

- `iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py`
- editing `algorithms/example_algorithm.py`
- inspecting `results/`

### 3. Run bundle

A single simulation run ends up under something like:

```text
results/20260323-123456-abcdef12-1234/
```

That run bundle contains files such as:

- `results.csv`
- `clinical_report.pdf`
- `audit/`
- `baseline/`
- `run_manifest.json`
- `run_metadata.json`

## Fastest Working Flow

### Installed SDK flow

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[mdmp]"

iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

### Source repo flow

```bash
cd /path/to/IINTS-SDK
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[mdmp]"

./scripts/run_live_stage_demo.sh
```

## Booth Demo Paths

If you only installed the SDK and do not have the repository checkout, export the same showable booth code with:

```bash
iints demo-export --output-dir iints_demo
cd iints_demo
python 07_live_stage_demo.py
```

That writes:

- `07_live_stage_demo.py`
- `RUN_ME_FIRST.txt`

If you use:

```bash
./scripts/run_live_stage_demo.sh
```

the default output folder is:

```text
<repo-root>/results/booth_demo_live/
```

The most useful files there are:

- `booth_demo_poster.png`
- `JURY_TALK_TRACK.md`
- `BEURS_LIVE_DEMO_SCRIPT.txt`
- `run_commands.md`

The three scenario folders are:

- `01_normal_run/`
- `02_meal_stress_test/`
- `03_supervisor_override/`

## Quick Troubleshooting

### `iints ai` or `iints demo-booth` is missing

```bash
iints-sdk-doctor
```

If needed:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[mdmp]"
hash -r
```

### `pip install -e ".[mdmp]"` fails

Move into the SDK repository root first:

```bash
cd /path/to/IINTS-SDK
python -m pip install -e ".[mdmp]"
```

### Wrong Python version

Current releases require Python `>=3.10`.

Check it:

```bash
python --version
```
