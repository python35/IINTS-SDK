# Installation And Paths

Use this page when the problem is not the science yet, but the environment:

1. how to install the SDK correctly
2. which folder a command belongs in

**Read before:** [Quickstart](QUICKSTART.md) if you only need the fastest first run.

**Read next:** [Getting Started](GETTING_STARTED.md) for the full first workflow, or [Troubleshooting](TROUBLESHOOTING.md) if something failed.

## Choose The Install Path

| If you are... | Use |
| --- | --- |
| trying the SDK on a normal laptop or workstation | released `full` install |
| preparing Raspberry Pi or UNO Q hardware | lighter `edge` install |
| modifying the SDK itself | source install from the repository root |

## The Short Rule

- `iints ...` commands can run from any working folder once the SDK is installed.
- `iints patient ...` commands can also run from any working folder once the SDK is installed.
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
python -m pip install -U "iints-sdk-python35[full,mdmp]"
```

Then verify:

```bash
iints doctor --smoke-run
python -c "import iints; print(iints.__version__)"
```

## Option 2: Install The Edge Runtime Profile

For Raspberry Pi or UNO Q style deployments, use the lighter edge profile:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"
```

Use this when you mainly need:

- `iints patient ...`
- `iints edge ...`
- local FastAPI dashboard
- SQLite runtime state
- CLI control
- UNO Q serial bridge support
- optional local AI review

Security defaults in this profile:

- the digital-patient dashboard API stays on loopback by default
- remote API binding now requires both `--allow-remote-api` and a bearer token source
- public dataset downloads without a published SHA-256 now require `--no-verify` so the CLI does not pretend to verify an unknown checksum

Important for UNO Q users:

- if `iints edge ...` says `No such command 'edge'`, your installed CLI is older than the current docs
- in that case, use the source install method below instead of the released package path
- the dedicated UNO Q guide follows the source-install path on purpose so the commands match exactly

Typical SBC bootstrap:

```bash
iints edge quickstart --board raspberry_pi
cd iints_pi_demo
iints edge status --project-dir .
```

UNO Q bootstrap:

```bash
iints edge quickstart --board uno_q
cd iints_uno_q_demo
```

Then upload `uno_q_bridge/iints_supervisor_bridge.ino` once and use `./test_uno_q_bridge.sh`.

## Option 3: Install From Source

Use this only if you are developing the SDK itself.

Exception:

- for `Arduino UNO Q`, this is currently the recommended path because it guarantees the `iints edge ...` commands match the docs

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
python -m pip install -U -e ".[full,mdmp]"
```

If you want the lighter edge runtime from source instead:

```bash
python -m pip install -U -e ".[edge,mdmp]"
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

## Optional: Turn A Raspberry Pi Into A Live Digital Patient

If you want a persistent expo or classroom rig, the SDK now includes:

- `iints patient start`
- `iints patient status`
- `iints patient inject-meal`
- `iints patient expo-reset`
- `iints patient review`

Recommended setup:

- Raspberry Pi 5
- Raspberry Pi OS Desktop (Bookworm or newer)
- Raspberry Pi Connect enabled

Fastest flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "iints-sdk-python35[edge,mdmp]"

iints quickstart --project-name iints_pi_demo
cd iints_pi_demo
iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x
```

Then open:

```text
http://127.0.0.1:8765/dashboard
```

Use Raspberry Pi Connect screen sharing from your laptop to present the live dashboard.

Remote presentation note:

- Raspberry Pi Connect does **not** require `--allow-remote-api`
- the safest demo path is still to keep the API on `127.0.0.1`
- only use remote API binding if another machine must talk to the dashboard directly

If you really do need a remote API bind, use a token-backed start command:

```bash
export IINTS_PATIENT_API_TOKEN="replace-this-with-a-random-secret"

iints patient start \
  --algo algorithms/example_algorithm.py \
  --workspace patient_runtime \
  --scenario-profile normal_day \
  --mode demo-time \
  --speed 60x \
  --api-host 0.0.0.0 \
  --allow-remote-api \
  --api-token-env IINTS_PATIENT_API_TOKEN
```

When token protection is enabled:

- browser reads of `/dashboard`, `/kiosk`, `/status`, and glucose history also require the token
- the simplest browser form is `http://<host>:8765/dashboard?token=<your-token>`
- command-line or scripted access should prefer `Authorization: Bearer <token>`

## Dataset Fetch Verification

`iints data fetch` is stricter now.

If a public source does **not** publish a pinned SHA-256, the SDK will no longer call that a verified download.

So this can now happen intentionally:

```bash
iints data fetch aide_t1d --output-dir data_packs/public/aide_t1d
```

If the registry entry has no pinned hash yet, the secure fallback is:

```bash
iints data fetch aide_t1d \
  --output-dir data_packs/public/aide_t1d \
  --no-verify
```

Use `--no-verify` only when:

- you trust the upstream source
- the dataset entry still lacks a published checksum
- you understand this is a trust decision, not a cryptographic verification

The long-term fix is to add a pinned SHA-256 to `src/iints/data/datasets.json`.

## Safer Nightscout And Tidepool Secrets

Prefer environment variables or files over plain CLI secrets.

Nightscout:

```bash
export IINTS_NIGHTSCOUT_SECRET="replace-me"
export IINTS_NIGHTSCOUT_TOKEN="replace-me"

iints import-nightscout \
  --url https://your-nightscout.example \
  --api-secret-env IINTS_NIGHTSCOUT_SECRET \
  --token-env IINTS_NIGHTSCOUT_TOKEN \
  --output-dir results/nightscout_import
```

Tidepool:

```bash
export IINTS_TIDEPOOL_TOKEN="replace-me"

iints import-tidepool \
  --base-url https://api.tidepool.org \
  --token-env IINTS_TIDEPOOL_TOKEN \
  --output-dir results/tidepool_import
```

Plain `--token` and `--api-secret` still work for compatibility, but the CLI now warns because those values can leak into shell history and process lists.

If this Pi will be left running unattended, export a ready-made systemd unit after the first start:

```bash
iints edge service --project-dir .
```

Full guide:

- `docs/DIGITAL_PATIENT_PI.md`

## Folder Map

There are three important places to keep straight:

### 1. SDK repository root

Example:

```text
/path/to/IINTS-SDK
```

Use this for:

- `python -m pip install -e ".[full,mdmp]"`
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

- `iints run --algo algorithms/example_algorithm.py --patient-config-path patients/stable_patient.yaml --scenario-path scenarios/clinic_safe_baseline.json --duration 1440`
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
python -m pip install -U "iints-sdk-python35[full,mdmp]"

iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints run --algo algorithms/example_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440
```

### Source repo flow

```bash
cd /path/to/IINTS-SDK
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U -e ".[full,mdmp]"

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
python -m pip install -U "iints-sdk-python35[full,mdmp]"
hash -r
```

### `pip install -e ".[full,mdmp]"` or `pip install -e ".[edge,mdmp]"` fails

Move into the SDK repository root first:

```bash
cd /path/to/IINTS-SDK
python -m pip install -e ".[full,mdmp]"
```

### Wrong Python version

Current releases require Python `>=3.10`.

Check it:

```bash
python --version
```

## Where To Go Next

| If you installed... | Continue with |
|---|---|
| the full SDK | [Getting Started](GETTING_STARTED.md) |
| the edge profile | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |
| local AI extras | [AI Assistant](AI_ASSISTANT.md) |
| data certification tools | [MDMP Quickstart](MDMP_QUICKSTART.md) |
| but something failed | [Troubleshooting](TROUBLESHOOTING.md) |
