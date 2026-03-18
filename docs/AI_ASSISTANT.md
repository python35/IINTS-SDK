# AI Assistant Guide

This guide explains how the local open-weight Ministral 3 AI layer works inside IINTS-AF.

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
- AI output is blocked unless MDMP verification succeeds first.

## What The AI Layer Does

The local AI assistant is designed for four narrow tasks:

- `explain`: explain a single simulation step in plain language
- `trends`: summarize glucose-oriented trends from a payload
- `anomalies`: call out unusual or safety-relevant patterns
- `report`: generate a short markdown run summary

The assistant is intentionally conservative. It explains simulation behavior; it does not produce treatment advice.

## Architecture

The flow is:

1. `iints ai ...` loads a JSON payload from disk.
2. `MDMPGuard` verifies the signed MDMP artifact and enforces the minimum grade.
3. `IINTSAssistant` selects the backend.
4. `OllamaBackend` checks that Ollama is reachable and that a local open Ministral 3 tag is installed.
5. The prompt is built from a fixed system instruction plus a serialized payload.
6. The response is wrapped with a hard-coded research-only disclaimer before output is shown or saved.

## Local Backend Behavior

The SDK defaults to local inference through Ollama.

Default model:

```bash
ministral-3:8b
```

Supported convenience aliases include:

- `ministral`
- `ministral-3`
- `ministral-3:8b`
- `ministral-8b`
- `ministral-8b-instruct`

If the alias is used, IINTS resolves it to the installed local Ollama tag before generation.

## Recommended Setup

Always work from an active virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[mdmp]"
```

Install and check the local model:

```bash
ollama pull ministral-3:8b
iints ai local-check --model ministral-3:8b
```

If the model is missing, the command fails with the exact `ollama pull ...` command to run next.
If your Ollama runtime is too old for the open Ministral 3 line, `local-check` now tells you that as well.
`local-check` also runs a tiny generation smoke-test by default, so it catches the common case where `/api/tags` works but the model crashes during real inference.
The current Ollama listing for `ministral-3` expects Ollama `0.13.1` or newer.

## Hardware Recommendations

Use this as a practical starting point:

| Model | Good Fit | Recommended System RAM | Recommended GPU VRAM | Approx Download |
|---|---|---:|---:|---:|
| `ministral-3:3b` | smaller laptops, CPU-first setups, entry-level edge boxes | 16 GB | 6 GB | ~3 GB |
| `ministral-3:8b` | balanced desktop or strong laptop | 24 GB | 10 GB | ~6 GB |
| `ministral-3:14b` | high-end workstation | 32 GB | 16 GB | ~10 GB |

General advice:

- Start with `ministral-3:8b` unless you have a specific reason to go smaller or larger.
- Choose `ministral-3:3b` if latency and memory matter more than answer quality.
- Choose `ministral-3:14b` only if your machine can comfortably absorb the extra RAM and latency.
- Run `iints ai models` in the CLI to see the same recommendations in a terminal-friendly table.
- If `ministral-3:8b` closes the connection during generation, try `ministral-3:3b` first before assuming something is wrong with the SDK.

## Recommended Workflow

After a run completes, prepare the run directory once:

```bash
iints ai prepare results/<run_id>
```

That command creates:

- `ai/report_payload.json`
- `ai/anomalies_payload.json`
- `ai/trends_payload.json`
- `ai/step_riskiest.json`
- `ai/step_latest.json`
- `ai/report.signed.mdmp` plus `ai/keys/` when the MDMP extra is installed

After that, you can point the AI commands directly at the run directory:

```bash
iints ai explain results/<run_id>
iints ai trends results/<run_id>
iints ai anomalies results/<run_id>
iints ai report results/<run_id> --output results/<run_id>/ai/ai_report.md
```

## Generation Commands

Prepared run directory mode:

```bash
iints ai explain results/<run_id>
iints ai trends results/<run_id>
iints ai anomalies results/<run_id>
iints ai report results/<run_id> --output results/<run_id>/ai/ai_report.md
```

Direct JSON mode:

```bash
iints ai explain results/step.json \
  --mdmp-cert results/report.signed.mdmp

iints ai trends results/glucose_payload.json \
  --mdmp-cert results/report.signed.mdmp

iints ai anomalies results/simulation_run.json \
  --mdmp-cert results/report.signed.mdmp

iints ai report results/simulation_run.json \
  --mdmp-cert results/report.signed.mdmp \
  --output results/ai_report.md
```

Useful options:

- `--mode local` to require Ollama explicitly
- `--model ministral-3:8b` or `--model ministral`
- `--model ministral-3:3b` for lighter machines
- `--model ministral-3:14b` for stronger workstations
- `--ollama-host http://127.0.0.1:11434` to override the endpoint
- `--timeout-seconds 120` for slower local hardware
- `--minimum-grade research_grade` to raise or lower the MDMP floor

## How Reliability Is Enforced

For local robustness, the SDK now does four checks before a real generation call:

- verifies that the Ollama HTTP endpoint is reachable
- verifies that a compatible local open Ministral 3 model is installed
- normalizes common local model aliases to the installed tag
- truncates oversized JSON payloads before prompt construction so large run artifacts do not overwhelm local inference
- flags too-old Ollama runtimes when they do not meet the minimum version expected for the open Ministral 3 line

If a generation succeeds, the response records the actual resolved model name used by the local backend.

## MDMP Guard Behavior

The AI assistant does not run on unsigned or insufficiently graded artifacts.

The guard enforces:

- signed MDMP verification
- minimum grade threshold
- hard-coded disclaimer injection on every response

That means the research-only warning is not dependent on the prompt and cannot be removed by changing prompt text alone.

## Troubleshooting

### Ollama Not Reachable

Run:

```bash
iints ai local-check --model ministral-3:8b
```

This now checks both basic reachability and a tiny real generation.

If the endpoint is wrong, retry with:

```bash
iints ai local-check --model ministral-3:8b --ollama-host http://127.0.0.1:11434
```

### Model Missing

Pull the model shown in the error output:

```bash
ollama pull ministral-3:8b
```

### Local Inference Is Slow

Increase the timeout:

```bash
iints ai report results/simulation_run.json \
  --mdmp-cert results/report.signed.mdmp \
  --timeout-seconds 180
```

If the server disconnects instead of timing out, the model may be too heavy for the machine at that moment. In that case try:

```bash
ollama pull ministral-3:3b
iints ai local-check --model ministral-3:3b
iints ai report results/<run_id> --model ministral-3:3b
```

### Large Run Payloads

The assistant now clips oversized payloads automatically before sending them to the model. If you want tighter control, pass a smaller JSON summary rather than a full raw run dump.

### No `report.signed.mdmp` In My Run Folder

That is now expected for a fresh raw run. The easiest fix is:

```bash
iints ai prepare results/<run_id>
```

This creates a local development certificate for AI use when the MDMP extra is available, so you do not have to hand-build `step.json` and `report.signed.mdmp` yourself.

### `No such command 'ai'`

If the CLI says `No such command 'ai'`, the most common cause is a legacy `iints` package still being installed beside `iints-sdk-python35`. That older package can shadow the newer SDK command tree.

Run the install doctor:

```bash
iints-sdk-doctor
```

If it reports a package ownership conflict, repair the environment:

```bash
python -m pip uninstall -y iints iints-sdk-python35
python -m pip install -U "iints-sdk-python35[mdmp]==1.1.2"
hash -r
```

Then retry:

```bash
iints ai models
iints ai local-check --model ministral-3:8b
```
