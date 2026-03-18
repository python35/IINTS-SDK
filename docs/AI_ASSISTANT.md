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
The current Ollama listing for `ministral-3` expects Ollama `0.13.1` or newer.

## Generation Commands

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

### Large Run Payloads

The assistant now clips oversized payloads automatically before sending them to the model. If you want tighter control, pass a smaller JSON summary rather than a full raw run dump.
