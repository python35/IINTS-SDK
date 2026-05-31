# Mistral Model Migration

Mistral is retiring several older Serverless API model IDs in 2026. IINTS keeps the local AI path on open local Ollama models, but the SDK now documents the safe cloud-side replacements so external scripts and future API integrations do not keep pointing at deprecated IDs.

!!! warning "IINTS scope"
    IINTS local AI remains research-only and MDMP-gated. These model names are for explanation, review, and documentation workflows, not medical dosing authority.

## Current Defaults

| Use | Model | Extra setting |
| --- | --- | --- |
| Local SDK default | `ministral-3:8b` through Ollama | local only |
| Smaller local fallback | `ministral-3:3b` through Ollama | local only |
| Cloud small/reasoning replacement | `mistral-small-latest` | `reasoning_effort="high"` |
| Cloud stronger replacement | `mistral-medium-3-5` | `reasoning_effort="high"` |
| OCR replacement | `mistral-ocr-latest` | n/a |
| Moderation replacement | `mistral-moderation-2603` | n/a |
| Transcription replacement | `voxtral-mini-latest` | n/a |

## Migration Table

| Deprecated model ID | Replacement |
| --- | --- |
| `devstral-small-2507`, `devstral-small-latest` | `mistral-small-latest` with `reasoning_effort="high"` |
| `devstral-medium-2507`, `devstral-2512`, `devstral-latest` | `mistral-medium-3-5` with `reasoning_effort="high"` |
| `magistral-small-2509` | `mistral-small-latest` with `reasoning_effort="high"` |
| `magistral-medium-2509`, `magistral-medium-latest` | `mistral-medium-3-5` with `reasoning_effort="high"` |
| `mistral-large-2411`, `pixtral-large-2411`, `mistral-medium-2505`, `mistral-medium-2508` | `mistral-medium-3-5` |
| `mistral-small-2506` | `mistral-small-latest` |
| `open-mistral-nemo-2407` | `ministral-8b-2512` for Serverless API use |
| `mistral-ocr-2505` | `mistral-ocr-latest` |
| `mistral-moderation-2411`, `mistral-moderation-latest` | `mistral-moderation-2603` |
| `voxtral-mini-2507` | `voxtral-mini-latest` |

## SDK Behavior

The local SDK path is not affected by the Serverless API retirement because `iints ai ...` defaults to local Ollama:

```bash
iints ai local-check --model ministral-3:8b
```

The SDK now also exposes migration metadata through:

```bash
iints ai models
```

If a future cloud backend receives a deprecated Serverless model ID, the catalog maps it to the replacement model before it is used. Cloud fallback is still intentionally disabled in this SDK build; external Mistral clients should use the replacement IDs above.

## Official Mistral References

- Mistral adjustable reasoning: `mistral-small-latest` and `mistral-medium-3-5` support `reasoning_effort`.
- Mistral Small 4 model card: `mistral-small-2603`, exposed through `mistral-small-latest`.
- Mistral Medium 3.5 model card: `mistral-medium-3-5`.
