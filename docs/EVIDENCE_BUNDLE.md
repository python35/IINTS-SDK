# Research Evidence Bundle

Use this page when you need a public, inspectable folder that explains what an IINTS run proves.

The evidence bundle is designed for demos, EUCYS-style project review, engineering handoff, and reproducibility checks. It is not clinical evidence and must not be presented as medical-device validation.

## What It Builds

```bash
iints evidence build \
  --run normal=results/live_demo/results/01_normal_run \
  --run stress=results/live_demo/results/02_meal_stress \
  --run supervisor=results/live_demo/results/03_supervisor_override \
  --output-dir results/live_demo/evidence_bundle
```

The output folder contains:

| File | Meaning |
| --- | --- |
| `README.md` | Human-readable summary of the attached runs |
| `MODEL_CARD.md` | Research-only AI/model framing and safety gate summary |
| `evidence_summary.json` | Machine-readable evidence index |
| `run_index.csv` | Table with rows, mean glucose, TIR, and hypo percentages |
| `runs/*/RUN_CARD.json` | Per-run artifact and metric card |
| `runs/*/artifacts/` | Copied CSVs, manifests, reports, and safety JSON where available |

## Live Demo Integration

The easiest path is still:

```bash
iints demo --audience jury --output-dir results/live_demo
```

By default this now builds:

```text
results/live_demo/
  PRESENTER_GUIDE.md
  DEMO_CUE_CARD.md
  DEMO_ARTIFACTS.md
  DEMO_RUN_LOG.txt
  evidence_bundle/
```

If you need a faster rehearsal without the evidence folder:

```bash
iints demo --audience jury --output-dir results/live_demo --no-evidence
```

## Add Local AI Evidence

After training a local AI lab:

```bash
iints research train-local-ai \
  --run day1=results/jetson_research_day \
  --run day2=results/jetson_research_day_2 \
  --output-dir results/local_ai_lab
```

Attach that evidence to the public bundle:

```bash
iints evidence build \
  --run day1=results/jetson_research_day \
  --run day2=results/jetson_research_day_2 \
  --local-ai-dir results/local_ai_lab \
  --output-dir results/public_evidence
```

The model card will include the local AI safety gate status when `LOCAL_AI_RESEARCH_SUMMARY.json` is present.

## Add Pico Pump Bench Evidence

After compiling and bench-testing a Pico bundle:

```bash
iints pump compile \
  --algorithm algorithms/pico_bench_algorithm.py \
  --output-dir bundles/pico_bench_bundle

iints pump bench-test \
  --bundle-dir bundles/pico_bench_bundle \
  --output-json bundles/pico_bench_bundle/bench_test_report.json
```

Attach the pump bundle:

```bash
iints evidence build \
  --run one_day=results/one_day_sim \
  --pump-bundle-dir bundles/pico_bench_bundle \
  --output-dir results/public_evidence
```

## Required Wording

Use this exact framing in public demos:

> This is a research and educational evidence bundle. It shows reproducibility, software behavior, and safety-supervisor boundaries. It is not clinical evidence, not a certified medical-device submission, and not dosing advice.

## Best Practice

- Run `iints run-doctor` before long simulations.
- Keep every run folder with its `run_manifest.json`.
- Attach local AI summaries only after safety gates run.
- Attach Pico pump bundles only after `iints pump bench-test` passes.
- Present the evidence bundle next to the normal docs site or public Obsidian vault, not as a replacement for scientific validation.
