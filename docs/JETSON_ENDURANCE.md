# Jetson Endurance Mode

Jetson Endurance Mode turns an NVIDIA Jetson into a headless scientific study machine. It now supports two deliberately different modes:

- `accelerated` endurance: many simulated hours as fast as the device can compute them
- `wall-clock` research: one simulated minute follows one real minute, so `1d` genuinely occupies about 24 hours

Configure it once, let the study run, and collect one reproducible results folder at the end.

## When To Use

Use accelerated mode when you want to test safety behavior under long-running adversarial physiological scenarios:

- overnight sanity checks
- weekend stress tests
- 7-day EUCYS-style robustness studies
- multi-week regression studies on edge hardware

Use wall-clock mode when you want to operate the Jetson as an actual research station:

- a 24-hour acquisition run that really lasts 24 hours
- repeated hardware telemetry over the same real period as the simulated study
- a training-ready dataset export after the run
- clean separation between data acquisition and later predictor training

For Raspberry Pi demos and realistic long-study behavior, use `iints edge long-study`. Jetson Endurance Mode is more aggressive: it targets supervisor limits, sensor failures, double meals, exercise hypoglycemia, and cold-start behavior.

## What You Get

- headless long-run execution
- hardware monitoring
- resumable checkpoints, partial CSV persistence, and daily summaries
- periodic hardware telemetry samples during the run
- worst-case event logs
- publication-ready export artifacts
- a `research/` bundle with predictor-training rows and lineage metadata

## Quick Start

```bash
iints jetson doctor

iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --predictor models/lstm_predictor.pt \
  --duration 7d \
  --output-dir results/jetson_7day \
  --profile mixed_adversarial \
  --seed 42
```

For a true one-day research run that occupies one real day:

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --predictor models/lstm_predictor.pt \
  --duration 1d \
  --output-dir results/jetson_research_day \
  --profile normal \
  --wall-clock \
  --seed 42
```

If you want the machine to complete the full post-run research loop automatically, add `--finalize-research`:

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --duration 1d \
  --output-dir results/jetson_research_day \
  --profile normal \
  --wall-clock \
  --finalize-research
```

Check progress:

```bash
iints jetson endurance status --output-dir results/jetson_7day
iints jetson endurance monitor --output-dir results/jetson_7day --watch
```

Stop safely:

```bash
iints jetson endurance stop \
  --output-dir results/jetson_7day \
  --generate-report
```

Export the full evidence bundle:

```bash
iints jetson endurance export \
  --output-dir results/jetson_7day \
  --output results/jetson_7day_export.zip
```

## Durations

Durations always describe the **study horizon**. The execution mode controls how quickly that horizon is traversed:

| Mode | Meaning of `--duration 1d` |
|---|---|
| default accelerated mode | simulate 1440 minutes as fast as possible |
| `--wall-clock` mode | spend about 24 real hours executing the 1440-minute study |

With a 5-minute step size:

| Duration | Steps |
|---|---:|
| `1h` | 12 |
| `6h` | 72 |
| `24h` | 288 |
| `3d` | 864 |
| `7d` | 2016 |
| `14d` | 4032 |
| `30d` | 8640 |

Supported units are `m`, `h`, `d`, and `w`.

## Profiles

| Profile | Purpose |
|---|---|
| `normal` | Regular meal patterns. |
| `stress` | Normal days plus exercise and under-reported carbohydrates. |
| `adversarial` | Rotating edge cases such as dropout, artifact storms, and IOB pressure. |
| `mixed_adversarial` | Recommended default; combines normal, stress, and adversarial events. |
| `sensor_failure` | Repeated sensor artifacts and implausible CGM values. |
| `nighttime_risk` | Night drift and low-glucose risk during sleep. |
| `custom` | YAML-defined stress events. |

Custom profile example:

```yaml
stress_events:
  - start_time: 120
    event_type: meal
    value: 80
    reported_value: 45
    absorption_delay_minutes: 20
  - start_time: 180
    event_type: sensor_error
    value: 250
```

Run it with:

```bash
iints jetson endurance start \
  --algo algorithms/example_algorithm.py \
  --duration 24h \
  --profile custom \
  --custom-profile profiles/my_jetson_profile.yaml \
  --output-dir results/custom_jetson
```

## Output Layout

```text
results/jetson_7day/
  protocol/
    test_config.yaml
    hardware_info.json
  raw/
    steps.csv
    interventions.csv
    critical_events.csv
    hardware_metrics.csv
  daily/
    day_01_summary.json
    day_02_summary.json
  snapshots/
    snapshot_000360m.json
    snapshot_000720m.json
  final/
    test_summary.json
    tir_timeseries.csv
    supervisor_analysis.json
    worst_case_events.json
    ENDURANCE_REPORT.md
    ENDURANCE_REPORT.pdf
    main_figure.png
  research/
    predictor_training.csv
    controller_teacher_dataset.csv
    training_manifest.json
    README.md
    models/
      linear_controller.json
      neural_controller.pt
      predictor/
    evaluation/
      closed_loop_runs.csv
      closed_loop_summary.json
      CONTROL_EVALUATION_REPORT.md
    research_pipeline_summary.json
    RESEARCH_PIPELINE_REPORT.md
```

Important files:

- `protocol/test_config.yaml` records exact settings and generated stress events.
- `protocol/hardware_info.json` records platform, thermal, CUDA, and Jetson probe data.
- `raw/steps.csv` contains every simulation step.
- `raw/interventions.csv` contains safety-supervisor interventions.
- `raw/critical_events.csv` contains glucose values below 54 mg/dL.
- `raw/hardware_metrics.csv` stores periodic thermal/CUDA probe snapshots.
- `final/test_summary.json` contains TIR, confidence interval, failure-rate proxy, and performance metrics.
- `final/ENDURANCE_REPORT.md` is the human-readable summary for review.
- `final/main_figure.png` is the main glucose trace figure.
- `research/predictor_training.csv` is a standardized dataset slice for the glucose-predictor training pipeline.
- `research/controller_teacher_dataset.csv` contains safety-supervised insulin-action labels for controller-policy research.
- `research/training_manifest.json` records lineage, columns, and a reproducible example training command.
- `research/RESEARCH_PIPELINE_REPORT.md` records what the automatic post-run training/evaluation step actually produced.

## Research Mode Versus AI Training

This distinction matters:

| Component | Current role |
|---|---|
| Jetson wall-clock mode | acquires a real-duration study bundle |
| post-run research finalizer | trains local models and evaluates them after acquisition |
| predictor-training pipeline | trains or fine-tunes the glucose forecasting model from exported rows |
| Ministral / Ollama local AI | explains, reviews, and summarizes runs |

The current SDK does **not** fine-tune Ministral online during the same run. That would mix language-model adaptation with physiological control research and is not a clean design. Instead, the endurance runner exports physiological datasets, and the post-run finalizer trains the relevant numeric models after acquisition is complete.

Example:

```bash
PYTHONPATH=src python3 research/train_predictor.py \
  --data results/jetson_research_day/research/predictor_training.csv \
  --config research/configs/predictor_multimodal_dual_guard.yaml \
  --out models/jetson_research_day_predictor
```

Controller imitation baseline:

```bash
iints research train-controller \
  --data results/jetson_research_day/research/controller_teacher_dataset.csv \
  --output models/jetson_research_day_controller.json
```

Full post-run bundle finalization:

```bash
iints jetson endurance finalize-research \
  --output-dir results/jetson_research_day
```

That command writes:

- a linear imitation controller
- a PyTorch neural controller
- a predictor model when the exported time-series is long enough
- a held-out closed-loop comparison against the clinical baseline
- `research/RESEARCH_PIPELINE_REPORT.md`

For a stronger research workflow across several acquisition days, use the local AI lab:

```bash
iints research local-ai-lab \
  --run day1=results/jetson_research_day \
  --run day2=results/jetson_research_day_2 \
  --output-dir results/local_ai_lab
```

That command creates one combined training workspace with:

- merged predictor rows for glucose forecasting
- merged controller-teacher rows for policy learning
- `LOCAL_AI_DATASET_CARD.json` with source lineage
- linear and optional neural controller models
- optional predictor training
- held-out controller evaluation

If a future project adds Ministral fine-tuning, that should be a separate explicit pipeline with its own dataset governance and validation, not an invisible side effect of the endurance runner.

## Systemd Service

For runs longer than 24 hours, generate a service file:

```bash
iints jetson endurance install-service \
  --algo algorithms/example_algorithm.py \
  --predictor models/lstm_predictor.pt \
  --duration 7d \
  --output-dir results/jetson_7day
```

Then install it on the Jetson:

```bash
sudo cp results/jetson_7day/protocol/iints-jetson-endurance.service /etc/systemd/system/iints-jetson-endurance.service
sudo systemctl daemon-reload
sudo systemctl enable iints-jetson-endurance
sudo systemctl start iints-jetson-endurance
```

The generated service uses `--resume`, so a restarted run continues from the
latest checkpoint when possible. `status.json` now also records the last
checkpoint minute, resume count, elapsed wall-clock time, and estimated wall
time remaining.

For a real-time research service, add `--wall-clock` when generating the unit:

```bash
iints jetson endurance install-service \
  --algo algorithms/example_algorithm.py \
  --duration 1d \
  --output-dir results/jetson_research_day \
  --wall-clock
```

## Scientific Claim Pattern

A useful EUCYS-style summary should be concrete:

```text
We ran a 7-day continuous adversarial stress test on an NVIDIA Jetson,
executing 2016 simulation steps with mixed adversarial scenario classes.
The deterministic safety supervisor intercepted unsafe decisions, and all
critical events were recorded in the reproducible evidence bundle.
```

Always report the exact numbers from `final/test_summary.json`; do not hard-code claims before the run is complete.

## AI Red-Team Realism Audit

After an endurance run, you can scan the generated CSV for impossible physiology and optionally ask local Ollama/Ministral to explain only the suspicious window:

```bash
python3 -m iints.tools.ai_realism_auditor \
  results/jetson_7day/raw/steps.csv \
  --report results/jetson_7day/final/AI_REALISM_AUDIT.md \
  --no-ai
```

For the full educational AdvancedMetabolicModel stress demo with FFA and ketones:

```bash
PYTHONPATH=src python3 examples/jetson_endurance_test.py \
  --days 14 \
  --output results/red_team/endurance_data.csv \
  --inject-demo-glitch

PYTHONPATH=src python3 -m iints.tools.ai_realism_auditor \
  results/red_team/endurance_data.csv \
  --report results/red_team/AI_REALISM_AUDIT.md \
  --no-ai
```

See `AI_RED_TEAM_AUDITOR.md` for the full workflow and LaTeX model equations.

