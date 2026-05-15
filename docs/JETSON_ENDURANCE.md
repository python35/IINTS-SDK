# Jetson Endurance Mode

Jetson Endurance Mode turns an NVIDIA Jetson into a headless scientific stress-test machine. It is for long adversarial runs, not for dashboards or stage demos.

Configure it once, let the study run, and collect one reproducible results folder at the end.

## When To Use

Use this mode when you want to test safety behavior under long-running adversarial physiological scenarios:

- overnight sanity checks
- weekend stress tests
- 7-day EUCYS-style robustness studies
- multi-week regression studies on edge hardware

For Raspberry Pi demos and realistic long-study behavior, use `iints edge long-study`. Jetson Endurance Mode is more aggressive: it targets supervisor limits, sensor failures, double meals, exercise hypoglycemia, and cold-start behavior.

## What You Get

- headless long-run execution
- hardware monitoring
- daily summaries and snapshots
- worst-case event logs
- publication-ready export artifacts

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

Durations are converted to simulation minutes. With a 5-minute step size:

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
  daily/
    day_01_summary.json
    day_02_summary.json
  snapshots/
    snapshot_24h.json
    snapshot_48h.json
  final/
    test_summary.json
    tir_timeseries.csv
    supervisor_analysis.json
    worst_case_events.json
    ENDURANCE_REPORT.md
    ENDURANCE_REPORT.pdf
    main_figure.png
```

Important files:

- `protocol/test_config.yaml` records exact settings and generated stress events.
- `protocol/hardware_info.json` records platform, thermal, CUDA, and Jetson probe data.
- `raw/steps.csv` contains every simulation step.
- `raw/interventions.csv` contains safety-supervisor interventions.
- `raw/critical_events.csv` contains glucose values below 54 mg/dL.
- `final/test_summary.json` contains TIR, confidence interval, failure-rate proxy, and performance metrics.
- `final/ENDURANCE_REPORT.md` is the human-readable summary for review.
- `final/main_figure.png` is the main glucose trace figure.

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

The generated service uses `--resume`, so a restarted run continues from the latest snapshot when possible.

## Scientific Claim Pattern

A useful EUCYS-style summary should be concrete:

```text
We ran a 7-day continuous adversarial stress test on an NVIDIA Jetson,
executing 2016 simulation steps with mixed adversarial scenario classes.
The deterministic safety supervisor intercepted unsafe decisions, and all
critical events were recorded in the reproducible evidence bundle.
```

Always report the exact numbers from `final/test_summary.json`; do not hard-code claims before the run is complete.
