# Raspberry Pi Pico Pump Lab

!!! warning "Bench-only research workflow"
    This page is for algorithm development, simulation, and disconnected hardware-in-the-loop tests. It is not a medical-device workflow, not an insulin delivery system, and must not be connected to a person, animal, insulin reservoir, or infusion set.

The Pico Pump Lab gives the SDK one clear path from an algorithm idea to a safe bench target:

1. write or import an IINTS algorithm
2. test it in simulation and validation profiles
3. package a reproducible bench bundle
4. copy locked firmware to a Raspberry Pi Pico-style board
5. verify the board over USB serial

The generated firmware is deliberately non-actuating. It accepts only `PING`, `STATUS`, and `LOCKOUT`; it rejects `DOSE`, `BOLUS`, `BASAL`, and `PRIME`.

## Why This Exists

If you are designing a PCB or pump-like demonstrator, the SDK should be the place where the risky logic is tested first. The hardware should only receive a bundle after the algorithm has a manifest, source snapshot, safety contract, and reproducibility metadata.

The FDA notes that infusion pump software controls important safety functions and that model-based and early verification methods help evaluate software before finished hardware exists. IINTS follows that idea for research: simulate first, package evidence second, touch hardware last.

## Create A Lab

```bash
iints pump init --output-dir iints_pico_pump_lab
cd iints_pico_pump_lab
```

This creates:

| Path | Purpose |
| --- | --- |
| `algorithms/pico_bench_algorithm.py` | Conservative starter SDK algorithm |
| `safety_contract.json` | Zero-delivery bench contract |
| `firmware/pico_pump_bench/code.py` | Non-actuating Pico/CircuitPython firmware |
| `bundles/` | Packaged algorithm/firmware bundles |
| `scripts/package_bench_bundle.sh` | One-command packaging helper |
| `README.md` | Local workflow notes |

## Simulate First

Use the normal SDK run path before packaging for hardware:

```bash
iints run \
  --algo algorithms/pico_bench_algorithm.py \
  --patient-config-path patients/stable_patient.yaml \
  --scenario-path scenarios/clinic_safe_baseline.json \
  --duration 1440 \
  --output-dir evidence/one_day_sim
```

Then validate the result:

```bash
iints validate-run \
  --run-dir evidence/one_day_sim \
  --profile clinic_safe_demo
```

## Package A Bench Bundle

```bash
iints pump compile \
  --algorithm algorithms/pico_bench_algorithm.py \
  --output-dir bundles/pico_bench_bundle \
  --safety-contract safety_contract.json
```

The bundle contains:

| File | Meaning |
| --- | --- |
| `algorithm.py` | Algorithm source snapshot |
| `firmware/code.py` | Locked USB serial firmware |
| `safety_contract.json` | Contract with `hardware_actuation_enabled=false` |
| `manifest.json` | Hashes, creation time, target, and upload confirmation |

## Bench-Test Before Upload

```bash
iints pump bench-test \
  --bundle-dir bundles/pico_bench_bundle \
  --output-json bundles/pico_bench_bundle/bench_test_report.json
```

This checks:

| Check | Meaning |
| --- | --- |
| manifest | Bundle metadata exists and targets `raspberry_pi_pico_bench` |
| algorithm | Algorithm source snapshot is present |
| firmware | Locked non-actuating firmware is present |
| safety_contract | Zero-delivery contract is present |
| algorithm_sha256 | Algorithm hash matches the manifest |
| contract_lockout | `hardware_actuation_enabled=false` and dose limits remain zero |

Optionally add `--port /dev/ttyACM0` after upload to run the serial `PING`, `STATUS`, and `LOCKOUT` smoke test.

## Upload To A Pico-Style Board

First do a dry run:

```bash
iints pump upload \
  --bundle-dir bundles/pico_bench_bundle \
  --mount-dir /Volumes/CIRCUITPY \
  --bench-only-confirm "I understand this is bench-only and not for human use"
```

If the copy plan is correct, run the write:

```bash
iints pump upload \
  --bundle-dir bundles/pico_bench_bundle \
  --mount-dir /Volumes/CIRCUITPY \
  --bench-only-confirm "I understand this is bench-only and not for human use" \
  --write
```

If your Pico appears as `RPI-RP2`, it is in BOOTSEL/UF2 mode. The SDK will refuse that mount because it is not a writable Python filesystem. Install a Python-capable Pico runtime first, then use a writable board drive such as `CIRCUITPY`, or test against a local folder.

## Serial Test

```bash
iints edge pump serial-test --port /dev/ttyACM0
```

Expected responses are JSON lines showing:

```json
{"device":"iints-pico-pump-bench","hardware_actuation_enabled":false,"locked":true,"ok":true}
```

## Safety Contract Defaults

| Field | Default | Why |
| --- | ---: | --- |
| `hardware_actuation_enabled` | `false` | Prevents this workflow from being used as real delivery firmware |
| `max_command_units` | `0.0` | No single command may represent real insulin delivery |
| `max_units_per_hour` | `0.0` | No hourly delivery is allowed in bench firmware |
| `requires_external_supervisor` | `true` | The Pico is not the safety authority |
| `requires_manual_bench_arm` | `true` | Any later hardware actuation must be a separate explicit bench step |

## What Comes Later

A future real actuator path must be a separate project with formal requirements, independent review, hazard analysis, traceability, alarm handling, occlusion/flow detection, dose accuracy testing, cybersecurity review, and regulatory guidance. Keep this SDK path as the simulation and evidence generator.

## References

- FDA, [Infusion Pump Software Safety Research](https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-software-safety-research-fda)
- FDA, [Infusion Pumps](https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/infusion-pumps)
- Raspberry Pi, [Pico-series microcontrollers documentation](https://www.raspberrypi.com/documentation/microcontrollers/raspberry-pi-pico.html)
