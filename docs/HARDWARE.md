# Hardware Hub

Use this page when the question is no longer “how do I run the SDK?” but “where should this run?”

## Pick The Device

| Device | Best for | Start with |
| --- | --- | --- |
| Raspberry Pi | live digital-patient demos, kiosk viewing, long-running edge studies | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |
| Arduino UNO Q | simple physical bridge demos with a thin microcontroller layer | [Arduino UNO Q Setup](ARDUINO_UNO_Q.md) |
| NVIDIA Jetson | accelerated adversarial endurance studies and GPU-backed stress testing | [Jetson Endurance Mode](JETSON_ENDURANCE.md) |

## Fastest Safe Entry Point

```bash
iints edge quickstart --board raspberry_pi
iints edge quickstart --board uno_q
```

For Jetson:

```bash
iints jetson doctor
iints jetson endurance start --duration 7d --profile mixed_adversarial --output-dir results/jetson_7day
```

## Hardware Routes

### Raspberry Pi

```text
Edge Hardware Matrix
  -> Raspberry Pi Digital Patient
  -> Remote Deploy & Pi Connect
  -> Maker Faire Pi Mode if you need a booth setup
```

### Arduino UNO Q

```text
Edge Hardware Matrix
  -> Arduino UNO Q Setup
  -> upload bridge sketch once
  -> run Linux-side patient runtime
```

### Jetson

```text
Jetson Endurance Mode
  -> doctor
  -> endurance start
  -> status / monitor / export
```

## Read Next

- [Edge Hardware & SBC Matrix](EDGE_HARDWARE.md)
- [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)
- [Arduino UNO Q Setup](ARDUINO_UNO_Q.md)
- [Jetson Endurance Mode](JETSON_ENDURANCE.md)
- [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md)
