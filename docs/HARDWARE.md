# Hardware Hub

Use this page when you already know the SDK workflow and need to decide where it should run.

## Choose The Device

| If you need... | Best fit | Start with |
| --- | --- | --- |
| a stable live digital-patient demo | Raspberry Pi | [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md) |
| visible hardware feedback with a small bridge layer | Arduino UNO Q | [Arduino UNO Q Setup](ARDUINO_UNO_Q.md) |
| accelerated stress testing over long simulated periods | NVIDIA Jetson | [Jetson Endurance Mode](JETSON_ENDURANCE.md) |

## Fastest Safe Start

Raspberry Pi:

```bash
iints edge quickstart --board raspberry_pi
```

Arduino UNO Q:

```bash
iints edge quickstart --board uno_q
```

Jetson:

```bash
iints jetson doctor
iints jetson endurance start --duration 7d --profile mixed_adversarial --output-dir results/jetson_7day
```

## Device Routes

### Raspberry Pi

```text
Edge Hardware Matrix
  -> Raspberry Pi Digital Patient
  -> Remote Deploy & Pi Connect
  -> Maker Faire Pi Mode if needed
```

### Arduino UNO Q

```text
Edge Hardware Matrix
  -> Arduino UNO Q Setup
  -> upload bridge sketch once
  -> run the Linux-side patient runtime
```

### NVIDIA Jetson

```text
Jetson Endurance Mode
  -> doctor
  -> endurance start
  -> status / monitor / export
```

## Read Next

- [Edge Hardware Matrix](EDGE_HARDWARE.md)
- [Raspberry Pi Digital Patient](DIGITAL_PATIENT_PI.md)
- [Arduino UNO Q Setup](ARDUINO_UNO_Q.md)
- [Jetson Endurance Mode](JETSON_ENDURANCE.md)
- [Remote Deploy & Pi Connect](EDGE_REMOTE_DEPLOY.md)
