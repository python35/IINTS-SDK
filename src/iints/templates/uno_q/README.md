# UNO Q Supervisor Bridge

This folder contains a simple STM32 sketch for Arduino UNO Q.

Use case:

- Linux side runs the IINTS digital patient runtime
- supervisor status is mirrored to the MCU over serial
- the MCU drives a green LED, red LED, and buzzer

Serial messages:

- `OK`
- `OVERRIDE`
- `CRITICAL`

Recommended expo mapping:

- green LED: algorithm operating normally
- red LED: supervisor intervened
- buzzer: critical intervention / urgent safety event
