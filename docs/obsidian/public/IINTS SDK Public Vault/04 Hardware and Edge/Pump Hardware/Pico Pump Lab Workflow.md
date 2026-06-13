# Pico Pump Lab Workflow

The pump workflow is: algorithm in SDK → simulation → validation → manifest → zero-delivery safety contract → locked Pico bench bundle → USB serial test.

## Commands

```bash
iints edge pump init --output-dir iints_pico_pump_lab
iints edge pump package --algorithm iints_pico_pump_lab/algorithms/pico_bench_algorithm.py --output-dir iints_pico_pump_lab/bundles/pico_bench_bundle --safety-contract iints_pico_pump_lab/safety_contract.json
iints edge pump upload --bundle-dir iints_pico_pump_lab/bundles/pico_bench_bundle --mount-dir /Volumes/CIRCUITPY --bench-only-confirm "I understand this is bench-only and not for human use"
```

Source: [Pico Pump Lab docs](../20%20Official%20Documentation/docs/PICO_PUMP_LAB.md)
