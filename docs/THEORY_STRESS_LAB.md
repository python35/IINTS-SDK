# IINTS-AF Theory Stress Lab

The Theory Stress Lab is a deterministic scientific red-team mode for the SDK. It is designed for Jetson and CI runs that continuously look for weak points in the physiological and safety theories encoded in IINTS-AF.

It is **pre-clinical simulation QA**. It is not medical validation and must not be used for patient care.

## Run It

```bash
iints jetson theory-stress run --output-dir results/theory_stress_lab --profile jetson --seed 42
```

For a longer Jetson pass, repeat the configured suite with shifted seeds:

```bash
iints jetson theory-stress run \
  --output-dir results/theory_stress_lab_overnight \
  --profile jetson \
  --seed 42 \
  --repeats 100
```

For CI gating, fail when any invariant fails:

```bash
iints jetson theory-stress run --profile ci --output-dir results/theory_stress_ci --fail-on-weakness
```

The module can also run directly:

```bash
python -m iints.tools.theory_stress_lab --output-dir results/theory_stress_lab --profile ci
```

## Outputs

The tool writes:

- `summary.md`: human-readable report for research notes
- `checks.json`: machine-readable check payload
- `weakness_rankings.csv`: ranked weak points sorted by lowest score

## Current v1 Checks

- `no_negative_states`: glucose, FFA, ketones, beta mass, IOB, and COB must remain finite and non-negative
- `hypo_blocks_insulin`: Safety Supervisor must block insulin during hypoglycemia
- `iob_limits_bolus`: high IOB must cap extra bolus requests using mass balance
- `pump_failure_raises_ffa_ketones`: insulin absence should raise FFA and ketone pressure
- `sensor_lag_is_bounded`: CGM lag should be visible but not teleporting
- `exercise_does_not_create_impossible_crash`: exercise + IOB should not create non-physiological glucose motion
- `meal_response_has_plausible_peak`: meal absorption should peak in a plausible size/time window
- `illness_increases_insulin_need_without_exploding`: illness should raise glucose pressure without numerical explosion

## Interpretation

A failed check is not automatically a product failure. It is a lead for model improvement. The intended workflow is:

1. run the lab on Jetson or CI
2. inspect `weakness_rankings.csv`
3. inspect the model or parameter causing the top weak point
4. adjust the physiological theory or safety bound
5. rerun the lab

This makes the Jetson a local scientific bug-hunter for the SDK.
