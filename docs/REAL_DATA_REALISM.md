# Real-Data Realism Gate

IINTS now separates two questions:

- **Plausibility:** does a trace look physiologically plausible?
- **Evidence readiness:** is it strong enough for public results, local-AI training, or EUCYS-style claims?

The normal realism validator still returns `likely_realistic`, `needs_review`, or `likely_unrealistic`. The stricter gate in `iints.data.realism_governance` adds a second layer for research evidence.

## Strict Gate

Use `review_real_data_realism(report)` after `validate_realism_dataset(...)`.

The strict profile blocks traces when:

- the base verdict is not `likely_realistic`;
- any realism check failed;
- the trace is shorter than 20 hours for daily physiology claims;
- impossible values, rapid sensor jumps, or long gaps are present;
- time below 54 mg/dL is too high for a normal-reference trace;
- meals are present but insulin annotations were lost;
- no empirical reference profile was used;
- the reference envelope fails.

## Data Evidence Strategy

The SDK registry now ranks real-data sources by evidence use:

- `tier_1_calibration_reference`: strongest current candidates for simulator calibration and realism envelopes.
- `tier_2_external_training_candidate`: useful after import, MDMP certification, and external validation.
- `tier_3_controlled_validation_candidate`: strong independent validation once access is approved.
- `demo_only`: good for quickstarts, not final realism claims.

In code:

```python
from iints.data.evidence import rank_real_data_sources

for source in rank_real_data_sources():
    print(source["id"], source["tier"], source["use_case"])
```

## Important Boundary

Passing the strict gate does **not** mean the SDK is clinically validated. It means the trace is better suited for research evidence than a basic synthetic demo trace.
