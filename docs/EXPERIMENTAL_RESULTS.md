# Executed experiments and results

Every number here is read from a committed evidence artefact, not retyped from
memory. Sources:

| Artefact | Contents |
| --- | --- |
| `results/eucys_2026/EUCYS_RESULTS_TABLE.csv` | Three-arm simulation benchmark, 3600 runs |
| `apps/iints-tauri/frontend/evidence/crossfold_evidence.json` | Held-out subject forecasting, 6 subjects, SHA-256 pinned models and data |
| `arm_comparison.json` | Training-loss comparison, 2 subjects |

Read this before quoting any performance figure. Two results are unfavourable
and both are load-bearing.

---

## 1. Primary outcome: the forecaster is worse than persistence at catching hypoglycaemia

Pre-specified primary outcome, from `crossfold_evidence.json`: the share of
windows whose reference endpoint is below 70 mg/dL for which the forecast
endpoint is also below it. 3133 hypoglycaemic windows across 6 held-out
subjects. Cluster-level *t* intervals over subjects — the effective sample size
is 6, never 3133.

| | Estimate | 95% CI | Excludes 0 |
| --- | ---: | ---: | :---: |
| LSTM model | 3.47 % | [−1.53, 8.47] | no |
| Persistence baseline | 17.13 % | [6.53, 27.74] | yes |
| **Model − persistence** | **−13.66 pp** | **[−23.81, −3.51]** | **yes** |

`model_better_at_95pct: false`. The model is **significantly worse** than a
naive "glucose will stay where it is" baseline at the task that matters most.

The mechanism is in the same file. The predicted rate of change has a standard
deviation of **0.184 mg/dL/min** against **0.942** in the reference — a rate
attenuation of **0.195**. The network is heavily over-smoothed, so it flattens
precisely the excursions a hypoglycaemia alarm depends on.

This result must lead any presentation of the forecaster. A summary that opens
with the Clarke-zone figures and reaches this on a later page is misleading by
construction.

## 2. Held-out forecasting, secondary outcomes

Same 6 subjects, 76 504 forecast windows, 120-minute horizon.

| Metric | Model | 95% CI |
| --- | ---: | ---: |
| Clarke zone A | 51.11 % | [40.60, 61.62] |
| Clarke hazardous | 7.17 % | [3.53, 10.81] |
| Directionally accurate | 74.55 % | [67.87, 81.24] |
| Absolute error | 39.17 mg/dL | [34.67, 43.68] |
| Signed error | 5.59 mg/dL | [−6.26, 17.44] |

Paired against persistence, six distinct outcomes; **two exclude zero**:

| Outcome | Model − persistence | 95% CI | Model better |
| --- | ---: | ---: | :---: |
| Absolute error | −3.55 mg/dL | [−5.12, −1.99] | **yes** |
| Clarke hazardous | −1.56 pp | [−2.86, −0.27] | **yes** |
| Clarke zone A | +2.48 pp | [−0.14, +5.11] | no |
| Directionally accurate | +0.57 pp | [−0.50, +1.63] | no |
| Signed error | +5.71 mg/dL | [−6.35, +17.76] | no |
| Reversed trend | 0.00 pp | [0.00, 0.00] | no |

`directional_erroneous` is reported separately in the JSON but is numerically
identical to `clarke_hazardous`; counting both would double-count one result.

**Reading.** The model is modestly more accurate on average and makes fewer
hazardous errors, while failing the primary hypoglycaemia-detection outcome.
Those are consistent: over-smoothing lowers mean error and avoids wild
mistakes, at the cost of never predicting the sharp drops.

## 3. Three-arm simulation benchmark

3600 runs: 6 profiles × 4 scenario families × 3 arms × 5 algorithm paths ×
10 seeds, 1200 runs per arm.

| Arm | TIR 70–180 | TBR <70 | TAR >180 | Mean glucose | Supervisor interventions | Runs with severe hypo | Terminated early |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean, certified | 90.30 % | 6.45 % | 3.25 % | 115.9 mg/dL | 177.19 | **743 / 1200** | 0 |
| Corrupted, uncertified | 73.19 % | 3.69 % | 23.12 % | 141.7 mg/dL | 219.72 | 560 / 1200 | 0 |
| Supervisor-off ablation | 80.78 % | 3.88 % | 15.35 % | 140.2 mg/dL | 198.02 | **193 / 1200** | 0 |

Clean minus corrupted is **17.11 percentage points** of time in range.

**The unfavourable result.** The clean, certified arm has the highest
time-in-range *and* the highest hypoglycaemia burden: 6.45 % below 70 mg/dL,
above the 4 % consensus limit, and severe hypoglycaemia in **62 % of its runs**
against 16 % in the supervisor-off ablation. The supervisor-enabled arms carry
roughly four times the severe-hypoglycaemia rate of the ablation.

This is not a rounding artefact and should not be presented as one. Two
readings are consistent with the data and this benchmark cannot separate them:
the supervisor's interventions may be driving glucose down, or the arms may
differ in ways that co-vary with the supervisor setting. Resolving it requires
disaggregation by profile, scenario, algorithm and seed, which has not been
done.

Note also that the corrupted arm shows *less* time below range than the clean
arm. That is not corruption helping — corruption pushes runs hyperglycaemic
(23.12 % above 180), which mechanically reduces time below. The two columns
have to be read together.

## 4. Training-loss comparison

`arm_comparison.json` compares `safety_weighted` against `band_weighted` on a
shared split, holding twelve hyperparameters constant.

Only **2 held-out subjects** were available. The artefact therefore reports
point estimates and **omits every interval**, recording why:

> 2 subjects is below the minimum of 3; a cluster-level interval here would be
> uninformative and would read as more evidence than the design contains.

No conclusion about the better loss is supported by this experiment.

## What these results support

- The platform executes a large, locked, reproducible experiment matrix and
  exports results that can be traced to protocol and run artefacts.
- Clean and deliberately corrupted data conditions produce a large, measurable
  difference in outcome.
- The forecaster beats a naive baseline on average error and hazardous-error
  rate.

## What they do not support

- That the forecaster is useful for hypoglycaemia warning. On the primary
  outcome it is significantly worse than persistence.
- That any algorithm here is clinically safe, or that the virtual population
  represents a clinical population.
- That the supervisor improves glycaemic outcomes. On this benchmark the
  supervisor-enabled arms have a substantially higher severe-hypoglycaemia rate,
  unexplained.
- That either training loss is preferable — the comparison is underpowered by
  design and says so.
- Anything about a real dose for a real person.
