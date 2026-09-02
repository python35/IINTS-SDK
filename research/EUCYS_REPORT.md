# IINTS-AF EUCYS Research Report

**Working title:** A reproducible platform for evaluating safe insulin decision systems on virtual patients

**Author:** Runebob Baers

**Competition:** EUCYS 2026

**Version:** Draft 0.4

**Status:** Updated with the May 8, 2026 final multi-seed benchmark results from `results/eucys_2026`

> "Code shouldn't be a secret when it's managing a life."

## Abstract

This report studies whether open-source simulation and deterministic safety supervision can make insulin delivery algorithm development safer and more transparent for researchers and patients. IINTS-AF is an open research SDK that combines virtual-patient simulation, benchmark execution, protocol bundles, explainable artifacts, and local edge deployment tooling in one platform. The final benchmark bundle analyzed in this report contains 3600 runs across 6 virtual-patient profiles, 4 scenario families, 3 study arms, 5 algorithms, and 10 fixed seeds. Across the full bundle, the mean Time in Range (70-180 mg/dL) was 81.42% (95% CI 81.07 to 81.78). Because the 3600 runs are repeated measurements of only 6 virtual patients, every interval in this report is taken over those 6 profiles rather than over runs, and candidate-vs-baseline differences are paired within profile before the interval is formed (Section 4.5).

The jury-facing clean certified arm shows the strongest candidate result: `ExampleAlgorithm` achieved 93.04% Time in Range with 4.20% time below range and 116.97 mean interventions, the highest clean-arm Time in Range and the lowest intervention burden of the five algorithms. That advantage is established against three of the four baselines: the paired profile-level differences against `Clinical Baseline` (+4.03 pp, 95% CI 3.00 to 5.06), `CorrectionBolus` (+4.03 pp, 95% CI 3.35 to 4.70) and `PIDController` (+4.41 pp, 95% CI 2.50 to 6.30) exclude zero, while the +1.22 pp difference against `Standard Pump` does not (95% CI -1.22 to 3.66). On this benchmark the candidate is therefore not distinguishable from `Standard Pump` on Time in Range. Across the full stress bundle, `Correction Bolus` had the highest mean Time in Range (83.23%), while `ExampleAlgorithm` followed closely (82.40%) with substantially fewer interventions and less low-glucose exposure. Clean certified conditions outperformed corrupted uncertified conditions by 17.11 Time-in-Range points, supporting the importance of trustworthy inputs and controlled evaluation conditions. The supervisor-off ablation remained a trade-off result: clean supervision improved Time in Range and reduced hyperglycemia compared with supervisor-off, but it also increased low-glucose exposure in this benchmark. These results support the main contribution of IINTS-AF as a reproducible benchmark platform for understanding insulin decision systems, not merely as a single algorithm demo.

## Key Findings

- The final benchmark executed `3600` locked runs over `10` fixed seeds, making the study substantially more reproducible than a single-seed demo.
- In the clean certified arm used for the main figure, `ExampleAlgorithm` reached `93.04%` Time in Range and had the lowest intervention burden of all included controllers. Its TIR advantage is established over three of the four baselines; against `Standard Pump` the `+1.22`-point gap has an interval from `-1.22` to `+3.66` and stays undecided.
- Every interval in this report is taken over the `6` virtual patients rather than over the `3600` runs, because runs sharing a profile are repeated measurements of the same simulated body. That is what leaves the top-two ordering open.
- Across the full stress bundle, `Correction Bolus` had the highest mean TIR (`83.23%`), while `ExampleAlgorithm` was close behind (`82.40%`) with much lower intervention count and lower low-glucose exposure.
- Clean certified conditions outperformed corrupted uncertified conditions by `17.11` Time-in-Range points, supporting the importance of trustworthy evaluation inputs.
- The supervisor-off ablation remained mixed, which strengthens the honesty of the platform: the safety layer should be evaluated as a trade-off mechanism, not marketed as a universal performance booster.

## 1. Research Question

**Primary question**

Can open-source simulation and deterministic safety supervision make insulin delivery algorithm development safer and more transparent for researchers and patients?

**Secondary questions**
- How does the candidate algorithm compare against baseline controllers?
- How robust are the results across multiple patient profiles, scenario types, and fixed random seeds?
- How often does the safety layer intervene, and under which benchmark conditions?
- Can the full workflow be reproduced from one documented command sequence?

## 2. Hypothesis

The main hypothesis is that a structured benchmark platform with fixed scenarios, fixed seeds, explicit baselines, and safety metrics will reveal meaningful differences between insulin decision systems that are not visible in one-off demo runs.

Sub-hypotheses:
- **H1:** Certified benchmark conditions produce more reliable and defendable evidence than corrupted uncertified conditions.
- **H2:** The safety-supervised arm reduces risk without unacceptable performance loss.
- **H3:** The benchmark structure is reproducible across fixed seeds and protocol bundles.

## 3. Motivation

Insulin algorithms are often demonstrated in isolated examples, but those examples do not always show whether the setup is reproducible, safety-aware, or robust across different virtual-patient conditions. In a life-critical domain, that opacity matters. The contribution of this project is therefore not just a new insulin algorithm, but a platform for understanding insulin algorithms.

IINTS-AF was built to support three layers:
- simulation of insulin algorithms on virtual patients
- certification and auditability of datasets and outputs
- explainable review through reports, summaries, and study bundles

In one sentence, the scientific contribution is:

**I did not just build an insulin controller. I built a reproducible system to test, compare, and understand insulin decision systems.**

## 4. Methodology

### 4.1 Platform

The experiments in this report use the IINTS-AF SDK, which provides:
- virtual-patient simulation
- baseline comparisons
- study protocol generation
- audit trails and manifests
- edge deployment paths for Raspberry Pi, UNO Q, and Jetson demonstration setups

### 4.2 Study Design

This report reflects the final multi-seed benchmark bundle at:
- bundle root: `results/eucys_2026`
- protocol: `results/eucys_2026/protocol/STUDY_PROTOCOL.md`
- study design JSON: `results/eucys_2026/protocol/study_design.json`

The locked design inside that bundle is:
- preset: `eucys`
- profile set: `clinic_safe_core`
- patient profiles: `clinic_safe_baseline`, `clinic_safe_stress_meal`, `clinic_safe_hypo_prone`, `clinic_safe_hyper_challenge`, `clinic_safe_pizza`, `clinic_safe_midnight`
- scenario families: `baseline_day`, `meal_challenge`, `exercise_challenge`, `supervisor_override`
- study arms: `clean_certified`, `corrupted_uncertified`, `supervisor_off_ablation`
- algorithms: 5
- seed policy: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- total matrix size: `3600` runs

The full matrix corresponds to:
- `6 profiles x 4 scenarios x 3 arms x 5 algorithms x 10 seeds`

### 4.3 Compared Algorithms

The analyzed final bundle includes these algorithms:
- Candidate algorithm: `ExampleAlgorithm`
- Baseline: `Clinical Baseline`
- Baseline: `PID Controller`
- Baseline: `Standard Pump`
- Baseline: `Correction Bolus`

The inclusion of `Clinical Baseline` is important because it functions as a clinician-style heuristic comparator rather than only as a classical control-engineering baseline.

### 4.4 Metrics

Primary metrics used in this report:
- Time in Range (70-180 mg/dL)
- Time Below Range (<70 mg/dL)
- Time Below 54 mg/dL
- Time Above Range (>180 mg/dL)
- Time Above 250 mg/dL
- mean glucose
- coefficient of variation
- supervisor interventions

Additional benchmark indicators used from the bundle:
- severe hypo run count
- terminated early run count
- paired candidate-vs-baseline delta summaries
- worst-case and highest-intervention run summaries

### 4.5 Statistical Analysis

The study summary reports:
- per-algorithm aggregate means
- standard deviations
- medians and extrema
- 95% confidence intervals
- paired candidate-vs-baseline delta summaries

The benchmark is a fully crossed matrix of 6 virtual-patient profiles x 4 scenario
families x 3 study arms x 5 algorithms x 10 seeds = 3600 runs. The runs are
therefore not independent observations: each profile contributes 600 of them, and
seeds are repeated measurements inside a profile rather than new patients.

Intervals in this report are cluster-level t intervals over the 6 profiles: the
runs are reduced to one mean per profile, and the interval is formed on those 6
values. Candidate-vs-baseline differences are first paired within
(arm, profile, scenario, seed) blocks, which removes between-patient variation
from the contrast, and the interval is then taken over the per-profile mean
differences. An interval is reported only when at least
`MIN_CLUSTERS_FOR_INTERVAL` profiles contribute; below that the point estimate is
printed and the interval is withheld with the reason stated, because with 2 or 3
clusters the interval is not informative. The implementation is
`iints.analysis.clustered_inference`, applied in `iints.analysis.study_analysis`
and pinned by `tests/analysis/test_clustered_reporting.py`.

An earlier version of this report computed intervals as mean +/- `1.96 *
standard error` over all runs. That treats 3600 runs as 3600 independent
patients and understates the width: for the full-bundle Time in Range it gives
+/-0.33 points against +/-0.36 for the clustered interval, and for the paired
contrast against `Standard Pump` it gives an interval about 3.3 times too
narrow, which is the difference between reporting that comparison as
established and reporting it as undecided. Where pairing removes a large
between-patient variance the clustered interval can also be *narrower* than the
naive one - the `CorrectionBolus` contrast is an example - so the correction is
not a uniform widening.

No multiplicity correction is applied. The per-algorithm intervals are reported
for description; the ranking across five algorithms should not be read as a
sequence of hypothesis tests.

Sample sizes in the final benchmark:
- total runs: `3600`
- runs per study arm: `1200`
- runs per algorithm: `720`
- paired comparisons per baseline against the candidate: `720` full bundle, `240` per arm
- independent clusters carrying every interval: `6` virtual-patient profiles

All numbers in Section 6 are regenerated by
`research/export_eucys_statistics.py` into
`research/eucys_pack/eucys_statistics.json`, which records the SHA-256 of each
input summary and the interval method used for every reported quantity.

## 5. Reproducibility

This section is mandatory for the final submission.

### 5.1 Reproducibility Command Path

The final locked benchmark is reproducible with the public helper workflow:

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --no-prepare-ai
```

For the local final run, the exact source-tree-backed CLI path was:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.mplt_eucys" \
  python3 -c 'from iints.cli.cli import app; app()' -- run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --no-prepare-ai
```

The packaged competition artifacts were then refreshed with:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.mplt_eucys" \
  python3 -c 'from iints.cli.cli import app; app()' -- eucys-results results/eucys_2026
```

### 5.2 Provenance

The strongest provenance identifiers available during report writing are:
- benchmark run date: `2026-05-08`
- base repository commit used for the final benchmark: `cf8b3004f3dcaa0c3b486a1825b291f716008b99`
- active source-tree package version during the final run: `1.5.4`
- runtime: `Python 3.13.9` on `macOS-26.2-arm64-arm-64bit-Mach-O`
- locked seed set in the analyzed bundle: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- locked protocol bundle path: `results/eucys_2026/protocol`
- final study bundle path: `results/eucys_2026`

The benchmark outputs themselves should be treated as the strongest evidence artifact, because they include the protocol, study matrix, algorithm registry, summaries, and EUCYS packaging outputs in one place.

### 5.3 Output Artifacts

The key files for this report are:
- `results/eucys_2026/study_summary.json`
- `results/eucys_2026/EUCYS_SUMMARY.md`
- `results/eucys_2026/EUCYS_RESULTS_TABLE.csv`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_SUMMARY.md`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_ABSTRACT_FILLED.md`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png`
- `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.csv`
- `results/eucys_2026/protocol/study_design.json`
- `results/eucys_2026/protocol/study_matrix.csv`

## 6. Results

### 6.1 Overall Bundle Summary

Across all 3600 runs, the bundle produced:
- mean TIR 70-180: `81.42%`
- 95% CI for TIR: `81.07 to 81.78` (cluster-level t interval over 6 profiles)
- mean time below 70: `4.67%`
- mean time below 54: `1.18%`
- mean time above 180: `13.91%`
- mean time above 250: `4.23%`
- mean glucose: `132.61 mg/dL`
- mean coefficient of variation: `33.61`
- mean supervisor interventions: `198.31`
- terminated early runs: `0`

This overall bundle summary is useful as a whole-platform view, but the cleaner scientific comparisons in this report are made at two more specific levels:
- by algorithm across the full bundle
- by algorithm inside the clean certified arm used for the main figure

### 6.2 Aggregate Results By Algorithm Across The Full Bundle

| Algorithm | Runs | Mean TIR | 95% CI | <70 | >180 | Mean Glucose | CV | Mean Interventions | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Correction Bolus | 720 | 83.23 | 82.49 to 83.98 | 4.82 | 11.95 | 122.83 | 34.33 | 252.69 | Highest full-bundle mean TIR, but with high low-glucose exposure and the largest intervention load |
| ExampleAlgorithm | 720 | 82.40 | 81.70 to 83.11 | 2.56 | 15.04 | 144.56 | 30.70 | 109.43 | Candidate controller; second full-bundle TIR, lowest intervention burden and lower low-glucose exposure |
| Standard Pump | 720 | 82.22 | 81.48 to 82.97 | 3.22 | 14.56 | 137.73 | 30.88 | 204.75 | Strong comparator; its interval overlaps the candidate's, and the paired contrast does not separate them |
| Clinical Baseline | 720 | 80.56 | 79.23 to 81.88 | 3.49 | 15.96 | 136.33 | 34.06 | 223.19 | Clinician-style heuristic comparator with higher intervention burden |
| PID Controller | 720 | 78.69 | 77.73 to 79.65 | 9.28 | 12.03 | 121.60 | 38.10 | 201.48 | Lowest TIR and highest low-glucose exposure in the full bundle |

Intervals are cluster-level t intervals over the 6 virtual-patient profiles
(`n_clusters = 6` for every row). Overlapping intervals in this table do not by
themselves settle a comparison; the paired contrasts in Section 6.5 do, because
they remove between-patient variation.

### 6.3 Jury-Facing Main Figure: Clean Certified Arm

The EUCYS main figure is built from the clean certified arm, because it shows controller behavior under the most interpretable benchmark conditions before corruption or safety ablation are layered in.

| Algorithm | TIR 70-180 | Time <70 | Time >180 | Mean Interventions |
|---|---:|---:|---:|---:|
| ExampleAlgorithm | 93.04 | 4.20 | 2.77 | 116.97 |
| Standard Pump | 91.81 | 6.15 | 2.03 | 141.55 |
| Correction Bolus | 89.01 | 7.94 | 3.05 | 213.05 |
| Clinical Baseline | 89.01 | 6.45 | 4.54 | 172.70 |
| PID Controller | 88.63 | 7.52 | 3.85 | 241.66 |

Interpretation:
- `ExampleAlgorithm` achieved the highest clean-arm Time in Range (93.04%, 95% CI 91.80 to 94.27 over 6 profiles).
- It also required fewer interventions than every baseline in the main figure.
- The paired profile-level contrasts separate the candidate from `Clinical Baseline` (+4.03 pp, CI 3.00 to 5.06), `CorrectionBolus` (+4.03 pp, CI 3.35 to 4.70) and `PIDController` (+4.41 pp, CI 2.50 to 6.30).
- Against `Standard Pump` the clean-arm gap is +1.22 pp with an interval from -1.22 to +3.66, so the ordering of the top two rows in this table is not established by these 1200 runs. Adding seeds will not settle it; it needs more virtual patients.
- `Standard Pump` achieved slightly lower time above range than the candidate, and lower point-estimate TIR with higher low-glucose exposure and more interventions.
- The clean-arm result is the strongest jury-facing candidate result; the full-bundle result remains more nuanced because corrupted and supervisor-off stress conditions change the ranking.

![EUCYS main figure: clean certified arm comparison](../results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png)

Figure 1. Jury-facing benchmark comparison generated from `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png`. The underlying values are archived in `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.csv`.

### 6.4 Study Arms And Stress Conditions

| Arm | Runs | Mean TIR | 95% CI | Mean <70 | Mean >180 | Mean Interventions | Severe Hypo Runs | Early Terminations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_certified | 1200 | 90.30 | 89.80 to 90.80 | 6.45 | 3.25 | 177.19 | 743 | 0 |
| corrupted_uncertified | 1200 | 73.19 | 72.45 to 73.92 | 3.69 | 23.12 | 219.72 | 560 | 0 |
| supervisor_off_ablation | 1200 | 80.78 | 80.02 to 81.53 | 3.88 | 15.35 | 198.02 | 193 | 0 |

Key arm-level differences:
- clean certified vs corrupted uncertified TIR delta: `+17.11` points (95% CI `+16.58 to +17.64`)
- clean certified vs corrupted uncertified intervention delta: `-42.53` (95% CI `-47.37 to -37.69`)
- clean certified vs supervisor-off TIR delta: `+9.52` points (95% CI `+8.48 to +10.57`)
- clean certified vs supervisor-off time-above-range delta: `-12.10` points
- clean certified vs supervisor-off time-below-range delta: `+2.57` points (95% CI `+1.47 to +3.67`)

These arm contrasts are paired within (profile, scenario, seed, algorithm) and the
interval is then taken over the 6 profiles. The corrupted-arm TIR effect is in the
same direction in every profile, ranging from `+16.26` to `+17.60` points, so it
does not depend on one virtual patient.

The arm comparison matters because it separates three different questions:
- how controllers behave under clean benchmark conditions
- how they degrade under corrupted inputs
- how performance changes when the safety-supervision structure is removed

### 6.5 Candidate vs Baselines Across The Full Bundle

The candidate algorithm in the final benchmark was `ExampleAlgorithm`.

The deltas below are paired within (arm, profile, scenario, seed). For Time in
Range the profile-level 95% interval over the 6 paired per-profile differences is
given alongside the point estimate; the other metrics are point estimates only.

Against `PID Controller` (720 paired runs):
- TIR delta: `+3.71` (95% CI `+2.12 to +5.31`, excludes zero)
- <70 delta: `-6.73`
- >180 delta: `+3.01`
- intervention delta: `-92.05`
- mean glucose delta: `+22.96 mg/dL`
- CV delta: `-7.40`

Against `Standard Pump` (720 paired runs):
- TIR delta: `+0.18` (95% CI `-1.17 to +1.53`, includes zero)
- <70 delta: `-0.66`
- >180 delta: `+0.48`
- intervention delta: `-95.32`
- mean glucose delta: `+6.83 mg/dL`
- CV delta: `-0.17`

Against `Correction Bolus` (720 paired runs):
- TIR delta: `-0.83` (95% CI `-1.72 to +0.06`, includes zero)
- <70 delta: `-2.27`
- >180 delta: `+3.10`
- intervention delta: `-143.26`
- mean glucose delta: `+21.72 mg/dL`
- CV delta: `-3.62`

Against `Clinical Baseline` (720 paired runs):
- TIR delta: `+1.85` (95% CI `+1.19 to +2.50`, excludes zero)
- <70 delta: `-0.93`
- >180 delta: `-0.91`
- intervention delta: `-113.76`
- mean glucose delta: `+8.23 mg/dL`
- CV delta: `-3.35`

Interpretation:
- the candidate achieved the strongest clean-arm result, but not the highest full-bundle TIR
- `Correction Bolus` led the full bundle on mean TIR, while the candidate reduced low-glucose exposure and intervention count
- versus `PID Controller` and `Clinical Baseline` the TIR improvement is established across profiles (both intervals exclude zero), and it comes with lower low-glucose exposure and fewer interventions
- versus `Standard Pump` the full-bundle TIR difference is `+0.18` points with an interval from `-1.17` to `+1.53`: on this benchmark the two are not distinguishable on TIR, and the candidate's advantage over it rests on the intervention count (`-95.32`), not on TIR
- versus `Correction Bolus` the TIR difference is negative and its interval also includes zero, so neither direction is established for that pair
- the candidate's trade-off is higher mean glucose and, in several comparisons, more time above range

This trade-off is exactly the kind of result a benchmark platform should expose. A controller can look stronger on one metric while becoming more conservative or more permissive on another. The value of the platform is that those differences are measured explicitly instead of being hidden behind one attractive example trace.

### 6.6 Failure And Safety Signals

From the bundle-level failure analysis:
- terminated early runs: `0`
- severe hypo runs: `1496`
- hypo-exposed runs: `1351`
- severe hyper runs: `1436`
- supervisor-heavy runs: `3600`
- needs-review runs: `0`

The lowest-TIR runs were concentrated in the `corrupted_uncertified` arm, including:
- `PID Controller` on `clinic_safe_stress_meal` during `exercise_challenge` in `corrupted_uncertified` with TIR `41.87%`
- `Clinical Baseline` on `clinic_safe_hyper_challenge` during `supervisor_override` in `corrupted_uncertified` with TIR `46.71%`
- `Clinical Baseline` on `clinic_safe_stress_meal` during `meal_challenge` in `corrupted_uncertified` with TIR `47.75%`

This concentration of failures in the corrupted arm supports the claim that corrupted-input robustness is a meaningful and necessary stress dimension in the benchmark.

### 6.7 Hypothesis Status In The Final Benchmark

- **H1:** supported
  Clean certified conditions outperformed corrupted uncertified conditions by 17.11 TIR points (95% CI +16.58 to +17.64, paired within profile, scenario, seed and algorithm; interval over 6 profiles) and required 42.53 fewer interventions (95% CI -47.37 to -37.69). Both intervals exclude zero, and the effect is in the same direction in all 6 profiles.

- **H2:** mixed / partially supported
  Clean supervised conditions improved Time in Range by 9.52 points versus supervisor-off (95% CI +8.48 to +10.57) and lowered mean glucose by 25.74 mg/dL (95% CI -28.12 to -23.36), but they also increased time below 70 mg/dL by 2.57 points (95% CI +1.47 to +3.67). Both sides of that trade-off exclude zero, so it is a measured trade-off rather than an ambiguous one: the safety layer should be analyzed as a trade-off mechanism rather than described as an automatic risk reducer.

- **H3:** supported
  The final locked benchmark executed 10 fixed seeds over the full protocol matrix and produced a reproducible 3600-run bundle with protocol, matrix, summaries, and packaged EUCYS artifacts.

## 7. Discussion

The strongest result in the final benchmark is not simply that one controller “won,” but that the benchmark structure itself surfaced meaningful trade-offs across controllers, stress conditions, and seeds.

Key observations:
- `ExampleAlgorithm` achieved the strongest clean certified arm result and the lowest intervention burden in the jury-facing main figure.
- Across the full stress bundle, `Correction Bolus` achieved the highest mean TIR, showing that the final results should be presented honestly rather than as a one-controller victory lap.
- The candidate controller reduced low-glucose exposure and interventions versus every baseline, including the higher-TIR `Correction Bolus` baseline. Unlike the TIR ranking, these reductions are established for all four baselines: every paired profile-level interval excludes zero, from `-0.66` points of time below 70 mg/dL versus `Standard Pump` (95% CI -0.90 to -0.42) to `-6.73` versus `PID Controller` (95% CI -7.38 to -6.07), and from `-92.05` interventions versus `PID Controller` (95% CI -112.95 to -71.16) to `-143.26` versus `Correction Bolus` (95% CI -159.20 to -127.33).
- That candidate performance came with a clear cost: higher mean glucose and, in some comparisons, more time above range.
- The inclusion of `Clinical Baseline` strengthened the study because it added a clinician-style comparator rather than only classical engineering baselines.
- The corruption arm had a dramatic negative effect on TIR and intervention burden, which supports the importance of trustworthy evaluation conditions.
- The supervisor-off ablation result shows that safety layers can change the shape of risk rather than simply maximizing every glucose metric.

This is scientifically useful because it shows that a benchmark platform can reveal where an algorithm is conservative, where it is aggressive, and where its apparent strengths depend on the quality of the surrounding data and safety context.

For a science jury, that distinction matters. A project that only shows a strong controller is interesting. A project that explains when, why, and under which benchmark conditions a controller looks strong is much closer to a research contribution.

## 8. Limitations

This section remains essential.

- The platform is simulation-first and does not represent a clinical trial.
- The results do not imply medical safety in real patients.
- Virtual-patient behavior is still a model, not ground-truth biology.
- The candidate controller in this benchmark is still an example algorithm, useful for benchmarking and platform validation but not yet the strongest possible competition controller.
- Severe-hypo and severe-hyper counts are benchmark indicators, not clinical outcomes.
- The safety-layer interpretation remains mixed and should be analyzed with scenario-level review rather than with a simple good/bad label.
- External clinical validation is still outside the scope of this project.
- The statistical resolution of this benchmark is set by 6 virtual patients, not by 3600 runs. Every interval rests on 6 independent units, which is why the candidate-versus-`Standard Pump` comparison stays undecided even with 240 paired runs behind it. Adding seeds or scenarios sharpens the description of these 6 patients; only adding profiles narrows the intervals. A future bundle should widen the profile set before it adds seeds.
- No multiplicity correction is applied, so the algorithm ranking is descriptive rather than a set of tests.

These limitations are not side notes; they are part of the scientific framing. The project should therefore be presented as a preclinical evaluation framework, not as a deployable treatment system.

## 9. Ethics And Safety

- IINTS-AF is research software, not a medical device.
- The project does not provide treatment advice.
- Safety supervision and transparent evaluation are treated as first-class concerns.
- The goal is responsible evaluation before any real-world decision support is considered.
- The project is framed as an evaluation framework, not as a deployable dosing system.

## 10. Conclusion

The final multi-seed benchmark strengthens the main claim of the project: IINTS-AF is valuable as a reproducible platform for comparing insulin decision systems, not merely as a single algorithm demo. In the 3600-run benchmark, the candidate controller achieved the strongest clean certified arm result and reduced low-glucose exposure and intervention count relative to every included baseline. Across the full stress bundle, the ranking became more nuanced, with `Correction Bolus` achieving the highest full-bundle mean TIR. That nuance is not a weakness; it is the point of the platform. The benchmark makes trade-offs visible, measurable, and auditable across clean, corrupted, and safety-ablation conditions.

For EUCYS, the key contribution is therefore methodological as much as technical: IINTS-AF provides a transparent framework to test, compare, and understand insulin decision systems before any clinical use is considered.

## 11. Submission Checklist

Before the final EUCYS submission, confirm:
- [x] rerun the benchmark with the final seed set
- [x] include the clinical-style baseline in the locked final run
- [x] replace the pilot framing with the final multi-seed benchmark framing
- [x] export the main figure and final table into the report/PDF layout
- [x] record the exact runtime environment used for the final bundle in the submission package
- [x] keep limitations visible in the final paper
- [x] archive the final study bundle with protocol and result artifacts
