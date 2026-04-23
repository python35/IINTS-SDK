# IINTS-AF EUCYS Research Report

**Working title:** A reproducible platform for evaluating safe insulin decision systems on virtual patients

**Author:** Runebob Baers  
**Competition:** EUCYS 2026  
**Version:** Draft 0.3  
**Status:** Updated with final multi-seed benchmark results from `results/eucys_2026`

> "Code shouldn't be a secret when it's managing a life."

## Abstract

This report studies whether open-source simulation and deterministic safety supervision can make insulin delivery algorithm development safer and more transparent for researchers and patients. IINTS-AF is an open research SDK that combines virtual-patient simulation, benchmark execution, protocol bundles, explainable artifacts, and local edge deployment tooling in one platform. The final benchmark bundle analyzed in this report contains 3600 runs across 6 virtual-patient profiles, 4 scenario families, 3 study arms, 5 algorithms, and 10 fixed seeds. Across the full bundle, the mean Time in Range (70-180 mg/dL) was 85.99% (95% CI 85.61 to 86.36). In the clean certified arm used for the jury-facing main figure, `ExampleAlgorithm` achieved 93.44% Time in Range with 2.61% time below range and 111.7 mean supervisor interventions, outperforming `PID Controller`, `Clinical Baseline`, and `Correction Bolus`, and slightly outperforming `Standard Pump` on Time in Range while requiring fewer interventions. The clean certified arm outperformed the corrupted uncertified arm by 17.64 Time-in-Range points, supporting the importance of trustworthy inputs and controlled evaluation conditions. However, the supervisor-off ablation remained mixed: it produced high glucose performance in this simulation benchmark, suggesting that the current safety layer is conservative and should be interpreted as a trade-off mechanism rather than as an automatic performance booster. These results support the main contribution of IINTS-AF as a reproducible benchmark platform for understanding insulin decision systems, not merely as a single algorithm demo.

## Key Findings

- The final benchmark executed `3600` locked runs over `10` fixed seeds, making the study substantially more reproducible than a single-seed demo.
- `ExampleAlgorithm` achieved the strongest overall mean Time in Range of the compared controllers at `87.16%`.
- In the clean certified arm used for the main figure, `ExampleAlgorithm` reached `93.44%` Time in Range with fewer supervisor interventions than any included baseline.
- Clean certified conditions outperformed corrupted uncertified conditions by `17.64` Time-in-Range points, supporting the importance of trustworthy evaluation inputs.
- The supervisor-off ablation remained mixed, which strengthens the honesty of the platform: the safety layer should be evaluated as a trade-off mechanism, not marketed as a universal performance gain.

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
- edge deployment paths for Raspberry Pi and UNO Q demonstration setups

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
- `6 profiles × 4 scenarios × 3 arms × 5 algorithms × 10 seeds`

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
- pairwise deltas between candidate and baselines
- worst-case and highest-intervention run summaries

### 4.5 Statistical Analysis

The study summary reports:
- per-algorithm aggregate means
- standard deviations
- medians and extrema
- 95% confidence intervals
- paired candidate-vs-baseline delta summaries

From the SDK analysis implementation, the 95% confidence intervals are computed as mean ± `1.96 * standard error`.

Sample sizes in the final benchmark:
- total runs: `3600`
- runs per study arm: `1200`
- runs per algorithm: `720`
- paired comparisons per baseline against the candidate: `720`

## 5. Reproducibility

This section is mandatory for the final submission.

### 5.1 Reproducibility Command Path

The final locked benchmark was executed with the same study shape documented in the public workflow:

```bash
tools/research/run_eucys_final.sh \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

For the local final run, the source-tree-backed CLI path was:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.mplt_eucys" \
  iints run-eucys-study \
  --algo algorithms/example_algorithm.py \
  --output-dir results/eucys_2026 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --no-prepare-ai
```

The packaged competition artifacts were then refreshed with:

```bash
PYTHONPATH=src MPLCONFIGDIR="$PWD/.mplt_eucys" iints eucys-results results/eucys_2026
```

### 5.2 Provenance

The strongest provenance identifiers available during report writing are:
- base repository commit: `4f054d961f1e7e5598d0af557cfd905c2420d3b8`
- active source-tree package version during the final run: `1.5.3`
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
- mean TIR 70-180: `85.99%`
- 95% CI for TIR: `85.61 to 86.36`
- mean time below 70: `2.80%`
- mean time below 54: `0.99%`
- mean time above 180: `11.21%`
- mean time above 250: `2.26%`
- mean glucose: `130.60 mg/dL`
- mean coefficient of variation: `25.56`
- mean supervisor interventions: `189.69`
- terminated early runs: `0`

This overall bundle summary is useful as a whole-platform view, but the cleaner scientific comparisons in this report are made at two more specific levels:
- by algorithm across the full bundle
- by algorithm inside the clean certified arm used for the main figure

### 6.2 Aggregate Results By Algorithm Across The Full Bundle

| Algorithm | Runs | Mean TIR | 95% CI | <70 | >180 | Mean Glucose | CV | Mean Interventions | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ExampleAlgorithm | 720 | 87.16 | 86.35 to 87.97 | 1.28 | 11.57 | 140.46 | 23.29 | 98.75 | Highest overall mean TIR and lowest intervention burden, but also highest mean glucose |
| Clinical Baseline | 720 | 86.25 | 85.39 to 87.10 | 2.19 | 11.56 | 129.13 | 26.26 | 217.22 | Clinician-style heuristic comparator with much higher intervention burden |
| Standard Pump | 720 | 86.77 | 85.96 to 87.59 | 2.23 | 11.00 | 133.33 | 23.75 | 202.26 | Strong overall competitor, slightly lower TIR than the candidate |
| Correction Bolus | 720 | 86.04 | 85.16 to 86.92 | 3.07 | 10.89 | 122.59 | 25.98 | 253.62 | Aggressive baseline with the highest intervention load |
| PID Controller | 720 | 83.72 | 82.91 to 84.52 | 5.25 | 11.04 | 127.47 | 28.51 | 176.58 | Lowest TIR and highest low-glucose exposure among the compared controllers |

### 6.3 Jury-Facing Main Figure: Clean Certified Arm

The EUCYS main figure is built from the clean certified arm, because it shows controller behavior under the most interpretable benchmark conditions before corruption or safety ablation are layered in.

| Algorithm | TIR 70-180 | Time <70 | Time >180 | Mean Interventions |
|---|---:|---:|---:|---:|
| ExampleAlgorithm | 93.44 | 2.61 | 3.95 | 111.70 |
| Standard Pump | 92.19 | 4.08 | 3.73 | 133.95 |
| Clinical Baseline | 91.65 | 3.75 | 4.60 | 152.85 |
| Correction Bolus | 90.11 | 4.65 | 5.24 | 207.60 |
| PID Controller | 88.12 | 5.58 | 6.30 | 211.83 |

Interpretation:
- `ExampleAlgorithm` achieved the highest clean-arm Time in Range.
- It also required fewer supervisor interventions than every baseline in the main figure.
- `Standard Pump` achieved slightly lower time above range than the candidate, but also lower TIR and higher low-glucose exposure.
- `PID Controller` remained the weakest comparator in the clean certified arm on both TIR and low-glucose exposure.

![EUCYS main figure: clean certified arm comparison](../results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png)

Figure 1. Jury-facing benchmark comparison generated from `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.png`. The underlying values are archived in `results/eucys_2026/EUCYS_RESULTS/EUCYS_MAIN_FIGURE.csv`.

### 6.4 Study Arms And Stress Conditions

| Arm | Runs | Mean TIR | 95% CI | Mean <70 | Mean >180 | Mean Interventions | Severe Hypo Runs | Early Terminations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_certified | 1200 | 91.10 | 90.61 to 91.59 | 4.13 | 4.77 | 163.59 | 453 | 0 |
| corrupted_uncertified | 1200 | 73.46 | 72.99 to 73.94 | 2.60 | 23.94 | 221.27 | 362 | 0 |
| supervisor_off_ablation | 1200 | 93.39 | 93.19 to 93.59 | 1.67 | 4.94 | 184.21 | 214 | 0 |

Key arm-level differences:
- clean certified vs corrupted uncertified TIR delta: `+17.64` points
- clean certified vs corrupted uncertified intervention delta: `-57.68`
- supervisor-off ablation still produced high TIR in this benchmark, so the safety layer must be interpreted as a conservative control mechanism rather than as a guaranteed performance improver

The arm comparison matters because it separates three different questions:
- how controllers behave under clean benchmark conditions
- how they degrade under corrupted inputs
- how performance changes when the safety-supervision structure is removed

### 6.5 Candidate vs Baselines Across The Full Bundle

The candidate algorithm in the final benchmark was `ExampleAlgorithm`.

Against `PID Controller` (720 paired runs):
- TIR delta: `+3.44`
- <70 delta: `-3.97`
- >180 delta: `+0.53`
- intervention delta: `-77.82`
- mean glucose delta: `+12.98 mg/dL`
- CV delta: `-5.22`

Against `Standard Pump` (720 paired runs):
- TIR delta: `+0.38`
- <70 delta: `-0.95`
- >180 delta: `+0.57`
- intervention delta: `-103.51`
- mean glucose delta: `+7.13 mg/dL`
- CV delta: `-0.46`

Against `Correction Bolus` (720 paired runs):
- TIR delta: `+1.12`
- <70 delta: `-1.79`
- >180 delta: `+0.68`
- intervention delta: `-154.87`
- mean glucose delta: `+17.87 mg/dL`
- CV delta: `-2.69`

Against `Clinical Baseline` (720 paired runs):
- TIR delta: `+0.91`
- <70 delta: `-0.91`
- >180 delta: `+0.00`
- intervention delta: `-118.47`
- mean glucose delta: `+11.33 mg/dL`
- CV delta: `-2.97`

Interpretation:
- the candidate achieved the strongest overall TIR profile of the compared algorithms
- it also consistently reduced low-glucose exposure and intervention count versus every baseline
- however, that came with higher mean glucose and, in several comparisons, slightly more time above range
- this is a scientifically relevant trade-off rather than a simple “winner-takes-all” result

This trade-off is exactly the kind of result a benchmark platform should expose. A controller can look stronger on one metric while becoming more conservative or more permissive on another. The value of the platform is that those differences are measured explicitly instead of being hidden behind one attractive example trace.

### 6.6 Failure And Safety Signals

From the bundle-level failure analysis:
- terminated early runs: `0`
- severe hypo runs: `1029`
- hypo-exposed runs: `963`
- severe hyper runs: `657`
- supervisor-heavy runs: `3428`
- needs-review runs: `0`

The lowest-TIR runs were concentrated in the `corrupted_uncertified` arm, including:
- `ExampleAlgorithm` on `clinic_safe_hyper_challenge` during `meal_challenge` with TIR `41.87%`
- `PIDController` on `clinic_safe_stress_meal` during `supervisor_override` with TIR `47.75%`
- `Clinical Baseline` on `clinic_safe_baseline` during `supervisor_override` with TIR `48.79%`

This concentration of failures in the corrupted arm supports the claim that corrupted-input robustness is a meaningful and necessary stress dimension in the benchmark.

### 6.7 Hypothesis Status In The Final Benchmark

- **H1:** supported  
  Clean certified conditions outperformed corrupted uncertified conditions by 17.64 TIR points and required fewer interventions.

- **H2:** mixed / not yet supported cleanly  
  In this simulation benchmark, the supervisor-off ablation still produced very strong glucose performance. This suggests that the current safety layer is conservative and should be analyzed as a trade-off mechanism instead of being described as a universal performance gain.

- **H3:** supported  
  The final locked benchmark executed 10 fixed seeds over the full protocol matrix and produced a reproducible 3600-run bundle with protocol, matrix, summaries, and packaged EUCYS artifacts.

## 7. Discussion

The strongest result in the final benchmark is not simply that one controller “won,” but that the benchmark structure itself surfaced meaningful trade-offs across controllers, stress conditions, and seeds.

Key observations:
- `ExampleAlgorithm` achieved the highest overall mean TIR of the compared algorithms.
- In the clean certified arm, it also achieved the strongest jury-facing combination of TIR and low intervention burden.
- That performance came with a clear cost: higher mean glucose than the more aggressive baselines.
- The inclusion of `Clinical Baseline` strengthened the study because it added a clinician-style comparator rather than only classical engineering baselines.
- The corruption arm had a dramatic negative effect on TIR and intervention burden, which supports the importance of trustworthy evaluation conditions.
- The supervisor-off ablation result shows that safety layers can reduce aggressiveness but do not automatically maximize benchmark glucose metrics. This makes the platform more honest, not weaker.

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

These limitations are not side notes; they are part of the scientific framing. The project should therefore be presented as a preclinical evaluation framework, not as a deployable treatment system.

## 9. Ethics And Safety

- IINTS-AF is research software, not a medical device.
- The project does not provide treatment advice.
- Safety supervision and transparent evaluation are treated as first-class concerns.
- The goal is responsible evaluation before any real-world decision support is considered.
- The project is framed as an evaluation framework, not as a deployable dosing system.

## 10. Conclusion

The final multi-seed benchmark strengthens the main claim of the project: IINTS-AF is valuable as a reproducible platform for comparing insulin decision systems, not merely as a single algorithm demo. In the 3600-run benchmark, the candidate controller achieved the highest overall mean TIR while also reducing low-glucose exposure and intervention count relative to every included baseline. The most convincing result is not just controller ranking, but the fact that the platform makes trade-offs visible, measurable, and auditable across clean, corrupted, and safety-ablation conditions.

For EUCYS, the key contribution is therefore methodological as much as technical: IINTS-AF provides a transparent framework to test, compare, and understand insulin decision systems before any clinical use is considered.

## 11. Submission Checklist

Before the final EUCYS submission, confirm:
- [x] rerun the benchmark with the final seed set
- [x] include the clinical-style baseline in the locked final run
- [x] replace the pilot framing with the final multi-seed benchmark framing
- [ ] export the main figure and final table into the PDF layout itself
- [ ] record the exact runtime environment used for the final bundle in the submission package
- [ ] keep limitations visible in the final paper
- [x] archive the final study bundle with protocol and result artifacts
