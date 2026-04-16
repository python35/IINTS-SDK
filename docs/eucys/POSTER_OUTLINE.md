# EUCYS Poster Outline

Use this page to build a poster that matches the actual IINTS-AF benchmark outputs.

The goal is simple:

- one central research question
- one architecture figure
- one killer comparison figure
- one core results table
- one limitations and ethics block

If the poster starts to feel crowded, remove extra material before you remove clarity.

## Poster Goal

Someone should be able to answer these five questions after reading the poster:

1. What problem is being studied?
2. What did the student build?
3. How was it tested fairly?
4. What did the results show?
5. What are the limits of the work?

## Recommended Title

**IINTS-AF: A Transparent and Safety-First Benchmark Platform for AI-Guided Insulin Decision Research**

## Recommended Poster Structure

### 1. Problem

Short text block:

> AI is increasingly used in glucose forecasting and insulin decision support, but performance alone is not enough. Evaluation must also include safety, explainability, and reproducibility before any real-world use can be justified.

### 2. Research Question

Use a single sentence:

> Can a transparent, safety-first benchmark platform evaluate AI-guided insulin decision systems more rigorously than single-run demonstrations, while supporting fair comparison against classical baselines?

### 3. System Architecture

Required figure:

`Sensor / CGM -> Forecast / Controller -> Safety Gate -> Final Action -> Audit Trail`

Caption suggestion:

> IINTS-AF separates prediction, candidate action, safety review, and final action so each decision can be inspected and reproduced.

### 4. Study Design

Show this as a compact block or mini table:

- profile set: `clinic_safe_core`
- study arms: `clean_certified`, `corrupted_uncertified`, `supervisor_off_ablation`
- scenario families: `baseline_day`, `meal_challenge`, `exercise_challenge`, `supervisor_override`
- algorithms: candidate + classical baselines
- shared seeds across all comparisons

Caption suggestion:

> Every comparison uses the same profiles, scenario families, and seeds to reduce cherry-picking.

### 5. Main Results Figure

This should be the most visually important part of the poster.

Recommended figure content:

- candidate vs `PID Controller`
- candidate vs `Standard Pump`
- candidate vs `Correction Bolus`
- metrics shown side-by-side:
  - `TIR 70-180`
  - `Time <70`
  - `Time >180`
  - `Safety interventions`

Preferred visual forms:

- grouped bars
- compact boxplots
- one summary plot, not four disconnected charts

Caption suggestion:

> The candidate algorithm should be interpreted by balancing glucose outcomes with safety interventions, not by TIR alone.

### 6. Core Results Table

Keep one table with only the strongest metrics:

| Algorithm | Mean TIR | Mean <70 | Mean >180 | Mean Glucose | Safety Interventions | Runs |
| --- | --- | --- | --- | --- | --- | --- |
| Candidate | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |
| PID Controller | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |
| Standard Pump | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |
| Correction Bolus | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` | `[x]` |

### 7. Safety And Ablation Block

This is where the poster becomes mature rather than flashy.

Show:

- candidate with safety gate
- candidate without safety gate
- optionally candidate with and without uncertainty layer

Short message:

> Removing safety may improve some glucose metrics in isolated cases, but it can also increase risk. This trade-off is part of the scientific result, not a side note.

### 8. Reproducibility Block

Use bullets, not a wall of text:

- fixed protocol bundle
- study matrix
- shared seeds
- run manifests
- machine-readable summaries
- exportable poster assets

Short message:

> The platform records enough metadata to reproduce and audit benchmark conclusions.

### 9. Limitations And Ethics

Keep this highly visible.

Suggested wording:

- simulation-first, not a clinical trial
- not a medical device
- depends on model assumptions and scenario design
- public datasets are used for plausibility, not as proof of clinical benefit
- safety and transparency are design goals, not guarantees of real-world readiness

### 10. Conclusion

Short and strong:

> IINTS-AF does not claim to replace clinical systems. Its contribution is a transparent, safety-first evaluation framework that makes AI-guided insulin decision research easier to test, compare, explain, and reproduce.

## Poster Layout Recommendation

If you have a standard portrait poster:

- top row: title, problem, research question
- upper middle: architecture + study design
- center: main results figure
- lower middle: core results table + safety/ablation block
- bottom: reproducibility + limitations + conclusion

## What To Avoid

- too many tiny figures
- unexplained acronyms
- saying “AI improves diabetes care” without a narrow, supported context
- hiding limitations in small text
- using a live demo screenshot as your main evidence

## Final Poster Checklist

Before printing, confirm:

- the title matches the real claim
- the central figure matches the actual study bundle
- the table values come from the final summary outputs
- the limitations are visible from normal viewing distance
- the conclusion does not overclaim
