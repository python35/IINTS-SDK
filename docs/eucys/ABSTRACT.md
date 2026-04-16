# EUCYS Abstract Draft

Use this page as the competition-ready abstract draft for IINTS-AF.

Important:

- keep the structure stable
- only replace the bracketed metrics after the final benchmark run
- do not upgrade the claims beyond what the study bundle actually shows

## Title Option

**IINTS-AF: A Transparent, Safety-First Evaluation Platform for AI-Guided Insulin Decision Research**

## Competition Abstract Draft

Type 1 diabetes care increasingly depends on software that interprets glucose measurements and helps determine insulin delivery. However, insulin-decision AI is difficult to evaluate responsibly, because strong glucose control is not enough on its own: safety, explainability, and reproducibility are equally important. In this project, I developed **IINTS-AF**, a local, simulation-first research platform for benchmarking insulin-decision algorithms against classical baselines under controlled study conditions.

The central research question is whether a transparent, safety-first benchmark can evaluate candidate insulin-decision algorithms more rigorously than isolated demo runs or single-metric comparisons. The platform combines virtual patient simulation, fixed study protocols, baseline comparison, safety-layer analysis, reproducible run manifests, and edge-ready deployment on single-board computers. Instead of evaluating a single “best-looking” run, IINTS-AF generates structured benchmark bundles across predefined patient profiles, scenario families, and random seeds. This allows candidate algorithms to be compared fairly with baseline controllers such as `PID Controller`, `Standard Pump`, and `Correction Bolus`. The study engine also supports subgroup analysis, safety summaries, calibration reporting, and uncertainty-aware evaluation when predictor outputs are available.

In the current benchmark configuration, the platform evaluated **[RUN_COUNT]** runs across **[PROFILE_COUNT]** patient profiles, **[SCENARIO_COUNT]** scenario families, and **[ALGORITHM_COUNT]** algorithms. The candidate algorithm achieved **[CANDIDATE_TIR]%** time in range compared with **[BASELINE_TIR]%** for the strongest baseline, while the safety analysis showed **[INTERVENTION_RESULT]**. These results suggest that performance claims in diabetes AI should always be interpreted together with safety outcomes, uncertainty, and reproducibility.

IINTS-AF is **not** a medical device and is **not** intended for clinical dosing. Its contribution is a transparent and reproducible evaluation framework that helps researchers test AI-guided insulin decision systems more rigorously before any real-world use is considered.

## Final Fill-In Checklist

Before you export the final abstract, replace:

- `[RUN_COUNT]`
- `[PROFILE_COUNT]`
- `[SCENARIO_COUNT]`
- `[ALGORITHM_COUNT]`
- `[CANDIDATE_TIR]`
- `[BASELINE_TIR]`
- `[INTERVENTION_RESULT]`

## Hard Rules

- Do not claim patient benefit unless you have evidence for that exact wording.
- Do not say “improves diabetes treatment” unless the results actually justify it.
- Prefer `evaluation platform`, `benchmark framework`, and `simulation-first study engine`.
- Keep the last paragraph explicit: this is research software, not a medical product.
