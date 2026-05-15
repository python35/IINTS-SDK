# IINTS-AF Jury Q&A

**Project:** IINTS-AF SDK

**Purpose:** Short answers to likely EUCYS judge questions.

**Audience:** Presenter preparation and live judging.

## 1. What did you actually build?

I built an open-source Python SDK for testing insulin decision algorithms on virtual patients. It includes simulation, baseline comparison, safety supervision, data-quality certification, analysis reports, documentation, and edge-device workflows.

## 2. Is this a medical device?

No. It is a research and simulation platform. It does not dose real patients and should not be used for treatment decisions.

## 3. What is the main scientific question?

Can open-source simulation and deterministic safety supervision make insulin algorithm development more reproducible, auditable, and transparent before real-world testing?

## 4. Why not just show one good glucose graph?

One graph can be misleading. IINTS-AF runs a matrix of patient profiles, scenarios, study arms, baselines, and seeds. The final EUCYS benchmark contains 3600 runs, so the result is harder to cherry-pick.

## 5. What is MDMP?

MDMP is the data-quality layer. It checks whether a CGM-style dataset follows an expected contract and whether the data looks usable enough for research. It helps prevent bad or corrupted data from silently becoming evidence.

## 6. What is the safety supervisor?

The safety supervisor is a deterministic guard around the algorithm. It can limit or reject unsafe decisions and logs why it intervened. It is deliberately transparent rather than a second black-box model.

## 7. What was the strongest result?

In the clean certified arm, the candidate algorithm reached 93.04% Time in Range with lower intervention burden than the included baselines. Across the full stress bundle, Correction Bolus had the highest mean Time in Range, while the candidate remained close with fewer interventions and less low-glucose exposure.

## 8. Why mention that another baseline wins one metric?

Because it is scientifically honest. A platform is stronger when it reveals trade-offs instead of hiding them. The claim is not that one algorithm wins everything; the claim is that the SDK makes these comparisons visible.

## 9. Why does corrupted data matter?

Algorithms can look good on clean data and fail when inputs are missing, noisy, or physiologically implausible. In the final benchmark, clean certified data outperformed corrupted uncertified data by 17.11 Time-in-Range points.

## 10. What makes the project reproducible?

The repo includes fixed seeds, CLI commands, protocol bundles, result tables, generated figures, markdown reports, PDFs, tests, and CI checks. A reviewer can rerun the benchmark path instead of trusting a screenshot.

## 11. How do you know the physiology is realistic?

The SDK is grounded in diabetes simulation and CGM-metric literature, but it does not claim perfect realism. Instead, it checks plausibility, reports limitations, and identifies real-data validation as a key next step.

## 12. Which sources support the physiology and metrics?

The source legend includes ADA Standards of Care 2026, the international Time in Range consensus, OhioT1DM, Dalla Man/Cobelli meal model literature, UVA/Padova simulator literature, and a 2024 Communications Medicine paper on generative T1D simulation.

## 13. What would you improve with more time?

I would add a stronger real-data realism dashboard, more public dataset adapters, a visual failure-case gallery, Docker-based reproducibility, and more explicit unit tests for physiological plausibility.

## 14. Why is this useful if it is not clinical yet?

Clinical tools should not start as clinical tools. They should first be tested in transparent, reproducible, pre-clinical environments. IINTS-AF helps create that environment.

## 15. What is your best one-sentence pitch?

IINTS-AF is an open research SDK that makes insulin algorithm testing reproducible, safety-aware, and honest about data quality before any real-world claim is made.
