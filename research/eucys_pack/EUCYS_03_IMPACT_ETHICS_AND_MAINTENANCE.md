# IINTS-AF Impact, Ethics, And Maintenance Brief

**Project:** IINTS-AF SDK

**Purpose:** Explain why the SDK matters, how the project stays honest, and what maintenance evidence supports it.

**Audience:** EUCYS judges, general science judges, ethics reviewers, project mentors.

## The Problem

Insulin algorithms sit in a high-risk domain. A beautiful graph is not enough evidence. A responsible project must also show how the data was checked, which baselines were used, what failed, how edge cases behaved, and whether the results can be reproduced.

Many student projects stop at a demo. IINTS-AF aims to go further: it creates the scaffolding needed to evaluate insulin decision systems in a reviewable way.

## Why This Matters

| Stakeholder | Why the SDK matters |
|---|---|
| Students and researchers | They can test algorithms without hiding logic in notebooks |
| Mentors and judges | They can inspect commands, outputs, and limitations |
| Open-source community | They can reproduce, critique, and improve the work |
| Patients indirectly | Safer research practices reduce overclaiming before real-world use |

The most important impact is methodological: making pre-clinical algorithm research less opaque.

## The Safety Philosophy

The project follows a conservative safety philosophy:

- separate algorithm output from safety supervision
- log interventions instead of hiding them
- evaluate corrupted data instead of assuming perfect inputs
- report low-glucose risk, not just average performance
- state limitations clearly
- never present the SDK as a medical device

This is important because software confidence should come from evidence, not from optimism.

## Ethics Boundary

IINTS-AF is for simulation, research, education, and reproducible benchmarking.

It is not for:

- real patient dosing
- replacing a clinician
- controlling an insulin pump in daily life
- making individual treatment recommendations
- claiming clinical efficacy without clinical evidence

That boundary should stay visible in the repository, the documentation, and all EUCYS materials.

## What Makes The Project Responsible

| Practice | Why it is responsible |
|---|---|
| Fixed seeds | Reduces cherry-picking risk |
| Baselines | Prevents comparing only against nothing |
| Corrupted-data arm | Shows sensitivity to bad inputs |
| Supervisor-off ablation | Shows what the safety layer changes |
| Limitations section | Prevents overclaiming |
| Source legend | Makes scientific assumptions traceable |
| CI checks | Shows the project is maintained, not frozen |

## Maintenance Evidence

After the final evidence refresh, the project passed the main local checks:

| Check | Result |
|---|---|
| Full Python test suite | 369 passed, 4 skipped |
| Flake8 | passed |
| Mypy | passed |
| MkDocs build | passed |
| MDMP sync check | passed |
| Git diff whitespace check | passed |

Remote GitHub Actions for the latest pushed evidence commit also completed successfully for Python Package CI, Health Badges, and Notebook Bake.

## Why Open Source Is Part Of The Science

Open source matters here because insulin-algorithm claims should be inspectable. A closed demo can look convincing while hiding assumptions. An open SDK can be challenged.

A judge can ask:

- Which command produced this result?
- Which seed set was used?
- Which baselines were compared?
- What happens when data is corrupted?
- What does the safety layer change?
- Which files prove the benchmark design?

IINTS-AF is designed so those questions have file-based answers.

## Honest Result Framing

The final benchmark is not presented as "my algorithm solves diabetes." That would be wrong.

The result is framed as:

- the SDK can run a large reproducible benchmark
- clean certified data performed better than corrupted uncertified data
- the candidate performed strongly in the clean arm
- another baseline had the highest full-bundle TIR
- the safety supervisor is a trade-off layer that must be measured, not blindly praised

This honest framing is stronger for EUCYS because it shows scientific maturity.

## Future Impact Roadmap

The best next improvements are:

| Next step | Why it helps |
|---|---|
| Real-data realism dashboard | Shows whether simulated curves resemble real CGM behavior |
| More public dataset adapters | Makes validation less dependent on one data source |
| Failure-case gallery | Turns edge cases into inspectable learning artifacts |
| Calibration analysis | Shows whether predictor uncertainty matches actual error |
| Reproducible Docker/Devcontainer | Makes judging and review easier on other machines |
| Smaller public sample bundle | Lets reviewers run a fast smoke benchmark |

## EUCYS-Level Contribution

The EUCYS contribution is interdisciplinary:

| Discipline | Contribution |
|---|---|
| Computer science | SDK architecture, CLI, tests, reports, reproducibility |
| Data science | data validation, metrics, uncertainty and benchmark design |
| Biomedical engineering | insulin-control simulation and safety-supervisor framing |
| Ethics | no medical overclaiming, transparent limitations, reproducible evidence |

## Core Impact Claim

The strongest impact claim is:

**IINTS-AF makes insulin-algorithm research easier to test, easier to criticize, and harder to overclaim.**
