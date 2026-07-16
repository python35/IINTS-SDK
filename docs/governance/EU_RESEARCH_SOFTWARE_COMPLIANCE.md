# EU Research Software Compliance Dossier

IINTS-AF is research and education software for pre-clinical diabetes technology simulation. It is **not a medical device**, not a treatment recommendation system, not an insulin dosing tool, and not intended for real-time patient care.

This dossier is a practical governance map for the SDK and the experimental Tauri/Rust desktop shell. It is not legal advice and it is not a conformity assessment. It exists so contributors, reviewers, university partners, and EUCYS-style judges can see how the project handles European research-software expectations.

## Regulatory Position

| Area | Current IINTS-AF position | Design consequence |
| --- | --- | --- |
| EU AI Act | Research-only simulator and local AI analysis workbench; not placed on the market as a clinical AI system. | Keep intended purpose narrow; document AI limitations; log runs; preserve human oversight. |
| Medical Devices Regulation (MDR) | Not intended for diagnosis, prevention, monitoring, prediction, prognosis, treatment, or alleviation of disease in a real person. | No real dosing use; no treatment advice; no claims of CE/MDR conformity; medical-device boundary review before any clinical use. |
| GDPR / health data | The SDK can process sensitive diabetes data locally when a user imports it. | Default local processing, no telemetry, data minimisation, pseudonymisation guidance, no public upload of private datasets. |
| Cyber Resilience Act / security | Open-source research software and downloadable desktop app need security-by-design practices even before formal product obligations mature. | Signed releases where possible, dependency scanning, SBOM, narrow Tauri permissions, vulnerability handling. |
| Research integrity | Outputs must be reproducible, reviewable, and not overclaimed. | Fixed seeds, manifests, hashes, reports, MDMP certificates, model cards, limitation docs. |

## Non-Negotiable Boundaries

The SDK and desktop app must always state these boundaries clearly:

- Research only.
- Educational simulation only.
- Not a medical device.
- Not for insulin dosing, treatment decisions, diagnosis, or real-time patient care.
- Outputs are hypotheses or research artifacts, not clinical instructions.
- Local AI may explain or critique runs, but deterministic SDK logs, formulas, and reports remain the source of truth.

## EU AI Act Readiness Map

The European Commission describes the AI Act as a risk-based framework and identifies strict obligations for high-risk AI systems, including risk mitigation, dataset quality, logging, technical documentation, deployer information, human oversight, robustness, cybersecurity, and accuracy.

IINTS-AF is not currently positioned as a high-risk clinical AI product. Still, the SDK uses those themes as a research governance checklist:

| AI Act theme | IINTS-AF control |
| --- | --- |
| Intended purpose | README, docs, app headers, report disclaimers, and release notes say research-only and not for treatment. |
| Risk management | Safety supervisor, deterministic numeric guards, run validation profiles, realism checks, and local AI red-team review. |
| Data governance | MDMP contracts, dataset registry, dataset license/citation checks, source library, manifest hashes. |
| Technical documentation | Paper technical dossier, physiology reference, architecture overview, formula registry, command reference. |
| Logging and traceability | Run manifests, CSV outputs, audit trails, MDMP certificates, desktop run history. |
| Transparency | Reports explain assumptions, limitations, and whether AI analysis was used. |
| Human oversight | AI suggestions are non-authoritative; deterministic checks and user review are required. |
| Accuracy/robustness/cybersecurity | Pytest, mypy, pip-audit, SBOM, deterministic replay checks, Tauri permission minimisation. |

## MDR / Medical Software Boundary

The MDR and MDCG software qualification/classification guidance become critical if the project changes intended purpose. The present intended purpose is deliberately outside clinical use.

A formal medical-device review is required before any of these are added:

- using IINTS-AF to decide insulin or glucagon delivery for a person;
- connecting output directly to a real pump for treatment;
- claiming diagnosis, monitoring, prediction, prognosis, treatment, or disease alleviation for an individual patient;
- claiming clinical performance, safety, CE marking, or MDR conformity;
- deploying the model into care workflows.

If any of those happen, the project must stop and create a separate regulated-product branch with regulatory expertise, quality management, clinical evaluation, risk management, cybersecurity, post-market surveillance, and conformity-assessment planning.

## GDPR / Health Data Controls

Diabetes data is health data. The SDK therefore follows these practical controls:

- No telemetry by default.
- Local-first processing by default.
- Do not commit private CGM, pump, genetics, or clinical files to GitHub.
- Prefer pseudonymous `subject_id` values; avoid names, e-mails, dates of birth, MRNs, addresses, and direct identifiers.
- Keep raw private datasets outside the repo.
- Public model cards and reports must describe data sources without exposing private records.
- MDMP certificates can certify data quality, but they do not make private data safe to publish.
- If real human-subject research is performed, obtain the needed ethics/IRB/DPO approval for the institution and country.

## Tauri / Rust Desktop Security Baseline

The Tauri shell should remain a narrow native boundary around the Python SDK:

- No shell plugin.
- No broad filesystem plugin.
- No arbitrary command execution from the frontend.
- No remote web content in the default app.
- Fixed Rust command allowlist only.
- Python bridge accepts fixed subcommands only.
- Long-running work should run through async Rust commands or background workers.
- Updates should be signed before automatic deployment is enabled.
- macOS notarization and Windows signing are release-hardening tasks, not clinical validation.

## Release Evidence Checklist

Before a public release, maintainers should check:

- Full test suite passes.
- Mypy passes for touched modules.
- Dependency audit is clean or documented.
- SBOM is generated.
- Desktop smoke tests pass on packaged apps.
- README and app still show research-only/not-medical-device boundaries.
- Tauri capabilities still do not include shell/filesystem/http/updater permissions unless separately risk-reviewed.
- Release notes do not claim clinical performance or regulatory approval.
- Any dataset used in examples has a license/citation and contains no private patient data.

## Machine-Checked Controls

The CI governance check covers the core repository controls:

```bash
python tools/ci/check_governance.py
```

That script checks licensing, SBOM shape, dataset registry metadata, manifest hashing, EU research-compliance documentation, privacy boundaries, README disclaimers, and Tauri permission minimisation.

The deeper governance pack is split into:

- [Intended Use & Claims Control](INTENDED_USE_AND_CLAIMS.md)
- [EU Research Risk Register](RISK_REGISTER.md)
- [DPIA-Lite for Diabetes Research Data](DPIA_LITE.md)
- [Tauri Desktop Threat Model](TAURI_THREAT_MODEL.md)
- [`EU_RESEARCH_CONTROL_MATRIX.json`](EU_RESEARCH_CONTROL_MATRIX.json)

The JSON control matrix is intentionally machine-readable so CI can verify that each named control still has evidence in the repository.

## Official Sources

- European Commission, AI Act overview: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- Regulation (EU) 2024/1689, Artificial Intelligence Act: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2017/745, Medical Device Regulation: https://eur-lex.europa.eu/eli/reg/2017/745/oj
- European Commission MDCG guidance list, including MDCG 2019-11 software qualification/classification: https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
- Regulation (EU) 2016/679, GDPR: https://eur-lex.europa.eu/eli/reg/2016/679/oj
- European Commission, Cyber Resilience Act overview: https://digital-strategy.ec.europa.eu/en/policies/cyber-resilience-act
