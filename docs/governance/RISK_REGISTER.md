# EU Research Risk Register

This risk register is for research-software governance. It is not an MDR ISO 14971 risk-management file and not a clinical safety case.

| ID | Risk | Impact | Current controls | Residual risk | Next action |
| --- | --- | --- | --- | --- | --- |
| R-001 | User mistakes research output for treatment advice | Could lead to unsafe real-world decisions | README/app/report disclaimers, terms, docs, AI prompt restrictions | Medium | Keep disclaimers in CI and UI; add onboarding warnings to future apps |
| R-002 | Private CGM/pump/genetic data committed to GitHub | GDPR/privacy breach | Privacy policy, dataset docs, `.gitignore`, local-first processing | Medium | Add optional secret/PHI scan workflow before releases |
| R-003 | Local AI hallucinates clinical interpretation | Misleading research conclusion | Deterministic run quality gate remains source of truth; AI marked advisory | Medium | Add structured AI output schema and contradiction checks |
| R-004 | Model or simulator physiology is overclaimed | Scientific credibility loss | Physiology reference, realism tests, limitations docs, source library | Medium | Keep calibrating against OhioT1DM and other public datasets |
| R-005 | Desktop app gains broad OS permissions | Security and privacy exposure | Tauri capability minimisation checked in CI | Low | Require security review before shell/fs/http/updater plugins |
| R-006 | Dependency vulnerability in app or SDK | Compromise or unsafe outputs | Dependabot, pip-audit, SBOM, version caps | Medium | Add Rust `cargo audit` when Cargo is available in CI |
| R-007 | Dataset license/provenance missing | Legal/research integrity issue | Dataset registry license/citation checks | Low | Extend registry to all downloaded HF/public datasets |
| R-008 | Formula drift between docs and code | Paper claims no longer match implementation | Formula registry, technical dossier, tests | Medium | Add formula-to-code traceability table per release |
| R-009 | Unsigned desktop updates or downloads | User trust/security warnings | Signing docs, no automatic updater enabled by default | Medium | Add notarized macOS and signed Windows releases |
| R-010 | Hugging Face model repo exposes private training data | Privacy breach | Model card privacy docs, public manifest only | Medium | Add dataset fingerprinting and no-raw-data release check |
| R-011 | Hardware demos are mistaken for therapeutic devices | Safety misunderstanding | Bench-only docs, no patient connection, hardware disclaimers | Medium | Add physical labels and app warning in hardware modes |
| R-012 | Scientific users treat single simulation as evidence | Invalid conclusion | Reports, evidence bundles, run history, seed recording | Medium | Encourage multi-seed studies and confidence intervals by default |

## Review Cadence

- Review before public release.
- Review after adding new data connectors, AI models, hardware control, desktop permissions, or updater logic.
- Review before any university, clinical, or public demo that uses real health data.
