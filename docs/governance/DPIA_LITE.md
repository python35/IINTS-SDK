# DPIA-Lite for Diabetes Research Data

This is a lightweight privacy impact checklist for local research use of IINTS-AF. A formal DPIA may still be required by a university, hospital, DPO, ethics board, or national law.

## Processing Summary

| Item | Default SDK position |
| --- | --- |
| Processing location | Local machine by default |
| Telemetry | None by default |
| Data upload | None unless the user explicitly uses a third-party service |
| Data types | CGM values, pump events, carbs, simulated physiology, optional research metadata |
| Sensitive data | Diabetes data can be health data under GDPR |
| Direct identifiers | Not required and should not be imported |
| Genetic data | Must be treated as highly sensitive if real human data is used |

## Data Minimisation Rules

- Use pseudonymous `subject_id` values.
- Do not include names, MRNs, addresses, phone numbers, e-mails, or exact dates of birth.
- Keep raw private datasets outside the repository.
- Export public artifacts with aggregate metrics and dataset fingerprints, not raw private records.
- Prefer synthetic or public properly licensed datasets for demos.

## Lawful Basis / Governance

The SDK does not choose a GDPR lawful basis for the user. The user or institution must determine it. For university or clinical research, check with the DPO/ethics board before processing real patient data.

## Security Measures

- Store private data in local protected folders.
- Avoid cloud-synced folders for sensitive raw data unless approved.
- Encrypt external drives when storing health data.
- Keep Hugging Face repos private unless all artifacts are safe to publish.
- Use MDMP certificates for data quality and provenance, not as a privacy guarantee.

## Retention

- Keep raw private data only as long as needed for the approved research purpose.
- Delete temporary run folders when no longer needed.
- Keep public model cards limited to metrics, source descriptions, and non-identifying manifests.

## When a Formal DPIA Is Needed

Escalate to a formal DPIA/DPO review if:

- identifiable health data is used;
- genetic data is used;
- data leaves the local machine;
- a cloud service, API, or shared model repository receives health data;
- the project is used in a clinical institution;
- the project affects real people or care decisions;
- large-scale patient datasets are processed.
