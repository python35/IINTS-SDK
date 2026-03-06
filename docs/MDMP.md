# IINTS Medical Data Management Protocol (MDMP) - Draft 1.0

The IINTS MDMP is a data-governance layer for AI research pipelines.  
It standardizes how datasets are validated, fingerprinted, and scored before model training or evaluation.

MDMP is designed for traceability and reproducibility in research settings.  
It does **not** certify a model for clinical use on its own.

## Scope

MDMP currently covers:
- tabular/time-series ingestion via data contracts
- structural and range validation
- deterministic dataset fingerprinting
- reproducible compliance scoring

Implemented in:
- `iints data contract-template`
- `iints data contract-run`

## Core Pillars

1. Structural Integrity
- Contracts define required columns, expected types, and physiologic ranges.
- Unit conversions can be applied deterministically (for example `mmol/L -> mg/dL`).

2. Chain of Trust
- Contract and dataset fingerprints are generated with SHA-256.
- Any source-data change produces a new fingerprint.

3. Compliance Scoring
- Every run returns `compliance_score` and pass/fail checks.
- MDMP grade is derived from score + hard compliance.

## MDMP Grades

- `clinical_grade`: compliant and score >= 90
- `research_grade`: score >= 75
- `draft`: below research threshold

Grade gating can be enforced in CI:

```bash
iints data contract-run data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade \
  --fail-on-noncompliant
```

Certification dashboard generation:

```bash
iints data mdmp-visualizer results/contract_data_report.json \
  --output-html results/mdmp_dashboard.html
```

## Recommended Workflow

1. Write `data_contract.yaml` for your dataset.
2. Run `iints data contract-run` and store JSON output.
3. Archive fingerprints with model artifacts.
4. Train/evaluate only on datasets meeting your minimum grade policy.

## Audit Artifacts

Keep these together per run:
- `contract_data_report.json`
- `validation_report.json`
- `sources_manifest.json`
- `SUMMARY.md`
- model artifact metadata (including seed, split, and commit SHA)

## Regulatory Positioning

MDMP is a research data-quality protocol that can support documentation practices expected by regulated environments.  
It is not itself a regulatory approval and does not turn the SDK into a medical device.
