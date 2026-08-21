# Data Certification Specification (Draft)

The IINTS MDMP is a data-governance layer for AI research pipelines.  
It standardizes how datasets are validated, fingerprinted, and scored before model training or evaluation.

For the complete operational guide (commands, rules, scoring, gates, and pitfalls), see `MDMP_FULL_GUIDE.md`.

MDMP is designed for traceability and reproducibility in research settings.  
It does **not** certify a model for clinical use on its own.

Public SDK surface:
- Python: `iints.data`
- CLI contracts and certification: `iints data ...`
- CLI protection and passport verification: `iints mdmp ...`

## Who This Page Is For

- Researchers defining pre-model data quality gates.
- Engineers implementing reproducible data pipelines.
- Reviewers checking traceability and dataset provenance evidence.

## Terminology Used Consistently In This Page

- `Contract`: machine-readable schema + constraint definition.
- `Validation run`: one execution of contract checks against a dataset.
- `Grade`: MDMP contract-check level (`draft`, `research_grade`, legacy `clinical_grade`).
- `Fingerprint`: deterministic SHA-256 hash for contract and dataset.

## Section Structure

This page is structured as:
- `Purpose`
- `When to use`
- `Commands`
- `Output / Artifacts`

## Scope

**Purpose**
- Define what MDMP currently governs inside IINTS.

**When to use**
- Before model training, fine-tuning, benchmarking, or report generation.

MDMP currently covers:
- tabular/time-series ingestion via data contracts
- structural and range validation
- deterministic dataset fingerprinting
- reproducible compliance scoring

Implemented in:
- `iints data certify-template`
- `iints data certify`

## Core Pillars

1. Structural Integrity
- Contracts define required columns, expected types, and physiologic ranges.
- Unit conversions can be applied deterministically (for example `mmol/L -> mg/dL`).

2. Deterministic Provenance
- Contract and dataset fingerprints use SHA-256.
- Any source-data change produces a different fingerprint.

3. Explicit Compliance Scoring
- Every validation report records individual checks, a score, and an MDMP grade.
- A grade summarizes contract conformance only; it is not a clinical certification.

4. Runtime Guardians
- `mdmp_gate` can enforce a contract before an in-memory research function runs.
- This provides a reproducible data-quality boundary inside a pipeline.

5. Authenticated Local-File Encryption
- ChaCha20-Poly1305 authenticated encryption protects confidentiality and integrity of local files.
- Human passphrases are processed with scrypt and a random per-envelope salt; exact 32-byte raw keys are also supported.
- New passphrase-protected envelopes require at least 12 UTF-8 bytes. Longer, randomly generated passphrases remain preferable.
- Associated authenticated data (AAD) can bind a study identifier or other context to the ciphertext.
- This feature protects files at rest. It does not replace TLS, access control, key management, backups, or host security.

6. Signed Data Passports
- Ed25519 signatures authenticate passport content and SHA-256 fingerprints.
- Signature format v2 also authenticates the signer label, key identifier, algorithm label, and signing timestamp. Existing v1 cards remain verifiable through an explicit compatibility path.
- Verification can use an explicit public key, the bundled research root, or an MDMP trust store.
- The SDK does not currently implement or claim ML-DSA or other post-quantum signatures. The legacy `--pqc` flag fails closed instead of creating a misleading label.

Encryption example:

```bash
iints mdmp encrypt-data --input data/my_cgm.csv --output data/my_cgm.csv.enc
iints mdmp decrypt-data --input data/my_cgm.csv.enc --output data/restored.csv
```

The commands prompt for a passphrase. Prefer the prompt or `--key-file`; avoid `--key` because command-line arguments can be exposed in shell history and process listings.

## MDMP Grades

- `clinical_grade`: legacy label for compliant and score >= 90; it does not
  mean clinically validated, unbiased, safe for care, or regulator-certified
- `research_grade`: score >= 75
- `draft`: below research threshold

Grade gating can be enforced in CI:

**Commands**

```bash
iints data certify data_contract.yaml data/my_cgm.csv \
  --min-mdmp-grade research_grade \
  --fail-on-noncompliant
```

Certification dashboard generation:

```bash
iints data certify-visualizer results/certification.json \
  --output-html results/mdmp_dashboard.html
```

Synthetic mirror generation:

```bash
iints data synthetic-mirror data/real.csv contracts/clinical_mdmp_contract.yaml \
  --output-csv data/synthetic_mirror.csv \
  --output-json audit/synthetic_mirror_report.json
```

Runtime function gate:

```python
from iints import mdmp_gate

@mdmp_gate("contracts/clinical_mdmp_contract.yaml", min_grade="clinical_grade")
def process_dataframe(df):
    ...
```

## Recommended Workflow

**Output / Artifacts**
- `certification.json`
- MDMP grade + compliance score
- contract and dataset fingerprints
- optional HTML dashboard for audit sharing

1. Write `data_contract.yaml` for your dataset.
2. Run `iints data certify` and store JSON output.
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
