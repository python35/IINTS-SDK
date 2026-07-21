# Academic Research Workbench

IINTS-AF can add a reviewable academic metadata layer to a completed simulation run. The goal is practical reproducibility: another researcher should be able to identify the software, configuration, seed, artifacts, checksums, and references without relying on a screenshot or an AI summary.

!!! warning "Scope"
    An academic package is not peer review, ethical approval, privacy clearance, clinical validation, or medical-device certification. It does not upload data. Review every artifact before sharing it.

## Create A Package

```bash
iints research academic-bundle results/my_run \
  --title "Baseline T1D reproducibility run" \
  --creator "Researcher Name" \
  --orcid "https://orcid.org/0000-0000-0000-0000" \
  --license "CC-BY-4.0"
```

To associate references explicitly, repeat `--source-id`:

```bash
iints research academic-bundle results/my_run \
  --source-id hovorka_2004_nmpc_t1d \
  --source-id attd_2019_time_in_range
```

When no source IDs are supplied, the exporter makes only conservative associations that can be inferred from run metadata and artifacts. These associations are candidates for human review, not proof that a paper validates the implementation.

The run-artifact licence defaults to `NOASSERTION`. The SDK code is Apache-2.0, but that software licence is not silently applied to real-data-derived CSV files, reports, or other research artifacts. Choose a data/output licence only when you have the right to do so.

The same operation is available in both desktop workbenches from the **Results** or **Reproducibility package** area.

## Generated Files

| File | Purpose |
| --- | --- |
| `ro-crate-metadata.json` | [RO-Crate 1.2](https://www.researchobject.org/ro-crate/specification/1.2/introduction.html) JSON-LD with run, software, file, checksum, creator, licence, and source entities |
| `academic_audit.json` | machine-readable checks for metadata, seed, configuration, manifest, revision, sources, attribution, licence, artifact inventory, and basic privacy markers |
| `academic_sources.json` | exact evidence-registry snapshot, registry hash, selected sources, and selection method |
| `ACADEMIC_BUNDLE.md` | short human review guide stored with the run |

Existing experimental files are not copied or rewritten. The exporter reads them to calculate SHA-256 checksums and then writes the four metadata files beside them.

## Readiness Status

| Status | Meaning |
| --- | --- |
| `ready` | all implemented required, recommended, and review checks passed |
| `needs_review` | required metadata exists, but one or more recommended or human-review checks remain |
| `incomplete` | a required artifact or metadata field is missing |

The score measures only the implemented checklist. A high score does not mean that a model is physiologically accurate or that a dataset may legally be shared.

## Academic Integration Levels

The app labels every external resource by maturity so a portal link cannot be mistaken for a functioning scientific integration.

| Level | Meaning | Current examples |
| --- | --- | --- |
| Integrated | the SDK calls a defined local/API workflow and writes a reviewable artifact | RO-Crate export; AlphaFold; SBML/libRoadRunner; COPASI; CellML/OpenCOR validation; FMI/FMPy; BindingDB |
| Partial | a useful query or render exists, but full versioned import and validation are not complete | GTEx expression, ChEMBL context, STRING networks, ClinVar context |
| Planned | a scientifically useful boundary is documented but not implemented | SED-ML protocol export, automatic Physiome repository import, structured PubMed capture |
| Portal | the app opens an official allowlisted resource; no evidence is ingested | RCSB PDB, UniProt, Human Protein Atlas, ClinicalTrials.gov, Zenodo |

## Standards Direction

- [FAIR4RS](https://www.nature.com/articles/s41597-022-01710-x) guides software citation, metadata, access, interoperability, and reuse. IINTS-AF is FAIR-oriented; it is not externally FAIR-certified.
- [RO-Crate 1.2](https://www.researchobject.org/ro-crate/specification/1.2/introduction.html) is implemented for run-level metadata and artifact inventory.
- [SED-ML Level 1 Version 5](https://sed-ml.org/) is the planned portable description for model/simulation/task/output protocols. IINTS-AF does not yet claim SED-ML compatibility.
- [SBML Level 3 Version 2](https://sbml.org/documents/specifications/) is supported for safe structural inspection of local reference files. Optional independent execution uses libRoadRunner. Current Python patient models are not claimed to be SBML models.
- [BioModels](https://www.biomodels.org/) is useful for model provenance and comparison. Automatic repository import is not implemented; local model files can be inspected only after the researcher has reviewed their source and licence.
- [Zenodo](https://developers.zenodo.org/) remains a manual publication route after privacy, licensing, and completeness review; the SDK never uploads automatically.

## Evidence Rules

1. Record the exact database, identifier, query, access date, and local artifact hash.
2. Prefer primary publications and official database records over summaries.
3. Keep structural confidence, variant assertions, expression evidence, and physiological parameters separate.
4. Never convert AlphaFold pLDDT or PAE directly into pathogenicity, insulin sensitivity, or a dosing parameter.
5. Never let external evidence or local AI silently alter the deterministic simulation configuration.
6. Preserve failed runs, exclusions, software versions, and changed assumptions.
7. Review direct and indirect identifiers before publishing any real-patient-derived data.

## Recommended Paper Workflow

```mermaid
flowchart LR
    A["Freeze protocol and hypotheses"] --> B["Run with recorded seeds"]
    B --> C["Validate raw artifacts"]
    C --> D["Create academic package"]
    D --> E["Resolve audit findings"]
    E --> F["Review privacy and licences"]
    F --> G["Archive code, environment, and approved outputs"]
```

Use deterministic metrics and raw time series as the numerical authority. Local AI can help critique or explain a run, but its response is a review note and is not included as ground truth.

For independent external equation-model checks, use [Mechanistic Reference Models](MECHANISTIC_REFERENCE_MODELS.md). Reference execution remains separate from IINTS calibration so a successful solver run cannot silently change the virtual patient.

For sensitivity tasks, CellML validation, physical-device FMUs, and measured affinity evidence, continue with [Cross-scale Reference Labs](CROSS_SCALE_REFERENCE_LABS.md). These outputs remain separate evidence layers and require explicit mappings before comparison.
