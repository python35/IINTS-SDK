# IINTS-AF Public Documentation (v0.1.22)

This document is the public, single‑entry documentation index for the IINTS‑AF SDK. It summarizes
**software**, **data**, **content**, and **AI system** documentation in one place.

Public docs site: `https://python35.github.io/IINTS-SDK/`

## Who This Page Is For

- External reviewers and collaborators who need a documentation map.
- Researchers preparing study or audit packages.
- Developers onboarding to the SDK documentation landscape.

## Terminology Used Consistently In Public Docs

- `Algorithm`: insulin-dosing logic under test.
- `Forecast model`: optional AI predictor (never final dosing authority).
- `Safety Supervisor`: deterministic safety layer.
- `Run bundle`: reproducible output folder with metrics and reports.
- `MDMP`: data-quality protocol for contract validation and grading.

## Section Structure

Each major chapter below follows:
- `Purpose`: what the chapter covers.
- `When to use`: when this information matters.
- `Commands / Entry points`: where to start.
- `Output / Artifacts`: what to keep for traceability.

For non-technical readers, start with:
- `PLAIN_LANGUAGE_GUIDE.md`
- `../README.md`

---

## 1) Software Documentation

**Purpose**
- Describe core SDK references for building and validating simulation workflows.

**When to use**
- When implementing a new algorithm, run workflow, or integration.

### Developer Docs
- **Primary guide**: `COMPREHENSIVE_GUIDE.md`
- **Technical details**: `TECHNICAL_README.md`
- **Data protocol (MDMP draft)**: `MDMP.md`
- **Evidence base**: `EVIDENCE_BASE.md`
- **API stability**: `../API_STABILITY.md`
- **Change history**: `../CHANGELOG.md`

### Technical Architecture
- **Core simulation**: `src/iints/core/` (Simulator, Supervisor, SafetyConfig)
- **Data ingestion**: `src/iints/data/` (registry, importers, parsers, quality checks)
- **Analysis & reporting**: `src/iints/analysis/` (metrics, reporting, baseline comparisons)
- **Emulation**: `src/iints/emulation/` (commercial pump emulators)
- **CLI**: `src/iints/cli/cli.py`

### User Guides
- **Quickstart**: `../README.md`
- **Notebooks**: `examples/notebooks/` (step‑by‑step walkthroughs)
- **Presets**: `src/iints/presets/presets.json`
- **AI Research Track**: `../research/README.md`

**Output / Artifacts**
- `run_metadata.json`, `run_manifest.json`, `validation_report.json`, and report outputs from run bundles.

### Research Checklist (Recommended)
- Use a fixed `seed` for every run (or record the auto‑seed in `run_metadata.json`).
- Archive `config.json`, `run_metadata.json`, `run_manifest.json`, and `results.csv` together.
- Keep `report.pdf` + `audit/` for reviewability.
- Cite datasets using `iints data cite <dataset_id>`.
- Export literature sources using `iints sources --output-json results/source_manifest.json`.
- Use `iints study-ready` when you want a ready-to-review bundle in one command.
- Use `iints data mdmp-visualizer` to turn `contract_data_report.json` into a shareable single-file audit dashboard.
- Use `iints init --template clinical-trial` for a ready-made MDMP scaffold (contract + demo data + audit folders).
- Use `iints data synthetic-mirror` to build privacy-safe synthetic datasets from validated source data.
- Prefer `iints mdmp ...` commands and `iints.mdmp` imports for protocol-specific workflows.
- Record the SDK version + git SHA from `run_metadata.json`.

---

## 2) Data Documentation

**Purpose**
- Define data schema, metadata sources, and data access commands.

**When to use**
- Before training, forecast evaluation, or reproducibility packaging.

### Data Dictionary (Standard Schema)
The IINTS standard schema for CGM time series uses these columns:
- `timestamp` (ISO8601 or epoch)
- `glucose` (mg/dL)
- `carbs` (grams)
- `insulin` (units)

Reference:
- `data_packs/DATA_SCHEMA.md`

### Metadata & Background
- Dataset registry: `src/iints/data/datasets.json`
- Bundled demo data: `src/iints/data/demo/demo_cgm.csv`

### Data Sources & Access Instructions
Use the CLI to discover and access datasets:
```bash
iints data list
iints data info <dataset_id>
iints data fetch <dataset_id> --output-dir data_packs/<dataset_id>
```

Registry documentation:
- `data_packs/DATASETS.md`

**Output / Artifacts**
- Dataset identifiers, source metadata, and schema references used in your study record.

---

## 3) Content Documentation

**Purpose**
- Explain non-code assets and required tools for notebooks and reports.

**When to use**
- When onboarding teams that consume outputs but do not modify SDK code.

For non‑code content (notebooks, PDF reports, plots), the following tools are required:

### Required Apps / Software
- **Python** 3.10+ (3.11+ recommended)
- **Jupyter / Colab** for notebooks
- **PDF reader** for reports
- **Terminal** for CLI use

### Hardware / Platform Compatibility
- macOS / Linux / Windows (tested on macOS + Ubuntu)
- GPU **not required** (Torch optional)

### Usage Instructions
- Notebooks: open `examples/notebooks/*.ipynb` locally or in Colab.
- Reports: generated via `iints run-full` or `iints presets run`.

**Output / Artifacts**
- Notebook outputs, generated reports, and presentation-ready visuals.

---

## 4) AI Systems Documentation

**Purpose**
- Clarify how AI predictor research is documented and bounded by safety logic.

**When to use**
- When preparing model cards, datasheets, and forecast evaluation evidence.

IINTS‑AF is a **simulation platform**. It ships an **optional AI research pipeline** but does not
bundle a production‑trained model. The following documentation applies when using AI algorithms
inside the SDK:

### Model Card (Template)
**Model Name**: IINTS Predictor (research track)
- **Location**: `research/` and `src/iints/research/`
- **Purpose**: Forecast glucose 30-120 minutes ahead for Safety Supervisor foresight
- **Training Data**: *Not bundled in SDK* (user‑provided or synthetic)
- **Evaluation**: Compare using built‑in metrics (TIR, CV, LBGI/HBGI, safety violations)
- **Model card**: `research/model_card.md`

### Model Architecture
- LSTM predictor (time‑series forecaster)
- Safety Supervisor remains deterministic and always gates dosing

### Datasheet / Training‑Evaluation Notes
- The SDK includes **dataset registry** + **import tools** to support training data ingestion.
- Users should document the specific dataset, preprocessing, and evaluation protocol
  used for any trained model.
- Datasheet template: `research/datasheet.md`

### Recommended Evaluation Outputs
- Safety report (`safety_report`)
- Audit trail (`audit_trail.jsonl`)
- Clinical metrics (TIR, CV, GMI, LBGI/HBGI)

**Output / Artifacts**
- `model_card.md`, `datasheet.md`, forecast metrics, and safety/audit traces.

---

## Canonical Entry Points (Public)

- `../README.md` (start here)
- `COMPREHENSIVE_GUIDE.md`
- `TECHNICAL_README.md`
- `data_packs/DATA_SCHEMA.md`
- `data_packs/DATASETS.md`

---

## Notes on Scope
- IINTS‑AF is **pre‑clinical** software for research and validation.
- It is **not** approved for clinical use.
