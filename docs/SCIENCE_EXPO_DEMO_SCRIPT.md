# IINTS-AF Science Expo Demo Script (v0.1.15)
**3–4 Minute Demo Flow**

---

## Opening Statement (15–20s)

> “IINTS‑AF is a safety‑first simulation SDK for insulin algorithms. It turns invisible dosing decisions into audit‑ready evidence, so researchers can validate controllers before touching a patient.”

---

## Demo Flow

### 1) Quickstart: Run a Preset (60s)
**Action**
```bash
iints quickstart --project-name iints_demo
cd iints_demo
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

**What to show**
- Results CSV in `results/presets/`
- PDF report path printed in terminal

**Key points**
- End‑to‑end in one command
- Safety supervisor intervenes automatically

---

### 2) Audit Trail + PDF (45s)
**Action**
Open the generated PDF and audit summary JSON.

**What to show**
- PDF report (graphs + summary)
- `audit_summary.json` explaining safety overrides

**Key points**
- “Every decision is traceable.”
- “This is pre‑clinical validation evidence.”

---

### 3) Baseline Comparison (45s)
**Action**
Point to `baseline_comparison.csv` in `results/presets/baseline/`.

**Key points**
- Compare against PID + standard pump baselines
- Clear head‑to‑head metrics in one file

---

### 4) Data Registry + Import (40s)
**Action**
```bash
iints data list
iints data info sample
iints data fetch sample --output-dir data_packs/sample
iints import-demo --output-dir results/demo_import
```

**Key points**
- Built‑in demo dataset for offline use
- Citation + integrity checks included

---

### 5) Optional: Parallel Batch Run (30s)
**Action**
```bash
iints run-parallel --scenario presets/clinic_safe_baseline.json --algo algorithms/example_algorithm.py --runs 8
```

**Key points**
- Batch evaluation across multiple seeds
- Simple scale‑out path for research

---

## Closing Statement (10–15s)

> “IINTS‑AF doesn’t just run a simulation — it makes safety and outcomes measurable, reproducible, and explainable.”

---

## Common Questions (Short Answers)

**“Is this safe for real patients?”**
No — it’s a pre‑clinical simulator for research and validation only.

**“What makes it different?”**
SafetyConfig + independent supervisor + audit trail + baseline comparisons in one SDK.

**“How do you show it works?”**
We generate PDF reports, audit logs, and baseline comparisons for every run.

---

## Pre‑Demo Checklist
- [ ] `iints quickstart` run complete
- [ ] Open one PDF report
- [ ] Show `audit_summary.json`
- [ ] Show baseline comparison CSV
- [ ] Have demo data pack ready (`iints data fetch sample`)
