# IINTS-AF SDK Documentation

<div class="iints-hero">
  <div class="iints-hero-left">
    <img src="assets/iints-logo.png" alt="IINTS-AF logo" />
  </div>
  <div class="iints-hero-right">
    <h2>Safety-first research platform for insulin algorithm validation</h2>
    <p>Use IINTS-AF to simulate algorithms, enforce deterministic safety checks, and produce audit-ready artifacts.</p>
  </div>
</div>

<div class="iints-actions">
  <a href="GETTING_STARTED/" class="iints-action">Get Started</a>
  <a href="MDMP_QUICKSTART/" class="iints-action">MDMP Quickstart</a>
  <a href="https://github.com/python35/IINTS-SDK/tree/main/examples/demos" class="iints-action">Demos</a>
</div>

## Choose Your Path

<div class="iints-grid">
  <a class="iints-card-link" href="PLAIN_LANGUAGE_GUIDE/">
    <div class="iints-card">
      <h3>New to IINTS</h3>
      <p>Read a simple explanation of what the SDK does and what it does not do.</p>
    </div>
  </a>
  <a class="iints-card-link" href="GETTING_STARTED/">
    <div class="iints-card">
      <h3>Build First Run</h3>
      <p>Install, run a baseline scenario, and inspect generated outputs.</p>
    </div>
  </a>
  <a class="iints-card-link" href="MDMP_QUICKSTART/">
    <div class="iints-card">
      <h3>Validate Data (MDMP)</h3>
      <p>Apply contracts, compute grades, and export dashboards for audits.</p>
    </div>
  </a>
  <a class="iints-card-link" href="TECHNICAL_README/">
    <div class="iints-card">
      <h3>Engineering Reference</h3>
      <p>Use the full command reference and technical integration details.</p>
    </div>
  </a>
</div>

## 10-Minute Quick Start

```bash
pip install iints-sdk-python35

iints doctor --smoke-run
iints quickstart --project-name iints_quickstart
cd iints_quickstart
iints presets run --name baseline_t1d --algo algorithms/example_algorithm.py
```

Expected outputs include:
- `results.csv`
- `clinical_report.pdf`
- `audit/`
- `run_manifest.json`

## MDMP in 60 Seconds

MDMP is the IINTS data-quality protocol.

- `Contract`: required columns, types, units, and value bounds.
- `Validation`: dataset is checked against contract rules.
- `Grading`: output receives `draft`, `research_grade`, or `clinical_grade`.
- `Fingerprinting`: deterministic hashes support reproducibility and audits.

Use MDMP with:

```bash
iints mdmp template --output-path mdmp_contract.yaml
iints mdmp validate mdmp_contract.yaml data/my_cgm.csv --output-json results/mdmp_report.json
iints mdmp visualizer results/mdmp_report.json --output-html results/mdmp_dashboard.html
```

## Scope

- Research use only.
- Not a medical device.
- No clinical dosing advice.
