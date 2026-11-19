# IINTS-AF Examples

This directory contains runnable examples, step-by-step demos, and Jupyter notebooks for experimenting with closed-loop insulin algorithms, safety supervision, and population studies.

> **Note:** All examples are for research and educational simulation only — not for clinical use or treatment decisions.

## Where To Start

| Path | Description |
| :--- | :--- |
| [`demos/`](demos/README.md) | Curated, numbered walkthrough scripts (`01_basic_simulation.py` through `07_live_stage_demo.py`). Best starting point for new researchers. |
| [`notebooks/`](notebooks/README.md) | Interactive Jupyter notebooks (`00_Quickstart.ipynb` through `08_Data_Registry_and_Import.ipynb`) covering presets, safety supervisor ablations, audit trails, and data import. |
| [`data/`](data/) | Small bundled sample CGM CSV (`demo_cgm.csv`) for quick offline runs. |

## Standalone Scripts

### Simulation & Algorithm Prototyping
- `main.py` — Interactive menu runner for core SDK workflows.
- `demo_quickstart_flow.py` — Minimal end-to-end simulation flow.
- `demo_showcase.py` — Multi-feature showcase exporting structured outputs.
- `experiment.py` — Configurable single-experiment harness.
- `broken_algorithm_example.py` — Intentionally faulty (10x overdose) controller demonstrating how the Safety Supervisor intervenes.
- `mytest_algorithm.py` — Minimal custom algorithm fixture used in CLI validation tests.

### Benchmarking, Population Studies & Analysis
- `comparative_benchmarking.py` — Head-to-head comparison of multiple dosing controllers.
- `population_runner.py` / `run_population_study.py` — Virtual cohort simulation and batch evaluation.
- `create_population_graphics.py` — Visualization utilities for cohort-level results.
- `clinical_reality_check.py` — Checking simulated outputs against glycemic reference envelopes.
- `uncertainty_quantification.py` — Monte Carlo / sensitivity uncertainty analysis.
- `run_comprehensive_analysis.py` / `run_final_analysis.py` — Full multi-metric analysis pipelines.
- `run_reverse_engineering.py` — Exploratory controller behavior inspection.
- `visualize_algorithm_xray.py` — Detailed per-step trace inspection of controller decisions.

### Reporting, Import & Edge / External Tools
- `audit_and_report.py` — Audit-trail verification and report generation.
- `generate_clinical_report.py` / `generate_excel_summary.py` — PDF and Excel report exports.
- `data_import.py` — Importing external CGM/pump CSV traces into the SDK schema.
- `jetson_endurance_test.py` — Multi-day fault-injected endurance run for edge/auditor testing.
- `matlab_control_integration.py` — Exporting traces for MATLAB control-system analysis.
- `manual_scenario_editor.py` / `master_dashboard.py` / `real_time_dashboard.py` — Interactive scenario and dashboard scripts.
- `validate_system.py`, `test_validation.py`, `test_autonomous_learning.py`, `test_comparative_benchmarking.py`, `test_terminal_app.py` — End-to-end script checks invoked from `main.py` or manually.
