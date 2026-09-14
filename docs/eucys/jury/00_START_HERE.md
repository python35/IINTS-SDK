# Start here

This folder is the evidence behind the project: an SDK for simulating type 1 diabetes
physiology and testing closed-loop insulin dosing algorithms against it.

Everything here is either a measurement taken from the code, an output of a simulation
run, or a document from the repository. Nothing has been re-typed by hand. Every file
appears in `MANIFEST.csv` with its size, its SHA-256 hash, and the path it came from,
so any single file can be traced back to its origin.

Read `PROVENANCE_NOTES.md` next. It states what each provenance class means, what was
found to be broken and fixed during this build, and what is deliberately *not* in here.

## The ten-minute route

1. `03_PHYSIOLOGICAL_MODEL/physiology_full_day.png` — one simulated day. The top panel
   is what a controller would see; the bottom panel is the differential-equation fluxes
   inside the model that produce it. This is the core of the project: not a curve fit,
   but a mechanistic model.
2. `02_PUMP_MECHANICS/pump_delivery_and_iob.png` — the pump side. Discrete boluses,
   continuous basal, and the resulting insulin on board. The third panel shows why
   dosing is hard: insulin delivered is not insulin acting, because the subcutaneous
   depot delays every unit.
3. `04_AI_ALGORITHMS/controller_comparison.png` — four shipped controllers on an
   identical day with an identical seed. The controller with the best time in range is
   also the only one that goes hypoglycaemic. This is the honest result, and the reason
   time in range alone is not a verdict.
4. `08_FAILURE_CASES/hypo_failure_case.png` — a run that fails. Glucose reaches
   59 mg/dL and the safety supervisor intervenes on 102 steps. A simulator that cannot
   produce bad outcomes cannot be used to look for them.
5. `07_VALIDATION/replay_check.json` — three identical runs of the same scenario
   produce byte-identical result hashes and safety hashes. Without this, none of the
   numbers above would mean anything.

## Measured during this build

| What | Value | File |
| --- | --- | --- |
| Test suite | 681 passed, 5 skipped, 161 s | `07_VALIDATION/pytest_summary.txt` |
| Deterministic replay | 3 runs, identical result and safety hashes | `07_VALIDATION/replay_check.json` |
| Simulation throughput | 141.4 steps/s, 7.07 ms mean step latency | `02_PUMP_MECHANICS/edge_benchmark.json` |
| Peak memory of one run | 453 MB | `02_PUMP_MECHANICS/edge_benchmark.json` |
| Source size | 332 Python files, 108 282 lines | `10_SOURCE_CODE/code_stats.csv` |
| Test size | 176 test files, 24 600 lines | `10_SOURCE_CODE/code_stats.csv` |
| History | 961 commits, 2026-02-01 to 2026-09-09 | `10_SOURCE_CODE/git_history.txt` |

The throughput figure was measured on the development laptop (Darwin, arm64), over 288
simulation steps. It is not a measurement on embedded hardware; see
`PROVENANCE_NOTES.md`.

## Simulated during this build

Five runs, all with seed 42, all reproducible with the scripts in `10_SOURCE_CODE/`.
Raw per-step traces and the metrics computed from them are in `06_RESULTS/` and
`08_FAILURE_CASES/`.

| Run | Time in range 70-180 | Time below 70 | Insulin delivered | Lowest glucose |
| --- | --- | --- | --- | --- |
| Clinical baseline | 56.7 % | 0.0 % | 40.1 U | 127 mg/dL |
| MPC | 50.5 % | 0.0 % | 34.1 U | 114 mg/dL |
| PID | 79.2 % | 6.2 % | 94.3 U | 67 mg/dL |
| Fixed basal-bolus | 43.6 % | 0.0 % | 32.6 U | 142 mg/dL |
| Clinical baseline, hypo-prone night | 91.0 % | 7.6 % | 11.0 U | 59 mg/dL |

The insulin column is what the pump *delivered*, which is not what the algorithm asked
for. In the clinical baseline run the algorithm commanded 43.3 U and the safety
supervisor withheld 3.2 U of it across 109 of 289 steps; in the hypo-prone night it
commanded 17.3 U and 6.4 U was withheld — and that run still reached 59 mg/dL. Panel c
of `02_PUMP_MECHANICS/pump_delivery_and_iob.png` shows this gap step by step.

Two things in this table are worth a question from a juror. PID buys its time in range
with 2.3 times the insulin of the clinical baseline and 6.2 % of the day below
70 mg/dL. And the hypo-prone night scores the highest time in range of all five runs
while also being the most dangerous — which is exactly the failure mode that a single
headline metric hides.

## Folder guide

| Folder | What is in it |
| --- | --- |
| `01_ARCHITECTURE` | How the SDK is put together: diagrams and the written architecture. |
| `02_PUMP_MECHANICS` | Pump delivery and insulin kinetics, plus the performance benchmark and the embedded-target design documents. |
| `03_PHYSIOLOGICAL_MODEL` | The physiology: which compartments and equations, and the reference briefs. |
| `04_AI_ALGORITHMS` | The controllers that ship with the SDK and their head-to-head comparison. |
| `05_EXPERIMENTS` | Experiment designs and protocols. |
| `06_RESULTS` | Result tables and the raw per-step traces behind the figures. |
| `07_VALIDATION` | Test suite output, determinism check, and the earlier validation against the Ohio dataset. |
| `08_FAILURE_CASES` | Runs that go wrong, on purpose: hypoglycaemia, supervisor overrides, worst-case sweeps. |
| `09_LITERATURE` | The formula registry: every equation in the model with its literature basis and its evidence class. |
| `10_SOURCE_CODE` | Code statistics, git history, licence, and the two scripts that produced the computed files in this folder. |

All figures and tables in this folder are in English; some documents copied from the
repository are in Dutch.
