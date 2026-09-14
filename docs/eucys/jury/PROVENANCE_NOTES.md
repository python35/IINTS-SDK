# Provenance notes

The point of this folder is that a juror can check any claim without taking a word on
trust. This document records how each file got here, what was found to be wrong along
the way, and what is not included.

## Provenance classes

Every row in `MANIFEST.csv` carries one of five classes.

| Class | Meaning |
| --- | --- |
| `MEASURED` | A measurement taken during this build: test runs, timing, hashes. Re-runnable. |
| `SIMULATED` | Output of a simulation run made during this build, with a fixed seed. Re-runnable. |
| `DERIVED` | Computed from earlier stored results in the repository, not re-run here. |
| `DOCUMENT` | Written text from the repository. A claim, not evidence. |
| `DIAGRAM` | A drawing. Illustrates a design; measures nothing. |

The distinction between the first two and the last two matters. `DOCUMENT` and
`DIAGRAM` files describe what the system is meant to do. Only `MEASURED` and
`SIMULATED` files show what it actually did, and only those two classes were produced
under observation during this build.

## Reproducing the computed files

`10_SOURCE_CODE/compute_runs.py` produced the five runs; `10_SOURCE_CODE/make_figures.py`
produced the four computed figures from those runs. Both read only the SDK and their own
run outputs. The runs use seed 42 and the SDK's own metrics and safety modules — no
metric in this folder was computed by hand.

Units on the figures are not typed in by hand either. The insulin and glucose flux units
come from the SDK's own compartment schema, queried at plotting time. An earlier draft of
the physiology figure carried a hand-written flux unit that was simply wrong; the schema
was the fix.

## A determinism bug found and fixed during this build

The determinism check in `07_VALIDATION/replay_check.json` passes now. It did not when
this build started.

Three repeats of the same scenario produced identical `results_hash` values but a
different `safety_hash` every time. The simulation was never the problem: the safety
report included the absolute path of the temporary directory that the run happened to
write to, and the run number was part of that path. The hash was therefore hashing the
location of the output rather than the output.

The filter that removes non-deterministic fields from a report only looked at the top
level of the structure, and the offending field was nested. Two things changed in
`src/iints/validation/replay.py`: the filter now normalises run paths at any depth, and
a regression test was added for the real case. The existing test had stubbed the
simulation with a fixed report, which is why the bug had stayed invisible.

The `PASS` in this folder is a post-fix measurement. A juror who checks out the
repository at an earlier commit will see it fail.

## A mislabelled figure found and fixed during this build

The first version of `02_PUMP_MECHANICS/pump_delivery_and_iob.png` called its insulin
panels "delivery". They were not. The daily totals on the figure disagreed with the
totals in `06_RESULTS/run_metrics.csv` by 3.2 U, which is what prompted the check.

In the per-step traces, `bolus_insulin_units` and `basal_insulin_units` are what the
algorithm *commands*, and `delivered_insulin_units` is what remains after the safety
supervisor approves it and the pump model executes it. This was verified in
`src/iints/core/simulator.py`, where the recommendation passes through
`safety_result["approved_insulin"]` and then `pump_model.deliver()` before being
recorded. The two quantities differ on every step where `safety_triggered` is set, and
nowhere else.

The figure now shows both, and the withheld amount is a panel of its own. The
discrepancy was a labelling error, not a simulation error, but a juror reading the old
figure would have been misled about what the pump did.

## The formula registry is honest about its own status

`09_LITERATURE/formula_registry.csv` lists all 16 equations in the physiological model
with their literature basis and an evidence class. The counts are:

- 0 canonical
- 7 adapted
- 9 heuristic

No equation in this model is a verbatim reproduction of a published, validated model.
Seven are adaptations of published models — Bergman, Hovorka, subcutaneous insulin
two-compartment kinetics, meal absorption, glucagon PK/PD, CGM observation. Nine are
heuristics: plausible functional forms chosen to produce realistic behaviour, including
the circadian dawn effect, exercise-driven glucose uptake, the hypoglycaemia rescue
response, and hypoglycaemia-associated autonomic failure memory.

This is a legitimate basis for an algorithm testbed and it is not a basis for any claim
about a real patient. The registry says so per equation, which is the right place for it
to be said.

## What is deliberately not in this folder

**Three figures planned for the poster.** `poster_loss_convergence.pdf`,
`poster_error_distribution.pdf` and `poster_supervisor_intervention.pdf` are referenced
in `EUCYS_POSTER_ASSETS/POSTER_TEXT_GUIDE.md` and `POSTER_FINAL_TEXT.md`, but no script
anywhere in the repository generates them. Their numbers cannot be traced to a run, so
they are not evidence and they are not here. Either regenerate them from a run whose
trace can be shipped, or leave them off the poster.

**Embedded hardware measurements.** `02_PUMP_MECHANICS` contains design documents for
Arduino, Jetson and FPGA targets. The only timing measurement in this folder was taken
on the development laptop. No number here was measured on any of those devices, and the
benchmark file records the platform it ran on.

**Clinical or human data.** Nothing in this folder comes from a person. Every trace is
simulator output. The Ohio dataset comparison in `07_VALIDATION` is a `DERIVED` result
stored earlier in the repository; it was not re-run during this build, and it compares
model predictions against a public research dataset, not against patients of this
project.

**Anything the SDK does not itself compute.** Where a number appears in a figure title
or in `00_START_HERE.md`, it was read out of a run file or a measurement file in this
folder. If a claim could not be traced to such a file, it was dropped rather than
softened.

## The limits a juror should press on

- The model is an adapted-and-heuristic construction, validated for behavioural realism
  rather than against a clinical cohort. Ask what would falsify it.
- The controller comparison is a single day, a single patient configuration and a single
  seed per controller. It is enough to show that the ranking depends on which metric you
  choose; it is not enough to rank the controllers.
- Determinism was verified for one scenario at one seed, repeated three times. That is
  the right check and it is a narrow one.
- The safety supervisor acts frequently — 102 to 204 steps depending on the run. Whether
  that is a well-tuned guard or a controller leaning on its safety net is a fair question,
  and the raw traces in `06_RESULTS` and `08_FAILURE_CASES` are what you would answer it
  with.
