# Jury Quick Reference

## One-sentence pitch

IINTS-AF is an open-source research workbench that makes diabetes-algorithm
experiments reproducible, safety-aware and inspectable before any real-world
claim is made.

## Thirty-second answer

The SDK creates a virtual patient, adds meals, insulin, exercise and sensor
effects, and lets an algorithm propose simulated actions. A separate
deterministic supervisor checks those actions. Every run writes the raw trace,
settings, safety events and reports. Optional AI explains the result but has no
numerical or dosing authority.

## Five facts to remember

| Fact | Short answer |
| --- | --- |
| What is it? | A pre-clinical simulation and research SDK |
| What is the key design? | Proposal and safety permission are separate |
| How many registered formulas? | 15 deterministic formulas in registry v5 |
| What is the EUCYS experiment? | Normal, stress and risk scenarios plus a larger locked benchmark |
| Is it medical software? | No; research and education only |

## Likely questions

### What did you personally build?

An integrated SDK and desktop workbench covering virtual-patient simulation,
algorithm interfaces, deterministic safety checks, data validation, AI research
workflows, reports, evidence bundles and bench-hardware adapters.

### What is scientifically new?

The contribution is the combination of transparent physiology, explicit
candidate-versus-safety authority, data-quality gates and reproducible evidence
packaging in one open research workflow.

### Why not use one existing simulator?

Existing simulators remain important references. IINTS-AF focuses on an open,
inspectable workflow around algorithms, safety, data, reporting, AI and
hardware. It must still be compared with independent reference simulators.

### Does the AI calculate the formulas?

No. Python code evaluates the registered equations. The language model only
receives fixed formulas and recorded metrics as explanation context.

### Can the AI deliver insulin?

No. An experimental controller can propose a simulated action, but deterministic
safety logic remains the final simulated authority. The SDK must not control a
real pump.

### How do you know the physiology is realistic?

The model is grounded in Bergman-, Hovorka- and meal-model literature and has
plausibility checks. It is still a research approximation. Stronger held-out
real-data calibration and independent validation are explicit next steps.

### What is the strongest result?

The platform executed a locked 3600-run matrix and exposed measurable
differences between clean and corrupted inputs. Equally important, it exposed
low-glucose and intervention burdens that require further research.

### Why report bad outcomes?

Because a scientific test platform should reveal failures rather than optimise
the presentation. A negative or concerning result helps identify the next
experiment.

### What does MDMP add?

It checks an explicit data contract, provenance and quality before data is used
as evidence. Its certificate documents the check; it is not a clinical
approval.

### What do AlphaFold and ClinVar add?

They add structural and variant-classification context. They do not
automatically determine patient insulin sensitivity or a quantitative disease
effect.

### How is the work reproducible?

Runs record patient, scenario, algorithm, duration, step, seed, software
version, raw trace, validation output and file manifests.

### What would you do next?

Investigate low-glucose failure clusters, normalise safety interventions by
episode and time, calibrate against held-out subjects, and compare with an
independent reference simulator.

## Four formulas to explain live

Meal delay:

\[
\frac{dD_3}{dt}=k_{\mathrm{empt}}D_2-k_{\mathrm{abs}}D_3
\]

Insulin delay:

\[
\frac{dS_2}{dt}=kS_1-kS_2
\]

Hovorka glucose mass:

\[
\frac{dQ_1}{dt}
=-(\mathrm{NIMGU}+F_R)-x_1Q_1+k_{12}Q_2
+\mathrm{EGP}+U_G
\]

Sensor lag:

\[
\tau_{\mathrm{ISF}}\frac{d\mathrm{ISF}}{dt}
=\mathrm{BG}_{\mathrm{lagged}}-\mathrm{ISF}
\]

## Demo command

```bash
iints demo eucys \
  --output-dir results/eucys_live \
  --skip-ai \
  --evidence
```

## Safe closing line

> IINTS-AF does not prove that an algorithm is safe for people. It makes a
> pre-clinical experiment easier to reproduce, inspect and challenge.
