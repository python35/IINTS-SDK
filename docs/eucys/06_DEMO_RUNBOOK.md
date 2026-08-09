# EUCYS Live Demonstration Runbook

## Demo objective

The demonstration should answer one question:

> Can IINTS-AF make a doubtful diabetes-algorithm action visible, check it with
> deterministic safety logic, and preserve the evidence?

The demo is not a terminal performance and not a claim that the virtual patient
is clinically validated.

## Recommended duration

| Segment | Time |
| --- | ---: |
| Problem and boundary | 1 minute |
| Architecture | 2 minutes |
| Scientific model | 2 minutes |
| Live three-scenario experiment | 4 minutes |
| Evidence and result interpretation | 2 minutes |
| Limitations and feedback question | 1 minute |

Total target: `12 minutes`.

## Prepare the day before

Run:

```bash
iints --version
iints doctor
iints demo eucys \
  --output-dir results/eucys_live \
  --dry-run
```

Then run the complete demonstration once:

```bash
iints demo eucys \
  --output-dir results/eucys_live \
  --skip-ai \
  --evidence
```

Use `--skip-ai` for the primary live path. AI is optional and adds startup,
model and timing uncertainty. If local Ollama is already verified, show its
explanation after the deterministic evidence.

## Verify before presenting

- The poster opens.
- `results.csv` exists for each scenario.
- The safety/risk scenario contains at least one clear intervention or flag.
- The report and CSV describe the same duration.
- The displayed metrics match the CSV-derived output.
- No patient-identifying path, username or private dataset appears on screen.
- The output folder contains `DEMO_STORY.md` and
  `EUCYS_EXPERIMENT_SCRIPT.md`.
- A backup copy of the poster and report is available without rerunning.

## Screen order

Open these in advance:

1. This runbook.
2. The architecture diagram.
3. The generated three-panel poster.
4. One safety visualisation or audit record.
5. One `results.csv` excerpt.
6. The evidence manifest.
7. The limitations page.

Do not begin with source code, folder paths or logs.

## Presenter script

### 0:00 - Boundary and problem

Say:

> This is a virtual research patient, not a real patient and not a medical
> device. Diabetes algorithms must react to delayed sensor data, food, active
> insulin and changing physiology. My project creates a safe place to test and
> challenge those algorithms before any real-world claim.

Show: title and the one-sentence project description.

### 1:00 - Research question

Say:

> My question is whether an open-source simulation workbench can make risky or
> unrealistic algorithm behaviour visible and reproducible.

Show: research question, hypothesis and three-scenario structure.

### 2:00 - Architecture

Say:

> The patient model calculates the hidden physiological state. The sensor turns
> that into a delayed CGM-like observation. The algorithm proposes an action,
> but a separate deterministic supervisor can approve, reduce or block it.

Show: end-to-end Mermaid diagram.

Point out:

- patient state and sensor observation are separate
- candidate and delivered action are separate
- AI explanation is outside numerical authority
- evidence is written after every run

### 4:00 - Scientific model

Say:

> Food and insulin are delayed through compartments. The richer model separates
> accessible glucose mass, insulin depots and three insulin-action channels.
> The formulas are evaluated by deterministic code. The language model never
> solves them.

Show: F04, F05, F06 and F15. Do not attempt to explain all 15 formulas live.

### 6:00 - Run the experiment

Run:

```bash
iints demo eucys --output-dir results/eucys_live --skip-ai --evidence
```

If it was already run, say:

> I ran the same packaged command before the meeting so that we do not depend
> on live rendering time. I will now show the generated artifacts and can rerun
> any step if requested.

### 6:30 - Normal day

Say:

> This first panel is the control condition. I use it to check whether the
> patient, sensor and controller are stable before adding stress.

Show: normal panel. Point to meal markers, glucose trace and target band.

### 7:15 - Stress test

Say:

> The second panel adds a harder disturbance. The important question is not
> whether the curve looks attractive, but whether timing, rate of change and
> insulin action remain plausible.

Show: stress panel and one event.

### 8:00 - Risk and supervisor

Say:

> In the third scenario the candidate reaches a state that should be questioned.
> The deterministic safety layer records what was proposed, what was accepted,
> and why it intervened.

Show: intervention marker, then one underlying audit row.

Avoid saying "the supervisor proves safety." Say "the supervisor made this
configured risk visible and changed the simulated action."

### 9:30 - Evidence

Say:

> The graph is only the front page. The research evidence is the raw trace,
> configuration, seed, manifest, validation output and safety record behind it.

Show:

```text
results/eucys_live/
  DEMO_STORY.md
  EUCYS_EXPERIMENT_SCRIPT.md
  results/
  evidence_bundle/
```

Open only one representative `results.csv` and one manifest.

### 10:30 - Result and honest limitation

Say:

> The larger benchmark proves that the workflow scales to a locked matrix and
> exposes large differences between clean and corrupted inputs. It also exposes
> concerning low-glucose and intervention burdens. I therefore present it as a
> failure-discovery research result, not as evidence that a controller is
> clinically safe.

Show: result table and limitations.

### 11:30 - Closing

Say:

> The contribution is a transparent pre-clinical workflow: define, simulate,
> stress, supervise, validate and document. My next step is stronger
> calibration against held-out real data and independent validation of the
> failure cases.

Ask:

> Which validation step would you prioritise before trusting this simulator as
> a stronger research benchmark?

## If a judge asks for the AI

Say:

> The local model reads the completed run and explains supplied metrics. It
> cannot change the CSV, calculate missing values or approve an action.

Only then run or show:

```bash
iints ai report results/eucys_live/results/<scenario>
```

Verify every number against deterministic artifacts.

## If a judge asks for molecular biology

Show one AlphaFold structure or PAE view and say:

> This is structural context. AlphaFold confidence is not converted into
> insulin sensitivity or disease severity. Patient-level effects require
> independent functional evidence and a declared scenario assumption.

## Failure plan

| Failure | Response |
| --- | --- |
| Command is not found | Use the packaged app or verified environment; show `iints --version` |
| Demo takes too long | Open the rehearsed output and state that it came from the same command |
| Ollama is unavailable | Continue; AI is optional |
| Poster fails to open | Open the PNG directly from the output folder |
| Report generation fails | Show CSV, manifest and validation JSON |
| Unexpected result | Do not hide it; label it and use it as a failure case |
| Question is outside evidence | Say what is known, what is assumed and what must be tested |

## What not to say

Avoid:

- "The model perfectly simulates a patient."
- "The AI calculates the biology."
- "The supervisor guarantees safety."
- "This result proves clinical performance."
- "AlphaFold predicts insulin resistance."
- "This can control a real pump."

Prefer:

- "This is a documented research approximation."
- "The equation is evaluated deterministically."
- "The configured supervisor blocked this simulated proposal."
- "This result motivates the next validation experiment."

## Packing checklist

- Laptop and charger.
- Offline copy of the repository and docs.
- Pre-generated poster, report and CSV.
- A PDF copy of this dossier.
- A one-page jury quick reference.
- Optional bench hardware with a non-actuation label.
- No private health data.
