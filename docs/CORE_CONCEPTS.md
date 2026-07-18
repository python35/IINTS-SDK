# Core Concepts

This page defines the terms used throughout IINTS-AF. Read it before changing patient parameters or interpreting a report.

## The Experiment Unit

An IINTS-AF run combines six explicit inputs:

| Input | Meaning |
| --- | --- |
| Patient model | The mathematical state and parameters of the virtual patient |
| Scenario | Timed events such as meals, exercise, stress, or sensor disturbances |
| Candidate algorithm | Experimental logic that proposes an action |
| Safety configuration | Deterministic limits that can restrict or reject a proposal |
| Execution settings | Duration, time step, seed, sensor settings, and output options |
| SDK version | The software implementation used to execute the run |

Changing any of these may change the result. A comparison is defensible only when the intended differences are controlled and documented.

## Virtual Patient

A virtual patient is a mathematical model, not a person. It represents selected glucose-insulin processes with state variables, parameters, and differential or discrete update equations.

The SDK includes several model families and extensions. They differ in complexity and intended use. A more complex model is not automatically more accurate; it needs calibration and validation for the question being studied.

Read [Physiology Reference](PHYSIOLOGY_REFERENCE.md) and [Formula Registry](FORMULA_REGISTRY.md) before making physiological claims.

## Scenario

A scenario describes what happens around the virtual patient. Examples include:

- a meal at a specified time
- a period of exercise
- stress or illness modifiers
- sensor noise, lag, bias, or dropout
- pump interruption or a hardware fault

The scenario is separate from the patient. This allows the same event schedule to be tested across patient profiles or algorithms.

## Candidate Algorithm

The candidate algorithm receives simulated observations and proposes an experimental action. It may be a fixed rule, PID controller, MPC experiment, learned model, or plugin.

Candidate output is not a treatment recommendation. It remains inside the simulation or bench-only workflow.

## Deterministic Safety Layer

The safety layer uses fixed, inspectable rules to check candidate actions. It can constrain, reject, or annotate behavior when configured limits are crossed.

This separation matters:

```text
candidate proposes -> deterministic rules check -> simulator applies allowed action
```

The safety layer is a research control, not proof of medical-device safety or regulatory compliance.

## Sensor And Pump Models

The simulated physiological glucose state and the reported sensor glucose are different concepts. Sensor models may add lag, filtering, noise, bias, and dropout.

Pump and transport models may represent delivery delay, quantisation, communication, or faults. They remain software or bench abstractions unless a specific hardware experiment states otherwise.

## Time Step And Seed

The **time step** controls how often the simulation advances. It affects event timing, numerical integration, and output resolution.

The **seed** controls pseudorandom components such as configured sensor noise or population sampling. Record it. Two runs are not a controlled comparison when their randomness changes unintentionally.

## Run Bundle

A run bundle is the evidence folder produced by an experiment. It normally contains a time-series file, report, metadata, hashes, and optional validation or audit artifacts.

The exact files depend on the command. Use [Understand A Run](RUN_OUTPUTS.md) to inspect them.

## Realism, Validation, And Clinical Validity

These terms are different:

- **Numerical validity:** the implementation runs without invalid states or solver failure.
- **Internal validation:** invariants and expected relationships hold in controlled tests.
- **Realism:** distributions or trajectories resemble the selected reference envelope.
- **External validation:** results are compared against independent real-world data not used for fitting.
- **Clinical validity:** the system is demonstrated to be suitable for a clinical purpose.

IINTS-AF supports the first four as research activities to varying degrees. It does not claim clinical validity.

## AI Roles

The SDK uses AI in clearly separated research roles:

1. explain or summarise completed runs
2. forecast glucose in dedicated prediction experiments
3. explore candidate controller behavior inside simulation
4. organise and compare large result collections

Authoritative formulas, safety thresholds, metrics, hashes, and simulation state transitions must remain deterministic code. AI output is untrusted commentary until checked against the recorded evidence.

Read [AI Assistant](AI_ASSISTANT.md) and [AI Safety Gates](LOCAL_AI_SAFETY_GATES.md) before enabling local models.

## MDMP Data Certification

MDMP is the SDK's data-contract and evidence workflow. It checks whether a dataset matches an explicit schema and quality policy, then records the result.

A certificate is evidence of checks performed under a named contract. It is not a clinical approval and does not make poor or biased data scientifically appropriate by itself.

Start with [Certification Quickstart](MDMP_QUICKSTART.md).
