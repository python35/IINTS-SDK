# Hypoglycemia Science Model

Use this page when you need to explain why IINTS-AF is moving beyond a simple carbohydrate-insulin simulator and toward a deeper physiology model for hypoglycemia research.

Scope:
- pre-clinical simulation and retrospective AI research only
- not a treatment recommendation engine
- not a certified medical device
- not a replacement for clinical judgement, emergency glucagon instructions, or insulin-pump labeling

The short version:
- basic glucose-insulin models are useful, but they can miss the body's defense systems against hypoglycemia
- IINTS-AF documents those missing systems explicitly instead of hiding them
- the experimental hypoglycemia layer is organized around counterregulation, HAAF memory, exogenous glucagon, and renal glucose clearance

!!! warning "Research boundary"
    This page documents model design and literature grounding. It does not mean the SDK has been clinically validated for hypoglycemia prediction, treatment, glucagon dosing, or insulin delivery.

## 1. Why This Layer Exists

Many diabetes simulators start with the dominant axis:

```text
meal carbohydrates -> glucose rise -> insulin action -> glucose fall
```

That is necessary, but it is incomplete when the research question is hypoglycemia.

A real person is not passive during a low. The body tries to defend itself through:

| Defense system | What it tries to do | Why it matters for IINTS-AF |
| --- | --- | --- |
| falling insulin / low endogenous insulin | reduce glucose-lowering pressure | people with type 1 diabetes using exogenous insulin cannot simply switch off already absorbed insulin |
| glucagon | raise hepatic glucose output | primary fast counterregulatory hormone in acute hypoglycemia |
| epinephrine/adrenaline | raise glucose production and symptoms/awareness | becomes especially important when glucagon response is impaired |
| cortisol and growth hormone | slower support over longer stress windows | useful for illness, prolonged lows, and stress-state research |
| behavior and awareness | make the person treat the low | reduced awareness changes safety risk even if glucose dynamics look similar |
| renal glucose excretion | remove glucose at high levels | matters for hyperglycemia tails, not immediate hypo rescue, but improves whole-day mass balance |

## 2. Current Status Labels

IINTS-AF uses explicit status labels so we do not overclaim.

| Layer | Status | Meaning |
| --- | --- | --- |
| CGM metrics and hypo bands | implemented | TIR/TBR/TAR and low-glucose thresholds are reported from simulation data |
| supervisor safety rails | implemented | software safety rules can block risky actions in research runs |
| Bergman-style insulin action | implemented | glucose, plasma insulin, and remote insulin-action states are available in the mechanistic model |
| Hovorka-style state exposure | experimental | richer glucose/insulin/action compartments are available for research simulations |
| stress/illness modifiers | experimental | supported models can raise glucose production and reduce insulin sensitivity under stress events |
| endogenous counterregulation | experimental | Hovorka-style runs can raise EGP during lows, but parameters are not clinically calibrated hormone assays |
| HAAF memory | experimental | repeated lows can raise a bounded memory state that blunts rescue response; not a clinical awareness predictor |
| exogenous glucagon PK/PD | experimental | dual-hormone simulations can pass glucagon through depot/plasma/effect states with research-only safety caps |
| renal glucose clearance | experimental | high-glucose loss uses a smooth threshold/splay-style function, not a personalized renal model |

## 3. Endogenous Counterregulation

The design goal is to model a hormone response that activates when glucose becomes low or is falling quickly.

A simple research-form equation can be:

```text
low_drive = sigmoid((G_counter_threshold - G) / G_width)
trend_drive = max(0, -dG_dt - fall_rate_threshold)
response_drive = low_drive + trend_weight * trend_drive
```

Then hormone states can follow first-order dynamics:

```text
dGlucagon_dt   = k_glucagon_on * response_drive - k_glucagon_off * Glucagon
dEpinephrine_dt = k_epi_on * response_drive - k_epi_off * Epinephrine
```

The physiological effect is not direct insulin dosing. It is a modification of glucose production and utilization:

```text
EGP_effect = base_EGP + glucagon_gain * Glucagon + epi_gain * Epinephrine
insulin_sensitivity_effective = insulin_sensitivity * (1 - epi_resistance_gain * Epinephrine)
```

Why this is useful:
- a patient can rise from a low without eating in some scenarios
- a patient with impaired counterregulation can continue downward despite the same insulin exposure
- the SDK can test whether an algorithm recognizes "falling fast with insulin active" as different from "stable low"

## 4. HAAF Memory

HAAF means repeated antecedent hypoglycemia can blunt the next autonomic response and reduce symptom awareness. In a simulator, that needs memory.

A practical research state can be:

```text
dHAAF_dt = low_exposure_gain * time_below_range - recovery_rate * HAAF
HAAF = clamp(HAAF, 0, 1)
```

The model then reduces the strength of future rescue responses:

```text
glucagon_gain_effective = glucagon_gain * (1 - haaf_glucagon_blunt * HAAF)
epi_gain_effective = epi_gain * (1 - haaf_epi_blunt * HAAF)
symptom_awareness_score = 1 - HAAF
```

This matters because two virtual patients can have the same current glucose but different risk:
- patient A had no recent lows and mounts a strong response
- patient B had repeated lows and has reduced warning/rescue response

That difference is exactly the kind of hidden risk a research SDK should make visible.

## 5. Exogenous Glucagon PK/PD

For dual-hormone pump research, the model needs to separate:

```text
glucagon delivered -> subcutaneous depot -> plasma glucagon -> hepatic glucose effect
```

A minimum model is:

```text
dSC_glucagon_dt = delivered_glucagon - k_abs * SC_glucagon
dPlasma_glucagon_dt = k_abs * SC_glucagon - k_clear * Plasma_glucagon
glucagon_effect = effect_gain * delayed_or_filtered(Plasma_glucagon)
```

Why this matters:
- glucagon does not act instantly
- repeated microdoses can stack
- dual-hormone algorithms must be tested against delay, dose history, and failure modes
- simulated glucagon should not be treated as emergency medical instruction

## 6. Renal Glucose Clearance

Renal glucose excretion mostly matters at high glucose, but it helps whole-day realism. It should not be modeled as a sudden vertical cutoff.

A smooth threshold/splay form is better:

```text
renal_fraction = sigmoid((G - renal_threshold) / renal_splay_width)
renal_loss = renal_clearance_gain * renal_fraction * max(0, G - renal_threshold_soft)
```

Why this matters:
- very high glucose should not drift forever without any clearance path
- different renal thresholds can explain different high-glucose tails
- the model remains continuous, which is safer for simulation and AI training

## 7. How Users Should Interpret This

When reading an IINTS-AF run, ask:

| Question | Where to look |
| --- | --- |
| Was glucose low, or just trending low? | glucose trace, derivative, CGM lag |
| Was insulin already active? | IOB, active insulin, insulin effect |
| Did the simulated body mount a rescue response? | counterregulation states when enabled |
| Was awareness blunted by recent lows? | HAAF memory state when enabled |
| Was glucagon delivered by a dual-hormone controller? | exogenous glucagon depot/plasma/effect states when enabled |
| Was high glucose partly cleared by renal excretion? | renal clearance state when enabled |

For public demos, phrase it like this:

```text
IINTS-AF separates insulin delivery, insulin action, sensor measurement,
and the body's own defense systems. That makes hypoglycemia research more
transparent than a curve-only simulator.
```

## 8. Source Trail

Core model foundations:
- Bergman et al., *Quantitative estimation of insulin sensitivity*. DOI: [10.1152/ajpendo.1979.236.6.E667](https://doi.org/10.1152/ajpendo.1979.236.6.E667)
- Hovorka et al., *Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes*. DOI: [10.1088/0967-3334/25/4/010](https://doi.org/10.1088/0967-3334/25/4/010)
- Dalla Man, Rizza, and Cobelli, *Meal simulation model of the glucose-insulin system*. DOI: [10.1109/TBME.2007.893506](https://doi.org/10.1109/TBME.2007.893506)

Hypoglycemia and HAAF:
- Gerich, Davis, and Lorenzi, *Hormonal mechanisms of recovery from insulin-induced hypoglycemia in man*. DOI: [10.1152/ajpendo.1979.236.4.E380](https://doi.org/10.1152/ajpendo.1979.236.4.E380)
- Cryer, *Mechanisms of Hypoglycemia-Associated Autonomic Failure in Diabetes*. DOI: [10.1056/NEJMra1215228](https://doi.org/10.1056/NEJMra1215228)
- Cryer, *Hypoglycemia-associated autonomic failure in diabetes*. DOI: [10.1016/B978-0-444-53491-0.00023-7](https://doi.org/10.1016/B978-0-444-53491-0.00023-7)

Glucagon and dual-hormone research:
- Lv, Breton, and Farhy, *Pharmacokinetics modeling of exogenous glucagon in type 1 diabetes mellitus patients*. DOI: [10.1089/dia.2013.0150](https://doi.org/10.1089/dia.2013.0150)
- Haidar et al., *Glucose-responsive insulin and glucagon delivery in adults with type 1 diabetes*. DOI: [10.1503/cmaj.121265](https://doi.org/10.1503/cmaj.121265)
- Haidar et al., *Pharmacokinetics of Insulin Aspart and Glucagon in Type 1 Diabetes during Closed-Loop Operation*. DOI: [10.1177/193229681300700610](https://doi.org/10.1177/193229681300700610)

Renal glucose handling:
- Hummel et al., *Physiology of renal glucose handling via SGLT1, SGLT2 and GLUT2*. DOI: [10.1007/s00125-018-4656-5](https://doi.org/10.1007/s00125-018-4656-5)
- DeFronzo et al., *Characterization of Renal Glucose Reabsorption in Response to Dapagliflozin*. DOI: [10.2337/dc13-0387](https://doi.org/10.2337/dc13-0387)
