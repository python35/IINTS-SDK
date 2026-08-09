# Scientific Mechanisms And Their Limits

IINTS-AF is a **research and educational simulator**, not a clinically validated
patient model or medical device. This page explains what the deterministic code
implements and, equally importantly, what each mechanism does **not** prove.

The authoritative equation list is the [Formula Registry](FORMULA_REGISTRY.md).
Every entry is labelled as:

- `canonical`: a direct published equation;
- `adapted`: a published model modified for SDK integration;
- `heuristic`: an explicit, testable research assumption.

Published equations do not automatically validate the combined implementation,
its parameters, or its use for an individual person.

## 1. Bergman-Style Glucose Balance (`F01`)

The Bergman mode uses a concentration-domain glucose balance with a remote
insulin-action state. Meal appearance, renal loss, exercise, stress, dawn and
glucagon terms are IINTS extensions. It is therefore **Bergman-style**, not an
unmodified implementation of the 1979 protocol model.

## 2. Remote Insulin Action (`F02`)

The action state follows insulin above or below a basal reference:

$$
\frac{dX}{dt}=-p_2X+p_{3,\mathrm{eff}}(I-I_{\mathrm{ref}}).
$$

Here, \(I_{\mathrm{ref}}\) is the pump-supported fasting concentration derived
from basal delivery, distribution volume and clearance. Allowing a negative
deviation matters after interrupted basal delivery; clipping the deviation at
zero would hide the loss of basal insulin action.

## 3. Plasma Insulin And Optional Graft Secretion (`F03`)

Plasma insulin is cleared by a first-order term and receives absorbed
subcutaneous insulin. Optional beta-cell or graft secretion is an **experimental
heuristic**, disabled in standard T1D profiles. It is not a transplantation
outcome predictor.

## 4. Subcutaneous Insulin PK (`F04`)

Two serial depots delay pump delivery before plasma appearance. The structure is
literature-based; insulin-type time constants are predefined research profiles,
not product-specific bioequivalence claims.

## 5. Meal Appearance (`F05`)

Hovorka mode uses its published two-compartment meal chain. Bergman and advanced
modes use an adapted stomach/gut chain. The latter must not be called an exact
Dalla Man implementation: it does not reproduce the complete Dalla Man meal
submodel or its nonlinear gastric-emptying function.

## 6. Hovorka Glucose Mass Balance (`F06`)

Accessible and non-accessible glucose are represented as mass compartments.
The base balance follows Hovorka structure, while stress, exercise, circadian,
renal, HAAF and glucagon terms are declared extensions. Mass is converted to
concentration only through the configured glucose distribution volume.

## 7. Hovorka Insulin-Action Channels (`F07`)

Three action channels affect distribution, disposal and endogenous glucose
production. Tissue and molecular-affinity scalars are scenario stressors, not
quantitative consequences inferred from AlphaFold, GTEx or ClinVar. Structural
confidence and gene-expression evidence remain contextual evidence only.

## 8. Stress And Exercise States (`F08`)

Stress and exercise are filtered scenario inputs that alter sensitivity and
glucose fluxes. They are not measured cortisol, adrenaline, AMPK or lactate
concentrations. Their coefficients require empirical calibration before a
specific population claim can be made.

## 9. Exercise/GLUT4 Abstraction (`F09`)

The bounded `GLUT4_active` state adds insulin-independent glucose uptake during
exercise. It captures a plausible direction of effect but is not a molecular
translocation model and cannot be validated by a protein structure image alone.

## 10. Circadian/Dawn Term (`F10`)

A gated Fourier profile can perturb endogenous glucose production. It is off or
weak in baseline profiles and is a configurable circadian stressor, not a model
of measured growth hormone or cortisol secretion.

## 11. Counter-Regulatory Rescue (`F11`)

Low glucose activates a bounded increase in endogenous glucose production. The
response is reduced by the antecedent-hypoglycaemia memory state. This tests the
direction of counter-regulation; it does not estimate a patient's glucagon or
epinephrine response.

## 12. Antecedent-Hypoglycaemia Memory (`F12`)

The HAAF-like state accumulates gradually during hypoglycaemia and recovers over
days. It is deliberately bounded and labelled heuristic. It must never be
reported as a diagnosis of impaired awareness or HAAF.

## 13. Exogenous Glucagon PK/PD (`F13`)

The two-state absorption and clearance equations use literature-informed ranges,
with exact mass conversion (`1 mg = 10^9 pg`). The bounded concentration-effect
coupling is an IINTS adaptation and has not been established as a dose-selection
model for patient care.

## 14. Renal Glucose Loss (`F14`)

A smooth softplus approximation replaces a discontinuous renal threshold. This
is numerically useful and represents threshold/splay behaviour qualitatively,
but renal threshold varies between and within people. It is not an eGFR or renal
disease model.

## 15. CGM Observation Model (`F15`)

Latent blood glucose passes through an explicit dead time and a first-order
blood-to-interstitial compartment before bias, drift, dropout and seeded noise
are applied. This separates physiology from measurement. It is a generic CGM
observation model, not a Dexcom-, Libre- or Medtronic-equivalent sensor model.

## Additional Advanced Stressors

The advanced model also exposes FFA, ketone, protein, fat, illness, menstrual
cycle, beta-cell and cannula-age states. These are hypothesis-generating stress
tests. In particular:

- ketone state is not a DKA diagnosis;
- menstrual phase is not inferred from hormone measurements;
- cannula ageing is not a device-specific failure probability;
- protein-to-glucose conversion is a simplified delayed flux;
- beta-cell/graft states are not transplant efficacy predictions.

## AI, Calibration And Validation Boundary

LLMs may summarize deterministic outputs but never calculate the ODEs, safety
limits or formula values. The glucose predictor's physiology-aware loss is a
regularizer, not a full ODE-residual PINN. Parameter calibration estimates a
small bounded profile from CGM and event data; those parameters are generally
not uniquely identifiable and require held-out and external validation.

Scientific use therefore requires reporting the model version, parameter set,
seed, input semantics, evidence class, numerical checks, held-out performance,
failure cases and limitations with every result.
