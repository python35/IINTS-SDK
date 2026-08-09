# IINTS-AF Formula Registry

Registry version: `iints-formula-registry-v5`

These formulas document deterministic SDK code. The local AI may explain them, but it must not derive, solve, or alter them.

Evidence classes: `canonical` means a direct published equation, `adapted` means a published model changed for SDK integration, and `heuristic` means an explicit research assumption that requires calibration and external validation.

No registry entry, including a canonical equation, establishes clinical validity for its parameterization or for the combined simulator.

## F01_BERGMAN_GLUCOSE_RHS: Bergman-style glucose concentration balance

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\frac{dG}{dt}=-(p_{1,\mathrm{eff}}+X)G+p_{1,\mathrm{eff}}G_{b,\mathrm{eff}}+R_a+D_{\mathrm{dawn}}-U_{\mathrm{exercise}}-F_R
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dG/dt = -(p1_eff + X)G + p1_eff*Gb_eff + Ra + dawn - U_exercise - F_R
```

</details>

Runtime/solved form: Integrated by scipy.solve_ivp over each simulator step; glucose transition is rate-guarded after integration.

Units: G in mg/dL, rates in mg/dL/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1152/ajpendo.1979.236.6.E667), [source 2](https://doi.org/10.1109/TBME.2007.893506)

Validation note: Research extension of Bergman minimal-model dynamics with added meal, renal, exercise, dawn, glucagon, and HAAF terms.

## F02_BERGMAN_REMOTE_INSULIN: Remote insulin action

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\frac{dX}{dt}=-p_2X+p_{3,\mathrm{eff}}(I-I_{\mathrm{ref}})
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dX/dt = -p2*X + p3_eff*(I - I_ref)
```

</details>

Runtime/solved form: First-order action compartment integrated inside the Bergman ODE RHS.

Units: X in 1/min, I in mU/L

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1152/ajpendo.1979.236.6.E667)

Validation note: X is a deviation-from-reference action state and can become negative when insulin falls below the pump-supported fasting reference concentration. I_ref is derived from basal delivery, distribution volume and clearance; the configured Ib is used only when basal delivery is zero.

## F03_PLASMA_INSULIN_BALANCE: Plasma insulin balance with optional graft secretion

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}q_{\mathrm{sec}}&=\gamma M_{\mathrm{graft}}\max\!\left(G-h,0\right)V_I\\\frac{dI}{dt}&=-nI+\frac{q_{\mathrm{sec}}(1-f_{\mathrm{subq}})+R_{a,I}}{V_I}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
q_sec=gamma*M_graft*max(G-h,0)*V_I; dI/dt=-n*I+(q_sec*(1-f_subq)+Ra_I)/V_I
```

</details>

Runtime/solved form: Integrated in Bergman mode; gamma defaults to 0 for T1D research profiles. If f_subq>0, graft secretion first enters the S1/S2 absorption chain.

Units: I in mU/L, Ra_I in mU/min, V_I in L

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1152/ajpendo.1979.236.6.E667)

Validation note: Stem-cell/islet graft secretion is an experimental abstraction and is disabled by default for T1D simulation.

## F04_SUBCUT_INSULIN_TWO_DEPOT_PK: Two-depot subcutaneous insulin absorption

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\begin{aligned}q_{\mathrm{sec}}&=\gamma M_{\mathrm{graft}}\max\!\left(G-h,0\right)V_I\\\frac{dS_1}{dt}&=u_I+q_{\mathrm{sec}}f_{\mathrm{subq}}-kS_1\\\frac{dS_2}{dt}&=kS_1-kS_2\\U_I&=kS_2\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
q_sec=gamma*M_graft*max(G-h,0)*V_I; dS1/dt = u_I + q_sec*f_subq - k*S1; dS2/dt = k*S1 - k*S2; U_I = k*S2
```

</details>

Runtime/solved form: Bergman uses k_a; Hovorka uses 1/t_max_I. The state equations are integrated each step.

Units: S1/S2 in mU, u_I/U_I in mU/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1088/0967-3334/25/4/010)

Validation note: Runtime code chooses the insulin absorption time constant deterministically from configured insulin type.

## F05_MEAL_ABSORPTION_CHAIN: Meal absorption and glucose appearance

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\begin{aligned}\frac{dD_1}{dt}&=-\frac{D_1}{t_{\max,G}}\\\frac{dD_2}{dt}&=\frac{D_1}{t_{\max,G}}-\frac{D_2}{t_{\max,G}}\\U_G&=A_G\frac{D_2}{t_{\max,G}}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
Hovorka: dD1/dt=-D1/tmaxG; dD2/dt=D1/tmaxG-D2/tmaxG; U_G=A_G*D2/tmaxG. Bergman adaptation: Q_sto1 -> Q_sto2 -> Q_gut.
```

</details>

Runtime/solved form: Hovorka mode uses the published two-compartment chain. Bergman and advanced modes retain an explicitly adapted stomach/gut chain.

Units: carbohydrate mass in mg; U_G/Ra in mg/min or mg/dL/min after volume scaling

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1088/0967-3334/25/4/010), [source 2](https://doi.org/10.1109/TBME.2007.893506)

Validation note: The Hovorka branch follows its published meal equations; the Bergman three-stage branch is an IINTS adaptation and requires dataset calibration.

## F06_HOVORKA_GLUCOSE_MASS_BALANCE: Hovorka-style accessible/non-accessible glucose mass balance

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\begin{aligned}\frac{dQ_1}{dt}&=-(\mathrm{NIMGU}+F_R)-x_1Q_1+k_{12}Q_2+\mathrm{EGP}_0\max\!\left(0,1-x_3+x_{\mathrm{gluc}}\right)+U_G+V_G r_{\mathrm{dawn}}\\\frac{dQ_2}{dt}&=x_1Q_1-(k_{12}+x_2)Q_2\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dQ1/dt = -(NIMGU + F_R) - x1*Q1 + k12*Q2 + EGP0*max(0, 1 - x3 + x_gluc) + U_G + V_G*dawn_rate; dQ2/dt = x1*Q1 - (k12 + x2)*Q2
```

</details>

Runtime/solved form: Integrated by scipy.solve_ivp; concentration is G = Q1 / V_G_dL after integration.

Units: Q1/Q2 in mg, G in mg/dL, mass fluxes in mg/min

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1088/0967-3334/25/4/010)

Validation note: Research Hovorka-style RHS with explicit extensions for glucagon, renal loss, exercise, stress, and a phenomenological dawn input.

## F07_HOVORKA_INSULIN_ACTION_CHANNELS: Hovorka-style insulin action channels

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}\frac{dx_1}{dt}&=-k_{a1}x_1+k_{b1}I\\\frac{dx_2}{dt}&=-k_{a2}x_2+k_{b2}I\\\frac{dx_3}{dt}&=-k_{a3}x_3+k_{b3}I,\qquad k_{b,i}\propto A_{\mathrm{affinity}}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dx1/dt=-ka1*x1+kb1*I; dx2/dt=-ka2*x2+kb2*I; dx3/dt=-ka3*x3+kb3*I; kb_i includes molecular_affinity_scalar
```

</details>

Runtime/solved form: kb1/kb2/kb3 are deterministic sensitivity products before ODE integration.

Units: x1/x2 in 1/min-like action states; x3 dimensionless research action

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1088/0967-3334/25/4/010)

Validation note: The Hovorka action-channel topology is literature based, but the overall, tissue-specific, and molecular-affinity scalars are explicit IINTS scenario assumptions. They are not inferred from AlphaFold confidence or ClinVar labels and require independent calibration before quantitative interpretation.

## F08_STRESS_EXERCISE_SENSITIVITY: Stress/exercise sensitivity and EGP multipliers

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}\frac{dH_{\mathrm{stress}}}{dt}&=\frac{H_{\mathrm{stress,target}}-H_{\mathrm{stress}}}{20}\\\frac{dH_{\mathrm{exercise}}}{dt}&=\frac{H_{\mathrm{exercise,target}}-H_{\mathrm{exercise}}}{10}\\S_{\mathrm{overall}}&=(1-0.7H_{\mathrm{stress}})(1+2H_{\mathrm{exercise}})\\\mathrm{EGP}_{\mathrm{stress}}&=1+0.5H_{\mathrm{stress}}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dH_stress/dt=(target_stress-H_stress)/20; dH_exercise/dt=(target_exercise-H_exercise)/10; S_overall=(1-0.7*H_stress)*(1+2*H_exercise); EGP_stress=1+0.5*H_stress
```

</details>

Runtime/solved form: Pseudo-hormone states are first-order deterministic filters of scenario inputs.

Units: dimensionless states, time constants in minutes

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1371/journal.pone.0248280), [source 2](https://doi.org/10.1152/ajpendo.00084.2021)

Validation note: The sources support the direction of exercise effects on glucose effectiveness, insulin sensitivity, and insulin-independent uptake. The IINTS filter time constants and multiplier coefficients are heuristic scenario parameters, not values identified or clinically validated by those studies. The stress branch is not a cortisol/adrenaline concentration model.

## F09_GLUT4_NIMGU_EXERCISE: Exercise-driven GLUT4/NIMGU state

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}\frac{d\mathrm{GLUT4}}{dt}&=k_{\mathrm{act}}H_{\mathrm{exercise}}(1-\mathrm{GLUT4})-k_{\mathrm{deact}}\mathrm{GLUT4}\\\mathrm{NIMGU}&=F_{01c}(1+1.5\,\mathrm{GLUT4})\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dGLUT4/dt = k_act*H_exercise*(1-GLUT4) - k_deact*GLUT4; NIMGU = F_01c*(1 + 1.5*GLUT4)
```

</details>

Runtime/solved form: Exercise can increase non-insulin-mediated glucose uptake without LLM calculation.

Units: GLUT4 dimensionless, NIMGU in mg/min

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1371/journal.pone.0248280), [source 2](https://doi.org/10.1152/ajpendo.00084.2021)

Validation note: The sources support exercise-mediated insulin-dependent and insulin-independent glucose utilization. GLUT4 is a bounded latent scenario state here; its activation/deactivation constants and 1.5 uptake multiplier are IINTS heuristics, not measured receptor abundance or a cell-level translocation assay.

## F10_CIRCADIAN_DAWN_EGP: Phenomenological dawn glucose-rate perturbation

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}w(t)&=\tfrac12\left[1+\cos\!\left(\frac{\pi(t-t_m)}{h}\right)\right]\quad (|t-t_m|\le h)\\r_{\mathrm{dawn}}(t)&=\frac{s_{\mathrm{dawn}}}{60}w(t)\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
w(t)=0.5*[1+cos(pi*(t-t_mid)/h)] inside the configured window, otherwise 0; dawn_rate=(s_dawn/60)*w(t)
```

</details>

Runtime/solved form: Computed directly in mg/dL/min. Concentration-domain models add the rate; Hovorka converts it to glucose mass flow using V_G.

Units: dawn_phenomenon_strength: mg/dL/hour; runtime rate: mg/dL/min

Implementation: `src/iints/core/patient/physiology.py:dawn_glucose_rate_mgdl_min`, `src/iints/core/patient/models.py:update`, `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`, `src/iints/core/patient/advanced_metabolic_model.py:_ode`

Literature basis: [source 1](https://pubmed.ncbi.nlm.nih.gov/35466006/), [source 2](https://doi.org/10.1089/dia.2015.0011)

Validation note: The sources support circadian variation in glucose regulation. The raised-cosine shape and configured peak rate are an IINTS scenario heuristic, not coefficients estimated from either study. The default is zero, and all backends use the same declared unit.

## F11_HYPO_RESCUE_MULTIPLIER: Endogenous hypoglycemia rescue multiplier

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}\Delta_{\mathrm{hypo}}&=\max(0,70-G)\\a&=\frac{\Delta_{\mathrm{hypo}}}{16+\Delta_{\mathrm{hypo}}}\\R_{\mathrm{rescue}}&=1+a(1-\mathrm{HAAF})\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
Delta=max(0,70-G); a=Delta/(16+Delta); R_rescue=1+a*(1-HAAF)
```

</details>

Runtime/solved form: Computed directly inside ODE RHS before effective EGP is assembled.

Units: G in mg/dL; multiplier dimensionless

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1152/ajpendo.2001.281.6.E1115), [source 2](https://doi.org/10.1210/jcem.84.5.5675)

Validation note: Captures the concept of blunted counterregulation; not a diagnostic HAAF model.

## F12_HAAF_MEMORY: Hypoglycemia-associated autonomic failure memory

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}s&=\operatorname{clip}\!\left(\frac{70-G}{16},0,1.5\right)\\\frac{d\mathrm{HAAF}}{dt}&=\frac{s(1-\mathrm{HAAF})}{360}-\frac{\mathrm{HAAF}}{4320}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
severity=clip((70-G)/16,0,1.5); dHAAF/dt=severity*(1-HAAF)/360-HAAF/4320
```

</details>

Runtime/solved form: Integrated as a bounded state. Material excursions outside [0, 1] fail the step; only solver-scale numerical tolerance is projected back.

Units: dimensionless memory state, rates in 1/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1152/ajpendo.2001.281.6.E1115), [source 2](https://doi.org/10.1210/jcem.84.5.5675)

Validation note: Research memory state only; never report as clinical hypo-awareness diagnosis.

## F13_EXOGENOUS_GLUCAGON_PKPD: Two-depot glucagon PK/PD effect on EGP

Category: `physiology`

Evidence class: `adapted`

Canonical expression:

$$
\begin{aligned}\frac{dY_1}{dt}&=u_G-k_1Y_1\\\frac{dY_2}{dt}&=k_1Y_1-k_2Y_2\\\Gamma&=\frac{k_2Y_2}{W\,Cl_{F,C}}\\a_G&=\frac{\Gamma}{CE_{50}+\Gamma}\\\frac{dx_{\mathrm{gluc}}}{dt}&=k_{aG}(S_Ga_G-x_{\mathrm{gluc}})\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
dY1/dt=u_G-k1*Y1; dY2/dt=k1*Y1-k2*Y2; Gamma=k2*Y2/(W*Cl_F,C); a_G=Gamma/(CE50+Gamma); dx_gluc/dt=k_aG*(S_G*a_G-x_gluc)
```

</details>

Runtime/solved form: Input doses are converted exactly from mg to pg. The two-state PK and clearance output follow Wendt et al.; the bounded effect compartment is an explicit IINTS adaptation.

Units: u_G in pg/min (1 mg = 10^9 pg), Y depots in pg, W in kg, clearance in mL/kg/min, Gamma and CE50 in pg/mL

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://doi.org/10.1089/dia.2013.0150), [source 2](https://doi.org/10.1177/1932296817693254)

Validation note: The PK structure and representative parameter ranges are literature-based. The bounded effect-compartment coupling remains an unvalidated IINTS adaptation.

## F14_SMOOTH_RENAL_CLEARANCE: Differentiable renal glucose clearance

Category: `physiology`

Evidence class: `heuristic`

Canonical expression:

$$
\begin{aligned}\operatorname{softplus}_s(z)&=s\ln\!\left(1+e^{z/s}\right)\\z&=G-162\\F_R&=c\,\operatorname{softplus}_s(G-162)\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
softplus(z)=s*log(1+exp(z/s)); z=G-162; F_R=c*softplus(G-162)
```

</details>

Runtime/solved form: Bergman uses concentration loss; Hovorka scales by V_G_dL for mass loss.

Units: G in mg/dL; F_R in mg/dL/min or mg/min after volume scaling

Implementation: `src/iints/core/patient/physiology.py:smooth_threshold_excess`, `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: [source 1](https://pubmed.ncbi.nlm.nih.gov/6714538/), [source 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC3781504/)

Validation note: Smooth approximation to avoid discontinuous renal cutoff; threshold is a configurable research approximation.

## F15_CGM_ISF_OBSERVATION: CGM blood-to-ISF lag and deterministic observation equation

Category: `sensor`

Evidence class: `adapted`

Canonical expression:

$$
\begin{aligned}\tau_{\mathrm{ISF}}\frac{d\mathrm{ISF}}{dt}&=\mathrm{BG}_{\mathrm{lagged}}-\mathrm{ISF}\\\alpha&=1-e^{-\Delta t/\tau_{\mathrm{ISF}}}\\\mathrm{ISF}_{\mathrm{next}}&=\mathrm{ISF}+\alpha(\mathrm{BG}_{\mathrm{lagged}}-\mathrm{ISF})\\\mathrm{CGM}&=\mathrm{ISF}+b+d+\varepsilon-o_{\mathrm{compression}}\end{aligned}
$$

<details>
<summary>Plain-text runtime notation</summary>

```text
tau_ISF*dISF/dt = BG_lagged - ISF; alpha=1-exp(-dt/tau_ISF); ISF_next = ISF + alpha*(BG_lagged-ISF); CGM = ISF + bias + drift + noise - compression_offset
```

</details>

Runtime/solved form: Exact constant-input first-order update; stochastic noise uses seeded RNG state, not AI.

Units: mg/dL

Implementation: `src/iints/core/devices/models.py:SensorModel.read`

Literature basis: [source 1](https://doi.org/10.1177/193229681000400507), [source 2](https://doi.org/10.1073/pnas.95.1.294)

Validation note: Models known CGM lag/noise qualitatively; seeded stochastic terms are reproducible when state is saved.
