# IINTS-AF Formula Registry

Registry version: `iints-formula-registry-v1`

These formulas are static SDK knowledge. The local AI may explain them, but it must not derive, solve, or alter them.

## F01_BERGMAN_GLUCOSE_RHS: Bergman-style glucose concentration balance

Category: `physiology`

Canonical expression:

```text
dG/dt = -(p1_eff + X)G + p1_eff*Gb_eff + Ra + dawn - U_exercise - F_R
```

Runtime/solved form: Integrated by scipy.solve_ivp over each simulator step; glucose transition is rate-guarded after integration.

Units: G in mg/dL, rates in mg/dL/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: https://doi.org/10.1152/ajpendo.1979.236.6.E667, https://arxiv.org/abs/1703.03134

Validation note: Research extension of Bergman minimal-model dynamics with added meal, renal, exercise, dawn, glucagon, and HAAF terms.

## F02_BERGMAN_REMOTE_INSULIN: Remote insulin action

Category: `physiology`

Canonical expression:

```text
dX/dt = -p2*X + p3_eff*max(I - Ib, 0)
```

Runtime/solved form: First-order action compartment integrated inside the Bergman ODE RHS.

Units: X in 1/min, I in mU/L

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: https://doi.org/10.1152/ajpendo.1979.236.6.E667, https://arxiv.org/abs/1703.03134

Validation note: Insulin action is deterministic and non-negative relative to basal insulin.

## F03_PLASMA_INSULIN_BALANCE: Plasma insulin balance with optional graft secretion

Category: `physiology`

Canonical expression:

```text
dI/dt = -n(I - Ib) + gamma*M_graft*max(G - h, 0)*(1-f_subq) + Ra_I / V_I
```

Runtime/solved form: Integrated in Bergman mode; gamma defaults to 0 for T1D research profiles. If f_subq>0, graft secretion first enters the S1/S2 absorption chain.

Units: I in mU/L, Ra_I in mU/min, V_I in L

Implementation: `src/iints/core/patient/bergman_model.py:_ode`

Literature basis: https://doi.org/10.1152/ajpendo.1979.236.6.E667

Validation note: Stem-cell/islet graft secretion is an experimental abstraction and is disabled by default for T1D simulation.

## F04_SUBCUT_INSULIN_TWO_DEPOT_PK: Two-depot subcutaneous insulin absorption

Category: `physiology`

Canonical expression:

```text
dS1/dt = u_I + gamma*M_graft*max(G-h,0)*f_subq - k*S1; dS2/dt = k*S1 - k*S2; U_I = k*S2
```

Runtime/solved form: Bergman uses k_a; Hovorka uses 1/t_max_I. The state equations are integrated each step.

Units: S1/S2 in mU, u_I/U_I in mU/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1088/0967-3334/25/4/010, https://arxiv.org/abs/2202.13938

Validation note: Runtime code chooses the insulin absorption time constant deterministically from configured insulin type.

## F05_MEAL_ABSORPTION_CHAIN: Three-compartment meal absorption and glucose appearance

Category: `physiology`

Canonical expression:

```text
dD1/dt=-k_solid*D1; dD2/dt=k_solid*D1-k_empt*D2; dD3/dt=k_empt*D2-k_abs*D3; U_G=k_abs*D3*A_G
```

Runtime/solved form: Meal mass is converted from grams to mg and pushed through deterministic stomach/gut compartments.

Units: carbohydrate mass in mg; U_G/Ra in mg/min or mg/dL/min after volume scaling

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1109/TBME.2007.893506, https://arxiv.org/abs/2307.16444

Validation note: The SDK uses an ODE meal-chain abstraction inspired by published meal absorption models.

## F06_HOVORKA_GLUCOSE_MASS_BALANCE: Hovorka-style accessible/non-accessible glucose mass balance

Category: `physiology`

Canonical expression:

```text
dQ1/dt = -(NIMGU + F_R) - x1*Q1 + k12*Q2 + EGP0*max(0, 1 - x3 + x_gluc) + U_G; dQ2/dt = x1*Q1 - (k12 + x2)*Q2
```

Runtime/solved form: Integrated by scipy.solve_ivp; concentration is G = Q1 / V_G_dL after integration.

Units: Q1/Q2 in mg, G in mg/dL, mass fluxes in mg/min

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1088/0967-3334/25/4/010, https://arxiv.org/abs/2202.13938

Validation note: Research Hovorka-style RHS with explicit extensions for glucagon, renal loss, exercise, stress, and circadian EGP.

## F07_HOVORKA_INSULIN_ACTION_CHANNELS: Hovorka-style insulin action channels

Category: `physiology`

Canonical expression:

```text
dx1/dt=-ka1*x1+kb1*I; dx2/dt=-ka2*x2+kb2*I; dx3/dt=-ka3*x3+kb3*I; kb_i includes molecular_affinity_scalar
```

Runtime/solved form: kb1/kb2/kb3 are deterministic sensitivity products before ODE integration.

Units: x1/x2 in 1/min-like action states; x3 dimensionless research action

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1088/0967-3334/25/4/010

Validation note: Stress/exercise change S_overall before the action channels are integrated.

## F08_STRESS_EXERCISE_SENSITIVITY: Stress/exercise sensitivity and EGP multipliers

Category: `physiology`

Canonical expression:

```text
dH_stress/dt=(target_stress-H_stress)/20; dH_exercise/dt=(target_exercise-H_exercise)/10; S_overall=(1-0.7*H_stress)*(1+2*H_exercise); EGP_stress=1+0.5*H_stress
```

Runtime/solved form: Pseudo-hormone states are first-order deterministic filters of scenario inputs.

Units: dimensionless states, time constants in minutes

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://arxiv.org/abs/2202.13938

Validation note: Research abstraction, not a clinical cortisol/adrenaline assay model.

## F09_GLUT4_NIMGU_EXERCISE: Exercise-driven GLUT4/NIMGU state

Category: `physiology`

Canonical expression:

```text
dGLUT4/dt = k_act*H_exercise*(1-GLUT4) - k_deact*GLUT4; NIMGU = F_01c*(1 + 1.5*GLUT4)
```

Runtime/solved form: Exercise can increase non-insulin-mediated glucose uptake without LLM calculation.

Units: GLUT4 dimensionless, NIMGU in mg/min

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://arxiv.org/abs/2202.13938

Validation note: Educational exercise physiology abstraction; not a cell-level GLUT4 translocation assay.

## F10_CIRCADIAN_DAWN_EGP: Gated circadian/dawn EGP multiplier

Category: `physiology`

Canonical expression:

```text
phi=2*pi*(t_day-t_dawn_mid)/1440; C(phi)=0.15*cos(phi)+0.05*cos(2*phi); EGP_circadian=1+s_dawn*C(phi)
```

Runtime/solved form: Computed directly from time of day; effect is gated by configured dawn_phenomenon_strength.

Units: dimensionless multiplier

Implementation: `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://arxiv.org/abs/2202.13938

Validation note: Off or weak by default to avoid hidden drift in baseline runs.

## F11_HYPO_RESCUE_MULTIPLIER: Endogenous hypoglycemia rescue multiplier

Category: `physiology`

Canonical expression:

```text
Delta_hypo=max(0,70-G); R_rescue=1+(Delta_hypo/10)*(1-HAAF)
```

Runtime/solved form: Computed directly inside ODE RHS before effective EGP is assembled.

Units: G in mg/dL; multiplier dimensionless

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1056/NEJMra1215228

Validation note: Captures the concept of blunted counterregulation; not a diagnostic HAAF model.

## F12_HAAF_MEMORY: Hypoglycemia-associated autonomic failure memory

Category: `physiology`

Canonical expression:

```text
dHAAF/dt = k_build*Delta_hypo*(1-HAAF) - k_decay*HAAF; k_decay=1/(24*60)
```

Runtime/solved form: Integrated as a bounded state and clipped to [0, 1] after solver steps.

Units: dimensionless memory state, rates in 1/min

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://doi.org/10.1056/NEJMra1215228

Validation note: Research memory state only; never report as clinical hypo-awareness diagnosis.

## F13_EXOGENOUS_GLUCAGON_PKPD: Two-depot glucagon PK/PD effect on EGP

Category: `physiology`

Canonical expression:

```text
dY1/dt=u_G-Y1/tmax_G; dY2/dt=Y1/tmax_G-Y2/tmax_G; dGamma/dt=(Y2/tmax_G)/V_Gamma-k_eG*Gamma; dx_gluc/dt=-k_aG*x_gluc+S_G*k_aG*Gamma
```

Runtime/solved form: Integrated only from deterministic glucagon requests after safety caps.

Units: Y depots in pg-like mass, Gamma in pg/mL, x_gluc dimensionless

Implementation: `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://arxiv.org/abs/2202.13938

Validation note: Dual-hormone simulation support only; not a real pump recommendation.

## F14_SMOOTH_RENAL_CLEARANCE: Differentiable renal glucose clearance

Category: `physiology`

Canonical expression:

```text
softplus(z)=s*log(1+exp(z/s)); z=G-162; F_R=c*softplus(G-162)
```

Runtime/solved form: Bergman uses concentration loss; Hovorka scales by V_G_dL for mass loss.

Units: G in mg/dL; F_R in mg/dL/min or mg/min after volume scaling

Implementation: `src/iints/core/patient/physiology.py:smooth_threshold_excess`, `src/iints/core/patient/bergman_model.py:_ode`, `src/iints/core/patient/hovorka_model.py:_ode`

Literature basis: https://en.wikipedia.org/wiki/Glycosuria, https://en.wikipedia.org/wiki/Renal_threshold

Validation note: Smooth approximation to avoid discontinuous renal cutoff; threshold is a configurable research approximation.

## F15_CGM_ISF_OBSERVATION: CGM blood-to-ISF lag and deterministic observation equation

Category: `sensor`

Canonical expression:

```text
tau_ISF*dISF/dt = BG_lagged - ISF; ISF_next = ISF + alpha*(BG_lagged-ISF); CGM = ISF + bias + drift + noise - compression_offset
```

Runtime/solved form: Euler low-pass update with bounded alpha; stochastic noise uses seeded RNG state, not AI.

Units: mg/dL

Implementation: `src/iints/core/devices/models.py:SensorModel.read`

Literature basis: https://en.wikipedia.org/wiki/Continuous_glucose_monitor

Validation note: Models known CGM lag/noise qualitatively; seeded stochastic terms are reproducible when state is saved.
