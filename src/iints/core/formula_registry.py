from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FormulaCategory = Literal["physiology", "sensor", "safety", "research_loss"]
EvidenceClass = Literal["canonical", "adapted", "heuristic"]

FORMULA_REGISTRY_VERSION = "iints-formula-registry-v5"


@dataclass(frozen=True)
class FormulaSpec:
    """Immutable description of a formula used by the SDK.

    This registry is documentation and AI-context metadata. Runtime numerical
    calculations remain in deterministic Python model code, not in LLM prompts.
    """

    formula_id: str
    title: str
    category: FormulaCategory
    canonical_expression: str
    latex_expression: str
    solved_or_runtime_form: str
    state_variables: tuple[str, ...]
    parameters: tuple[str, ...]
    units: str
    implementation_paths: tuple[str, ...]
    literature_basis: tuple[str, ...]
    validation_note: str
    evidence_class: EvidenceClass = "adapted"
    ai_policy: str = "LLM may explain only; never derive, solve, or alter this formula."

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FORMULAS: tuple[FormulaSpec, ...] = (
    FormulaSpec(
        formula_id="F01_BERGMAN_GLUCOSE_RHS",
        title="Bergman-style glucose concentration balance",
        category="physiology",
        canonical_expression=(
            "dG/dt = -(p1_eff + X)G + p1_eff*Gb_eff + Ra + dawn - U_exercise - F_R"
        ),
        latex_expression=(
            r"\frac{dG}{dt}=-(p_{1,\mathrm{eff}}+X)G"
            r"+p_{1,\mathrm{eff}}G_{b,\mathrm{eff}}+R_a+D_{\mathrm{dawn}}"
            r"-U_{\mathrm{exercise}}-F_R"
        ),
        solved_or_runtime_form=(
            "Integrated by scipy.solve_ivp over each simulator step; glucose transition is rate-guarded "
            "after integration."
        ),
        state_variables=("G", "X", "Q_gut", "HAAF", "x_gluc"),
        parameters=("p1_eff", "Gb_eff", "Ra", "dawn", "U_exercise", "F_R"),
        units="G in mg/dL, rates in mg/dL/min",
        implementation_paths=("src/iints/core/patient/bergman_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1152/ajpendo.1979.236.6.E667",
            "https://doi.org/10.1109/TBME.2007.893506",
        ),
        validation_note="Research extension of Bergman minimal-model dynamics with added meal, renal, exercise, dawn, glucagon, and HAAF terms.",
    ),
    FormulaSpec(
        formula_id="F02_BERGMAN_REMOTE_INSULIN",
        title="Remote insulin action",
        category="physiology",
        canonical_expression="dX/dt = -p2*X + p3_eff*(I - I_ref)",
        latex_expression=(
            r"\frac{dX}{dt}=-p_2X+p_{3,\mathrm{eff}}(I-I_{\mathrm{ref}})"
        ),
        solved_or_runtime_form="First-order action compartment integrated inside the Bergman ODE RHS.",
        state_variables=("X", "I"),
        parameters=("p2", "p3_eff", "I_ref"),
        units="X in 1/min, I in mU/L",
        implementation_paths=("src/iints/core/patient/bergman_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1152/ajpendo.1979.236.6.E667",
        ),
        validation_note=(
            "X is a deviation-from-reference action state and can become negative when "
            "insulin falls below the pump-supported fasting reference concentration. "
            "I_ref is derived from basal delivery, distribution volume and clearance; "
            "the configured Ib is used only when basal delivery is zero."
        ),
    ),
    FormulaSpec(
        formula_id="F03_PLASMA_INSULIN_BALANCE",
        title="Plasma insulin balance with optional graft secretion",
        category="physiology",
        canonical_expression=(
            "q_sec=gamma*M_graft*max(G-h,0)*V_I; "
            "dI/dt=-n*I+(q_sec*(1-f_subq)+Ra_I)/V_I"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"q_{\mathrm{sec}}&=\gamma M_{\mathrm{graft}}\max\!\left(G-h,0\right)V_I\\"
            r"\frac{dI}{dt}&=-nI+"
            r"\frac{q_{\mathrm{sec}}(1-f_{\mathrm{subq}})+R_{a,I}}{V_I}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Integrated in Bergman mode; gamma defaults to 0 for T1D research profiles. "
            "If f_subq>0, graft secretion first enters the S1/S2 absorption chain."
        ),
        state_variables=("I", "G", "S2", "M_graft"),
        parameters=("n", "gamma", "h", "Ra_I", "V_I", "f_subq"),
        units="I in mU/L, Ra_I in mU/min, V_I in L",
        implementation_paths=("src/iints/core/patient/bergman_model.py:_ode",),
        literature_basis=("https://doi.org/10.1152/ajpendo.1979.236.6.E667",),
        validation_note="Stem-cell/islet graft secretion is an experimental abstraction and is disabled by default for T1D simulation.",
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F04_SUBCUT_INSULIN_TWO_DEPOT_PK",
        title="Two-depot subcutaneous insulin absorption",
        category="physiology",
        canonical_expression=(
            "q_sec=gamma*M_graft*max(G-h,0)*V_I; "
            "dS1/dt = u_I + q_sec*f_subq - k*S1; "
            "dS2/dt = k*S1 - k*S2; U_I = k*S2"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"q_{\mathrm{sec}}&=\gamma M_{\mathrm{graft}}\max\!\left(G-h,0\right)V_I\\"
            r"\frac{dS_1}{dt}&=u_I+q_{\mathrm{sec}}f_{\mathrm{subq}}-kS_1\\"
            r"\frac{dS_2}{dt}&=kS_1-kS_2\\"
            r"U_I&=kS_2"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Bergman uses k_a; Hovorka uses 1/t_max_I. The state equations are integrated each step.",
        state_variables=("S1", "S2", "G", "M_graft"),
        parameters=("u_I", "k_a", "t_max_I", "gamma", "h", "f_subq"),
        units="S1/S2 in mU, u_I/U_I in mU/min",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1088/0967-3334/25/4/010",
        ),
        validation_note="Runtime code chooses the insulin absorption time constant deterministically from configured insulin type.",
    ),
    FormulaSpec(
        formula_id="F05_MEAL_ABSORPTION_CHAIN",
        title="Meal absorption and glucose appearance",
        category="physiology",
        canonical_expression=(
            "Hovorka: dD1/dt=-D1/tmaxG; dD2/dt=D1/tmaxG-D2/tmaxG; "
            "U_G=A_G*D2/tmaxG. Bergman adaptation: Q_sto1 -> Q_sto2 -> Q_gut."
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{dD_1}{dt}&=-\frac{D_1}{t_{\max,G}}\\"
            r"\frac{dD_2}{dt}&=\frac{D_1}{t_{\max,G}}-\frac{D_2}{t_{\max,G}}\\"
            r"U_G&=A_G\frac{D_2}{t_{\max,G}}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Hovorka mode uses the published two-compartment chain. Bergman and "
            "advanced modes retain an explicitly adapted stomach/gut chain."
        ),
        state_variables=("D1", "D2", "Q_sto1", "Q_sto2", "Q_gut"),
        parameters=("t_max_G", "A_G", "tau_meal", "k_abs", "f_bio"),
        units="carbohydrate mass in mg; U_G/Ra in mg/min or mg/dL/min after volume scaling",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1088/0967-3334/25/4/010",
            "https://doi.org/10.1109/TBME.2007.893506",
        ),
        validation_note=(
            "The Hovorka branch follows its published meal equations; the Bergman "
            "three-stage branch is an IINTS adaptation and requires dataset calibration."
        ),
    ),
    FormulaSpec(
        formula_id="F06_HOVORKA_GLUCOSE_MASS_BALANCE",
        title="Hovorka-style accessible/non-accessible glucose mass balance",
        category="physiology",
        canonical_expression=(
            "dQ1/dt = -(NIMGU + F_R) - x1*Q1 + k12*Q2 + "
            "EGP0*max(0, 1 - x3 + x_gluc) + U_G + V_G*dawn_rate; "
            "dQ2/dt = x1*Q1 - (k12 + x2)*Q2"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{dQ_1}{dt}&=-(\mathrm{NIMGU}+F_R)-x_1Q_1+k_{12}Q_2"
            r"+\mathrm{EGP}_0\max\!\left(0,1-x_3+x_{\mathrm{gluc}}\right)"
            r"+U_G+V_G r_{\mathrm{dawn}}\\"
            r"\frac{dQ_2}{dt}&=x_1Q_1-(k_{12}+x_2)Q_2"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Integrated by scipy.solve_ivp; concentration is G = Q1 / V_G_dL after integration.",
        state_variables=("Q1", "Q2", "x1", "x2", "x3", "x_gluc"),
        parameters=(
            "NIMGU",
            "F_R",
            "k12",
            "EGP0",
            "U_G",
            "V_G_dL",
            "dawn_rate",
        ),
        units="Q1/Q2 in mg, G in mg/dL, mass fluxes in mg/min",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1088/0967-3334/25/4/010",
        ),
        validation_note=(
            "Research Hovorka-style RHS with explicit extensions for glucagon, "
            "renal loss, exercise, stress, and a phenomenological dawn input."
        ),
    ),
    FormulaSpec(
        formula_id="F07_HOVORKA_INSULIN_ACTION_CHANNELS",
        title="Hovorka-style insulin action channels",
        category="physiology",
        canonical_expression=(
            "dx1/dt=-ka1*x1+kb1*I; dx2/dt=-ka2*x2+kb2*I; dx3/dt=-ka3*x3+kb3*I; "
            "kb_i includes molecular_affinity_scalar"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{dx_1}{dt}&=-k_{a1}x_1+k_{b1}I\\"
            r"\frac{dx_2}{dt}&=-k_{a2}x_2+k_{b2}I\\"
            r"\frac{dx_3}{dt}&=-k_{a3}x_3+k_{b3}I,"
            r"\qquad k_{b,i}\propto A_{\mathrm{affinity}}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="kb1/kb2/kb3 are deterministic sensitivity products before ODE integration.",
        state_variables=("x1", "x2", "x3", "I"),
        parameters=("ka1", "ka2", "ka3", "S_IT", "S_ID", "S_IE", "S_overall", "molecular_affinity_scalar"),
        units="x1/x2 in 1/min-like action states; x3 dimensionless research action",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=("https://doi.org/10.1088/0967-3334/25/4/010",),
        validation_note=(
            "The Hovorka action-channel topology is literature based, but the overall, "
            "tissue-specific, and molecular-affinity scalars are explicit IINTS scenario "
            "assumptions. They are not inferred from AlphaFold confidence or ClinVar labels "
            "and require independent calibration before quantitative interpretation."
        ),
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F08_STRESS_EXERCISE_SENSITIVITY",
        title="Stress/exercise sensitivity and EGP multipliers",
        category="physiology",
        canonical_expression=(
            "dH_stress/dt=(target_stress-H_stress)/20; dH_exercise/dt=(target_exercise-H_exercise)/10; "
            "S_overall=(1-0.7*H_stress)*(1+2*H_exercise); EGP_stress=1+0.5*H_stress"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{dH_{\mathrm{stress}}}{dt}&="
            r"\frac{H_{\mathrm{stress,target}}-H_{\mathrm{stress}}}{20}\\"
            r"\frac{dH_{\mathrm{exercise}}}{dt}&="
            r"\frac{H_{\mathrm{exercise,target}}-H_{\mathrm{exercise}}}{10}\\"
            r"S_{\mathrm{overall}}&=(1-0.7H_{\mathrm{stress}})(1+2H_{\mathrm{exercise}})\\"
            r"\mathrm{EGP}_{\mathrm{stress}}&=1+0.5H_{\mathrm{stress}}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Pseudo-hormone states are first-order deterministic filters of scenario inputs.",
        state_variables=("H_stress", "H_exercise"),
        parameters=("target_stress", "target_exercise"),
        units="dimensionless states, time constants in minutes",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1371/journal.pone.0248280",
            "https://doi.org/10.1152/ajpendo.00084.2021",
        ),
        validation_note=(
            "The sources support the direction of exercise effects on glucose effectiveness, "
            "insulin sensitivity, and insulin-independent uptake. The IINTS filter time "
            "constants and multiplier coefficients are heuristic scenario parameters, not "
            "values identified or clinically validated by those studies. The stress branch is "
            "not a cortisol/adrenaline concentration model."
        ),
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F09_GLUT4_NIMGU_EXERCISE",
        title="Exercise-driven GLUT4/NIMGU state",
        category="physiology",
        canonical_expression=(
            "dGLUT4/dt = k_act*H_exercise*(1-GLUT4) - k_deact*GLUT4; "
            "NIMGU = F_01c*(1 + 1.5*GLUT4)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{d\mathrm{GLUT4}}{dt}&=k_{\mathrm{act}}H_{\mathrm{exercise}}"
            r"(1-\mathrm{GLUT4})-k_{\mathrm{deact}}\mathrm{GLUT4}\\"
            r"\mathrm{NIMGU}&=F_{01c}(1+1.5\,\mathrm{GLUT4})"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Exercise can increase non-insulin-mediated glucose uptake without LLM calculation.",
        state_variables=("GLUT4", "H_exercise"),
        parameters=("k_act", "k_deact", "F_01c"),
        units="GLUT4 dimensionless, NIMGU in mg/min",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1371/journal.pone.0248280",
            "https://doi.org/10.1152/ajpendo.00084.2021",
        ),
        validation_note=(
            "The sources support exercise-mediated insulin-dependent and "
            "insulin-independent glucose utilization. GLUT4 is a bounded latent scenario "
            "state here; its activation/deactivation constants and 1.5 uptake multiplier are "
            "IINTS heuristics, not measured receptor abundance or a cell-level translocation "
            "assay."
        ),
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F10_CIRCADIAN_DAWN_EGP",
        title="Phenomenological dawn glucose-rate perturbation",
        category="physiology",
        canonical_expression=(
            "w(t)=0.5*[1+cos(pi*(t-t_mid)/h)] inside the configured window, "
            "otherwise 0; dawn_rate=(s_dawn/60)*w(t)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"w(t)&=\tfrac12\left[1+\cos\!\left(\frac{\pi(t-t_m)}{h}\right)\right]"
            r"\quad (|t-t_m|\le h)\\"
            r"r_{\mathrm{dawn}}(t)&=\frac{s_{\mathrm{dawn}}}{60}w(t)"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Computed directly in mg/dL/min. Concentration-domain models add "
            "the rate; Hovorka converts it to glucose mass flow using V_G."
        ),
        state_variables=("time_of_day_min",),
        parameters=("dawn_start_hour", "dawn_end_hour", "dawn_phenomenon_strength"),
        units="dawn_phenomenon_strength: mg/dL/hour; runtime rate: mg/dL/min",
        implementation_paths=(
            "src/iints/core/patient/physiology.py:dawn_glucose_rate_mgdl_min",
            "src/iints/core/patient/models.py:update",
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
            "src/iints/core/patient/advanced_metabolic_model.py:_ode",
        ),
        literature_basis=(
            "https://pubmed.ncbi.nlm.nih.gov/35466006/",
            "https://doi.org/10.1089/dia.2015.0011",
        ),
        validation_note=(
            "The sources support circadian variation in glucose regulation. "
            "The raised-cosine shape and configured peak rate are an IINTS "
            "scenario heuristic, not coefficients estimated from either study. "
            "The default is zero, and all backends use the same declared unit."
        ),
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F11_HYPO_RESCUE_MULTIPLIER",
        title="Endogenous hypoglycemia rescue multiplier",
        category="physiology",
        canonical_expression=(
            "Delta=max(0,70-G); a=Delta/(16+Delta); "
            "R_rescue=1+a*(1-HAAF)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\Delta_{\mathrm{hypo}}&=\max(0,70-G)\\"
            r"a&=\frac{\Delta_{\mathrm{hypo}}}{16+\Delta_{\mathrm{hypo}}}\\"
            r"R_{\mathrm{rescue}}&=1+a(1-\mathrm{HAAF})"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Computed directly inside ODE RHS before effective EGP is assembled.",
        state_variables=("G", "HAAF"),
        parameters=("hypoglycemia_threshold_mgdl",),
        units="G in mg/dL; multiplier dimensionless",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1152/ajpendo.2001.281.6.E1115",
            "https://doi.org/10.1210/jcem.84.5.5675",
        ),
        validation_note="Captures the concept of blunted counterregulation; not a diagnostic HAAF model.",
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F12_HAAF_MEMORY",
        title="Hypoglycemia-associated autonomic failure memory",
        category="physiology",
        canonical_expression=(
            "severity=clip((70-G)/16,0,1.5); "
            "dHAAF/dt=severity*(1-HAAF)/360-HAAF/4320"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"s&=\operatorname{clip}\!\left(\frac{70-G}{16},0,1.5\right)\\"
            r"\frac{d\mathrm{HAAF}}{dt}&=\frac{s(1-\mathrm{HAAF})}{360}"
            r"-\frac{\mathrm{HAAF}}{4320}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Integrated as a bounded state. Material excursions outside [0, 1] "
            "fail the step; only solver-scale numerical tolerance is projected back."
        ),
        state_variables=("HAAF", "G"),
        parameters=("k_build", "k_decay"),
        units="dimensionless memory state, rates in 1/min",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1152/ajpendo.2001.281.6.E1115",
            "https://doi.org/10.1210/jcem.84.5.5675",
        ),
        validation_note="Research memory state only; never report as clinical hypo-awareness diagnosis.",
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F13_EXOGENOUS_GLUCAGON_PKPD",
        title="Two-depot glucagon PK/PD effect on EGP",
        category="physiology",
        canonical_expression=(
            "dY1/dt=u_G-k1*Y1; dY2/dt=k1*Y1-k2*Y2; "
            "Gamma=k2*Y2/(W*Cl_F,C); a_G=Gamma/(CE50+Gamma); "
            "dx_gluc/dt=k_aG*(S_G*a_G-x_gluc)"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\frac{dY_1}{dt}&=u_G-k_1Y_1\\"
            r"\frac{dY_2}{dt}&=k_1Y_1-k_2Y_2\\"
            r"\Gamma&=\frac{k_2Y_2}{W\,Cl_{F,C}}\\"
            r"a_G&=\frac{\Gamma}{CE_{50}+\Gamma}\\"
            r"\frac{dx_{\mathrm{gluc}}}{dt}&=k_{aG}(S_Ga_G-x_{\mathrm{gluc}})"
            r"\end{aligned}"
        ),
        solved_or_runtime_form=(
            "Input doses are converted exactly from mg to pg. The two-state PK and "
            "clearance output follow Wendt et al.; the bounded effect compartment is "
            "an explicit IINTS adaptation."
        ),
        state_variables=("Y1", "Y2", "Gamma", "x_gluc"),
        parameters=("u_G", "k1", "k2", "W", "Cl_F,C", "CE50", "k_aG", "S_G"),
        units=(
            "u_G in pg/min (1 mg = 10^9 pg), Y depots in pg, W in kg, "
            "clearance in mL/kg/min, Gamma and CE50 in pg/mL"
        ),
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1089/dia.2013.0150",
            "https://doi.org/10.1177/1932296817693254",
        ),
        validation_note=(
            "The PK structure and representative parameter ranges are literature-based. "
            "The bounded effect-compartment coupling remains an unvalidated IINTS adaptation."
        ),
        evidence_class="adapted",
    ),
    FormulaSpec(
        formula_id="F14_SMOOTH_RENAL_CLEARANCE",
        title="Differentiable renal glucose clearance",
        category="physiology",
        canonical_expression="softplus(z)=s*log(1+exp(z/s)); z=G-162; F_R=c*softplus(G-162)",
        latex_expression=(
            r"\begin{aligned}"
            r"\operatorname{softplus}_s(z)&=s\ln\!\left(1+e^{z/s}\right)\\"
            r"z&=G-162\\"
            r"F_R&=c\,\operatorname{softplus}_s(G-162)"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Bergman uses concentration loss; Hovorka scales by V_G_dL for mass loss.",
        state_variables=("G",),
        parameters=("threshold_mgdl", "splay", "clearance_coefficient", "V_G_dL"),
        units="G in mg/dL; F_R in mg/dL/min or mg/min after volume scaling",
        implementation_paths=(
            "src/iints/core/patient/physiology.py:smooth_threshold_excess",
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://pubmed.ncbi.nlm.nih.gov/6714538/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC3781504/",
        ),
        validation_note="Smooth approximation to avoid discontinuous renal cutoff; threshold is a configurable research approximation.",
        evidence_class="heuristic",
    ),
    FormulaSpec(
        formula_id="F15_CGM_ISF_OBSERVATION",
        title="CGM blood-to-ISF lag and deterministic observation equation",
        category="sensor",
        canonical_expression=(
            "tau_ISF*dISF/dt = BG_lagged - ISF; "
            "alpha=1-exp(-dt/tau_ISF); ISF_next = ISF + alpha*(BG_lagged-ISF); "
            "CGM = ISF + bias + drift + noise - compression_offset"
        ),
        latex_expression=(
            r"\begin{aligned}"
            r"\tau_{\mathrm{ISF}}\frac{d\mathrm{ISF}}{dt}&="
            r"\mathrm{BG}_{\mathrm{lagged}}-\mathrm{ISF}\\"
            r"\alpha&=1-e^{-\Delta t/\tau_{\mathrm{ISF}}}\\"
            r"\mathrm{ISF}_{\mathrm{next}}&=\mathrm{ISF}+\alpha"
            r"(\mathrm{BG}_{\mathrm{lagged}}-\mathrm{ISF})\\"
            r"\mathrm{CGM}&=\mathrm{ISF}+b+d+\varepsilon-o_{\mathrm{compression}}"
            r"\end{aligned}"
        ),
        solved_or_runtime_form="Exact constant-input first-order update; stochastic noise uses seeded RNG state, not AI.",
        state_variables=("ISF", "BG_lagged", "CGM"),
        parameters=("tau_ISF", "lag_minutes", "bias", "drift", "noise", "compression_offset"),
        units="mg/dL",
        implementation_paths=("src/iints/core/devices/models.py:SensorModel.read",),
        literature_basis=(
            "https://doi.org/10.1177/193229681000400507",
            "https://doi.org/10.1073/pnas.95.1.294",
        ),
        validation_note="Models known CGM lag/noise qualitatively; seeded stochastic terms are reproducible when state is saved.",
    ),
)


def get_formula_registry() -> tuple[FormulaSpec, ...]:
    return FORMULAS


def get_formula(formula_id: str) -> FormulaSpec:
    for formula in FORMULAS:
        if formula.formula_id == formula_id:
            return formula
    raise KeyError(f"Unknown IINTS formula id: {formula_id}")


def formula_registry_dict() -> dict[str, object]:
    return {
        "registry_version": FORMULA_REGISTRY_VERSION,
        "formula_count": len(FORMULAS),
        "ai_numeric_authority": False,
        "formulas": [formula.to_dict() for formula in FORMULAS],
    }


def formula_context_for_ai() -> dict[str, object]:
    """Compact context for local LLM explanations.

    This is intentionally expression-only. It gives the model a fixed source of
    truth to mention, while all numeric evaluation remains in deterministic code.
    """

    return {
        "registry_version": FORMULA_REGISTRY_VERSION,
        "ai_formula_authority": False,
        "instruction": "Use these pre-registered formulas as immutable context. Do not derive or solve formulas.",
        "formulas": [
            {
                "id": formula.formula_id,
                "title": formula.title,
                "expression": formula.canonical_expression,
                "latex_expression": formula.latex_expression,
                "runtime_form": formula.solved_or_runtime_form,
                "units": formula.units,
                "evidence_class": formula.evidence_class,
            }
            for formula in FORMULAS
        ],
    }


def formula_registry_markdown() -> str:
    lines = [
        "# IINTS-AF Formula Registry",
        "",
        f"Registry version: `{FORMULA_REGISTRY_VERSION}`",
        "",
        "These formulas document deterministic SDK code. The local AI may explain them, "
        "but it must not derive, solve, or alter them.",
        "",
        "Evidence classes: `canonical` means a direct published equation, `adapted` means a "
        "published model changed for SDK integration, and `heuristic` means an explicit research "
        "assumption that requires calibration and external validation.",
        "",
        "No registry entry, including a canonical equation, establishes clinical validity for its "
        "parameterization or for the combined simulator.",
        "",
    ]
    for formula in FORMULAS:
        lines.extend(
            [
                f"## {formula.formula_id}: {formula.title}",
                "",
                f"Category: `{formula.category}`",
                "",
                f"Evidence class: `{formula.evidence_class}`",
                "",
                "Canonical expression:",
                "",
                "$$",
                formula.latex_expression,
                "$$",
                "",
                "<details>",
                "<summary>Plain-text runtime notation</summary>",
                "",
                f"```text\n{formula.canonical_expression}\n```",
                "",
                "</details>",
                "",
                f"Runtime/solved form: {formula.solved_or_runtime_form}",
                "",
                f"Units: {formula.units}",
                "",
                f"Implementation: {', '.join(f'`{path}`' for path in formula.implementation_paths)}",
                "",
                "Literature basis: "
                + ", ".join(
                    f"[source {index}]({source})"
                    for index, source in enumerate(formula.literature_basis, start=1)
                ),
                "",
                f"Validation note: {formula.validation_note}",
                "",
            ]
        )
    return "\n".join(lines)
