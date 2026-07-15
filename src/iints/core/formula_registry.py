from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FormulaCategory = Literal["physiology", "sensor", "safety", "research_loss"]

FORMULA_REGISTRY_VERSION = "iints-formula-registry-v1"


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
    solved_or_runtime_form: str
    state_variables: tuple[str, ...]
    parameters: tuple[str, ...]
    units: str
    implementation_paths: tuple[str, ...]
    literature_basis: tuple[str, ...]
    validation_note: str
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
            "https://arxiv.org/abs/1703.03134",
        ),
        validation_note="Research extension of Bergman minimal-model dynamics with added meal, renal, exercise, dawn, glucagon, and HAAF terms.",
    ),
    FormulaSpec(
        formula_id="F02_BERGMAN_REMOTE_INSULIN",
        title="Remote insulin action",
        category="physiology",
        canonical_expression="dX/dt = -p2*X + p3_eff*max(I - Ib, 0)",
        solved_or_runtime_form="First-order action compartment integrated inside the Bergman ODE RHS.",
        state_variables=("X", "I"),
        parameters=("p2", "p3_eff", "Ib"),
        units="X in 1/min, I in mU/L",
        implementation_paths=("src/iints/core/patient/bergman_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1152/ajpendo.1979.236.6.E667",
            "https://arxiv.org/abs/1703.03134",
        ),
        validation_note="Insulin action is deterministic and non-negative relative to basal insulin.",
    ),
    FormulaSpec(
        formula_id="F03_PLASMA_INSULIN_BALANCE",
        title="Plasma insulin balance with optional graft secretion",
        category="physiology",
        canonical_expression=(
            "dI/dt = -n(I - Ib) + gamma*M_graft*max(G - h, 0)*(1-f_subq) + Ra_I / V_I"
        ),
        solved_or_runtime_form=(
            "Integrated in Bergman mode; gamma defaults to 0 for T1D research profiles. "
            "If f_subq>0, graft secretion first enters the S1/S2 absorption chain."
        ),
        state_variables=("I", "G", "S2", "M_graft"),
        parameters=("n", "Ib", "gamma", "h", "Ra_I", "V_I", "f_subq"),
        units="I in mU/L, Ra_I in mU/min, V_I in L",
        implementation_paths=("src/iints/core/patient/bergman_model.py:_ode",),
        literature_basis=("https://doi.org/10.1152/ajpendo.1979.236.6.E667",),
        validation_note="Stem-cell/islet graft secretion is an experimental abstraction and is disabled by default for T1D simulation.",
    ),
    FormulaSpec(
        formula_id="F04_SUBCUT_INSULIN_TWO_DEPOT_PK",
        title="Two-depot subcutaneous insulin absorption",
        category="physiology",
        canonical_expression=(
            "dS1/dt = u_I + gamma*M_graft*max(G-h,0)*f_subq - k*S1; "
            "dS2/dt = k*S1 - k*S2; U_I = k*S2"
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
            "https://arxiv.org/abs/2202.13938",
        ),
        validation_note="Runtime code chooses the insulin absorption time constant deterministically from configured insulin type.",
    ),
    FormulaSpec(
        formula_id="F05_MEAL_ABSORPTION_CHAIN",
        title="Three-compartment meal absorption and glucose appearance",
        category="physiology",
        canonical_expression=(
            "dD1/dt=-k_solid*D1; dD2/dt=k_solid*D1-k_empt*D2; "
            "dD3/dt=k_empt*D2-k_abs*D3; U_G=k_abs*D3*A_G"
        ),
        solved_or_runtime_form="Meal mass is converted from grams to mg and pushed through deterministic stomach/gut compartments.",
        state_variables=("D1", "D2", "D3", "Q_sto1", "Q_sto2", "Q_gut"),
        parameters=("k_solid", "k_empt", "k_abs", "A_G", "f_bio"),
        units="carbohydrate mass in mg; U_G/Ra in mg/min or mg/dL/min after volume scaling",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=(
            "https://doi.org/10.1109/TBME.2007.893506",
            "https://arxiv.org/abs/2307.16444",
        ),
        validation_note="The SDK uses an ODE meal-chain abstraction inspired by published meal absorption models.",
    ),
    FormulaSpec(
        formula_id="F06_HOVORKA_GLUCOSE_MASS_BALANCE",
        title="Hovorka-style accessible/non-accessible glucose mass balance",
        category="physiology",
        canonical_expression=(
            "dQ1/dt = -(NIMGU + F_R) - x1*Q1 + k12*Q2 + EGP0*max(0, 1 - x3 + x_gluc) + U_G; "
            "dQ2/dt = x1*Q1 - (k12 + x2)*Q2"
        ),
        solved_or_runtime_form="Integrated by scipy.solve_ivp; concentration is G = Q1 / V_G_dL after integration.",
        state_variables=("Q1", "Q2", "x1", "x2", "x3", "x_gluc"),
        parameters=("NIMGU", "F_R", "k12", "EGP0", "U_G", "V_G_dL"),
        units="Q1/Q2 in mg, G in mg/dL, mass fluxes in mg/min",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=(
            "https://doi.org/10.1088/0967-3334/25/4/010",
            "https://arxiv.org/abs/2202.13938",
        ),
        validation_note="Research Hovorka-style RHS with explicit extensions for glucagon, renal loss, exercise, stress, and circadian EGP.",
    ),
    FormulaSpec(
        formula_id="F07_HOVORKA_INSULIN_ACTION_CHANNELS",
        title="Hovorka-style insulin action channels",
        category="physiology",
        canonical_expression=(
            "dx1/dt=-ka1*x1+kb1*I; dx2/dt=-ka2*x2+kb2*I; dx3/dt=-ka3*x3+kb3*I; "
            "kb_i includes molecular_affinity_scalar"
        ),
        solved_or_runtime_form="kb1/kb2/kb3 are deterministic sensitivity products before ODE integration.",
        state_variables=("x1", "x2", "x3", "I"),
        parameters=("ka1", "ka2", "ka3", "S_IT", "S_ID", "S_IE", "S_overall", "molecular_affinity_scalar"),
        units="x1/x2 in 1/min-like action states; x3 dimensionless research action",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=("https://doi.org/10.1088/0967-3334/25/4/010",),
        validation_note="Stress/exercise change S_overall before the action channels are integrated.",
    ),
    FormulaSpec(
        formula_id="F08_STRESS_EXERCISE_SENSITIVITY",
        title="Stress/exercise sensitivity and EGP multipliers",
        category="physiology",
        canonical_expression=(
            "dH_stress/dt=(target_stress-H_stress)/20; dH_exercise/dt=(target_exercise-H_exercise)/10; "
            "S_overall=(1-0.7*H_stress)*(1+2*H_exercise); EGP_stress=1+0.5*H_stress"
        ),
        solved_or_runtime_form="Pseudo-hormone states are first-order deterministic filters of scenario inputs.",
        state_variables=("H_stress", "H_exercise"),
        parameters=("target_stress", "target_exercise"),
        units="dimensionless states, time constants in minutes",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=("https://arxiv.org/abs/2202.13938",),
        validation_note="Research abstraction, not a clinical cortisol/adrenaline assay model.",
    ),
    FormulaSpec(
        formula_id="F09_GLUT4_NIMGU_EXERCISE",
        title="Exercise-driven GLUT4/NIMGU state",
        category="physiology",
        canonical_expression=(
            "dGLUT4/dt = k_act*H_exercise*(1-GLUT4) - k_deact*GLUT4; "
            "NIMGU = F_01c*(1 + 1.5*GLUT4)"
        ),
        solved_or_runtime_form="Exercise can increase non-insulin-mediated glucose uptake without LLM calculation.",
        state_variables=("GLUT4", "H_exercise"),
        parameters=("k_act", "k_deact", "F_01c"),
        units="GLUT4 dimensionless, NIMGU in mg/min",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=("https://arxiv.org/abs/2202.13938",),
        validation_note="Educational exercise physiology abstraction; not a cell-level GLUT4 translocation assay.",
    ),
    FormulaSpec(
        formula_id="F10_CIRCADIAN_DAWN_EGP",
        title="Gated circadian/dawn EGP multiplier",
        category="physiology",
        canonical_expression=(
            "phi=2*pi*(t_day-t_dawn_mid)/1440; C(phi)=0.15*cos(phi)+0.05*cos(2*phi); "
            "EGP_circadian=1+s_dawn*C(phi)"
        ),
        solved_or_runtime_form="Computed directly from time of day; effect is gated by configured dawn_phenomenon_strength.",
        state_variables=("time_of_day_min",),
        parameters=("dawn_start_hour", "dawn_end_hour", "dawn_phenomenon_strength"),
        units="dimensionless multiplier",
        implementation_paths=("src/iints/core/patient/hovorka_model.py:_ode",),
        literature_basis=("https://arxiv.org/abs/2202.13938",),
        validation_note="Off or weak by default to avoid hidden drift in baseline runs.",
    ),
    FormulaSpec(
        formula_id="F11_HYPO_RESCUE_MULTIPLIER",
        title="Endogenous hypoglycemia rescue multiplier",
        category="physiology",
        canonical_expression="Delta_hypo=max(0,70-G); R_rescue=1+(Delta_hypo/10)*(1-HAAF)",
        solved_or_runtime_form="Computed directly inside ODE RHS before effective EGP is assembled.",
        state_variables=("G", "HAAF"),
        parameters=("hypoglycemia_threshold_mgdl",),
        units="G in mg/dL; multiplier dimensionless",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=("https://doi.org/10.1056/NEJMra1215228",),
        validation_note="Captures the concept of blunted counterregulation; not a diagnostic HAAF model.",
    ),
    FormulaSpec(
        formula_id="F12_HAAF_MEMORY",
        title="Hypoglycemia-associated autonomic failure memory",
        category="physiology",
        canonical_expression="dHAAF/dt = k_build*Delta_hypo*(1-HAAF) - k_decay*HAAF; k_decay=1/(24*60)",
        solved_or_runtime_form="Integrated as a bounded state and clipped to [0, 1] after solver steps.",
        state_variables=("HAAF", "G"),
        parameters=("k_build", "k_decay"),
        units="dimensionless memory state, rates in 1/min",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=("https://doi.org/10.1056/NEJMra1215228",),
        validation_note="Research memory state only; never report as clinical hypo-awareness diagnosis.",
    ),
    FormulaSpec(
        formula_id="F13_EXOGENOUS_GLUCAGON_PKPD",
        title="Two-depot glucagon PK/PD effect on EGP",
        category="physiology",
        canonical_expression=(
            "dY1/dt=u_G-Y1/tmax_G; dY2/dt=Y1/tmax_G-Y2/tmax_G; "
            "dGamma/dt=(Y2/tmax_G)/V_Gamma-k_eG*Gamma; "
            "dx_gluc/dt=-k_aG*x_gluc+S_G*k_aG*Gamma"
        ),
        solved_or_runtime_form="Integrated only from deterministic glucagon requests after safety caps.",
        state_variables=("Y1", "Y2", "Gamma", "x_gluc"),
        parameters=("u_G", "tmax_G", "V_Gamma", "k_eG", "k_aG", "S_G"),
        units="Y depots in pg-like mass, Gamma in pg/mL, x_gluc dimensionless",
        implementation_paths=(
            "src/iints/core/patient/bergman_model.py:_ode",
            "src/iints/core/patient/hovorka_model.py:_ode",
        ),
        literature_basis=("https://arxiv.org/abs/2202.13938",),
        validation_note="Dual-hormone simulation support only; not a real pump recommendation.",
    ),
    FormulaSpec(
        formula_id="F14_SMOOTH_RENAL_CLEARANCE",
        title="Differentiable renal glucose clearance",
        category="physiology",
        canonical_expression="softplus(z)=s*log(1+exp(z/s)); z=G-162; F_R=c*softplus(G-162)",
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
            "https://en.wikipedia.org/wiki/Glycosuria",
            "https://en.wikipedia.org/wiki/Renal_threshold",
        ),
        validation_note="Smooth approximation to avoid discontinuous renal cutoff; threshold is a configurable research approximation.",
    ),
    FormulaSpec(
        formula_id="F15_CGM_ISF_OBSERVATION",
        title="CGM blood-to-ISF lag and deterministic observation equation",
        category="sensor",
        canonical_expression=(
            "tau_ISF*dISF/dt = BG_lagged - ISF; "
            "ISF_next = ISF + alpha*(BG_lagged-ISF); CGM = ISF + bias + drift + noise - compression_offset"
        ),
        solved_or_runtime_form="Euler low-pass update with bounded alpha; stochastic noise uses seeded RNG state, not AI.",
        state_variables=("ISF", "BG_lagged", "CGM"),
        parameters=("tau_ISF", "lag_minutes", "bias", "drift", "noise", "compression_offset"),
        units="mg/dL",
        implementation_paths=("src/iints/core/devices/models.py:SensorModel.read",),
        literature_basis=("https://en.wikipedia.org/wiki/Continuous_glucose_monitor",),
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
                "runtime_form": formula.solved_or_runtime_form,
                "units": formula.units,
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
        "These formulas are static SDK knowledge. The local AI may explain them, but it must not derive, solve, or alter them.",
        "",
    ]
    for formula in FORMULAS:
        lines.extend(
            [
                f"## {formula.formula_id}: {formula.title}",
                "",
                f"Category: `{formula.category}`",
                "",
                "Canonical expression:",
                "",
                f"```text\n{formula.canonical_expression}\n```",
                "",
                f"Runtime/solved form: {formula.solved_or_runtime_form}",
                "",
                f"Units: {formula.units}",
                "",
                f"Implementation: {', '.join(f'`{path}`' for path in formula.implementation_paths)}",
                "",
                f"Literature basis: {', '.join(formula.literature_basis)}",
                "",
                f"Validation note: {formula.validation_note}",
                "",
            ]
        )
    return "\n".join(lines)
