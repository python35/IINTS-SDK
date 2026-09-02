"""Compartment and flux schema for the IINTS patient models.

This module names what the patient ODEs actually integrate, so that a viewer or
report can render a compartment diagram without re-implementing the physiology
and without inventing structure the model does not contain.

Three distinctions carried here are load-bearing and must survive into any
visualisation:

`kind`
    ``pool`` is a mass compartment (mg, mU, pg) and ``concentration`` is an
    amount per volume. Only these two are physical contents that can be drawn as
    a filled volume. ``effect`` states are dimensionless or rate-valued remote
    action variables -- they have no volume and no content, and drawing them as a
    tank would imply a substance the model never tracks. ``legacy`` states are
    retained only so that older snapshots can be restored.

`provenance`
    ``canonical`` marks states from the published Hovorka (or Bergman) equations.
    ``extension`` marks states this project added. The patient model docstrings
    state plainly that the extensions are not part of the canonical model and are
    not clinically validated physiology; anything rendering them must say so
    rather than presenting them alongside canonical states as equals.

`rate_expression`
    The flux term exactly as it appears in the ODE. Fluxes here are deliberately
    not reduced to single constants, because most of them are state dependent
    (``x1 * Q1`` is not a rate constant). A viewer that needs numbers should use
    the instantaneous flux snapshot the model can emit, not a constant from this
    table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Compartment:
    """One state variable of a patient ODE."""

    key: str
    symbol: str
    label: str
    unit: str
    state_index: int
    kind: str  # pool | concentration | effect | legacy
    site: str  # subcutaneous | gut | plasma | periphery | signal
    provenance: str  # canonical | extension
    description: str

    @property
    def is_physical_content(self) -> bool:
        """True when the value is an amount or a concentration of a substance."""

        return self.kind in {"pool", "concentration"}


@dataclass(frozen=True)
class Flux:
    """One transfer term between compartments, or in/out of the patient."""

    key: str
    source: Optional[str]  # compartment key, or None for an external input
    target: Optional[str]  # compartment key, or None for elimination/clearance
    label: str
    unit: str
    rate_expression: str
    parameters: Tuple[str, ...]
    provenance: str
    description: str = ""
    recorded: bool = True
    """Whether the model emits an instantaneous numeric value for this flux.

    Discrete events -- a meal mass added at one step -- have no instantaneous
    rate, so they appear in the topology without a number. A test pins the
    recorded set against what the ODE actually writes, so the schema cannot
    drift away from the equations unnoticed.
    """


# --------------------------------------------------------------------------
# Adapted Hovorka model: 19 states, documented in hovorka_model.py
# --------------------------------------------------------------------------

HOVORKA_COMPARTMENTS: Tuple[Compartment, ...] = (
    Compartment(
        key="Q1",
        symbol="Q₁",
        label="Accessible glucose",
        unit="mg",
        state_index=0,
        kind="pool",
        site="plasma",
        provenance="canonical",
        description=(
            "Glucose mass in the accessible (plasma and rapidly equilibrating) "
            "space. Plasma glucose concentration is Q1 divided by the glucose "
            "distribution volume V_G."
        ),
    ),
    Compartment(
        key="Q2",
        symbol="Q₂",
        label="Non-accessible glucose",
        unit="mg",
        state_index=1,
        kind="pool",
        site="periphery",
        provenance="canonical",
        description="Glucose mass in the slowly equilibrating peripheral space.",
    ),
    Compartment(
        key="S1",
        symbol="S₁",
        label="Subcutaneous insulin depot 1",
        unit="mU",
        state_index=2,
        kind="pool",
        site="subcutaneous",
        provenance="canonical",
        description="First of two serial depots representing subcutaneous insulin absorption.",
    ),
    Compartment(
        key="S2",
        symbol="S₂",
        label="Subcutaneous insulin depot 2",
        unit="mU",
        state_index=3,
        kind="pool",
        site="subcutaneous",
        provenance="canonical",
        description="Second absorption depot; its outflow is the appearance rate into plasma.",
    ),
    Compartment(
        key="I",
        symbol="I",
        label="Plasma insulin",
        unit="mU/L",
        state_index=4,
        kind="concentration",
        site="plasma",
        provenance="canonical",
        description="Plasma insulin concentration driving the three remote action states.",
    ),
    Compartment(
        key="x1",
        symbol="x₁",
        label="Insulin action on distribution",
        unit="1/min",
        state_index=5,
        kind="effect",
        site="signal",
        provenance="canonical",
        description=(
            "Remote insulin action setting glucose transport from the accessible "
            "to the peripheral compartment. A rate, not a substance."
        ),
    ),
    Compartment(
        key="x2",
        symbol="x₂",
        label="Insulin action on disposal",
        unit="1/min",
        state_index=6,
        kind="effect",
        site="signal",
        provenance="canonical",
        description="Remote insulin action setting peripheral glucose disposal. A rate, not a substance.",
    ),
    Compartment(
        key="x3",
        symbol="x₃",
        label="Insulin action on EGP",
        unit="dimensionless",
        state_index=7,
        kind="effect",
        site="signal",
        provenance="canonical",
        description="Remote insulin action suppressing endogenous glucose production.",
    ),
    Compartment(
        key="D1",
        symbol="D₁",
        label="Meal absorption compartment 1",
        unit="mg",
        state_index=8,
        kind="pool",
        site="gut",
        provenance="canonical",
        description="Ingested carbohydrate mass entering the two-compartment absorption chain.",
    ),
    Compartment(
        key="D2",
        symbol="D₂",
        label="Meal absorption compartment 2",
        unit="mg",
        state_index=9,
        kind="pool",
        site="gut",
        provenance="canonical",
        description=(
            "Second absorption compartment. Its outflow times the bioavailability "
            "A_G is glucose appearance; the remaining fraction is not absorbed."
        ),
    ),
    Compartment(
        key="D3",
        symbol="D₃",
        label="Reserved legacy meal mass",
        unit="mg",
        state_index=10,
        kind="legacy",
        site="gut",
        provenance="extension",
        description=(
            "Retained so older 19-state snapshots restore; its derivative is "
            "identically zero and it carries no live meal mass."
        ),
    ),
    Compartment(
        key="H_stress",
        symbol="H_stress",
        label="Stress state",
        unit="dimensionless",
        state_index=11,
        kind="effect",
        site="signal",
        provenance="extension",
        description=(
            "First-order scenario stress state with a 20-minute time constant. It "
            "summarises an effect on insulin sensitivity and endogenous glucose "
            "production; it is not a measured adrenaline or cortisol concentration."
        ),
    ),
    Compartment(
        key="H_exercise",
        symbol="H_exercise",
        label="Exercise state",
        unit="dimensionless",
        state_index=12,
        kind="effect",
        site="signal",
        provenance="extension",
        description=(
            "First-order scenario exercise state with a 10-minute time constant. "
            "Not a measured AMPK or endorphin concentration."
        ),
    ),
    Compartment(
        key="Y1",
        symbol="Y₁",
        label="Subcutaneous glucagon depot 1",
        unit="pg",
        state_index=13,
        kind="pool",
        site="subcutaneous",
        provenance="extension",
        description="First depot of the bi-hormonal exogenous glucagon chain.",
    ),
    Compartment(
        key="Y2",
        symbol="Y₂",
        label="Subcutaneous glucagon depot 2",
        unit="pg",
        state_index=14,
        kind="pool",
        site="subcutaneous",
        provenance="extension",
        description="Second glucagon depot; its outflow sets the plasma glucagon concentration.",
    ),
    Compartment(
        key="Gamma",
        symbol="Γ",
        label="Plasma glucagon",
        unit="pg/mL",
        state_index=15,
        kind="concentration",
        site="plasma",
        provenance="extension",
        description=(
            "Plasma glucagon concentration. This state is integrated for reporting "
            "and tracks the algebraic expression k2*Y2/clearance; the glucagon "
            "effect on EGP is driven by that expression directly, not by this state."
        ),
    ),
    Compartment(
        key="x_gluc",
        symbol="x_gluc",
        label="Glucagon action on EGP",
        unit="dimensionless",
        state_index=16,
        kind="effect",
        site="signal",
        provenance="extension",
        description="Remote glucagon action raising endogenous glucose production.",
    ),
    Compartment(
        key="HAAF",
        symbol="HAAF",
        label="Antecedent hypoglycaemia memory",
        unit="dimensionless",
        state_index=17,
        kind="effect",
        site="signal",
        provenance="extension",
        description=(
            "Phenomenological memory state that blunts the counterregulatory "
            "rescue response after prior hypoglycaemia. Not a measured quantity."
        ),
    ),
    Compartment(
        key="GLUT4_active",
        symbol="GLUT4",
        label="Exercise-mediated uptake state",
        unit="dimensionless",
        state_index=18,
        kind="effect",
        site="signal",
        provenance="extension",
        description=(
            "Heuristic state raising insulin-independent glucose uptake during "
            "exercise. Inspired by skeletal-muscle uptake; not a GLUT4 assay model."
        ),
    ),
)


HOVORKA_FLUXES: Tuple[Flux, ...] = (
    Flux(
        key="insulin_infusion",
        source=None,
        target="S1",
        label="Pump insulin infusion",
        unit="mU/min",
        rate_expression="u_insulin",
        parameters=(),
        provenance="canonical",
        description="Delivered insulin entering the first subcutaneous depot.",
    ),
    Flux(
        key="insulin_depot_transfer",
        source="S1",
        target="S2",
        label="Depot transfer",
        unit="mU/min",
        rate_expression="S1 / t_max_I",
        parameters=("t_max_I", "insulin_type", "t_max_I_override"),
        provenance="canonical",
    ),
    Flux(
        key="insulin_appearance",
        source="S2",
        target="I",
        label="Insulin appearance in plasma",
        unit="mU/min",
        rate_expression="S2 / t_max_I  (divided by V_I for the concentration)",
        parameters=("t_max_I", "V_I_per_kg", "body_weight_kg"),
        provenance="canonical",
    ),
    Flux(
        key="insulin_elimination",
        source="I",
        target=None,
        label="Insulin elimination",
        unit="mU/L/min",
        rate_expression="k_e * I",
        parameters=("k_e",),
        provenance="canonical",
    ),
    Flux(
        key="insulin_action_x1",
        source="I",
        target="x1",
        label="Action activation (distribution)",
        unit="1/min²",
        rate_expression="-k_a1 * x1 + k_b1 * I",
        parameters=("k_a1", "S_IT"),
        provenance="canonical",
        description="Signal activation, not a mass transfer.",
    ),
    Flux(
        key="insulin_action_x2",
        source="I",
        target="x2",
        label="Action activation (disposal)",
        unit="1/min²",
        rate_expression="-k_a2 * x2 + k_b2 * I",
        parameters=("k_a2", "S_ID"),
        provenance="canonical",
        description="Signal activation, not a mass transfer.",
    ),
    Flux(
        key="insulin_action_x3",
        source="I",
        target="x3",
        label="Action activation (EGP)",
        unit="1/min",
        rate_expression="-k_a3 * x3 + k_b3 * I",
        parameters=("k_a3", "S_IE"),
        provenance="canonical",
        description="Signal activation, not a mass transfer.",
    ),
    Flux(
        key="meal_ingestion",
        source=None,
        target="D1",
        label="Carbohydrate ingestion",
        unit="mg",
        rate_expression="carb impulse added to D1 (grams x 1000)",
        parameters=(),
        provenance="canonical",
        description="Applied as a discrete mass addition at the meal step, not a continuous rate.",
        recorded=False,
    ),
    Flux(
        key="meal_transfer",
        source="D1",
        target="D2",
        label="Gastric transfer",
        unit="mg/min",
        rate_expression="D1 / t_max_G",
        parameters=("t_max_G",),
        provenance="canonical",
    ),
    Flux(
        key="glucose_appearance",
        source="D2",
        target="Q1",
        label="Glucose appearance",
        unit="mg/min",
        rate_expression="A_G * D2 / t_max_G",
        parameters=("A_G", "t_max_G"),
        provenance="canonical",
        description="The fraction (1 - A_G) of the transferred mass is not absorbed.",
    ),
    Flux(
        key="glucose_to_periphery",
        source="Q1",
        target="Q2",
        label="Transport to periphery",
        unit="mg/min",
        rate_expression="x1 * Q1",
        parameters=(),
        provenance="canonical",
        description="Insulin-dependent: the rate is the remote action state x1.",
    ),
    Flux(
        key="glucose_from_periphery",
        source="Q2",
        target="Q1",
        label="Return from periphery",
        unit="mg/min",
        rate_expression="k_12 * Q2",
        parameters=("k_12",),
        provenance="canonical",
    ),
    Flux(
        key="peripheral_disposal",
        source="Q2",
        target=None,
        label="Insulin-dependent disposal",
        unit="mg/min",
        rate_expression="x2 * Q2",
        parameters=(),
        provenance="canonical",
        description="Insulin-dependent: the rate is the remote action state x2.",
    ),
    Flux(
        key="nimgu",
        source="Q1",
        target=None,
        label="Insulin-independent uptake",
        unit="mg/min",
        rate_expression="F_01c * (1 + 1.5 * GLUT4_active)",
        parameters=("F_01c_per_kg", "body_weight_kg"),
        provenance="canonical",
        description=(
            "Canonical F_01c uptake, tapering below about 81 mg/dL. The GLUT4 "
            "enhancement factor is a project extension."
        ),
    ),
    Flux(
        key="renal_clearance",
        source="Q1",
        target=None,
        label="Renal clearance",
        unit="mg/min",
        rate_expression="0.003 * V_G * softplus(G - 162)",
        parameters=("V_G_per_kg", "body_weight_kg"),
        provenance="extension",
        description="Smooth glucosuria term above roughly 162 mg/dL, in place of a hard threshold.",
    ),
    Flux(
        key="endogenous_production",
        source=None,
        target="Q1",
        label="Endogenous glucose production",
        unit="mg/min",
        rate_expression="EGP_0 * max(0, 1 - x3 + x_gluc)",
        parameters=("EGP_0_per_kg", "body_weight_kg"),
        provenance="canonical",
        description=(
            "Hepatic output, suppressed by insulin action x3 and raised by glucagon "
            "action x_gluc, stress, and the counterregulatory rescue multiplier."
        ),
    ),
    Flux(
        key="dawn_flux",
        source=None,
        target="Q1",
        label="Dawn perturbation",
        unit="mg/min",
        rate_expression="dawn_rate(t) * V_G",
        parameters=("V_G_per_kg", "body_weight_kg"),
        provenance="extension",
        description="Phenomenological dawn term configured in mg/dL/hour.",
    ),
    Flux(
        key="glucagon_infusion",
        source=None,
        target="Y1",
        label="Glucagon delivery",
        unit="pg/min",
        rate_expression="u_glucagon",
        parameters=(),
        provenance="extension",
    ),
    Flux(
        key="glucagon_depot_transfer",
        source="Y1",
        target="Y2",
        label="Glucagon depot transfer",
        unit="pg/min",
        rate_expression="Y1 / t_max_glucagon",
        parameters=("t_max_glucagon",),
        provenance="extension",
    ),
    Flux(
        key="glucagon_appearance",
        source="Y2",
        target="Gamma",
        label="Glucagon appearance",
        unit="pg/min",
        rate_expression="k_e_glucagon * Y2 / clearance",
        parameters=("k_e_glucagon", "glucagon_clearance_ml_kg_min", "body_weight_kg"),
        provenance="extension",
    ),
    Flux(
        key="glucagon_action",
        source="Gamma",
        target="x_gluc",
        label="Glucagon action activation",
        unit="1/min",
        rate_expression="k_a_glucagon * (S_glucagon * activation - x_gluc)",
        parameters=("k_a_glucagon", "S_glucagon", "glucagon_ec50_pg_ml"),
        provenance="extension",
        description="Driven by the algebraic plasma concentration, not by the Gamma state.",
    ),
)


# --------------------------------------------------------------------------
# Bergman model: state vector documented in bergman_model.py
# --------------------------------------------------------------------------

BERGMAN_COMPARTMENTS: Tuple[Compartment, ...] = (
    Compartment(
        key="G",
        symbol="G",
        label="Plasma glucose",
        unit="mg/dL",
        state_index=0,
        kind="concentration",
        site="plasma",
        provenance="canonical",
        description="Plasma glucose concentration; the Bergman model integrates concentration directly.",
    ),
    Compartment(
        key="X",
        symbol="X",
        label="Remote insulin action",
        unit="1/min",
        state_index=1,
        kind="effect",
        site="signal",
        provenance="canonical",
        description="Insulin action in the remote compartment. A rate, not a substance.",
    ),
    Compartment(
        key="I",
        symbol="I",
        label="Plasma insulin",
        unit="mU/L",
        state_index=2,
        kind="concentration",
        site="plasma",
        provenance="canonical",
        description="Plasma insulin concentration.",
    ),
    Compartment(
        key="Q_sto1",
        symbol="Q_sto1",
        label="Stomach (solid)",
        unit="mg",
        state_index=3,
        kind="pool",
        site="gut",
        provenance="canonical",
        description="Solid-phase gastric content.",
    ),
    Compartment(
        key="Q_sto2",
        symbol="Q_sto2",
        label="Stomach (liquid)",
        unit="mg",
        state_index=4,
        kind="pool",
        site="gut",
        provenance="canonical",
        description="Liquid-phase gastric content.",
    ),
    Compartment(
        key="Q_gut",
        symbol="Q_gut",
        label="Intestine",
        unit="mg",
        state_index=5,
        kind="pool",
        site="gut",
        provenance="canonical",
        description="Intestinal glucose mass; its outflow is glucose appearance.",
    ),
    Compartment(
        key="S1",
        symbol="S₁",
        label="Subcutaneous insulin depot 1",
        unit="mU",
        state_index=6,
        kind="pool",
        site="subcutaneous",
        provenance="canonical",
        description="First subcutaneous insulin depot.",
    ),
    Compartment(
        key="S2",
        symbol="S₂",
        label="Subcutaneous insulin depot 2",
        unit="mU",
        state_index=7,
        kind="pool",
        site="subcutaneous",
        provenance="canonical",
        description="Second subcutaneous insulin depot.",
    ),
    Compartment(
        key="Y1",
        symbol="Y₁",
        label="Subcutaneous glucagon depot 1",
        unit="pg",
        state_index=8,
        kind="pool",
        site="subcutaneous",
        provenance="extension",
        description="First depot of the exogenous glucagon chain.",
    ),
    Compartment(
        key="Y2",
        symbol="Y₂",
        label="Subcutaneous glucagon depot 2",
        unit="pg",
        state_index=9,
        kind="pool",
        site="subcutaneous",
        provenance="extension",
        description="Second glucagon depot.",
    ),
    Compartment(
        key="Gamma",
        symbol="Γ",
        label="Plasma glucagon",
        unit="pg/mL",
        state_index=10,
        kind="concentration",
        site="plasma",
        provenance="extension",
        description="Plasma glucagon concentration.",
    ),
    Compartment(
        key="x_gluc",
        symbol="x_gluc",
        label="Glucagon action on EGP",
        unit="dimensionless",
        state_index=11,
        kind="effect",
        site="signal",
        provenance="extension",
        description="Remote glucagon action raising endogenous glucose production.",
    ),
    Compartment(
        key="HAAF",
        symbol="HAAF",
        label="Antecedent hypoglycaemia memory",
        unit="dimensionless",
        state_index=12,
        kind="effect",
        site="signal",
        provenance="extension",
        description="Phenomenological memory state blunting counterregulation. Not a measured quantity.",
    ),
    Compartment(
        key="M_graft",
        symbol="M_graft",
        label="Stem-cell graft mass",
        unit="dimensionless",
        state_index=13,
        kind="effect",
        site="signal",
        provenance="extension",
        description=(
            "Engrafted beta-cell mass fraction for the stem-cell transplant "
            "experiment. A research construct, not a clinical measurement."
        ),
    ),
)


BERGMAN_FLUXES: Tuple[Flux, ...] = (
    Flux(
        key="meal_ingestion",
        source=None,
        target="Q_sto1",
        label="Carbohydrate ingestion",
        unit="mg",
        rate_expression="carb impulse added to Q_sto1 (grams x 1000)",
        parameters=(),
        provenance="canonical",
        description="Applied as a discrete mass addition at the meal step, not a continuous rate.",
        recorded=False,
    ),
    Flux(
        key="gastric_liquefaction",
        source="Q_sto1",
        target="Q_sto2",
        label="Solid to liquid emptying",
        unit="mg/min",
        rate_expression="1.5 * Q_sto1 / tau_meal",
        parameters=("tau_meal",),
        provenance="canonical",
        description="The 1.5 factor is this project's three-stage adaptation of the two-stage chain.",
    ),
    Flux(
        key="gastric_emptying",
        source="Q_sto2",
        target="Q_gut",
        label="Gastric emptying",
        unit="mg/min",
        rate_expression="Q_sto2 / tau_meal",
        parameters=("tau_meal",),
        provenance="canonical",
    ),
    Flux(
        key="glucose_appearance",
        source="Q_gut",
        target="G",
        label="Glucose appearance",
        unit="mg/dL/min",
        rate_expression="k_abs * Q_gut / V_G",
        parameters=("k_abs", "Vg", "body_weight_kg"),
        provenance="canonical",
        description="Per volume, because this backend integrates glucose as a concentration.",
    ),
    Flux(
        key="insulin_infusion",
        source=None,
        target="S1",
        label="Pump insulin infusion",
        unit="mU/min",
        rate_expression="u_insulin",
        parameters=(),
        provenance="canonical",
    ),
    Flux(
        key="insulin_depot_transfer",
        source="S1",
        target="S2",
        label="Depot transfer",
        unit="mU/min",
        rate_expression="k_a * S1",
        parameters=("k_a",),
        provenance="canonical",
    ),
    Flux(
        key="insulin_appearance",
        source="S2",
        target="I",
        label="Insulin appearance in plasma",
        unit="mU/min",
        rate_expression="k_a * S2  (divided by V_I for the concentration)",
        parameters=("k_a", "Vi", "body_weight_kg"),
        provenance="canonical",
    ),
    Flux(
        key="insulin_elimination",
        source="I",
        target=None,
        label="Insulin elimination",
        unit="mU/L/min",
        rate_expression="n * I",
        parameters=("n",),
        provenance="canonical",
    ),
    Flux(
        key="insulin_action",
        source="I",
        target="X",
        label="Remote action activation",
        unit="1/min²",
        rate_expression="-p2 * X + p3_eff * (I - I_basal)",
        parameters=("p2", "p3"),
        provenance="canonical",
        description="Signal activation, not a mass transfer. Stress and exercise scale p3.",
    ),
    Flux(
        key="glucose_uptake",
        source="G",
        target=None,
        label="Glucose disappearance",
        unit="mg/dL/min",
        rate_expression="(p1_eff + X) * G",
        parameters=("p1",),
        provenance="canonical",
        description=(
            "Combines insulin-independent clearance p1 and insulin-dependent "
            "clearance X in one term; the minimal model does not separate them. "
            "Goes negative when remote insulin action X falls below its "
            "reference, in which case the term is a net appearance and a "
            "directional view must reverse the arrow rather than clamp it."
        ),
    ),
    Flux(
        key="basal_production",
        source=None,
        target="G",
        label="Endogenous glucose production",
        unit="mg/dL/min",
        rate_expression="p1_eff * Gb_eff",
        parameters=("p1", "Gb"),
        provenance="canonical",
        description=(
            "Hepatic output toward the basal set point, scaled by stress, the "
            "counterregulatory rescue multiplier, and glucagon action."
        ),
    ),
    Flux(
        key="renal_clearance",
        source="G",
        target=None,
        label="Renal clearance",
        unit="mg/dL/min",
        rate_expression="0.003 * softplus(G - 162)",
        parameters=(),
        provenance="extension",
        description="Smooth glucosuria term above roughly 162 mg/dL.",
    ),
    Flux(
        key="exercise_uptake",
        source="G",
        target=None,
        label="Exercise glucose uptake",
        unit="mg/dL/min",
        rate_expression="intensity * exercise_glucose_consumption_rate",
        parameters=("exercise_glucose_consumption_rate",),
        provenance="extension",
        description="Insulin-independent uptake applied while an exercise scenario is active.",
    ),
    Flux(
        key="dawn_flux",
        source=None,
        target="G",
        label="Dawn perturbation",
        unit="mg/dL/min",
        rate_expression="dawn_rate(t)",
        parameters=(),
        provenance="extension",
        description="Phenomenological dawn term configured in mg/dL/hour.",
    ),
    Flux(
        key="glucagon_infusion",
        source=None,
        target="Y1",
        label="Glucagon delivery",
        unit="pg/min",
        rate_expression="u_glucagon",
        parameters=(),
        provenance="extension",
    ),
    Flux(
        key="glucagon_depot_transfer",
        source="Y1",
        target="Y2",
        label="Glucagon depot transfer",
        unit="pg/min",
        rate_expression="Y1 / t_max_glucagon",
        parameters=("t_max_glucagon",),
        provenance="extension",
    ),
    Flux(
        key="glucagon_appearance",
        source="Y2",
        target="Gamma",
        label="Glucagon appearance",
        unit="pg/mL",
        rate_expression="k_e_glucagon * Y2 / clearance",
        parameters=("k_e_glucagon", "glucagon_clearance_ml_kg_min", "body_weight_kg"),
        provenance="extension",
    ),
    Flux(
        key="glucagon_action",
        source="Gamma",
        target="x_gluc",
        label="Glucagon action activation",
        unit="1/min",
        rate_expression="k_a_glucagon * (S_glucagon * activation - x_gluc)",
        parameters=("k_a_glucagon", "S_glucagon", "glucagon_ec50_pg_ml"),
        provenance="extension",
        description="Driven by the algebraic plasma concentration, not by the Gamma state.",
    ),
    Flux(
        key="islet_secretion_subq",
        source="M_graft",
        target="S1",
        label="Graft secretion into depot",
        unit="mU/min",
        rate_expression="gamma * M_graft * max(G - h, 0) * V_I * subq_fraction",
        parameters=("gamma", "h", "stem_cell_subq_fraction"),
        provenance="extension",
        description="Stem-cell transplant experiment; zero unless a graft is configured.",
    ),
    Flux(
        key="islet_secretion_plasma",
        source="M_graft",
        target="I",
        label="Graft secretion into plasma",
        unit="mU/L/min",
        rate_expression="gamma * M_graft * max(G - h, 0) * (1 - subq_fraction)",
        parameters=("gamma", "h", "stem_cell_subq_fraction"),
        provenance="extension",
        description="Stem-cell transplant experiment; zero unless a graft is configured.",
    ),
    Flux(
        key="graft_rejection",
        source="M_graft",
        target=None,
        label="Immune rejection",
        unit="1/min",
        rate_expression="immune_rejection_rate * M_graft",
        parameters=("immune_rejection_rate",),
        provenance="extension",
        description="First-order decay of the graft mass fraction.",
    ),
)


MODEL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "hovorka": {
        "model_label": "Adapted Hovorka (19 states)",
        "compartments": HOVORKA_COMPARTMENTS,
        "fluxes": HOVORKA_FLUXES,
    },
    "bergman": {
        "model_label": "Bergman minimal model, extended (14 states)",
        "compartments": BERGMAN_COMPARTMENTS,
        "fluxes": BERGMAN_FLUXES,
    },
}


def compartment_schema(model_key: str) -> Dict[str, Any]:
    """Return the serialisable compartment schema for a patient model key."""

    try:
        schema = MODEL_SCHEMAS[model_key]
    except KeyError as error:
        raise KeyError(
            f"No compartment schema for patient model '{model_key}'. "
            f"Known keys: {sorted(MODEL_SCHEMAS)}"
        ) from error
    return {
        "model_key": model_key,
        "model_label": schema["model_label"],
        "compartments": [_compartment_payload(item) for item in schema["compartments"]],
        "fluxes": [_flux_payload(item) for item in schema["fluxes"]],
    }


def schema_for_model(model: Any) -> Optional[Dict[str, Any]]:
    """Return a patient model's own compartment schema, or None if it has none.

    The schema is asked of the model rather than inferred from a name, so a run
    exported with a backend that publishes no schema carries no schema file at
    all. A consumer then knows the layout is unavailable instead of drawing
    another model's compartments.
    """

    describe = getattr(model, "describe_compartments", None)
    if not callable(describe):
        return None
    return describe()


def _compartment_payload(compartment: Compartment) -> Dict[str, Any]:
    return {
        "key": compartment.key,
        "symbol": compartment.symbol,
        "label": compartment.label,
        "unit": compartment.unit,
        "state_index": compartment.state_index,
        "kind": compartment.kind,
        "site": compartment.site,
        "provenance": compartment.provenance,
        "description": compartment.description,
    }


def _flux_payload(flux: Flux) -> Dict[str, Any]:
    return {
        "key": flux.key,
        "source": flux.source,
        "target": flux.target,
        "label": flux.label,
        "unit": flux.unit,
        "rate_expression": flux.rate_expression,
        "parameters": list(flux.parameters),
        "provenance": flux.provenance,
        "description": flux.description,
        "recorded": flux.recorded,
    }


__all__ = [
    "Compartment",
    "Flux",
    "HOVORKA_COMPARTMENTS",
    "HOVORKA_FLUXES",
    "BERGMAN_COMPARTMENTS",
    "MODEL_SCHEMAS",
    "compartment_schema",
    "schema_for_model",
]
