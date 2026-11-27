"""A. Experiment Designer & Pre-Registration Approval Gate for IINTS Research Lab.

Converts structured scientific questions into reviewable, pre-registered
``ResearchPlan`` artifacts with explicit hypotheses, control conditions,
missing-data contracts, confounders, and statistical protocols.

Enforces a hard human-in-the-loop gate: an experiment remains ``status='proposed'``
until explicitly approved via ``approve_research_plan()``, locking a SHA-256
pre-registration digest before any simulation is executed.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
import yaml

from iints.research.scenarios import (
    generate_boundary_scenarios,
    generate_dropout_conditions,
    generate_virtual_cohort,
)
from iints.sdk2.contracts import (
    RESEARCH_ONLY_DISCLAIMER,
    AlgorithmSpec,
    EvaluationSpec,
    MissingDataContract,
    PatientSpec,
    ScenarioSpec,
    SensorFaultSpec,
)
from iints.sdk2.reproducibility import sha256_json

__all__ = [
    "PlanNotApprovedError",
    "HypothesisSpec",
    "StatisticalProtocolSpec",
    "ResearchPlan",
    "list_research_templates",
    "propose_research_plan",
    "approve_research_plan",
    "save_research_plan",
    "load_research_plan",
]


class PlanNotApprovedError(RuntimeError):
    """Raised when the runner is asked to execute an unapproved ResearchPlan."""


class HypothesisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    primary_endpoint: str
    direction: str


class StatisticalProtocolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_of_analysis: Literal["virtual_patient"] = "virtual_patient"
    paired_against_control: bool = True
    primary_metric: str = "mae"
    primary_horizon_minutes: int = 30
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    nonparametric_test: Literal["wilcoxon_signed_rank"] = "wilcoxon_signed_rank"
    multiplicity_correction: Literal["holm_bonferroni"] = "holm_bonferroni"


class ResearchPlan(BaseModel):
    """Pre-registered, human-reviewable scientific experiment design."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    template_id: str
    title: str
    research_question: str
    disclaimer: str = RESEARCH_ONLY_DISCLAIMER
    status: Literal["proposed", "approved"] = "proposed"
    approved_by: Optional[str] = None
    approved_utc: Optional[str] = None
    plan_lock_sha256: Optional[str] = None
    master_seed: int = 42
    replicates_per_profile: int = Field(default=1, ge=1, le=20)
    control_condition_id: str = "dropout_0m"
    hypotheses: List[HypothesisSpec]
    independent_variables: List[str]
    dependent_variables: List[str]
    confounders_and_controls: List[str]
    patients: List[PatientSpec]
    scenarios: List[ScenarioSpec]
    sensor_conditions: List[SensorFaultSpec]
    algorithms: List[AlgorithmSpec]
    evaluation: EvaluationSpec
    statistical_protocol: StatisticalProtocolSpec = Field(default_factory=StatisticalProtocolSpec)

    def design_Payload_without_lock(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        for k in ("status", "approved_by", "approved_utc", "plan_lock_sha256"):
            data.pop(k, None)
        return data

    def total_simulation_cells(self) -> int:
        return (
            len(self.patients)
            * len(self.scenarios)
            * len(self.sensor_conditions)
            * int(self.replicates_per_profile)
        )


def list_research_templates() -> List[Dict[str, str]]:
    return [
        {
            "template_id": "study_001_cgm_dropout",
            "direction": "Direction A — Robustness Benchmark",
            "title": "Evaluating the Robustness of Glucose Forecasting Algorithms Under Simulated CGM Data Loss",
            "question": (
                "How does intermittent CGM signal loss (0, 5, 15, 30 min) affect 30-minute "
                "glucose forecasting accuracy across unseen virtual patients and boundary scenarios?"
            ),
        },
        {
            "template_id": "study_002_inclusive_stale_ui",
            "direction": "Direction B — Inclusive Interaction & Display Hazard Benchmark",
            "title": "Blank Missingness vs Unflagged Stale CGM Display Under Signal Loss",
            "question": (
                "How do blank missingness ('nan_blank') versus stale held readings ('stale_hold') "
                "affect forecasting error and missed-hypoglycaemia hazard when stale flags are hidden vs exposed?"
            ),
        },
        {
            "template_id": "study_003_cross_simulator",
            "direction": "Direction C — Cross-Simulator Validation",
            "title": "Invariance of Forecaster Rankings Across Independent Physiological Simulators",
            "question": (
                "Do conclusions about glucose forecaster accuracy and CGM-dropout robustness hold "
                "when the underlying virtual patient model changes (Hovorka vs Bergman vs Simple ODE)?"
            ),
        },
    ]


def propose_research_plan(
    template_id: str = "study_001_cgm_dropout",
    *,
    plan_id: Optional[str] = None,
    n_profiles: int = 100,
    seed: int = 42,
    duration_hours: float = 24.0,
    include_ood_scenarios: bool = True,
) -> ResearchPlan:
    """Create a reviewable ``ResearchPlan`` in ``status='proposed'`` (never auto-executed)."""
    scenarios = generate_boundary_scenarios(duration_hours=duration_hours)
    if not include_ood_scenarios:
        scenarios = [s for s in scenarios if s.regime == "in_distribution"]

    default_algos = [
        AlgorithmSpec(model_id="persistence_v1"),
        AlgorithmSpec(model_id="ar2_v1"),
        AlgorithmSpec(model_id="ridge_ml_v1"),
        AlgorithmSpec(model_id="hovorka_graybox_v2"),
    ]

    if template_id == "study_001_cgm_dropout":
        cohort = generate_virtual_cohort(n_profiles=n_profiles, seed=seed, unseen_fraction=0.30)
        conditions = generate_dropout_conditions([0, 5, 15, 30], display_mode="nan_blank")
        return ResearchPlan(
            plan_id=plan_id or "study_001_cgm_dropout",
            template_id=template_id,
            title="Study 001 — Robustness of Glucose Forecasting Algorithms Under Simulated CGM Data Loss",
            research_question=(
                "How does intermittent CGM signal loss (0, 5, 15, and 30 minutes) affect "
                "30-minute glucose forecasting accuracy across virtual patient profiles and OOD boundary scenarios?"
            ),
            master_seed=seed,
            control_condition_id="dropout_0m",
            hypotheses=[
                HypothesisSpec(
                    id="H1_degradation",
                    statement=(
                        "30-minute forecasting MAE increases monotonically with CGM dropout duration "
                        "(5, 15, 30 min) relative to the 0-minute clean control within the same virtual patient."
                    ),
                    primary_endpoint="paired_delta_mae_30m",
                    direction="delta_mae > 0 as dropout_minutes increases",
                ),
                HypothesisSpec(
                    id="H2_differential_robustness",
                    statement=(
                        "Mechanistic state-space forecasting (hovorka_graybox_v2) exhibits smaller paired MAE "
                        "degradation during 15–30 min CGM dropouts than pure sensor-extrapolation baselines."
                    ),
                    primary_endpoint="dropout_mae_30m",
                    direction="hovorka_graybox_v2 < persistence_v1, ar2_v1 during dropout",
                ),
                HypothesisSpec(
                    id="H3_unseen_and_ood_generalization",
                    statement=(
                        "Algorithm robustness rankings persist on strictly held-out unseen virtual profiles "
                        "and adversarial post-prandial / nocturnal descent boundary scenarios."
                    ),
                    primary_endpoint="rank_concordance_unseen_and_ood",
                    direction="consistent ordering across dev, unseen, and ood_stress",
                ),
            ],
            independent_variables=["sensor.dropout_minutes (0, 5, 15, 30)", "algorithm.model_id", "patient.split", "scenario.regime"],
            dependent_variables=["mae", "rmse", "dropout_mae", "dropout_recovery_mae", "clarke_ab_pct", "coverage_90_pct"],
            confounders_and_controls=[
                "Identical underlying patient blood-glucose trajectory across all 4 dropout arms via hierarchical SHA-256 seeding.",
                "Pre-registered missing-data contract: no truth columns exposed; causal history strictly truncated at origin.",
                "30% strictly held-out unseen patient profiles never used during development.",
                "Within-patient paired differences computed first to eliminate timepoint pseudoreplication.",
            ],
            patients=cohort,
            scenarios=scenarios,
            sensor_conditions=conditions,
            algorithms=default_algos,
            evaluation=EvaluationSpec(
                prediction_horizon_minutes=30,
                extra_horizons_minutes=[60],
                missing_data_contract=MissingDataContract(
                    allow_last_valid_observation=True,
                    expose_stale_flag_to_algorithm=True,
                    expose_insulin_and_announced_carbs=True,
                    impute_before_algorithm="none",
                    reference_target="true_plasma_glucose",
                ),
            ),
        )

    if template_id == "study_002_inclusive_stale_ui":
        cohort = generate_virtual_cohort(n_profiles=max(6, n_profiles), seed=seed, unseen_fraction=0.30)
        conditions = [
            SensorFaultSpec(condition_id="clean_0m", dropout_minutes=0, display_mode="nan_blank"),
            SensorFaultSpec(
                condition_id="dropout_20m_nan_blank",
                dropout_minutes=20,
                dropout_at_hour=8.5,
                recurring_dropout_every_hours=6.0,
                display_mode="nan_blank",
            ),
            SensorFaultSpec(
                condition_id="dropout_20m_stale_hold",
                dropout_minutes=20,
                dropout_at_hour=8.5,
                recurring_dropout_every_hours=6.0,
                display_mode="stale_hold",
            ),
        ]
        return ResearchPlan(
            plan_id=plan_id or "study_002_inclusive_stale_ui",
            template_id=template_id,
            title="Study 002 — Blank Missingness vs Unflagged Stale Display Hazard Benchmark",
            research_question=(
                "Does displaying outdated held CGM values ('stale_hold') without explicit staleness handling "
                "degrade forecast accuracy and increase missed-hypoglycaemia risk compared with explicit NaN gaps?"
            ),
            master_seed=seed,
            control_condition_id="clean_0m",
            hypotheses=[
                HypothesisSpec(
                    id="H1_stale_hazard",
                    statement="Unflagged stale CGM display increases prediction error during rapid glycaemic transitions.",
                    primary_endpoint="dropout_mae_30m",
                    direction="stale_hold >= nan_blank for staleness-blind estimators",
                )
            ],
            independent_variables=["sensor.display_mode (nan_blank vs stale_hold)", "algorithm.model_id"],
            dependent_variables=["mae", "dropout_mae", "missed_hypo_pct", "coverage_90_pct"],
            confounders_and_controls=[
                "Note: Software benchmarks detect algorithmic/interface data-handling hazards; real human participants are required for clinical accessibility claims."
            ],
            patients=cohort,
            scenarios=scenarios,
            sensor_conditions=conditions,
            algorithms=default_algos,
            evaluation=EvaluationSpec(
                prediction_horizon_minutes=30,
                missing_data_contract=MissingDataContract(
                    expose_stale_flag_to_algorithm=False,
                    reference_target="true_plasma_glucose",
                ),
            ),
        )

    if template_id == "study_003_cross_simulator":
        per_sim = max(2, n_profiles // 3)
        cross_simulator_cohort: List[PatientSpec] = []
        for sim_model in ("hovorka_equilibrium", "bergman", "iints_ode"):
            sub = generate_virtual_cohort(n_profiles=per_sim, seed=seed, unseen_fraction=0.30, model=sim_model)  # type: ignore[arg-type]
            for p in sub:
                cross_simulator_cohort.append(p.model_copy(update={"profile": f"{sim_model}_{p.profile}"}))
        conditions = generate_dropout_conditions([0, 15, 30], display_mode="nan_blank")
        return ResearchPlan(
            plan_id=plan_id or "study_003_cross_simulator",
            template_id=template_id,
            title="Study 003 — Cross-Simulator Validation of Glucose Forecaster Robustness",
            research_question=(
                "Are forecaster rankings under CGM signal loss invariant across Hovorka, Bergman, and Simple ODE virtual patients?"
            ),
            master_seed=seed,
            control_condition_id="dropout_0m",
            hypotheses=[
                HypothesisSpec(
                    id="H1_simulator_invariance",
                    statement="Relative algorithm rank ordering at 30-minute horizon remains concordant across simulator families.",
                    primary_endpoint="cross_simulator_spearman_rho",
                    direction="rho > 0.7",
                )
            ],
            independent_variables=["patient.model (hovorka_equilibrium, bergman, iints_ode)", "sensor.dropout_minutes"],
            dependent_variables=["mae", "rmse", "dropout_mae"],
            confounders_and_controls=[
                "Matched meal schedules, sensor noise seeds, and missing-data contract across all three simulator backends."
            ],
            patients=cross_simulator_cohort,
            scenarios=scenarios[:2],
            sensor_conditions=conditions,
            algorithms=default_algos,
            evaluation=EvaluationSpec(prediction_horizon_minutes=30),
        )

    raise ValueError(f"Unknown research template_id '{template_id}'.")


def approve_research_plan(plan: ResearchPlan, *, reviewer: str = "principal_investigator") -> ResearchPlan:
    """Explicitly approve and cryptographically lock a proposed ``ResearchPlan``."""
    if not reviewer or not reviewer.strip():
        raise ValueError("reviewer must be a non-empty identifier")
    digest = sha256_json(plan.design_Payload_without_lock())
    return plan.model_copy(
        update={
            "status": "approved",
            "approved_by": reviewer.strip(),
            "approved_utc": datetime.now(timezone.utc).isoformat(),
            "plan_lock_sha256": digest,
        }
    )


def save_research_plan(plan: ResearchPlan, path: Union[str, Path]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return dest


def load_research_plan(path: Union[str, Path]) -> ResearchPlan:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ResearchPlan.model_validate(raw)
