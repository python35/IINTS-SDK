from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from iints.scenarios.study_pack import build_eucys_study_pack

DEFAULT_PROFILE_SET = "clinic_safe_core"
DEFAULT_BASELINE_ALGORITHMS = [
    "Clinical Baseline",
    "PID Controller",
    "Standard Pump",
    "Correction Bolus",
]

DEFAULT_RESEARCH_QUESTION = (
    "Can a transparent, safety-first, reproducible evaluation workflow compare insulin "
    "control algorithms fairly across patient profiles, disturbance scenarios, and study arms?"
)

DEFAULT_HYPOTHESES = [
    {
        "id": "H1",
        "statement": "Certified data produces more reliable closed-loop evaluation summaries than uncertified or deliberately corrupted data.",
    },
    {
        "id": "H2",
        "statement": "The safety supervisor reduces severe hypo exposure and early terminations without an unacceptable loss in time in range.",
    },
    {
        "id": "H3",
        "statement": "Controller ranking should remain reproducible across the same seeds, patient profiles, and scenario families.",
    },
]

DEFAULT_METRICS = [
    "tir_70_180",
    "tir_below_70",
    "tir_below_54",
    "tir_above_180",
    "tir_above_250",
    "mean_glucose",
    "cv",
    "gmi",
    "supervisor_interventions",
    "terminated_early",
    "predictor_uncertainty_mean",
    "predictor_uncertainty_p95",
    "predictor_uncertainty_max",
]

_PROFILE_SET_DEFINITIONS: dict[str, list[dict[str, str]]] = {
    DEFAULT_PROFILE_SET: [
        {
            "profile_id": "clinic_safe_baseline",
            "label": "Clinic Safe Baseline",
            "description": "Reference patient profile with stable day structure.",
        },
        {
            "profile_id": "clinic_safe_stress_meal",
            "label": "Clinic Safe Stress Meal",
            "description": "Higher meal challenge profile for post-prandial stress.",
        },
        {
            "profile_id": "clinic_safe_hypo_prone",
            "label": "Clinic Safe Hypo-Prone",
            "description": "More fragile overnight profile for hypo-risk benchmarking.",
        },
        {
            "profile_id": "clinic_safe_hyper_challenge",
            "label": "Clinic Safe Hyper Challenge",
            "description": "Large-meal profile used to probe correction behavior.",
        },
        {
            "profile_id": "clinic_safe_pizza",
            "label": "Clinic Safe Pizza",
            "description": "Delayed-absorption profile for late post-meal rises.",
        },
        {
            "profile_id": "clinic_safe_midnight",
            "label": "Clinic Safe Midnight",
            "description": "Overnight crash-risk profile after evening exertion.",
        },
    ],
}


@dataclass(frozen=True)
class StudyProfileSpec:
    profile_id: str
    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "description": self.description,
        }


@dataclass(frozen=True)
class StudyAlgorithmSpec:
    algorithm_id: str
    display_name: str
    role: str
    source_type: str
    source_ref: str

    def to_dict(self) -> dict[str, str]:
        return {
            "algorithm_id": self.algorithm_id,
            "display_name": self.display_name,
            "role": self.role,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class StudyArmSpec:
    arm_id: str
    label: str
    data_condition: str
    expected_certification: str
    supervisor_enabled: bool
    corruption_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "label": self.label,
            "data_condition": self.data_condition,
            "expected_certification": self.expected_certification,
            "supervisor_enabled": self.supervisor_enabled,
            "corruption_modes": list(self.corruption_modes),
        }


@dataclass(frozen=True)
class StudyMatrixRow:
    arm_id: str
    condition_group: str
    profile_id: str
    profile_label: str
    algorithm_id: str
    algorithm_label: str
    algorithm_role: str
    algorithm_source_type: str
    scenario_slug: str
    scenario_label: str
    seed: int
    recommended_duration_minutes: int
    supervisor_enabled: bool
    expected_certification: str
    corruption_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "condition_group": self.condition_group,
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "algorithm_id": self.algorithm_id,
            "algorithm_label": self.algorithm_label,
            "algorithm_role": self.algorithm_role,
            "algorithm_source_type": self.algorithm_source_type,
            "scenario_slug": self.scenario_slug,
            "scenario_label": self.scenario_label,
            "seed": self.seed,
            "recommended_duration_minutes": self.recommended_duration_minutes,
            "supervisor_enabled": self.supervisor_enabled,
            "expected_certification": self.expected_certification,
            "corruption_modes": list(self.corruption_modes),
        }


@dataclass(frozen=True)
class StudyDesignPayload:
    preset: str
    title: str
    research_question: str
    hypotheses: list[dict[str, str]]
    profile_set: str
    profiles: list[StudyProfileSpec]
    algorithms: list[StudyAlgorithmSpec]
    scenarios: list[dict[str, Any]]
    study_arms: list[StudyArmSpec]
    metrics: list[str]
    seed_policy: dict[str, Any]
    corruption_plan: list[dict[str, Any]]
    statistics_plan: dict[str, Any]
    external_validation: dict[str, Any]
    reproducibility_checklist: list[str]
    recommended_commands: list[str]
    matrix_rows: list[StudyMatrixRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "title": self.title,
            "research_question": self.research_question,
            "hypotheses": [dict(item) for item in self.hypotheses],
            "profile_set": self.profile_set,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "algorithms": [algorithm.to_dict() for algorithm in self.algorithms],
            "scenarios": [dict(item) for item in self.scenarios],
            "study_arms": [arm.to_dict() for arm in self.study_arms],
            "metrics": list(self.metrics),
            "seed_policy": dict(self.seed_policy),
            "corruption_plan": [dict(item) for item in self.corruption_plan],
            "statistics_plan": dict(self.statistics_plan),
            "external_validation": dict(self.external_validation),
            "reproducibility_checklist": list(self.reproducibility_checklist),
            "recommended_commands": list(self.recommended_commands),
            "matrix_rows": [row.to_dict() for row in self.matrix_rows],
        }


def slugify_study_token(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "item"


def resolve_profile_specs(profile_set: str = DEFAULT_PROFILE_SET) -> list[StudyProfileSpec]:
    try:
        raw_profiles = _PROFILE_SET_DEFINITIONS[profile_set]
    except KeyError as exc:
        available = ", ".join(sorted(_PROFILE_SET_DEFINITIONS))
        raise ValueError(f"Unknown profile_set '{profile_set}'. Available: {available}") from exc
    return [StudyProfileSpec(**item) for item in raw_profiles]


def build_algorithm_registry(
    *,
    candidate_algorithm: str,
    candidate_source_type: str = "path",
    candidate_source_ref: str | None = None,
    include_default_baselines: bool = True,
    extra_algorithms: list[str] | None = None,
) -> list[StudyAlgorithmSpec]:
    registry: list[StudyAlgorithmSpec] = [
        StudyAlgorithmSpec(
            algorithm_id=slugify_study_token(candidate_algorithm),
            display_name=candidate_algorithm,
            role="candidate",
            source_type=candidate_source_type,
            source_ref=candidate_source_ref or candidate_algorithm,
        )
    ]

    if include_default_baselines:
        for name in DEFAULT_BASELINE_ALGORITHMS:
            registry.append(
                StudyAlgorithmSpec(
                    algorithm_id=slugify_study_token(name),
                    display_name=name,
                    role="baseline",
                    source_type="plugin",
                    source_ref=name,
                )
            )

    for name in extra_algorithms or []:
        normalized = name.strip()
        if not normalized:
            continue
        registry.append(
            StudyAlgorithmSpec(
                algorithm_id=slugify_study_token(normalized),
                display_name=normalized,
                role="baseline" if normalized in DEFAULT_BASELINE_ALGORITHMS else "comparison",
                source_type="plugin",
                source_ref=normalized,
            )
        )

    deduped: list[StudyAlgorithmSpec] = []
    seen: set[str] = set()
    for item in registry:
        if item.algorithm_id in seen:
            continue
        deduped.append(item)
        seen.add(item.algorithm_id)
    return deduped


def build_study_design_payload(
    *,
    preset: str = "default",
    title: str = "IINTS Scientific Validation Protocol",
    primary_hypothesis: str | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    candidate_algorithm: str = "your_algorithm",
    include_default_baselines: bool = True,
    extra_algorithms: list[str] | None = None,
    profile_set: str = DEFAULT_PROFILE_SET,
    external_reference_label: str = "CareLink personal workbench metrics",
    candidate_source_type: str = "path",
    candidate_source_ref: str | None = None,
) -> StudyDesignPayload:
    normalized_preset = preset.strip().lower()
    if normalized_preset not in {"default", "eucys"}:
        raise ValueError("preset must be 'default' or 'eucys'")

    pack = build_eucys_study_pack(seeds=seeds)
    scenario_catalog = [dict(item) for item in pack["scenarios"]]
    if scenarios:
        requested = {item.strip() for item in scenarios if item.strip()}
        scenario_catalog = [item for item in scenario_catalog if item["slug"] in requested]
        if not scenario_catalog:
            raise ValueError("No study scenarios matched the requested list")

    study_arms = [StudyArmSpec(**arm) for arm in pack["study_arms"]]
    profiles = resolve_profile_specs(profile_set)
    algorithms = build_algorithm_registry(
        candidate_algorithm=candidate_algorithm,
        candidate_source_type=candidate_source_type,
        candidate_source_ref=candidate_source_ref,
        include_default_baselines=include_default_baselines,
        extra_algorithms=extra_algorithms,
    )
    seed_list = list(seeds or pack["seeds"])
    hypotheses = list(DEFAULT_HYPOTHESES)
    if primary_hypothesis is not None:
        hypotheses[0] = {"id": "H1", "statement": primary_hypothesis}
    elif normalized_preset == "eucys":
        hypotheses[0] = {
            "id": "H1",
            "statement": "Certified data produces more reliable and defendable evaluation evidence than deliberately corrupted uncertified data.",
        }
        if title == "IINTS Scientific Validation Protocol":
            title = "IINTS EUCYS Scientific Validation Protocol"

    corruption_plan = [
        {
            "mode": mode,
            "purpose": {
                "timestamp_shift": "Stress provenance and temporal consistency checks.",
                "missing_block": "Simulate sensor outages or dropped rows.",
                "duplicate_rows": "Simulate export duplication or stitching bugs.",
                "glucose_spikes": "Stress plausibility filters and realism review.",
                "drop_meal_annotations": "Test how missing meal context changes evaluation.",
                "unit_scale_error": "Expose unit-mismatch errors and impossible ranges.",
                "meal_annotation_mismatch": "Probe robustness against meal-label drift.",
                "sensor_error_injection": "Reveal failure handling under implausible sensor excursions.",
            }.get(mode, "Controlled corruption operator for scientific ablation."),
        }
        for mode in sorted({mode for arm in study_arms for mode in arm.corruption_modes} | {"duplicate_rows", "drop_meal_annotations", "unit_scale_error"})
    ]

    matrix_rows: list[StudyMatrixRow] = []
    for arm in study_arms:
        for profile in profiles:
            for algorithm in algorithms:
                for scenario in scenario_catalog:
                    for seed in seed_list:
                        matrix_rows.append(
                            StudyMatrixRow(
                                arm_id=arm.arm_id,
                                condition_group=arm.arm_id,
                                profile_id=profile.profile_id,
                                profile_label=profile.label,
                                algorithm_id=algorithm.algorithm_id,
                                algorithm_label=algorithm.display_name,
                                algorithm_role=algorithm.role,
                                algorithm_source_type=algorithm.source_type,
                                scenario_slug=str(scenario["slug"]),
                                scenario_label=str(scenario["label"]),
                                seed=int(seed),
                                recommended_duration_minutes=int(scenario["recommended_duration_minutes"]),
                                supervisor_enabled=arm.supervisor_enabled,
                                expected_certification=arm.expected_certification,
                                corruption_modes=list(arm.corruption_modes),
                            )
                        )

    recommended_commands = [
        "iints scenarios export-study-pack --preset eucys --output-dir scenarios/eucys_pack",
        f"iints study-protocol --preset {normalized_preset} --profile-set {profile_set} --output-dir results/study_protocol",
        "iints run-study --algo algorithms/example_algorithm.py --preset eucys --output-dir results/study_bundle",
        "iints analyze results/study_bundle/study_clean --output-json results/study_bundle/study_clean/study_summary.json",
        "iints compare-study results/study_bundle/study_clean results/study_bundle/study_corrupted --output-json results/study_bundle/comparisons/clean_vs_corrupted.json",
        "iints poster-study results/study_bundle/study_clean/study_summary.json --output-path results/study_bundle/study_clean/study_poster.png",
    ]
    if normalized_preset != "eucys":
        recommended_commands[1] = f"iints study-protocol --preset default --profile-set {profile_set} --output-dir results/study_protocol"
        recommended_commands[2] = "iints run-study --algo algorithms/example_algorithm.py --preset default --output-dir results/study_bundle"

    statistics_plan = {
        "descriptive": ["mean", "median", "std", "min", "max", "95% confidence interval"],
        "comparative": ["difference in means", "Cohen's d"],
        "failure_analysis": [
            "terminated_early_runs",
            "severe_hypo_runs",
            "supervisor_heavy_runs",
            "worst_tir_runs",
        ],
    }

    return StudyDesignPayload(
        preset=normalized_preset,
        title=title,
        research_question=DEFAULT_RESEARCH_QUESTION,
        hypotheses=hypotheses,
        profile_set=profile_set,
        profiles=profiles,
        algorithms=algorithms,
        scenarios=scenario_catalog,
        study_arms=study_arms,
        metrics=list(DEFAULT_METRICS),
        seed_policy={
            "seeds": seed_list,
            "note": "Reuse the same seeds across all algorithms, profiles, and study arms to keep paired comparisons fair.",
        },
        corruption_plan=corruption_plan,
        statistics_plan=statistics_plan,
        external_validation={
            "reference_label": external_reference_label,
            "compare_metrics": [
                "mean_glucose_mgdl",
                "cv_pct",
                "time_in_range_70_180_pct",
                "time_below_70_pct",
                "time_above_180_pct",
            ],
            "note": "Use imported CareLink-style real data only as a plausibility reference, not as a clinical efficacy claim.",
        },
        reproducibility_checklist=[
            "Export the official study pack.",
            "Write the protocol bundle before running scenarios.",
            "Reuse the same seeds across conditions.",
            "Store certification JSON next to each run.",
            "Keep the study matrix and algorithm registry in the bundle root.",
            "Run analyze, compare-study, and poster-study on the final bundle.",
        ],
        recommended_commands=recommended_commands,
        matrix_rows=matrix_rows,
    )
