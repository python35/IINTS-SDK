"""IINTS Research Lab — Hypothesis-Driven Reproducible Experiment Environment."""

from iints.research.designer import (
    HypothesisSpec,
    PlanNotApprovedError,
    ResearchPlan,
    StatisticalProtocolSpec,
    approve_research_plan,
    list_research_templates,
    load_research_plan,
    propose_research_plan,
    save_research_plan,
)
from iints.research.reports import (
    generate_verifiable_report_bundle,
    plot_study_dashboard,
    render_scientific_report_markdown,
)
from iints.research.runner import execute_research_plan, load_study_cell_metrics
from iints.research.scenarios import (
    generate_boundary_scenarios,
    generate_dropout_conditions,
    generate_virtual_cohort,
)
from iints.research.statistics import (
    analyze_research_results,
    holm_bonferroni,
    paired_bootstrap_ci,
)

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
    "generate_virtual_cohort",
    "generate_boundary_scenarios",
    "generate_dropout_conditions",
    "execute_research_plan",
    "load_study_cell_metrics",
    "holm_bonferroni",
    "paired_bootstrap_ci",
    "analyze_research_results",
    "render_scientific_report_markdown",
    "plot_study_dashboard",
    "generate_verifiable_report_bundle",
]
