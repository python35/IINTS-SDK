from .clinical_metrics import ClinicalMetricsCalculator, ClinicalMetricsResult
from .baseline import compute_metrics, run_baseline_comparison, write_baseline_comparison
from .study_protocol import (
    build_study_protocol_payload,
    render_study_protocol_markdown,
    write_study_protocol_bundle,
)
from .study_analysis import (
    analyze_run_directory,
    analyze_study_directory,
    compare_studies,
    load_study_summary,
    quality_badges_for_metrics,
    StudyComparison,
    StudyRunSummary,
    StudySummary,
)


def _missing_reports_dependency(feature: str, exc: Exception) -> None:
    raise ImportError(
        f"{feature} requires the optional reporting stack. Install "
        f"'iints-sdk-python35[reports]' or 'iints-sdk-python35[full]'."
    ) from exc


try:
    from .booth_demo import build_booth_demo
except Exception as exc:  # pragma: no cover - optional reports stack
    _build_booth_demo_exc = exc

    def build_booth_demo(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("build_booth_demo()", _build_booth_demo_exc)


try:
    from .carelink_workbench import build_carelink_workbench
except Exception as exc:  # pragma: no cover - optional reports stack
    _build_carelink_workbench_exc = exc

    def build_carelink_workbench(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("build_carelink_workbench()", _build_carelink_workbench_exc)


try:
    from .poster import generate_results_poster
except Exception as exc:  # pragma: no cover - optional reports stack
    _generate_results_poster_exc = exc

    def generate_results_poster(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("generate_results_poster()", _generate_results_poster_exc)


try:
    from .reporting import ClinicalReportGenerator
except Exception as exc:  # pragma: no cover - optional reports stack
    _clinical_report_generator_exc = exc

    class ClinicalReportGenerator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _missing_reports_dependency("ClinicalReportGenerator", _clinical_report_generator_exc)


try:
    from .study_poster import generate_study_poster
except Exception as exc:  # pragma: no cover - optional reports stack
    _generate_study_poster_exc = exc

    def generate_study_poster(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("generate_study_poster()", _generate_study_poster_exc)


__all__ = [
    "analyze_run_directory",
    "analyze_study_directory",
    "build_booth_demo",
    "build_carelink_workbench",
    "ClinicalMetricsCalculator",
    "ClinicalMetricsResult",
    "ClinicalReportGenerator",
    "compute_metrics",
    "compare_studies",
    "generate_results_poster",
    "generate_study_poster",
    "build_study_protocol_payload",
    "render_study_protocol_markdown",
    "write_study_protocol_bundle",
    "load_study_summary",
    "quality_badges_for_metrics",
    "run_baseline_comparison",
    "StudyComparison",
    "StudyRunSummary",
    "StudySummary",
    "write_baseline_comparison",
]
