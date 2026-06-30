# src/iints/__init__.py

import pandas as pd # Required for type hints like pd.DataFrame
from typing import Any, Optional

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("iints-sdk-python35")
except PackageNotFoundError:  # pragma: no cover - source tree fallback
    __version__ = "1.5.20"

# Note to developers: this SDK is currently maintained by a single author.
# Please report bugs via GitHub issues and feel free to contribute fixes via PRs.


def _missing_reports_dependency(feature: str, exc: Exception) -> None:
    raise ImportError(
        f"{feature} requires the optional reporting stack. Install "
        f"'iints-sdk-python35[reports]' or 'iints-sdk-python35[full]'."
    ) from exc

# API Components for Algorithm Development
from .api.base_algorithm import (
    InsulinAlgorithm,
    AlgorithmInput,
    AlgorithmResult,
    AlgorithmMetadata,
    WhyLogEntry,
)

# Core Simulation Components
from .core.simulator import Simulator, StressEvent, SimulationLimitError
from .core.patient.models import PatientModel
from .core.patient.patient_factory import PatientFactory
from .core.patient.profile import PatientProfile
try:
    from .core.device_manager import DeviceManager
except Exception:  # pragma: no cover - fallback if torch/device manager import fails
    class DeviceManager:  # type: ignore
        def __init__(self):
            self._device = "cpu"

        def get_device(self):
            return self._device
from .core.safety import SafetyConfig, SafetySupervisor
from .core.devices.models import SENSOR_PROFILES, SensorModel, PumpModel, create_sensor_model
from .core.algorithms.standard_pump_algo import StandardPumpAlgorithm
from .core.algorithms.mock_algorithms import (
    ConstantDoseAlgorithm,
    RandomDoseAlgorithm,
    RunawayAIAlgorithm,
    StackingAIAlgorithm,
)

# Data Handling
from .data.ingestor import DataIngestor
from .data.importer import (
    ImportResult,
    export_demo_csv,
    export_standard_csv,
    guess_column_mapping,
    import_carelink_csv,
    import_carelink_timeline,
    import_cgm_csv,
    import_cgm_dataframe,
    load_carelink_event_log,
    load_demo_dataframe,
    scenario_from_csv,
    scenario_from_dataframe,
    summarize_carelink_csv,
)
from .data.nightscout import NightscoutConfig, import_nightscout
from .data.tidepool import TidepoolClient, TidepoolConfig, import_tidepool, load_openapi_spec
from .data.guardians import mdmp_gate, MDMPGateError
from .data.synthetic_mirror import generate_synthetic_mirror, SyntheticMirrorArtifact
from .data.study_corruption import AVAILABLE_STUDY_CORRUPTIONS, apply_study_corruptions, write_corrupted_study_csv
from .analysis.metrics import generate_benchmark_metrics # Added for benchmark
from .analysis.study_protocol import build_study_protocol_payload, render_study_protocol_markdown, write_study_protocol_bundle
from .analysis.edge_efficiency import EnergyEstimate, estimate_energy_per_decision
from .ai import AIResponse, IINTSAssistant, MDMPGuard
from .live_patient import (
    create_edge_bundle,
    export_edge_setup,
    LivePatientDaemon,
    PatientRuntimeConfig,
    create_patient_app,
    export_uno_q_bridge,
    get_runtime_scenario_profile,
    list_runtime_scenario_profiles,
    run_edge_benchmark,
    summarize_edge_workspace,
    write_edge_update_script,
)
from .scenarios import ScenarioGeneratorConfig, generate_random_scenario


def run_simulation(*args: Any, **kwargs: Any) -> Any:
    from .highlevel import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)


def run_full(*args: Any, **kwargs: Any) -> Any:
    from .highlevel import run_full as _run_full

    return _run_full(*args, **kwargs)


def run_population(*args: Any, **kwargs: Any) -> Any:
    from .highlevel import run_population as _run_population

    return _run_population(*args, **kwargs)

try:
    from .analysis.booth_demo import build_booth_demo
except Exception as exc:  # pragma: no cover - optional reports stack
    _build_booth_demo_exc = exc

    def build_booth_demo(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("build_booth_demo()", _build_booth_demo_exc)

try:
    from .analysis.carelink_workbench import build_carelink_workbench
except Exception as exc:  # pragma: no cover - optional reports stack
    _build_carelink_workbench_exc = exc

    def build_carelink_workbench(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("build_carelink_workbench()", _build_carelink_workbench_exc)

try:
    from .analysis.poster import generate_results_poster
except Exception as exc:  # pragma: no cover - optional reports stack
    _generate_results_poster_exc = exc

    def generate_results_poster(*args, **kwargs):  # type: ignore[misc,no-redef]
        _missing_reports_dependency("generate_results_poster()", _generate_results_poster_exc)

try:
    from .analysis.reporting import ClinicalReportGenerator
except Exception as exc:  # pragma: no cover - optional reports stack
    _clinical_report_generator_exc = exc

    class ClinicalReportGenerator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _missing_reports_dependency("ClinicalReportGenerator", _clinical_report_generator_exc)

# Population testing
from .population import (
    PopulationGenerator,
    PopulationConfig,
    ParameterDistribution,
    PopulationRunner,
    PopulationResult,
    PatientResult,
)

# Bergman Minimal Model (ODE-based patient)
try:
    from .core.patient.bergman_model import BergmanPatientModel
except ImportError:  # pragma: no cover - scipy may not be installed
    BergmanPatientModel = None  # type: ignore[assignment,misc]

# Placeholder for Reporting/Analysis
# This will be further developed in a dedicated module (e.g., iints.analysis.reporting)
def generate_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None, safety_report: Optional[dict] = None) -> Optional[str]:
    """
    Generate a clinical PDF report from simulation results.
    """
    if output_path is None:
        return None
    generator = ClinicalReportGenerator()
    return generator.generate_pdf(simulation_results, safety_report or {}, output_path)

def generate_quickstart_report(
    simulation_results: 'pd.DataFrame',
    output_path: Optional[str] = None,
    safety_report: Optional[dict] = None,
) -> Optional[str]:
    """
    Generate a concise Quickstart PDF report from simulation results.
    """
    if output_path is None:
        return None
    generator = ClinicalReportGenerator()
    return generator.generate_pdf(
        simulation_results,
        safety_report or {},
        output_path,
        title="IINTS-AF Quickstart Report",
    )

def generate_demo_report(
    simulation_results: 'pd.DataFrame',
    output_path: Optional[str] = None,
    safety_report: Optional[dict] = None,
) -> Optional[str]:
    """
    Generate a demo-friendly PDF with big visuals (Maker Faire style).
    """
    if output_path is None:
        return None
    generator = ClinicalReportGenerator()
    return generator.generate_demo_pdf(
        simulation_results,
        safety_report or {},
        output_path,
        title="IINTS-AF Demo Report",
    )

def generate_agp_report(
    simulation_results: 'pd.DataFrame',
    output_path: Optional[str] = None,
    safety_report: Optional[dict] = None,
    subject_name: str = "Research simulation",
    summary_json_path: Optional[str] = None,
) -> Optional[str]:
    """
    Generate an AGP-style research PDF report from dense CGM/simulation data.
    """
    if output_path is None:
        return None
    generator = ClinicalReportGenerator()
    return generator.generate_agp_pdf(
        simulation_results,
        output_path,
        subject_name=subject_name,
        safety_report=safety_report or {},
        summary_json_path=summary_json_path,
    )

def generate_agp_assets(
    simulation_results: 'pd.DataFrame',
    output_dir: Optional[str] = None,
    subject_name: str = "Research simulation",
    summary_json_path: Optional[str] = None,
    export_svg: bool = True,
) -> Optional[dict]:
    """
    Export AGP-style PNG/SVG assets and summary JSON from dense CGM/simulation data.
    """
    if output_dir is None:
        return None
    generator = ClinicalReportGenerator()
    return generator.export_agp_assets(
        simulation_results,
        output_dir,
        subject_name=subject_name,
        summary_json_path=summary_json_path,
        export_svg=export_svg,
    )

# You can also define __all__ to explicitly control what gets imported with `from iints import *`
__all__ = [
    # API
    "InsulinAlgorithm", "AlgorithmInput", "AlgorithmResult", "AlgorithmMetadata", "WhyLogEntry",
    # Core
    "Simulator", "StressEvent", "PatientModel", "DeviceManager",
    "PatientFactory",
    "PatientProfile",
    "SimulationLimitError",
    "SafetySupervisor",
    "SafetyConfig",
    "SensorModel",
    "PumpModel",
    "SENSOR_PROFILES",
    "create_sensor_model",
    "StandardPumpAlgorithm",
    "ConstantDoseAlgorithm",
    "RandomDoseAlgorithm",
    "RunawayAIAlgorithm",
    "StackingAIAlgorithm",
    # Data
    "DataIngestor",
    "ImportResult",
    "export_demo_csv",
    "export_standard_csv",
    "guess_column_mapping",
    "import_carelink_csv",
    "import_carelink_timeline",
    "import_cgm_csv",
    "import_cgm_dataframe",
    "load_carelink_event_log",
    "load_demo_dataframe",
    "scenario_from_csv",
    "scenario_from_dataframe",
    "summarize_carelink_csv",
    "NightscoutConfig",
    "import_nightscout",
    "TidepoolClient",
    "TidepoolConfig",
    "import_tidepool",
    "load_openapi_spec",
    "mdmp_gate",
    "MDMPGateError",
    "generate_synthetic_mirror",
    "SyntheticMirrorArtifact",
    "AVAILABLE_STUDY_CORRUPTIONS",
    "apply_study_corruptions",
    "write_corrupted_study_csv",
    # Analysis Metrics
    "generate_benchmark_metrics",
    "build_booth_demo",
    "build_carelink_workbench",
    "build_study_protocol_payload",
    "render_study_protocol_markdown",
    "write_study_protocol_bundle",
    "ClinicalReportGenerator",
    "EnergyEstimate",
    "estimate_energy_per_decision",
    "AIResponse",
    "IINTSAssistant",
    "MDMPGuard",
    "create_edge_bundle",
    "export_edge_setup",
    "LivePatientDaemon",
    "PatientRuntimeConfig",
    "create_patient_app",
    "export_uno_q_bridge",
    "get_runtime_scenario_profile",
    "list_runtime_scenario_profiles",
    "run_edge_benchmark",
    "summarize_edge_workspace",
    "write_edge_update_script",
    # Reporting
    "generate_report",
    "generate_quickstart_report",
    "generate_demo_report",
    "generate_agp_report",
    "generate_agp_assets",
    "generate_results_poster",
    # High-level API
    "run_simulation",
    "run_full",
    "run_population",
    "ScenarioGeneratorConfig",
    "generate_random_scenario",
    # Population testing
    "PopulationGenerator",
    "PopulationConfig",
    "ParameterDistribution",
    "PopulationRunner",
    "PopulationResult",
    "PatientResult",
    # Bergman model
    "BergmanPatientModel",
]
