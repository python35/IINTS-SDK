# src/iints/__init__.py

import pandas as pd # Required for type hints like pd.DataFrame
from typing import Optional

__version__ = "0.0.18"

# Note to developers: this SDK is currently maintained by a single author.
# Please report bugs via GitHub issues and feel free to contribute fixes via PRs.

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
from .core.devices.models import SensorModel, PumpModel
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
    import_cgm_csv,
    import_cgm_dataframe,
    load_demo_dataframe,
    scenario_from_csv,
    scenario_from_dataframe,
)
from .data.nightscout import NightscoutConfig, import_nightscout
from .data.tidepool import TidepoolClient, load_openapi_spec
from .analysis.metrics import generate_benchmark_metrics # Added for benchmark
from .analysis.reporting import ClinicalReportGenerator
from .analysis.edge_efficiency import EnergyEstimate, estimate_energy_per_decision
from .highlevel import run_simulation, run_full, run_population
from .scenarios import ScenarioGeneratorConfig, generate_random_scenario

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

# You can also define __all__ to explicitly control what gets imported with `from iints import *`
__all__ = [
    # API
    "InsulinAlgorithm", "AlgorithmInput", "AlgorithmResult", "AlgorithmMetadata", "WhyLogEntry",
    # Core
    "Simulator", "StressEvent", "PatientModel", "DeviceManager",
    "PatientProfile",
    "SimulationLimitError",
    "SafetySupervisor",
    "SafetyConfig",
    "SensorModel",
    "PumpModel",
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
    "import_cgm_csv",
    "import_cgm_dataframe",
    "load_demo_dataframe",
    "scenario_from_csv",
    "scenario_from_dataframe",
    "NightscoutConfig",
    "import_nightscout",
    "TidepoolClient",
    "load_openapi_spec",
    # Analysis Metrics
    "generate_benchmark_metrics",
    "ClinicalReportGenerator",
    "EnergyEstimate",
    "estimate_energy_per_decision",
    # Reporting
    "generate_report",
    "generate_quickstart_report",
    "generate_demo_report",
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
