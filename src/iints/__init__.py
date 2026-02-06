# src/iints/__init__.py

import pandas as pd # Required for type hints like pd.DataFrame
from typing import Optional

__version__ = "0.1.3"

# API Components for Algorithm Development
from .api.base_algorithm import (
    InsulinAlgorithm,
    AlgorithmInput,
    AlgorithmResult,
    AlgorithmMetadata,
    WhyLogEntry,
)

# Core Simulation Components
from .core.simulator import Simulator, StressEvent
from .core.patient.models import PatientModel
from .core.device_manager import DeviceManager
from .core.safety import SafetySupervisor
from .core.devices.models import SensorModel, PumpModel
from .core.algorithms.standard_pump_algo import StandardPumpAlgorithm

# Data Handling
from .data.ingestor import DataIngestor
from .analysis.metrics import generate_benchmark_metrics # Added for benchmark
from .analysis.reporting import ClinicalReportGenerator
from .highlevel import run_simulation

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

# You can also define __all__ to explicitly control what gets imported with `from iints import *`
__all__ = [
    # API
    "InsulinAlgorithm", "AlgorithmInput", "AlgorithmResult", "AlgorithmMetadata", "WhyLogEntry",
    # Core
    "Simulator", "StressEvent", "PatientModel", "DeviceManager",
    "SafetySupervisor",
    "SensorModel",
    "PumpModel",
    "StandardPumpAlgorithm",
    # Data
    "DataIngestor",
    # Analysis Metrics
    "generate_benchmark_metrics",
    "ClinicalReportGenerator",
    # Reporting
    "generate_report",
    # High-level API
    "run_simulation",
]
