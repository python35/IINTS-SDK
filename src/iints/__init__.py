# src/iints/__init__.py

import pandas as pd # Required for type hints like pd.DataFrame
from typing import Optional

__version__ = "0.1.2"

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
from .core.algorithms.standard_pump_algo import StandardPumpAlgorithm

# Data Handling
from .data.ingestor import DataIngestor
from .analysis.metrics import generate_benchmark_metrics # Added for benchmark

# Placeholder for Reporting/Analysis
# This will be further developed in a dedicated module (e.g., iints.analysis.reporting)
def generate_report(simulation_results: 'pd.DataFrame', output_path: Optional[str] = None) -> None:
    """
    Placeholder function to generate a simulation report.
    (Detailed implementation to follow)
    """
    print("Generating simulation report... (placeholder)")
    if output_path:
        print(f"Report would be saved to: {output_path}")
    # In a real implementation, this would involve plotting, summarizing, etc.
    pass

# You can also define __all__ to explicitly control what gets imported with `from iints import *`
__all__ = [
    # API
    "InsulinAlgorithm", "AlgorithmInput", "AlgorithmResult", "AlgorithmMetadata", "WhyLogEntry",
    # Core
    "Simulator", "StressEvent", "PatientModel", "DeviceManager",
    "SafetySupervisor",
    "StandardPumpAlgorithm",
    # Data
    "DataIngestor",
    # Analysis Metrics
    "generate_benchmark_metrics",
    # Reporting
    "generate_report",
]
