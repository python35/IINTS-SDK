# IINTS-AF SDK: Comprehensive Guide

This document provides a comprehensive overview of the IINTS-AF (Intelligent Insulin Titration System for Artificial Pancreas) SDK. It covers installation, project structure, key APIs, command-line interface usage, and development guidelines.

## 1. Introduction to IINTS-AF SDK

The IINTS-AF SDK is designed for researchers and developers to create, simulate, and benchmark insulin delivery algorithms for artificial pancreas systems. It provides a robust framework for patient simulation, data handling, and algorithm evaluation, alongside tools for documentation and continuous integration.

## 2. Installation

### 2.1 System Requirements

*   Python 3.8 or higher.
*   `pip` package manager.

### 2.2 Installing the SDK

To install the IINTS-AF SDK, you typically use `pip`. If you have received a distributable package (a `.whl` file), you can install it locally:

```bash
pip install path/to/iints-0.1.0-py3-none-any.whl
```

If you wish to install from the project source in editable mode (for development):

```bash
pip install -e .
```

This will install the `iints` package and its dependencies.

## 3. Project Structure

The core of the SDK's source code is located within the `src/iints/` directory. This structure ensures that all modules are properly packaged and discoverable when the SDK is installed.

```
src/
└── iints/
    ├── __init__.py             # Package initialization, re-exports key APIs
    ├── analysis/               # Modules for clinical metrics, algorithm analysis (e.g., algorithm_xray, clinical_metrics)
    ├── api/                    # Core API for algorithms (e.g., base_algorithm.py for InsulinAlgorithm)
    ├── cli/                    # Command-Line Interface definitions (e.g., cli.py for 'iints' command)
    ├── core/                   # Core simulation and patient management logic
    │   ├── algorithms/         # Implementations of various insulin algorithms
    │   ├── patient/            # Patient models and factory for creating patient instances
    │   ├── safety/             # Safety supervisor logic
    │   ├── simulation/         # Simulation setup and scenario parsing
    │   └── simulator.py        # The main simulator engine
    ├── data/                   # Data handling, ingestors, quality checkers, universal parsers
    ├── emulation/              # Modules for emulating existing commercial pumps
    ├── learning/               # Modules related to autonomous optimization and learning systems
    └── visualization/          # Tools for data visualization (e.g., cockpit, uncertainty_cloud)
```

## 4. Key API Components

The SDK provides several key classes and functions for interaction. The most important ones are often re-exported directly in `iints/__init__.py` for easy access.

### `iints.api.base_algorithm.InsulinAlgorithm`

This is the abstract base class for all insulin delivery algorithms. Any custom algorithm you develop must inherit from this class and implement its abstract methods.

**Core Methods:**

*   `get_algorithm_metadata() -> AlgorithmMetadata`: Provides descriptive metadata for your algorithm.
*   `predict_insulin(data: AlgorithmInput) -> Dict[str, float]`: The main method where your algorithm calculates and returns insulin doses based on current patient data.
*   `reset()`: Resets the algorithm's internal state for a new simulation.

### `iints.core.simulator.Simulator`

The central class for running simulations. It orchestrates the patient model, algorithm, and safety supervisor over time.

**Core Methods:**

*   `run()`: Executes a single simulation run.
*   `run_batch()`: Executes multiple simulations, often used for benchmarking.
*   `add_stress_event()`: Adds predefined stress events (e.g., missed meal, exercise) to a simulation.

### `iints.data.ingestor.DataIngestor`

Handles the loading and processing of various patient data formats.

### `iints.analysis.metrics.generate_benchmark_metrics`

A function to compute various clinical and performance metrics from simulation results.

## 5. Command-Line Interface (CLI)

The SDK provides a `iints` command-line tool for common tasks.

```bash
iints --help
```

**Available Commands:**

*   `new-algo`: Creates a new algorithm template file (`template_algorithm.py`) based on the `InsulinAlgorithm` base class. This is your starting point for custom algorithms.
    ```bash
    iints new-algo MyCustomAlgorithm.py
    ```
*   `run`: Runs an IINTS-AF simulation using a specified algorithm and patient configuration.
    ```bash
    iints run --algorithm MyCustomAlgorithm --patient-config path/to/patient_config.yaml --scenario path/to/scenario.json
    ```
*   `benchmark`: Runs a series of simulations to benchmark an AI algorithm against a standard pump across multiple patient configurations and scenarios.
*   `docs`: (Future functionality or internal-only) Potentially used for documentation tasks.

## 6. Creating Custom Algorithms

1.  **Generate Template**: Use `iints new-algo YourAlgorithmName.py` to create a template.
2.  **Implement Logic**: Fill in your insulin delivery logic within the `predict_insulin` method of your new algorithm class, inheriting from `InsulinAlgorithm`.
3.  **Run**: Use `iints run --algorithm YourAlgorithmName ...` to test your algorithm in simulations.

## 7. Data Formats

The SDK expects patient data in a standardized format, often managed through `iints.data.ingestor.DataIngestor` and `iints.data.universal_parser.UniversalParser`. Key data points typically include timestamps, glucose readings, insulin doses, and carbohydrate intake. Specific details can be found in `data_packs/DATA_SCHEMA.md`.

## 8. Development Workflow

### 8.1 Versioning

The SDK uses semantic versioning. The current version is defined in `pyproject.toml`.

*   To update the version for a new release, edit the `version` field in `pyproject.toml` (e.g., from `0.1.0` to `0.1.1`).
*   It is good practice to use `git tag vX.Y.Z` to mark releases in your version control history.

### 8.2 Continuous Integration (CI) with GitHub Actions

A GitHub Actions workflow (`.github/workflows/python-package.yml`) has been set up to automate the build and testing process.

*   **Triggers**: The workflow runs automatically on `push` to the `main` branch and on every `pull_request`.
*   **Steps**: It checks out the code, sets up multiple Python versions, installs dependencies, builds the SDK, runs basic installation tests, performs linting with Flake8, and type checking with MyPy.
*   **Artifacts**: Built `.whl` and `.tar.gz` files are uploaded as artifacts, which you can download from the GitHub Actions run summary.

This ensures that every change to the codebase is automatically validated, preventing regressions and maintaining code quality.

### 8.3 Change Log

A `CHANGELOG.md` file has been created in the project root. It is recommended to update this file with a concise summary of changes for each new version, helping users understand what's new or fixed in each release.

## 9. API Documentation

Comprehensive API documentation, generated using Sphinx, is available in HTML format.

*   **Location**: `docssphinx/_build/html/index.html`
*   **How to Build**: From the `docssphinx/` directory, run `make html`. Ensure your Python environment (or the `PYTHONPATH` if building manually) is correctly configured to include the project's `src` directory.

This documentation details all classes, methods, and functions within the `iints` package, extracted directly from their docstrings.
