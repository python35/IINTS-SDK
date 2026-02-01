# IINTS-AF SDK: Intelligent Insulin Titration System for Artificial Pancreas

![Build Status](https://github.com/python35/IINTS-SDK/actions/workflows/python-package.yml/badge.svg)

## Overview

The **IINTS-AF SDK** (Intelligent Insulin Titration System for Artificial Pancreas) is a comprehensive Python Software Development Kit designed for researchers and developers. It provides a robust framework for creating, simulating, benchmarking, and evaluating advanced insulin delivery algorithms for artificial pancreas systems.

IINTS-AF bridges the gap between academic AI research and safe, clinical-grade insulin delivery simulation.

Whether you're developing novel AI-driven insulin controllers, comparing existing algorithms, or conducting in-depth patient simulations, the IINTS-AF SDK offers the tools and infrastructure to accelerate your research.

## Features

*   **Algorithm Development**: Easily create and integrate custom insulin delivery algorithms by extending the `InsulinAlgorithm` base class.
*   **Patient Simulation**: Simulate realistic patient responses using various patient models and configurable scenarios.
*   **Benchmarking & Evaluation**: Compare algorithm performance against baselines and across diverse patient populations using built-in clinical metrics and analysis tools.
*   **Safety Supervisor**: Integrate safety constraints and monitoring to ensure algorithms operate within physiological limits.
*   **Data Handling**: Tools for ingesting, parsing, and quality-checking diverse patient data formats.
*   **Command-Line Interface (CLI)**: Streamlined CLI for running simulations, benchmarking, and algorithm scaffolding.
*   **Comprehensive Documentation**: Auto-generated API documentation detailing all SDK components.
*   **Continuous Integration**: GitHub Actions workflow to ensure code quality, build, and test on every change.

## Installation

### System Requirements

*   Python 3.8 or higher
*   `pip` package manager

### From PyPI (Recommended for Users)

Once the SDK is officially released on PyPI, you can install it directly:

```bash
pip install iints-af # (Placeholder: actual name might vary)
```

### From Source (For Developers)

To install the SDK directly from the source code for development purposes:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/IINTS.git # Replace with actual repo URL
    cd IINTS
    ```
2.  **Install in editable mode:**
    ```bash
    pip install -e .
    ```

This will install the `iints` package and all its necessary dependencies.

## Quick Start: Using the CLI

For the quickest start, see a minimal simulation example:

```python
import iints
# Start direct een simulatie met 2 regels
sim = iints.Simulator(algo="PID", patient="ohio_559")
sim.run()
```

The `iints` command-line tool is your primary interface for interacting with the SDK.

### 1. Create a New Algorithm

Start developing your own insulin delivery algorithm using the provided template:

```bash
iints new-algo MyAIAlgorithm.py
```

This will create a new file `MyAIAlgorithm.py` in your current directory, pre-filled with the necessary structure to get started. Implement your logic in the `predict_insulin` method.

### 2. Run a Simulation

Simulate your algorithm with a predefined patient and scenario:

```bash
# Example (assuming example_scenario.json and a patient config exist)
iints run --algorithm MyAIAlgorithm --patient-config data_packs/public/ohio_t1dm/patient_559/patient_559_config.yaml --scenario scenarios/example_scenario.json
```

Replace `MyAIAlgorithm` with the name of your algorithm's class (if you changed the filename, point to the file path). Adjust patient and scenario paths as needed.

### 3. Run Benchmarking

Compare your algorithm's performance against others:

```bash
iints benchmark --algorithms MyAIAlgorithm StandardPumpAlgorithm --patient-set public --metrics all
```

This command will run simulations across a defined patient set and output performance metrics.

## Key Concepts

*   **`InsulinAlgorithm`**: The base class for all insulin control algorithms. Your custom algorithms must inherit from this.
*   **`Simulator`**: The core engine that runs patient simulations over time, interacting with your algorithm and a patient model.
*   **`DataIngestor`**: Handles the loading and standardization of various patient data types.
*   **`AlgorithmInput`**: Data structure passed to your algorithm at each time step, containing current glucose, IOB, carb intake, etc.

## Documentation

*   **Comprehensive Guide**: For a deeper dive into the SDK's architecture, APIs, and development workflows, refer to the `SDK_COMPREHENSIVE_GUIDE.md` file in this repository.
*   **API Reference (HTML)**: Detailed, auto-generated API documentation from code docstrings is available.
    *   To view it, navigate to `docssphinx/_build/html/index.html` after building it locally.
    *   To build the documentation locally, run `cd docssphinx && make html` from the project root.

## Development & Contribution

The project utilizes GitHub Actions for Continuous Integration. Every push and pull request triggers automated builds, tests, linting (Flake8), and type checking (MyPy) across multiple Python versions to ensure code quality and stability.

*   **Versioning**: Semantic versioning is used. Update the `version` in `pyproject.toml` for new releases.
*   **Changelog**: Refer to `CHANGELOG.md` for a history of changes between versions.

Contributions are welcome! Please refer to the `CONTRIBUTING.md` (if available) for guidelines.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
