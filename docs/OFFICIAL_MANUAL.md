# IINTS-AF SDK: The Official Manual (v0.1.0)

**Intelligent Insulin Titration System - Autonomous Framework**

The IINTS-AF SDK is a professional research framework for designing, simulating, and validating autonomous insulin delivery algorithms. It provides a standardized "Digital Twin" environment to test AI models against virtual patients before any clinical deployment.

---

## 1. Introduction

### The Problem
Modern AI (like LSTM networks) is excellent at predicting glucose trends but lacks the inherent safety guarantees required for medical devices. A "black box" cannot be trusted with lethal medication like insulin.

### The Solution: Dual-Guard Architecture
This SDK implements the **Dual-Guard Architecture**, a safety-first design pattern that separates "intelligence" from "safety":

1.  **The Neural Layer (The Brain)**: Your custom algorithm (PID, LSTM, Fuzzy Logic) that optimizes for Time In Range (TIR). It suggests an insulin dose.
2.  **The Supervisor Layer (The Cage)**: A deterministic, mathematically proven safety guard that validates every suggestion against physiological limits (e.g., Insulin on Board, max bolus).

### What this SDK is for
*   **Research**: Benchmarking new control algorithms against standard baselines.
*   **Validation**: Stress-testing algorithms with "impossible" scenarios (missed meals, sensor errors).
*   **Education**: Teaching control theory and biological modeling.

---

## 2. Installation

### Prerequisites
*   Python 3.8 or higher
*   pip (Python Package Manager)

### Installation Methods

#### Option A: User Installation (Recommended)
If you just want to use the SDK to build algorithms:
```bash
pip install iints-sdk-python35
```
*(Note: Replace with actual PyPI package name if different)*

#### Option B: Developer Installation
If you want to modify the SDK core or contribute:
```bash
git clone https://github.com/python35/IINTS-SDK.git
cd IINTS-SDK
pip install -e .
```

### Verification
Verify the installation by checking the CLI version:
```bash
iints --version
# Should output: iints-sdk-python35 version: 0.1.0
```

---

## 3. Getting Started

The SDK uses a workspace-based approach. You don't run scripts inside the SDK folder; you create your own research project.

### Step 1: Initialize a Project
Create a new directory for your research:

```bash
iints init --project-name my_diabetes_research
cd my_diabetes_research
```

This creates the following structure:
*   `algorithms/`: Store your custom python algorithms here.
*   `scenarios/`: JSON files defining stress tests (meals, exercise).
*   `data/`: Patient configuration files.
*   `results/`: Simulation outputs (CSV, logs).
*   `README.md`: Project documentation.

### Step 2: Run the Example
The project comes with a working example. Run it to see the simulator in action:

```bash
iints run --algo algorithms/example_algorithm.py --scenario-path scenarios/example_scenario.json
```

**What happens?**
1.  The SDK loads the `ExampleAlgorithm`.
2.  It loads the `Standard Meal Challenge` scenario.
3.  It simulates 12 hours (default) of virtual patient life.
4.  It saves the results to `results/` and prints a summary.

---

## 4. Core Concepts

### 4.1 The Virtual Patient
The SDK uses differential equations to model glucose metabolism.
*   **Carb Absorption**: How fast food turns into blood glucose.
*   **Insulin Kinetics**: How fast insulin lowers blood glucose.
*   **Liver Production**: Background glucose release.

You can configure patients via YAML files in `src/iints/data/virtual_patients/` (or your local `data/` folder).

### 4.2 The Safety Supervisor
This is the "Digital Cage". It runs *after* your algorithm but *before* the patient.
It checks:
*   **Hypoglycemia Prediction**: Will this dose kill the patient?
*   **Insulin Stacking**: Is there already too much active insulin (IOB)?
*   **Hard Limits**: Is the dose > 5 Units? (Configurable).

If the Supervisor detects danger, it **overrides** your algorithm and logs a `SafetyViolation`.

### 4.3 Stress Events
Algorithms fail at the edges. Stress events simulate these edges:
*   `meal`: Standard food intake.
*   `missed_meal`: Patient took insulin but didn't eat (dangerous!).
*   `sensor_error`: CGM reports false values.
*   `exercise`: Increases insulin sensitivity rapidly.

---

## 5. Developing Algorithms

This is the main task for researchers. You write a Python class that decides how much insulin to give.

### The `InsulinAlgorithm` Class
All algorithms must inherit from `iints.InsulinAlgorithm`.

```python
from iints import InsulinAlgorithm, AlgorithmInput, AlgorithmMetadata
from typing import Dict, Any

class MySmartAlgo(InsulinAlgorithm):
    
    def get_algorithm_metadata(self):
        return AlgorithmMetadata(
            name="MySmartAlgo",
            author="Dr. Doe",
            description="A PID controller with AI enhancements",
            algorithm_type="hybrid"
        )

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        """
        Called every 5 minutes.
        
        Args:
            data.current_glucose (float): CGM reading (mg/dL)
            data.insulin_on_board (float): Active insulin (Units)
            data.carb_intake (float): Carbs announced (grams)
            data.glucose_trend (float): Rate of change
        """
        
        # 1. Your Logic
        dose = 0.0
        if data.current_glucose > 180:
            dose = 1.0
        
        # 2. Log your reasoning (Crucial for Explainable AI)
        self._log_reason("Glucose is high", "correction", dose)
        
        # 3. Return the decision
        return {
            "total_insulin_delivered": dose
        }
```

### Creating a New Algorithm
Use the CLI generator:
```bash
iints new-algo SuperController --author "Jane Doe" --output-dir algorithms/
```

---

## 6. Scenarios & Stress Testing

Scenarios are JSON files that define "What happens to the patient today?".

**Example: `scenarios/workout_day.json`**
```json
{
  "scenario_name": "Workout Wednesday",
  "stress_events": [
    {
      "start_time": 60,
      "event_type": "meal",
      "value": 45, 
      "absorption_delay_minutes": 15
    },
    {
      "start_time": 300,
      "event_type": "exercise",
      "value": 0.5,
      "duration": 45
    },
    {
      "start_time": 600,
      "event_type": "sensor_error",
      "value": 400,
      "duration": 30
    }
  ]
}
```

**Event Types:**
*   `meal`: `value` = grams of carbs.
*   `exercise`: `value` = intensity (0.0 - 1.0).
*   `sensor_error`: `value` = false glucose reading reported to algo.
*   `missed_bolus`: Patient eats but forgets to tell the pump (handled via unannounced meals in simulator).

---

## 7. Benchmarking

How do you know if your AI is better than a standard pump?
The `benchmark` command runs a head-to-head battle.

```bash
iints benchmark \
  --algo-to-benchmark algorithms/my_ai.py \
  --scenarios-dir scenarios/ \
  --patient-configs-dir src/iints/data/virtual_patients/
```

**Output:**
A comparison table showing:
*   **TIR (Time In Range)**: Goal > 70%.
*   **Hypo (<70 mg/dL)**: Goal < 4%.
*   **Safety Violations**: How many times the Supervisor had to intervene.

---

## 8. CLI Reference

### `iints init`
Initialize a new project workspace.
*   `--project-name`: Name of the folder.

### `iints new-algo`
Create a Python template for a new algorithm.
*   `NAME`: Class name of the algorithm.
*   `--author`: Your name.
*   `--output-dir`: Where to save the file.

### `iints run`
Execute a single simulation.
*   `--algo`: Path to algorithm file.
*   --patient-config-name`: YAML file name (default: `default`).
*   `--scenario-path`: JSON scenario file.
*   `--duration`: Minutes (default 720).
*   `--time-step`: Minutes (default 5).

### `iints benchmark`
Run a batch study.
*   `--algo-to-benchmark`: Your AI file.
*   `--scenarios-dir`: Folder with JSON scenarios.

### `iints docs algo`
Generate technical documentation for an algorithm file.

---

## 9. Troubleshooting

**Q: My algorithm crashes with "ImportError".**
A: Ensure your algorithm file imports from `iints`. The CLI handles the path, but your IDE might need you to install the package locally (`pip install -e .`).

**Q: The simulation results show flat glucose.**
A: Did you provide a scenario with meals? If the patient doesn't eat, glucose stays flat (basal equilibrium).

**Q: How do I change the patient's insulin sensitivity?**
A: Edit the patient YAML file (e.g., `src/iints/data/virtual_patients/default.yaml`) or create a copy in your project `data/` folder and point to it.

---

## 10. Glossary

*   **TIR (Time In Range)**: Percentage of time glucose is 70-180 mg/dL.
*   **IOB (Insulin On Board)**: Active insulin remaining in the body.
*   **ISF (Insulin Sensitivity Factor)**: How much 1 unit drops glucose (mg/dL/U).
*   **ICR (Insulin to Carb Ratio)**: How many grams of carbs 1 unit covers (g/U).
*   **Dual-Guard**: The architecture splitting AI and Safety.
