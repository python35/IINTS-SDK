# IINTS-AF: Complete Developer & Researcher Guide

**The Ultimate Guide to Intelligent Insulin Therapy Systems - Autonomous Framework**

*A comprehensive pre-clinical research platform for diabetes management algorithm validation, AI learning systems, and medical device development*

---

## Table of Contents

1. [Introduction & Philosophy](#introduction--philosophy)
2. [Quick Start Guide](#quick-start-guide)
3. [Platform Architecture](#platform-architecture)
4. [Core Concepts](#core-concepts)
5. [Module Reference](#module-reference)
6. [Algorithm Development](#algorithm-development)
7. [Patient Simulation](#patient-simulation)
8. [Clinical Analysis](#clinical-analysis)
9. [AI & Machine Learning](#ai--machine-learning)
10. [Safety Systems](#safety-systems)
11. [Data Management](#data-management)
12. [Visualization & Reporting](#visualization--reporting)
13. [Edge AI & Hardware](#edge-ai--hardware)
14. [Research Applications](#research-applications)
15. [Advanced Features](#advanced-features)
16. [Troubleshooting](#troubleshooting)
17. [Contributing](#contributing)
18. [Clinical Standards](#clinical-standards)
19. [Legal & Compliance](#legal--compliance)

---

## Introduction & Philosophy

### What is IINTS-AF?

IINTS-AF (Insuline Is Not The Solution - Autonomous Framework) is a **pre-clinical research platform** designed for validating AI-based diabetes management algorithms. It functions as a digital-twin-style sandbox that enables safe experimentation on virtual patient models before any real-world deployment consideration.

### Core Philosophy

**"Safety First, Transparency Always, Learning Never Stops"**

- **Pre-Clinical Focus**: This is NOT a medical device - it's a research instrument
- **Explainable AI**: Every decision must be traceable and understandable
- **Safety Constraints**: Hard limits prevent physiologically dangerous outputs
- **Continuous Learning**: AI systems that adapt and improve over time
- **Clinical Standards**: Aligned with industry practices (Medtronic, ADA guidelines)

### Who Should Use This?

- **Medical AI Researchers** - Algorithm validation and publication
- **Biomedical Engineering Students** - Learning diabetes management systems
- **Clinical Algorithm Developers** - Pre-clinical testing and validation
- **Regulatory Compliance Teams** - Safety and transparency verification
- **Academic Institutions** - Teaching AI in healthcare

---

## Quick Start Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/python35/IINTS-AF.git
cd IINTS-AF

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/validate_system.py
```

### First Run

```bash
# Launch the main interface
python iints.py

# Or run a quick demo
python iints.py --demo

# Or run system validation
python scripts/validate_system.py
```

### Your First Analysis

```bash
# Run a complete clinical validation
python iints.py
# Select option [1] - In-Silico Clinical Validation

# Generate Excel population study
python scripts/generate_excel_summary.py

# Create scientific visualizations
python scripts/create_population_graphics.py
```

---

## Platform Architecture

### Directory Structure

```
IINTS-AF/
 bin/                     # Main control interfaces
    master_dashboard.py  # Professional terminal interface
    main.py             # Legacy interface
    experiment.py       # Research experiment runner
    population_runner.py # Population study runner
 src/                     # Core framework modules
    algorithm/           # Insulin delivery algorithms
       base_algorithm.py      # Abstract base class
       lstm_algorithm.py      # LSTM neural network
       pid_controller.py      # PID control system
       hybrid_algorithm.py    # Hybrid AI+PID
       fixed_basal_bolus.py   # Traditional therapy
    analysis/            # Clinical analysis tools
       clinical_tir_analyzer.py    # 5-zone TIR analysis
       explainable_ai.py          # Decision transparency
       edge_performance_monitor.py # Hardware validation
       diabetes_metrics.py        # Clinical metrics
    patient/             # Patient simulation
       models.py               # Physiological models
       patient_factory.py      # Patient generation
    simulation/          # Simulation engine
       simulator.py           # Main simulation loop
    safety/              # Safety systems
       supervisor.py          # Safety supervisor
    learning/            # AI learning systems
        autonomous_optimizer.py # Self-improving AI
        learning_system.py     # Learning framework
 scripts/                 # Analysis and utility scripts
 data_packs/             # Clinical datasets
 results/                # Generated outputs
 tools/                  # Import and utility tools
 iints.py               # Main launcher
```

### Core Components

1. **Simulation Engine** (`src/simulation/simulator.py`)
   - Main simulation loop
   - Event handling (meals, stress, exercise)
   - Time-series data generation

2. **Algorithm Framework** (`src/algorithm/`)
   - Pluggable algorithm architecture
   - LSTM neural networks
   - Traditional control systems
   - Hybrid approaches

3. **Patient Models** (`src/patient/`)
   - Physiological simulation
   - Individual patient variability
   - Population diversity

4. **Safety Systems** (`src/safety/`)
   - Hard constraint enforcement
   - Risk assessment
   - Override mechanisms

5. **Analysis Tools** (`src/analysis/`)
   - Clinical metrics calculation
   - TIR analysis
   - Performance monitoring
   - Explainable AI

---

## Core Concepts

### Time in Range (TIR) Analysis

IINTS-AF uses the **Medtronic 5-Zone Classification**:

| Zone | Range (mg/dL) | Clinical Name | Target |
|------|---------------|---------------|---------|
| 1 | < 54 | Very Low | < 1% |
| 2 | 54-69 | Low | < 4% |
| 3 | 70-180 | Target Range | > 70% |
| 4 | 181-250 | High | < 25% |
| 5 | > 250 | Very High | < 5% |

### Safety Architecture

**Three-Layer Safety System:**

1. **Algorithm Layer**: Your AI/control algorithm
2. **Safety Supervisor**: Hard constraint enforcement
3. **Physiological Limits**: Absolute safety bounds

```python
# Example safety constraint
if predicted_insulin > MAX_BOLUS_UNITS:
    actual_insulin = MAX_BOLUS_UNITS
    safety_override = True
    log_safety_event("Excessive bolus prevented")
```

### Decision Transparency

Every insulin delivery decision includes:
- **Glucose Context**: Current value, trend, history
- **Clinical Reasoning**: Why this decision was made
- **Risk Assessment**: Potential outcomes
- **Confidence Score**: AI certainty level
- **Safety Status**: Any overrides or constraints

---

## Module Reference

### 1. In-Silico Clinical Validation

**Purpose**: Complete clinical validation with safety verification

**What it does**:
- Loads patient data (Ohio T1DM dataset)
- Runs neural network adaptation
- Performs 5-zone TIR analysis
- Validates safety constraints
- Generates clinical assessment

**Usage**:
```bash
python iints.py
# Select [1]
```

**Outputs**:
- TIR percentages by zone
- Clinical risk assessment
- Safety validation report

### 2. Population Study Analytics

**Purpose**: Statistical analysis across multiple patients

**What it does**:
- Tests algorithms on diverse patient population
- Generates Excel reports with 4 professional sheets
- Calculates statistical significance (p-values)
- Creates algorithm comparison matrices

**Usage**:
```bash
python scripts/generate_excel_summary.py
```

**Outputs**:
- `results/IINTS_Summary_*.xlsx` with sheets:
  - Population Study: TIR improvements
  - Decision Audit: AI decision tracking
  - Statistical Analysis: P-values and success rates
  - Algorithm Comparison: Performance matrix

### 3. Scientific Visualization Generator

**Purpose**: Publication-ready graphics for presentations

**What it does**:
- Creates 4-panel scientific figures
- Generates algorithm performance heatmaps
- Produces statistical summary plots
- Formats for Science Expo presentations

**Usage**:
```bash
python scripts/create_population_graphics.py
```

**Outputs**:
- `results/population_plots/population_study_overview.png`
- `results/population_plots/clinical_scenario_heatmap.png`
- `results/population_plots/statistical_summary.png`

### 4. Clinical Documentation System

**Purpose**: Hospital-grade PDF reports for clinicians

**What it does**:
- Generates Medtronic CareLink-style reports
- Includes 5-zone TIR analysis
- Shows learning curve validation
- Uses professional medical terminology

**Usage**:
```bash
python scripts/generate_clinical_report.py
```

**Outputs**:
- `results/clinical_reports/Clinical_Report_*.pdf`

### 5. MATLAB Control Theory Export

**Purpose**: University-level control theory analysis

**What it does**:
- Exports system response data
- Generates MATLAB analysis scripts
- Enables Bode plots and stability analysis
- Supports system identification

**Usage**:
```bash
python scripts/matlab_control_integration.py
```

**Outputs**:
- `results/matlab_analysis/system_data.mat`
- `results/matlab_analysis/analyze_stability.m`

### 6. Professional TIR Analysis

**Purpose**: Detailed 5-zone glucose classification

**What it does**:
- Analyzes glucose data using clinical standards
- Provides zone-by-zone breakdown
- Generates clinical risk assessment
- Calculates TIR quality metrics

**Usage**: Available in master dashboard [6]

### 7. Clinical Decision Audit System

**Purpose**: Explainable AI decision tracking

**What it does**:
- Logs every insulin delivery decision
- Provides clinical reasoning for each choice
- Tracks confidence levels
- Generates audit trails

**Usage**: Available in master dashboard [7]

### 8. Edge AI Performance Validation

**Purpose**: Hardware performance testing (Jetson Nano)

**What it does**:
- Measures inference latency
- Tests embedded device compatibility
- Validates real-time performance
- Generates hardware assessment

**Usage**: Available in master dashboard [8]

### 9. Comparative Algorithm Benchmarking

**Purpose**: Head-to-head algorithm comparison

**What it does**:
- Tests multiple algorithms simultaneously
- Calculates statistical significance
- Generates performance matrices
- Creates comparison visualizations

**Usage**:
```bash
python scripts/test_comparative_benchmarking.py
```

### 10. AI Uncertainty Quantification

**Purpose**: Advanced ML with confidence intervals

**What it does**:
- Implements Monte Carlo Dropout
- Separates epistemic vs aleatoric uncertainty
- Detects "I don't know" situations
- Provides confidence scoring

**Usage**:
```bash
python scripts/uncertainty_quantification.py
```

### 11. Real-Time Glucose Dashboard

**Purpose**: Live glucose monitoring simulation

**What it does**:
- Real-time glucose curve visualization
- Live TIR meter updates
- AI uncertainty visualization
- Insulin delivery tracking

**Usage**:
```bash
python scripts/real_time_dashboard.py
```

---

## Algorithm Development

### Creating Your Own Algorithm

1. **Inherit from BaseAlgorithm**:

```python
from src.algorithm.base_algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, settings):
        super().__init__(settings)
        # Initialize your algorithm
    
    def calculate_insulin(self, glucose_mgdl, glucose_trend, time_minutes):
        # Your algorithm logic here
        insulin_units = your_calculation(glucose_mgdl, glucose_trend)
        
        # Return with confidence score
        return {
            'insulin_units': insulin_units,
            'confidence': 0.85,
            'reasoning': "Your explanation here"
        }
```

2. **Register Your Algorithm**:

```python
# In src/algorithm/__init__.py
from .my_algorithm import MyAlgorithm

AVAILABLE_ALGORITHMS = {
    'my_algorithm': MyAlgorithm,
    'lstm': LSTMInsulinAlgorithm,
    'pid': PIDController,
    # ... other algorithms
}
```

3. **Test Your Algorithm**:

```python
from src.patient.patient_factory import PatientFactory
from src.simulation.simulator import Simulator

# Create test patient
patient = PatientFactory.create_patient('custom', initial_glucose=120)

# Initialize your algorithm
algorithm = MyAlgorithm({'param1': value1})

# Run simulation
simulator = Simulator(patient, algorithm)
results = simulator.run(duration_minutes=240)

# Analyze results
print(f"Final glucose: {results['glucose_actual_mgdl'].iloc[-1]}")
```

### Algorithm Types Included

1. **LSTM Neural Network** (`lstm_algorithm.py`)
   - Deep learning approach
   - Learns from historical patterns
   - Adapts to individual patients

2. **PID Controller** (`pid_controller.py`)
   - Traditional control theory
   - Proportional-Integral-Derivative control
   - Industry standard approach

3. **Hybrid System** (`hybrid_algorithm.py`)
   - Combines AI and traditional control
   - AI for pattern recognition
   - PID for stable control

4. **Fixed Basal-Bolus** (`fixed_basal_bolus.py`)
   - Traditional diabetes therapy
   - Fixed insulin schedules
   - Baseline comparison

---

## Patient Simulation

### Patient Model Architecture

The patient simulation is based on physiological models that simulate:
- **Glucose absorption** from meals
- **Insulin action** (rapid, intermediate, long-acting)
- **Hepatic glucose production**
- **Peripheral glucose uptake**
- **Individual variability** in insulin sensitivity

### Creating Patients

```python
from src.patient.patient_factory import PatientFactory

# Standard patient types
patient = PatientFactory.create_patient('adult_type1')
patient = PatientFactory.create_patient('adolescent')
patient = PatientFactory.create_patient('insulin_resistant')

# Custom patient
patient = PatientFactory.create_patient('custom', 
    initial_glucose=150,
    insulin_sensitivity=0.8,
    carb_ratio=12.0
)

# Get diverse population
patients = PatientFactory.get_patient_diversity_set()
```

### Patient Parameters

Key parameters you can modify:
- **insulin_sensitivity**: How much glucose drops per unit insulin
- **carb_ratio**: Grams of carbs covered by 1 unit insulin
- **basal_rate**: Background insulin needs (units/hour)
- **glucose_target**: Target glucose level (mg/dL)
- **dawn_phenomenon**: Early morning glucose rise
- **stress_response**: Response to stress events

### Stress Events

Simulate real-world challenges:

```python
from src.simulation.simulator import StressEvent

# Meal events
simulator.add_stress_event(StressEvent(60, 'meal', 45))  # 45g carbs at 60 min

# Exercise events
simulator.add_stress_event(StressEvent(120, 'exercise', 30))  # 30 min exercise

# Illness/stress
simulator.add_stress_event(StressEvent(180, 'stress', 2.0))  # 2x glucose rise

# Dawn phenomenon
simulator.add_stress_event(StressEvent(360, 'dawn', 1.5))  # 1.5x basal needs
```

---

## Clinical Analysis

### TIR Analysis Deep Dive

The clinical TIR analyzer provides comprehensive glucose zone analysis:

```python
from src.analysis.clinical_tir_analyzer import ClinicalTIRAnalyzer

analyzer = ClinicalTIRAnalyzer()
glucose_data = [120, 150, 180, 200, 160, 140, 110, 95, 85, 75]

analysis = analyzer.analyze_glucose_zones(glucose_data)

# Results include:
# - Percentage in each zone
# - Clinical assessment
# - Risk level determination
# - TIR quality rating
```

### Clinical Metrics

Standard diabetes management metrics:
- **Time in Range (TIR)**: 70-180 mg/dL
- **Time Below Range (TBR)**: < 70 mg/dL
- **Time Above Range (TAR)**: > 180 mg/dL
- **Glucose Variability**: Standard deviation, CV%
- **Hypoglycemia Risk**: < 54 mg/dL episodes
- **Hyperglycemia Severity**: > 250 mg/dL duration

### Statistical Analysis

Built-in statistical testing:
- **Paired t-tests** for before/after comparisons
- **ANOVA** for multiple algorithm comparison
- **Effect size** calculation (Cohen's d)
- **Confidence intervals** for all metrics
- **P-value** calculation with significance testing

---

## AI & Machine Learning

### LSTM Neural Network

The LSTM algorithm learns patterns in glucose data:

**Architecture**:
- Input: 7 features (glucose, trend, time, etc.)
- Hidden: 50 LSTM units
- Output: Insulin recommendation + confidence

**Training Process**:
```python
from src.learning.autonomous_optimizer import AutonomousOptimizer

optimizer = AutonomousOptimizer()
optimizer.train_on_patient_data(patient_data)
optimizer.save_improved_model()
```

**Features Used**:
1. Current glucose (mg/dL)
2. Glucose trend (mg/dL/min)
3. Time since last meal (minutes)
4. Active insulin on board (units)
5. Time of day (circadian factor)
6. Recent glucose variability
7. Historical pattern similarity

### Autonomous Learning

The system can improve itself through experience:

```python
# Test autonomous learning
python scripts/test_autonomous_learning.py

# Validate learning effectiveness
python scripts/clinical_reality_check.py
```

**Learning Process**:
1. **Experience Collection**: Gather simulation results
2. **Performance Evaluation**: Calculate TIR improvements
3. **Model Update**: Retrain neural network weights
4. **Validation**: Test on unseen patients
5. **Reality Check**: Verify actual behavior change

### Uncertainty Quantification

Advanced ML techniques for confidence estimation:

**Monte Carlo Dropout**:
- Multiple forward passes with dropout
- Estimates epistemic uncertainty
- Identifies model confidence

**Bayesian Neural Networks**:
- Probabilistic weight distributions
- Uncertainty propagation
- Principled confidence intervals

---

## Safety Systems

### Safety Supervisor Architecture

The safety supervisor enforces hard constraints:

```python
from src.safety.supervisor import SafetySupervisor

supervisor = SafetySupervisor()

# Check proposed insulin dose
safe_dose = supervisor.validate_insulin_dose(
    proposed_dose=2.5,
    current_glucose=120,
    active_insulin=1.2,
    time_since_last_dose=30
)
```

### Safety Constraints

**Hard Limits**:
- Maximum single bolus: 10 units
- Maximum hourly insulin: 15 units
- Minimum time between doses: 15 minutes
- Hypoglycemia prevention: No insulin if glucose < 80 mg/dL

**Risk Assessment**:
- Predicted glucose trajectory
- Insulin stacking prevention
- Meal timing consideration
- Exercise impact evaluation

### Safety Validation

Every algorithm decision is validated:

```python
# Safety check example
if glucose_mgdl < HYPOGLYCEMIA_THRESHOLD:
    insulin_units = 0.0
    safety_override = True
    reasoning = "Hypoglycemia prevention - insulin delivery suspended"
```

---

## Data Management

### Data Schema

All datasets conform to a unified schema:

```python
# Standard data format
{
    'timestamp': datetime,
    'glucose_mgdl': float,
    'insulin_units': float,
    'carbs_grams': float,
    'patient_id': str,
    'event_type': str  # 'meal', 'exercise', 'stress'
}
```

### Supported Datasets

1. **Ohio T1DM Dataset**
   - 12 patients with Type 1 diabetes
   - 8 weeks of continuous glucose monitoring
   - Meal and insulin annotations

2. **Synthetic Patients**
   - Generated using physiological models
   - Diverse population characteristics
   - Controlled experimental conditions

3. **Custom Data Import**
   - CSV file import
   - Nightscout export compatibility
   - OpenAPS data integration

### Data Import Tools

```bash
# Import Ohio T1DM data
python tools/import_ohio.py /path/to/ohio/dataset

# Import Nightscout export
python tools/import_openaps.py /path/to/nightscout.json

# Import custom CSV
python bin/data_import.py --file mydata.csv --name my_study
```

---

## Visualization & Reporting

### Scientific Visualizations

Publication-ready graphics with:
- **Professional styling** (Nature/Science journal standards)
- **Statistical annotations** (p-values, confidence intervals)
- **Clinical context** (TIR zones, safety thresholds)
- **Multi-panel layouts** for comprehensive analysis

### Report Types

1. **Excel Population Studies**
   - 4 professional sheets
   - Statistical analysis
   - Algorithm comparisons
   - Clinical metrics

2. **PDF Clinical Reports**
   - Hospital-grade formatting
   - Medtronic CareLink style
   - Professional medical terminology
   - Safety validation summaries

3. **MATLAB Analysis**
   - Control theory exports
   - System identification data
   - Bode plot generation
   - Stability analysis

### Custom Visualizations

Create your own plots:

```python
import matplotlib.pyplot as plt
from src.analysis.diabetes_metrics import calculate_tir

# Load simulation results
results = pd.read_csv('results/simulation_data.csv')

# Calculate TIR
tir_percentage = calculate_tir(results['glucose_mgdl'])

# Create custom plot
plt.figure(figsize=(12, 8))
plt.plot(results['time_minutes'], results['glucose_mgdl'])
plt.axhline(y=70, color='red', linestyle='--', label='Hypoglycemia')
plt.axhline(y=180, color='orange', linestyle='--', label='Hyperglycemia')
plt.fill_between(results['time_minutes'], 70, 180, alpha=0.2, color='green')
plt.title(f'Glucose Profile (TIR: {tir_percentage:.1f}%)')
plt.xlabel('Time (minutes)')
plt.ylabel('Glucose (mg/dL)')
plt.legend()
plt.savefig('my_glucose_plot.png', dpi=300, bbox_inches='tight')
```

---

## Edge AI & Hardware

### Jetson Nano Support

IINTS-AF is optimized for embedded deployment:

**Performance Monitoring**:
```python
from src.analysis.edge_performance_monitor import EdgeAIPerformanceMonitor

monitor = EdgeAIPerformanceMonitor()
monitor.start_monitoring()

# Measure inference latency
stats = monitor.measure_inference_latency(algorithm.predict, test_data)

print(f"Mean latency: {stats['latency_statistics']['mean_ms']:.3f} ms")
print(f"95th percentile: {stats['latency_statistics']['p95_ms']:.3f} ms")
```

**Hardware Requirements**:
- **Minimum**: NVIDIA Jetson Nano (4GB)
- **Recommended**: Jetson Xavier NX
- **Memory**: 4GB RAM minimum
- **Storage**: 32GB SD card minimum

### Real-Time Performance

Target performance metrics:
- **Inference latency**: < 10ms
- **Decision frequency**: Every 5 minutes
- **Memory usage**: < 2GB
- **Power consumption**: < 10W

### Deployment Considerations

1. **Model Optimization**:
   - TensorRT optimization
   - Quantization (FP16/INT8)
   - Pruning for edge deployment

2. **Safety Validation**:
   - Hardware failure detection
   - Graceful degradation
   - Backup control systems

---

## Research Applications

### Academic Research

**Publication Support**:
- Statistical rigor (p-values, confidence intervals)
- Reproducible experiments (fixed random seeds)
- Professional visualizations
- Clinical terminology alignment

**Research Questions**:
- Algorithm comparison studies
- Population diversity analysis
- Safety system validation
- Learning system effectiveness

### Medical Device Development

**Pre-Clinical Testing**:
- Algorithm validation
- Safety system verification
- Edge AI performance testing
- Regulatory compliance preparation

**Development Workflow**:
1. Algorithm development and testing
2. Population study validation
3. Safety system integration
4. Edge deployment optimization
5. Clinical documentation generation

### Educational Applications

**Teaching Diabetes Management**:
- Interactive simulations
- Algorithm comparison
- Safety system demonstration
- Clinical decision making

**Course Integration**:
- Biomedical engineering curricula
- Medical AI courses
- Control systems classes
- Clinical informatics programs

---

## Advanced Features

### Monte Carlo Experiments

Run multiple experiments with statistical analysis:

```python
from src.analysis.clinical_benchmark import run_monte_carlo_study

results = run_monte_carlo_study(
    algorithms=['lstm', 'pid', 'hybrid'],
    patients=PatientFactory.get_patient_diversity_set(),
    scenarios=['standard_meal', 'high_carb', 'exercise'],
    iterations=100,
    random_seed=42
)

# Results include statistical significance testing
print(f"LSTM vs PID p-value: {results['statistical_tests']['lstm_vs_pid']['p_value']}")
```

### Circadian Rhythm Modeling

Account for daily glucose patterns:

```python
# Circadian factors automatically included
patient = PatientFactory.create_patient('adult_type1')
patient.enable_circadian_variation(
    dawn_phenomenon=True,
    evening_insulin_resistance=True,
    sleep_glucose_stability=True
)
```

### Meal Detection

Automatic meal detection from glucose patterns:

```python
from src.analysis.sensor_filtering import detect_meals

glucose_data = results['glucose_mgdl']
detected_meals = detect_meals(glucose_data, threshold=20)  # 20 mg/dL rise

for meal in detected_meals:
    print(f"Meal detected at {meal['time']}: {meal['estimated_carbs']}g")
```

### Custom Stress Events

Create complex scenarios:

```python
# Multi-day simulation with various events
simulator = Simulator(patient, algorithm)

# Day 1: Normal meals
simulator.add_stress_event(StressEvent(480, 'meal', 45))   # Breakfast
simulator.add_stress_event(StressEvent(720, 'meal', 60))   # Lunch
simulator.add_stress_event(StressEvent(1080, 'meal', 75))  # Dinner

# Day 2: Exercise and stress
simulator.add_stress_event(StressEvent(1560, 'exercise', 45))  # Morning run
simulator.add_stress_event(StressEvent(1680, 'stress', 1.8))   # Work stress

# Day 3: Illness
simulator.add_stress_event(StressEvent(2160, 'illness', 2.5))  # Sick day

results = simulator.run(duration_minutes=4320)  # 3 days
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **LSTM Model Not Found**:
   ```bash
   # Check if model file exists
   ls src/algorithm/trained_lstm_model.pth
   
   # If missing, train a new model
   python scripts/test_autonomous_learning.py
   ```

3. **Memory Issues**:
   ```bash
   # Reduce simulation duration
   results = simulator.run(duration_minutes=240)  # Instead of 1440
   
   # Use fewer patients in population studies
   patients = PatientFactory.get_patient_diversity_set()[:3]
   ```

4. **Visualization Errors**:
   ```bash
   # Install additional plotting dependencies
   pip install seaborn plotly
   
   # Check display backend
   python -c "import matplotlib; print(matplotlib.get_backend())"
   ```

### Performance Optimization

1. **Speed Up Simulations**:
   ```python
   # Use larger time steps (less precision, faster execution)
   simulator = Simulator(patient, algorithm, time_step_minutes=5)
   
   # Reduce logging verbosity
   simulator.set_logging_level('WARNING')
   ```

2. **Memory Optimization**:
   ```python
   # Process patients sequentially instead of parallel
   for patient in patients:
       results = run_single_patient_study(patient)
       save_results(results)
       del results  # Free memory
   ```

3. **GPU Acceleration**:
   ```python
   # Enable GPU for LSTM if available
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   ```

### Debugging Tips

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Intermediate Results**:
   ```python
   # Save intermediate data
   results.to_csv('debug_results.csv')
   
   # Plot glucose trajectory
   plt.plot(results['glucose_mgdl'])
   plt.show()
   ```

3. **Validate Input Data**:
   ```python
   # Check data ranges
   print(f"Glucose range: {results['glucose_mgdl'].min()}-{results['glucose_mgdl'].max()}")
   print(f"Insulin range: {results['insulin_units'].min()}-{results['insulin_units'].max()}")
   ```

---

## Contributing

### Development Setup

1. **Fork the Repository**:
   ```bash
   git clone https://github.com/yourusername/IINTS-AF.git
   cd IINTS-AF
   ```

2. **Create Development Environment**:
   ```bash
   python -m venv iints_dev
   source iints_dev/bin/activate  # Linux/Mac
   # or
   iints_dev\Scripts\activate     # Windows
   
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/
   python scripts/validate_system.py
   ```

### Code Style

Follow PEP 8 with these additions:
- **Docstrings**: All functions must have docstrings
- **Type hints**: Use type hints for function parameters
- **Comments**: Explain clinical reasoning, not just code
- **Variable names**: Use medical terminology when appropriate

```python
def calculate_insulin_dose(glucose_mgdl: float, 
                          carb_grams: float,
                          insulin_sensitivity: float) -> Dict[str, float]:
    """
    Calculate insulin dose using carbohydrate ratio method.
    
    Args:
        glucose_mgdl: Current blood glucose in mg/dL
        carb_grams: Carbohydrates to be consumed in grams
        insulin_sensitivity: Patient's insulin sensitivity factor
    
    Returns:
        Dictionary containing insulin dose and clinical reasoning
    """
    # Clinical calculation logic here
    pass
```

### Adding New Algorithms

1. **Create Algorithm Class**:
   ```python
   # src/algorithm/my_new_algorithm.py
   from .base_algorithm import BaseAlgorithm
   
   class MyNewAlgorithm(BaseAlgorithm):
       def calculate_insulin(self, glucose_mgdl, glucose_trend, time_minutes):
           # Your algorithm implementation
           return {
               'insulin_units': calculated_dose,
               'confidence': confidence_score,
               'reasoning': clinical_explanation
           }
   ```

2. **Add Tests**:
   ```python
   # tests/test_my_new_algorithm.py
   def test_my_new_algorithm():
       algorithm = MyNewAlgorithm({})
       result = algorithm.calculate_insulin(120, 0.5, 60)
       assert result['insulin_units'] >= 0
       assert 0 <= result['confidence'] <= 1
   ```

3. **Update Documentation**:
   - Add algorithm description to README
   - Include clinical rationale
   - Provide usage examples

### Submitting Changes

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/my-new-algorithm
   ```

2. **Make Changes and Test**:
   ```bash
   python scripts/validate_system.py
   python -m pytest tests/
   ```

3. **Submit Pull Request**:
   - Clear description of changes
   - Clinical justification for new features
   - Test results and validation data

---

## Clinical Standards

### Regulatory Alignment

IINTS-AF aligns with established clinical standards:

**FDA Guidance**:
- Software as Medical Device (SaMD) principles
- Clinical evaluation requirements
- Risk management (ISO 14971)
- Software lifecycle processes (IEC 62304)

**Clinical Guidelines**:
- ADA Standards of Medical Care
- ISPAD Clinical Practice Consensus Guidelines
- Endocrine Society Clinical Practice Guidelines

**Industry Standards**:
- Medtronic CareLink reporting format
- Dexcom Clarity analysis methods
- Abbott FreeStyle Libre metrics

### Clinical Terminology

Standard medical terminology used throughout:
- **Hypoglycemia**: < 70 mg/dL (3.9 mmol/L)
- **Severe Hypoglycemia**: < 54 mg/dL (3.0 mmol/L)
- **Hyperglycemia**: > 180 mg/dL (10.0 mmol/L)
- **Severe Hyperglycemia**: > 250 mg/dL (13.9 mmol/L)
- **Time in Range**: 70-180 mg/dL (3.9-10.0 mmol/L)

### Quality Metrics

Clinical quality indicators:
- **TIR Target**: > 70% for most adults
- **Hypoglycemia Target**: < 4% below 70 mg/dL, < 1% below 54 mg/dL
- **Glucose Variability**: CV < 36%
- **Data Sufficiency**: ≥ 70% CGM data availability

---

## Legal & Compliance

### Important Disclaimers

[WARN] **CRITICAL SAFETY NOTICE**:

**THIS IS NOT A MEDICAL DEVICE**
- IINTS-AF is a research platform only
- Not approved for clinical use
- Not validated for patient care
- For educational and research purposes only

**NO MEDICAL ADVICE**
- Does not provide medical advice
- Does not replace clinical judgment
- Not intended for treatment decisions
- Consult healthcare providers for medical care

**RESEARCH USE ONLY**
- Pre-clinical validation platform
- Algorithm development and testing
- Educational demonstrations
- Academic research applications

### Liability Limitations

Users acknowledge:
- Software provided "as is"
- No warranties of any kind
- Users assume all risks
- Developers not liable for any damages
- Independent validation required for any clinical application

### Data Privacy

**No Patient Data**:
- Uses synthetic and anonymized datasets only
- No real patient information processed
- All data generated for research purposes
- Complies with HIPAA principles by design

**Open Source Commitment**:
- All code publicly available
- Transparent algorithms
- Reproducible research
- Community-driven development

### Regulatory Considerations

If you plan to develop medical devices based on this platform:
- Consult regulatory experts
- Follow FDA/CE marking requirements
- Conduct proper clinical trials
- Obtain necessary approvals
- Implement quality management systems

---

## Conclusion

IINTS-AF represents a comprehensive platform for diabetes management algorithm research and development. By providing a safe, transparent, and clinically-aligned environment, it enables researchers, students, and developers to advance the field of automated insulin delivery.

The platform's commitment to safety, explainability, and clinical standards makes it an ideal foundation for:
- Academic research and publication
- Medical device development
- Educational applications
- Regulatory compliance preparation

We encourage the community to contribute, extend, and improve this platform to advance diabetes care through technology.

---

## Support & Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Documentation**: Comprehensive guides and examples
- **Examples**: Real-world use cases and tutorials

**Remember**: This is a research platform. Always consult healthcare professionals for medical decisions.

---

*IINTS-AF: Transforming diabetes management AI from experimental code into a professional research instrument suitable for academic publication and regulatory review.*

**Version**: 1.0.0  
**Last Updated**: 2026  
**License**: Research Use Only  
**Maintainer**: IINTS-AF Development Team
