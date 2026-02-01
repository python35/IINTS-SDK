# Commercial Algorithm Emulation References

This document provides references for the commercial insulin pump emulators implemented in IINTS-AF.

## Overview

The emulation module implements the control algorithms of major commercial insulin pumps based on:
- Published clinical studies
- FDA regulatory documentation
- User manuals and technical specifications
- Patent filings

This enables researchers to:
1. Compare new algorithms against established commercial systems
2. Identify areas where commercial pumps underperform
3. Demonstrate the value of new approaches

---

## Medtronic MiniMed 780G with SmartGuard

### Key Features
- Target glucose: 100-120 mg/dL (configurable)
- Hybrid closed-loop with auto-basal and auto-correction
- Predictive Low Glucose Suspend (PLGS)
- Automatic correction boluses every 5 minutes

### Clinical Evidence
- **Bergenstal et al. (2020)** - "Hybrid Closed-Loop Therapy in Adults"
  - NEJM, DOI: 10.1056/NEJMoa2003479
  - Showed 71.2% Time-in-Range
  
### Regulatory
- FDA 510(k) K193510 (December 2020)
- CE Mark approval (2020)

### Technical Parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Target Glucose | 120 mg/dL (default) | User Guide |
| Low Suspend | 70 mg/dL | Clinical Study |
| Max Bolus | 10 U | Specifications |
| Max Basal | 6 U/hr | Specifications |
| Insulin Duration | 4 hours | User Guide |

### References
1. Medtronic 780G User Guide (2020)
2. Bergenstal RM, et al. NEJM 2020;382:209-220
3. FDA 510(k) Summary K193510

---

## Tandem t:slim X2 with Control-IQ

### Key Features
- Target glucose: 112.5 mg/dL (day), 140 mg/dL (exercise)
- Hybrid closed-loop with PLGS and Predictive High Glucose Assist (PHGS)
- Exercise mode with higher target
- 6-hour insulin duration (longer than typical)

### Clinical Evidence
- **Brown et al. (2019)** - "Control-IQ Technology"
  - Diabetes Technology & Therapeutics, DOI: 10.1089/dia.2019.0226
  - Showed 71% Time-in-Range in iDCL Trial
  
### Regulatory
- FDA 510(k) K191289 (December 2019)
- Control-IQ approved for ages 6+

### Technical Parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Target Glucose | 112.5 mg/dL | Clinical Study |
| Low Limit | 70 mg/dL | User Guide |
| Max Delivery | 3 U/hour auto | Specifications |
| Insulin Duration | 6 hours | User Guide |
| Correction Frequency | Hourly | User Guide |

### References
1. Brown SA, et al. Diabetes Technol Ther 2019;21(S1):A-15
2. iDCL Trial (NCT03563313)
3. FDA 510(k) Summary K191289
4. Control-IQ User Guide (2020)

---

## Omnipod 5 with Horizon Algorithm

### Key Features
- Target glucose: 100-150 mg/dL (configurable)
- Adaptive learning algorithm
- Activity and Sleep modes
- Tubeless patch pump design

### Clinical Evidence
- **ASSERT Trial** (2021) - Omnipod 5 Pivotal Study
  - Showed 74% Time-in-Range
  - Adults with type 1 diabetes
  
- **ONSET Trial** (2022) - Type 2 Diabetes Study
  - First automated insulin delivery in type 2

### Regulatory
- FDA 510(k) K203467 (January 2021)
- First tubeless AID system approved

### Technical Parameters
| Parameter | Value | Source |
|-----------|-------|--------|
| Target Glucose | 110 mg/dL (default) | User Guide |
| Low Threshold | 67 mg/dL | Clinical Study |
| Max Bolus | 30 U | Specifications |
| Max Basal | 3 U/hour | Specifications |
| Insulin Duration | 5 hours | User Guide |

### References
1. ASSERT Trial Results (2021)
2. ONSET Trial Results (2022)
3. FDA 510(k) Summary K203467
4. Omnipod 5 User Guide (2021)

---

## Comparison Matrix

| Feature | Medtronic 780G | Tandem Control-IQ | Omnipod 5 |
|---------|----------------|-------------------|-----------|
| Target Glucose | 100-120 mg/dL | 112.5/140 mg/dL | 100-150 mg/dL |
| Insulin Duration | 4 hours | 6 hours | 5 hours |
| Low Threshold | 70 mg/dL | 70 mg/dL | 67 mg/dL |
| Exercise Mode | No | Yes | Yes |
| Sleep Mode | No | No | Yes |
| Tubeless | No | No | Yes |
| Auto Corrections | Every 5 min | Hourly | Every 5 min |

---

## Methodology

### Emulation Approach

1. **Parameter Extraction**: Core parameters (PID gains, targets, limits) extracted from clinical studies and regulatory documents.

2. **Behavioral Modeling**: Decision logic implemented based on:
   - Algorithm descriptions in publications
   - User behavior observations
   - Safety constraint patterns

3. **Validation**: Emulator behavior compared against published clinical outcomes.

### Limitations

1. **Proprietary Algorithms**: Exact algorithm details are trade secrets.
   - Emulations are approximations based on observable behavior.

2. **Parameter Ranges**: Some parameters have ranges in documentation.
   - We use mid-range values as defaults.

3. **Individual Variation**: Commercial pumps adapt to individual users.
   - Emulators represent "typical" behavior.

---

## Using Emulators for Research

### 1. Compare Against Baseline
```python
from src.emulation import get_emulator

# Run commercial pump on data
pump = get_emulator('medtronic_780g')
pump_result = pump.emulate_decision(glucose=180, velocity=1.0, ...)

# Run new algorithm
new_result = new_algorithm.predict(data)

# Compare
print(f"780G: {pump_result.insulin_delivered:.2f} U")
print(f"New: {new_result['total_insulin_delivered']:.2f} U")
```

### 2. Identify Failure Modes
```python
# Find scenarios where commercial pump underperforms
for scenario in scenarios:
    pump_result = pump.emulate_decision(...)
    if pump_result.predicted_glucose < 54:  # Severe hypo
        print(f"Hypo risk at {scenario}")
```

### 3. Demonstrate Improvement
```python
# Show how new algorithm handles edge cases
for scenario in edge_cases:
    pump_result = pump.emulate_decision(...)
    new_result = new_algorithm.predict(...)
    
    print(f"Scenario: {scenario}")
    print(f"  780G TIR: {pump_metrics.tir:.1f}%")
    print(f"  New TIR: {new_metrics.tir:.1f}%")
```

---

## Contributing

To add a new pump emulator:

1. Create `src/emulation/{pump_name}.py` with:
   - `PumpBehavior` dataclass
   - `PumpNameEmulator` class inheriting from `LegacyEmulator`
   - `_get_default_behavior()` method
   - `get_sources()` method
   - `emulate_decision()` method

2. Update `src/emulation/__init__.py` to export new emulator

3. Add tests in `tests/test_emulation.py`

4. Document sources in this file

---

## Disclaimer

These emulators are for research purposes only. They do not represent the actual performance of commercial devices and should not be used for clinical decision-making.

