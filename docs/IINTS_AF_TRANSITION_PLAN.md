# IINTS-AF Pre-Clinical Research Framework - Transition Plan

## Executive Summary
Transition IINTS-AF from a single-algorithm project to a **Pre-clinical Research Framework** with four core work packages that enable universal algorithm testing, multi-source data ingestion, commercial pump emulation, and clinical-grade visualization.

---

## Phase 0: Current State Analysis

### Existing Assets [OK]
| Component | Location | Status |
|-----------|----------|--------|
| Base Algorithm Template | `src/algorithm/base_algorithm.py` | [OK] Exists, well-designed |
| Data Adapter | `src/data/adapter.py` | [OK] Basic Ohio support |
| Algorithm X-Ray | `src/analysis/algorithm_xray.py` | [OK] Reasoning logs exist |
| Explainable AI | `src/analysis/explainable_ai.py` | [OK] Audit trail exists |
| Reverse Engineering | `scripts/run_reverse_engineering.py` | [WARN] Basic pump profiles |
| PID Controller | `src/algorithm/pid_controller.py` | [OK] Reference implementation |

### Gap Analysis
| Work Package | Current Gap | Priority |
|--------------|-------------|----------|
| Data Bridge | No generic CSV/JSON parser, no data quality scoring | HIGH |
| Algorithm Engine | No multi-algorithm battle mode | HIGH |
| Legacy Emulators | No whitepaper-based implementations | MEDIUM |
| UI/Cockpit | No uncertainty cloud visualization | MEDIUM |

---

## Work Package 1: Universal Data Bridge (Data Team)

### Objective
Build an "Ingestion Engine" that accepts ANY CSV/JSON format and translates it to `[Time, Glucose, Carbs, Insulin]`, plus a "Data Quality Checker" with confidence scoring.

### Deliverables

#### 1.1 Universal Parser (`src/data/universal_parser.py`)
```python
class UniversalParser:
    """Parses any CSV/JSON to standard IINTS format"""
    
    def detect_schema(self, df: pd.DataFrame) -> Dict:
        """Auto-detect column mapping"""
        
    def parse(self, file_path: str) -> StandardDataPack:
        """Convert to [Time, Glucose, Carbs, Insulin]"""
```

#### 1.2 Column Mapper (`src/data/column_mapper.py`)
```python
class ColumnMapper:
    """Maps various column names to standard format"""
    STANDARD_COLUMNS = ['timestamp', 'glucose', 'carbs', 'insulin']
    
    ALIASES = {
        'glucose': ['bg', 'glucose_mg_dl', 'glucose_mgdl', 'sensor_glucose', 'sg', 'cbg'],
        'carbs': ['carbohydrates', 'cho', 'carb_intake', 'meal_carbs'],
        'insulin': ['insulin_delivered', 'insulin_units', 'bolus', 'total_insulin']
    }
```

#### 1.3 Data Quality Checker (`src/data/quality_checker.py`)
```python
class DataQualityChecker:
    """Validates data quality and returns confidence score"""
    
    def check_completeness(self, df) -> QualityReport:
        """Report missing data percentage with time ranges"""
        
    def calculate_confidence_score(self) -> float:
        """Return 0.0-1.0 confidence score"""
```

### Implementation Tasks
1. Create `UniversalParser` class with auto-detection
2. Build `ColumnMapper` with extensive alias database
3. Implement `DataQualityChecker` with gap detection
4. Add warnings: "[WARN] 10% data missing between 14:00-16:00, confidence: 0.72"
5. Update `src/data/__init__.py` to export new modules

---

## Work Package 2: Plug-and-Play Algorithm Engine (AI Team)

### Objective
Transform IINTS-AF into an open platform where ANY algorithm can be plugged in via a template. Enable "Battle Mode" to run multiple algorithms on identical patient data.

### Deliverables

#### 2.1 Enhanced Base Class (`src/algorithm/base_algorithm.py` - EXTEND)
```python
class InsulinAlgorithm(ABC):
    """Enhanced with battle mode support"""
    
    @abstractmethod
    def predict_insulin(self, data: AlgorithmInput) -> AlgorithmResult:
        """Returns insulin prediction with uncertainty"""
        
    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        """Returns name, version, author, paper_reference"""
```

#### 2.2 Battle Runner (`src/algorithm/battle_runner.py`)
```python
class BattleRunner:
    """Runs multiple algorithms on identical data"""
    
    def run_battle(self, 
                   algorithms: List[InsulinAlgorithm],
                   data: StandardDataPack,
                   metrics: List[str] = ['tir', 'gmi', 'cv']) -> BattleReport:
        """Compare algorithms head-to-head"""
```

#### 2.3 Standard Metrics (`src/analysis/clinical_metrics.py`)
```python
class ClinicalMetrics:
    """Standard metrics for algorithm comparison"""
    
    def calculate_tir(self, glucose_series) -> float:
        """Time-in-Range 70-180 mg/dL"""
        
    def calculate_time_in_tight_range(self, glucose_series) -> float:
        """Time-in-Range 70-140 mg/dL"""
```

### Implementation Tasks
1. Extend `InsulinAlgorithm` with `get_algorithm_metadata()`
2. Create `AlgorithmResult` dataclass with uncertainty
3. Build `BattleRunner` for multi-algorithm comparison
4. Implement clinical metrics: TIR, GMI, CV, hypoglycemia index
5. Add command: `iints.py --battle lstm pid hybrid --scenario standard_meal`

---

## Work Package 3: Reverse Engineering Library (Research Team)

### Objective
Build "Legacy Emulators" based on clinical whitepapers for Medtronic 780G, Tandem Control-IQ, etc. Enable comparison: "This is what current pumps do vs. where they fail."

### Deliverables

#### 3.1 Legacy Emulator Base (`src/emulation/legacy_base.py`)
```python
class LegacyEmulator(ABC):
    """Base class for commercial pump emulation"""
    
    @abstractmethod
    def get_safety_limits(self) -> SafetyLimits:
        """Return pump-specific safety constraints"""
        
    @abstractmethod
    def get_pid_parameters(self) -> PIDParameters:
        """Return PID tuning from whitepaper"""
```

#### 3.2 Medtronic 780G Emulator (`src/emulation/medtronic_780g.py`)
```python
class Medtronic780GEmulator(LegacyEmulator):
    """Emulates Medtronic 780G with SmartGuard"""
    # Based on: https://www.medtronicdiabetes.com/770g-780g-user-guide
    # PID settings from clinical studies
```

#### 3.3 Tandem Control-IQ Emulator (`src/emulation/tandem_controliq.py`)
```python
class TandemControlIQEmulator(LegacyEmulator):
    """Emulates Tandem Control-IQ algorithm"""
    # Based on: Brown et al. (2019) - Diabetes Technology & Therapeutics
```

#### 3.4 Whitepaper References (`docs/emulation_references.md`)
```markdown
# Commercial Algorithm References

## Medtronic 780G
- User Guide: [Link]
- Clinical Study: [Link]
- PID Parameters: ...

## Tandem Control-IQ  
- Brown et al. (2019)
- Clinical Outcomes: ...
```

### Implementation Tasks
1. Create `src/emulation/` directory structure
2. Build `LegacyEmulator` base class
3. Implement Medtronic 780G with whitepaper parameters
4. Implement Tandem Control-IQ with published settings
5. Add Omnipod 5 emulator
6. Create comparison visualization: "Legacy vs. New AI"

---

## Work Package 4: Clinical Control Center UI (Frontend Team)

### Objective
Build a professional medical cockpit with "X-Ray" visualization and "Uncertainty Cloud" - not a hobby app.

### Deliverables

#### 4.1 Reasoning Log Enhancement (`src/analysis/reasoning_log.py`)
```python
class ReasoningLog:
    """Enhanced clinical reasoning display"""
    
    def explain_decision(self, decision: AlgorithmResult) -> str:
        """Returns: '2 units delivered because glucose rising >2 mg/dL/min'"""
```

#### 4.2 Uncertainty Cloud Visualizer (`src/visualization/uncertainty_cloud.py`)
```python
class UncertaintyCloud:
    """Visualizes AI confidence as shadow around glucose line"""
    
    def plot_cloud(self, 
                   glucose: pd.Series, 
                   predictions: np.ndarray,
                   confidence: np.ndarray) -> Figure:
        """Create glucose plot with uncertainty envelope"""
```

#### 4.3 Dashboard Components (`src/visualization/cockpit.py`)
```python
class ClinicalCockpit:
    """Professional medical dashboard"""
    
    def render(self, simulation_data: pd.DataFrame) -> Dashboard:
        """Render full clinical control center"""
```

### Implementation Tasks
1. Enhance `WhyLogEntry` with clinical reasoning explanations
2. Create `UncertaintyCloud` visualization with confidence bands
3. Build `ClinicalCockpit` dashboard with:
   - Glucose trend with uncertainty cloud
   - Reasoning log panel
   - Algorithm personality metrics
   - Battle mode comparison
4. Add command: `iints.py --dashboard --battle`

---

## File Structure Changes

```
src/
 algorithm/
    base_algorithm.py      # ENHANCE: Add metadata
    battle_runner.py       # NEW: Multi-algorithm comparison
    ...
 analysis/
    algorithm_xray.py      # ENHANCE: Add battle mode
    clinical_metrics.py    # NEW: TIR, GMI, CV calculations
    explainable_ai.py      # KEEP: Audit trail
 data/
    adapter.py             # ENHANCE: Keep existing Ohio support
    universal_parser.py    # NEW: Generic CSV/JSON parser
    column_mapper.py       # NEW: Column alias mapping
    quality_checker.py     # NEW: Data quality scoring
 emulation/                 # NEW: Legacy pump emulators
    __init__.py
    legacy_base.py
    medtronic_780g.py
    tandem_controliq.py
    omnipod_5.py
 visualization/             # NEW: UI components
     __init__.py
     uncertainty_cloud.py
     cockpit.py
```

---

## CLI Commands (New)

```bash
# Data ingestion
python iints.py --import data.csv --format auto
python iints.py --validate my_dataset --quality_check

# Battle mode
python iints.py --battle lstm pid hybrid --scenario standard_meal
python iints.py --battle --visualize --report

# Legacy comparison
python iints.py --legacy medtronic_780g --compare_with lstm
python iints.py --emulate controliq --scenario unannounced_meal

# Dashboard
python iints.py --dashboard
python iints.py --dashboard --battle --legacy
```

---

## Testing Requirements

### Unit Tests
- [ ] `test_universal_parser.py` - Various CSV/JSON formats
- [ ] `test_quality_checker.py` - Gap detection accuracy
- [ ] `test_battle_runner.py` - Algorithm comparison logic
- [ ] `test_legacy_emulators.py` - Pump behavior accuracy

### Integration Tests
- [ ] End-to-end battle mode pipeline
- [ ] Data quality scoring in simulation
- [ ] Legacy emulator vs. new AI comparison

---

## Documentation Updates

1. **Update README.md** with new architecture diagram
2. **Create docs/ALGORITHM_TEMPLATE.md** for external contributors
3. **Create docs/LEGACY_EMULATOR_REFERENCES.md** with whitepaper citations
4. **Update docs/COMPREHENSIVE_GUIDE.md** with battle mode usage

---

## Timeline Estimate

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| 1 | Data Bridge | 2-3 days |
| 2 | Algorithm Engine | 3-4 days |
| 3 | Legacy Emulators | 4-5 days |
| 4 | UI/Cockpit | 3-4 days |
| 5 | Testing & Docs | 2-3 days |
| **Total** | | **14-19 days** |

---

## Success Criteria

1. [OK] Any CSV/JSON can be imported with auto-detection
2. [OK] Data quality warnings shown: "[WARN] 10% gap detected, confidence 0.72"
3. [OK] Multiple algorithms run simultaneously on identical data
4. [OK] Battle report shows: "LSTM TIR: 78% vs PID TIR: 65%"
5. [OK] Legacy emulators match published clinical behavior
6. [OK] Uncertainty cloud visible around glucose predictions
7. [OK] Reasoning log explains: "Why 2 units? Glucose rising >2 mg/dL/min"

---

## Next Steps

1. [OK] Plan approved by team
2. [ ] Create `src/emulation/` directory
3. [ ] Implement `src/data/universal_parser.py`
4. [ ] Implement `src/algorithm/battle_runner.py`
5. [ ] Implement `src/visualization/uncertainty_cloud.py`
6. [ ] Run battle mode demonstration
7. [ ] Validate legacy emulator accuracy