from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AlgorithmInput:
    """Dataclass for inputs to the insulin prediction algorithm."""
    current_glucose: float
    time_step: float
    insulin_on_board: float = 0.0
    carb_intake: float = 0.0
    patient_state: Dict[str, Any] = field(default_factory=dict)
    current_time: float = 0.0 # Added current_time
    carbs_on_board: float = 0.0
    isf: Optional[float] = None
    icr: Optional[float] = None
    dia_minutes: Optional[float] = None
    basal_rate_u_per_hr: Optional[float] = None
    glucose_trend_mgdl_min: Optional[float] = None
    predicted_glucose_30min: Optional[float] = None


@dataclass
class AlgorithmMetadata:
    """Metadata for algorithm registration and identification"""
    name: str
    version: str = "1.0.0"
    author: str = "IINTS-AF Team"
    paper_reference: Optional[str] = None
    description: str = ""
    algorithm_type: str = "rule_based"  # 'rule_based', 'ml', 'hybrid'
    requires_training: bool = False
    supported_scenarios: List[str] = field(default_factory=lambda: [
        'standard_meal', 'unannounced_meal', 'exercise', 
        'stress', 'hypoglycemia', 'hyperglycemia'
    ])
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'paper_reference': self.paper_reference,
            'description': self.description,
            'algorithm_type': self.algorithm_type,
            'requires_training': self.requires_training,
            'supported_scenarios': self.supported_scenarios
        }


@dataclass
class AlgorithmResult:
    """Result of an insulin prediction with uncertainty"""
    total_insulin_delivered: float
    bolus_insulin: float = 0.0
    basal_insulin: float = 0.0
    correction_bolus: float = 0.0
    meal_bolus: float = 0.0
    
    # Uncertainty quantification
    uncertainty: float = 0.0  # 0.0 = certain, 1.0 = very uncertain
    confidence_interval: tuple = (0.0, 0.0)  # (lower, upper)
    
    # Clinical reasoning
    primary_reason: str = ""
    secondary_reasons: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    
    # Metadata
    algorithm_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'total_insulin_delivered': self.total_insulin_delivered,
            'bolus_insulin': self.bolus_insulin,
            'basal_insulin': self.basal_insulin,
            'correction_bolus': self.correction_bolus,
            'meal_bolus': self.meal_bolus,
            'uncertainty': self.uncertainty,
            'confidence_interval': self.confidence_interval,
            'primary_reason': self.primary_reason,
            'secondary_reasons': self.secondary_reasons,
            'safety_constraints': self.safety_constraints,
            'algorithm_name': self.algorithm_name,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class WhyLogEntry:
    """Single entry in the Why Log explaining a decision reason"""
    reason: str
    category: str  # 'glucose_level', 'velocity', 'insulin_on_board', 'safety', 'context'
    value: Any = None
    clinical_impact: str = ""

    def to_dict(self) -> Dict:
        return {
            'reason': self.reason,
            'category': self.category,
            'value': self.value,
            'clinical_impact': self.clinical_impact
        }

class InsulinAlgorithm(ABC):
    """
    Abstract base class for insulin delivery algorithms.

    All specific insulin algorithms used in the simulation framework should
    inherit from this class and implement its abstract methods.
    
    This class supports the Plug-and-Play architecture, allowing any algorithm
    to be registered and compared in Battle Mode.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """
        Initializes the algorithm with its specific settings.

        Args:
            settings (Dict[str, Any]): A dictionary of algorithm-specific parameters.
        """
        self.settings = settings if settings is not None else {}
        self.state: Dict[str, Any] = {}  # To store internal algorithm state across calls
        self.why_log: List[WhyLogEntry] = []  # Decision reasoning log
        self._metadata: Optional[AlgorithmMetadata] = None  # Lazy-loaded metadata
        self.isf = self.settings.get('isf', 50.0)  # Default ISF
        self.icr = self.settings.get('icr', 10.0)  # Default ICR

    def set_isf(self, isf: float):
        """Set the Insulin Sensitivity Factor (mg/dL per unit)."""
        if isf <= 0:
            raise ValueError("ISF must be a positive value.")
        self.isf = isf
        
    def set_icr(self, icr: float):
        """Set the Insulin-to-Carb Ratio (grams per unit)."""
        if icr <= 0:
            raise ValueError("ICR must be a positive value.")
        self.icr = icr
        
    def get_algorithm_metadata(self) -> AlgorithmMetadata:
        """
        Get algorithm metadata. Override in subclasses for custom metadata.
        
        Returns:
            AlgorithmMetadata: Information about the algorithm for registration
        """
        if self._metadata is None:
            # Default metadata - override in subclasses
            self._metadata = AlgorithmMetadata(
                name=self.__class__.__name__,
                version="1.0.0",
                author="IINTS-AF Team",
                description=f"Insulin algorithm: {self.__class__.__name__}",
                algorithm_type="rule_based"
            )
        return self._metadata
    
    def set_algorithm_metadata(self, metadata: AlgorithmMetadata):
        """Set custom algorithm metadata"""
        self._metadata = metadata
    
    def calculate_uncertainty(self, data: AlgorithmInput) -> float:
        """
        Calculate uncertainty score for the current prediction.
        
        Override in subclasses to implement custom uncertainty quantification.
        
        Args:
            data: Current algorithm input
            
        Returns:
            float: Uncertainty score between 0.0 (certain) and 1.0 (very uncertain)
        """
        # Default: low uncertainty for rule-based algorithms
        return 0.1
    
    def calculate_confidence_interval(self, 
                                      data: AlgorithmInput,
                                      prediction: float,
                                      uncertainty: float) -> tuple:
        """
        Calculate confidence interval for the prediction.
        
        Args:
            data: Current algorithm input
            prediction: Predicted insulin dose
            uncertainty: Uncertainty score
            
        Returns:
            tuple: (lower_bound, upper_bound) for the prediction
        """
        # Default: ±20% of prediction based on uncertainty
        margin = prediction * 0.2 * (1 + uncertainty)
        return (max(0, prediction - margin), prediction + margin)
    
    def explain_prediction(self, 
                          data: AlgorithmInput, 
                          prediction: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for the prediction.
        
        This is used for the Reasoning Log in the Clinical Control Center.
        
        Args:
            data: Algorithm input
            prediction: Prediction dictionary
            
        Returns:
            str: Explanation like "2 units delivered because glucose rising >2 mg/dL/min"
        """
        reasons = []
        
        # Glucose level reasoning
        if data.current_glucose < 70:
            reasons.append(f"glucose critically low at {data.current_glucose:.0f} mg/dL")
        elif data.current_glucose < 100:
            reasons.append(f"glucose approaching low at {data.current_glucose:.0f} mg/dL")
        elif data.current_glucose > 180:
            reasons.append(f"glucose elevated at {data.current_glucose:.0f} mg/dL")
        else:
            reasons.append(f"glucose in target range at {data.current_glucose:.0f} mg/dL")
        
        # Insulin on board
        if data.insulin_on_board > 2.0:
            reasons.append(f"high insulin on board ({data.insulin_on_board:.1f} U)")
        elif data.insulin_on_board > 0.5:
            reasons.append(f"moderate insulin on board ({data.insulin_on_board:.1f} U)")
        
        # Carbs
        if data.carb_intake > 0:
            reasons.append(f"meal detected ({data.carb_intake:.0f}g carbs)")
        
        if prediction.get('total_insulin_delivered', 0) > 0:
            return f"{prediction['total_insulin_delivered']:.2f} units delivered because " + ", ".join(reasons)
        else:
            return f"No insulin delivered: " + ", ".join(reasons)

    @abstractmethod
    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        """
        Calculates the insulin dose based on current physiological data.
        This is the primary method to be implemented by custom algorithms.

        Args:
            data (AlgorithmInput): A dataclass containing all relevant data for the decision.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated insulin doses.
                            A key 'total_insulin_delivered' is expected by the simulator.
                            (e.g., {'total_insulin_delivered': 1.5, 'bolus_insulin': 1.0, 'basal_insulin': 0.5})
        """
        # Clear why_log at start of each calculation
        self.why_log = []
        raise NotImplementedError("Subclasses must implement predict_insulin method")
    
    def _log_reason(self, reason: str, category: str, value: Any = None, clinical_impact: str = ""):
        """Helper method to add reasoning to why_log"""
        entry = WhyLogEntry(
            reason=reason,
            category=category,
            value=value,
            clinical_impact=clinical_impact
        )
        self.why_log.append(entry)
    
    def get_why_log(self) -> List[WhyLogEntry]:
        """Get the decision reasoning log for the last calculation"""
        return self.why_log
    
    def get_why_log_text(self) -> str:
        """Get human-readable why log"""
        if not self.why_log:
            return "No decision reasoning available"
        
        text = "WHY_LOG:\n"
        for entry in self.why_log:
            text += f"- {entry.reason}"
            if entry.value is not None:
                text += f" (value: {entry.value})"
            if entry.clinical_impact:
                text += f" → {entry.clinical_impact}"
            text += "\n"
        return text

    def reset(self):
        """
        Resets the algorithm's internal state.
        This should be called at the start of each new simulation run.
        """
        self.state = {}
        self.why_log = []

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current internal state of the algorithm.
        """
        return self.state

    def set_state(self, state: Dict[str, Any]):
        """
        Sets the internal state of the algorithm.
        """
        self.state = state
