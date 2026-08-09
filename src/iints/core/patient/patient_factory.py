import numpy as np
from typing import Dict, Any, Optional
from .models import CustomPatientModel

_BERGMAN_ONLY_RESEARCH_PARAMETERS = (
    "stem_cell_engraftment_percent",
    "stem_cell_subq_fraction",
    "immune_rejection_rate",
)


def _without_bergman_only_parameters(
    kwargs: Dict[str, Any],
    *,
    patient_type: str,
) -> Dict[str, Any]:
    """Remove disabled Bergman-only defaults or reject an unsupported experiment."""

    normalized = dict(kwargs)
    enabled = {
        name: normalized.get(name)
        for name in _BERGMAN_ONLY_RESEARCH_PARAMETERS
        if normalized.get(name) not in (None, 0, 0.0)
    }
    if enabled:
        names = ", ".join(sorted(enabled))
        raise ValueError(
            f"Patient model '{patient_type}' does not support Bergman stem-cell "
            f"research parameters: {names}. Select patient_type='bergman'."
        )
    for name in _BERGMAN_ONLY_RESEARCH_PARAMETERS:
        normalized.pop(name, None)
    return normalized

try:
    from .bergman_model import BergmanPatientModel
    BERGMAN_AVAILABLE = True
except Exception:
    BergmanPatientModel = None  # type: ignore[assignment,misc]
    BERGMAN_AVAILABLE = False

try:
    from .hovorka_model import HovorkaPatientModel
    HOVORKA_AVAILABLE = True
except Exception:
    HovorkaPatientModel = None  # type: ignore[assignment,misc]
    HOVORKA_AVAILABLE = False


try:
    from .advanced_metabolic_model import AdvancedMetabolicModel
    ADVANCED_METABOLIC_AVAILABLE = True
except Exception:
    AdvancedMetabolicModel = None  # type: ignore[assignment,misc]
    ADVANCED_METABOLIC_AVAILABLE = False

try:
    from simglucose.patient.t1dpatient import Action as PatientAction
    from simglucose.patient.t1dpatient import T1DPatient
    SIMGLUCOSE_AVAILABLE = True
except ImportError:
    PatientAction = None  # type: ignore[assignment,misc]
    T1DPatient = None  # type: ignore[assignment,misc]
    SIMGLUCOSE_AVAILABLE = False

class PatientFactory:
    """Factory for creating different types of patient models."""
    
    SIMGLUCOSE_PATIENTS = [
        'adolescent#001', 'adolescent#002', 'adolescent#003', 'adolescent#004', 'adolescent#005',
        'adolescent#006', 'adolescent#007', 'adolescent#008', 'adolescent#009', 'adolescent#010',
        'adult#001', 'adult#002', 'adult#003', 'adult#004', 'adult#005',
        'adult#006', 'adult#007', 'adult#008', 'adult#009', 'adult#010',
        'child#001', 'child#002', 'child#003', 'child#004', 'child#005',
        'child#006', 'child#007', 'child#008', 'child#009', 'child#010'
    ]
    
    @staticmethod
    def create_patient(patient_type='auto', patient_id=None, initial_glucose=120.0, **kwargs):
        """Create a patient model based on type."""
        if patient_type == 'auto':
            if BERGMAN_AVAILABLE and BergmanPatientModel is not None:
                return BergmanPatientModel(initial_glucose=initial_glucose, **kwargs)
            if SIMGLUCOSE_AVAILABLE:
                simglucose_kwargs = _without_bergman_only_parameters(
                    kwargs, patient_type='simglucose'
                )
                patient_name = patient_id or PatientFactory.SIMGLUCOSE_PATIENTS[0]
                return SimglucosePatientWrapper(
                    patient_name, initial_glucose, **simglucose_kwargs
                )
            custom_kwargs = _without_bergman_only_parameters(kwargs, patient_type='custom')
            return CustomPatientModel(initial_glucose=initial_glucose, **custom_kwargs)
        if patient_type == 'custom':
            custom_kwargs = _without_bergman_only_parameters(kwargs, patient_type='custom')
            return CustomPatientModel(initial_glucose=initial_glucose, **custom_kwargs)
        elif patient_type == 'bergman':
            if not BERGMAN_AVAILABLE or BergmanPatientModel is None:
                raise ImportError("Bergman model requested but its dependencies are unavailable")
            return BergmanPatientModel(initial_glucose=initial_glucose, **kwargs)
        elif patient_type in {'advanced', 'advanced_metabolic'}:
            advanced_kwargs = _without_bergman_only_parameters(kwargs, patient_type='advanced')
            if not ADVANCED_METABOLIC_AVAILABLE or AdvancedMetabolicModel is None:
                raise ImportError(
                    "Advanced metabolic model requested but its dependencies are unavailable"
                )
            return AdvancedMetabolicModel(initial_glucose=initial_glucose, **advanced_kwargs)
        elif patient_type == 'hovorka':
            hovorka_kwargs = _without_bergman_only_parameters(kwargs, patient_type='hovorka')
            if not HOVORKA_AVAILABLE or HovorkaPatientModel is None:
                raise ImportError("Hovorka model requested but its dependencies are unavailable")
            return HovorkaPatientModel(initial_glucose=initial_glucose, **hovorka_kwargs)
        elif patient_type == 'simglucose':
            if not SIMGLUCOSE_AVAILABLE:
                raise ImportError("simglucose patient requested but simglucose is unavailable")

            simglucose_kwargs = _without_bergman_only_parameters(
                kwargs, patient_type='simglucose'
            )
            
            patient_name = patient_id or PatientFactory.SIMGLUCOSE_PATIENTS[0]
            return SimglucosePatientWrapper(
                patient_name, initial_glucose, **simglucose_kwargs
            )
        else:
            raise ValueError(f"Unknown patient type: {patient_type}")
    
    @staticmethod
    def get_patient_diversity_set():
        """Get a diverse set of patients for population studies."""
        if not SIMGLUCOSE_AVAILABLE:
            # Create diverse custom patients with different parameters
                return [
                    CustomPatientModel(initial_glucose=120, insulin_sensitivity=40),  # High sensitivity
                    CustomPatientModel(initial_glucose=120, insulin_sensitivity=60),  # Low sensitivity
                    CustomPatientModel(initial_glucose=120, carb_factor=8),          # Fast carb absorption
                    CustomPatientModel(initial_glucose=120, carb_factor=12),         # Slow carb absorption
                    CustomPatientModel(initial_glucose=120, glucose_decay_rate=0.03), # Slower homeostatic drift
                    CustomPatientModel(initial_glucose=120, glucose_decay_rate=0.07), # Faster homeostatic drift
                ]
        else:
            # Use the open-source simglucose virtual-patient implementation.
            selected_patients = [
                'adolescent#001', 'adolescent#005', 'adult#001', 
                'adult#005', 'child#001', 'child#005'
            ]
            return [SimglucosePatientWrapper(name) for name in selected_patients]

class SimglucosePatientWrapper:
    """Wrapper for simglucose patients to match CustomPatientModel interface."""
    
    def __init__(
        self,
        patient_name='adolescent#001',
        initial_glucose=120.0,
        basal_insulin_rate: float = 0.8,
        insulin_sensitivity: float = 50.0,
        carb_factor: float = 10.0,
        insulin_action_duration: float = 300.0,
        carb_absorption_duration_minutes: float = 240.0,
        **kwargs: Any,
    ):
        if not SIMGLUCOSE_AVAILABLE:
            raise ImportError("Simglucose not available")
        if kwargs:
            raise TypeError(
                f"Unsupported simglucose patient parameters: {sorted(kwargs)}"
            )
        numeric = {
            "initial_glucose": initial_glucose,
            "basal_insulin_rate": basal_insulin_rate,
            "insulin_sensitivity": insulin_sensitivity,
            "carb_factor": carb_factor,
            "insulin_action_duration": insulin_action_duration,
            "carb_absorption_duration_minutes": carb_absorption_duration_minutes,
        }
        if not all(np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("simglucose wrapper parameters must all be finite")
        if float(initial_glucose) < 20.0:
            raise ValueError("initial_glucose must be at least 20 mg/dL")
        if float(basal_insulin_rate) < 0.0:
            raise ValueError("basal_insulin_rate must be non-negative")
        for name in (
            "insulin_sensitivity",
            "carb_factor",
            "insulin_action_duration",
            "carb_absorption_duration_minutes",
        ):
            if float(numeric[name]) <= 0.0:
                raise ValueError(f"{name} must be positive")
            
        factory = getattr(T1DPatient, "withName", None) or getattr(T1DPatient, "make", None)
        if factory is None:
            raise RuntimeError("Unsupported simglucose version: no patient-name factory found")
        self.patient = factory(patient_name)
        self.patient_name = patient_name
        self.requested_initial_glucose = float(initial_glucose)
        self.basal_insulin_rate = float(basal_insulin_rate)
        self.insulin_sensitivity = float(insulin_sensitivity)
        self.carb_factor = float(carb_factor)
        self.insulin_action_duration = float(insulin_action_duration)
        self.carb_absorption_duration_minutes = float(carb_absorption_duration_minutes)
        self.reset()
    
    def reset(self):
        """Reset the simglucose environment."""
        self.patient.reset()
        self._last_info: Dict[str, Any] = {}
        self._active_insulin_doses: list[Dict[str, float]] = []
        self._active_carb_intakes: list[Dict[str, float]] = []
        self._insulin_on_board = 0.0
        self._carbs_on_board = 0.0
        self._last_unsupported_event: Optional[Dict[str, Any]] = None
    
    def get_current_glucose(self):
        """Get current glucose in mg/dL."""
        return float(self.patient.observation.Gsub)
    
    def update(self, time_step, delivered_insulin, carb_intake=0.0, **kwargs):
        """Update patient state."""
        delivered_glucagon = float(kwargs.pop("delivered_glucagon_mg", 0.0))
        current_time = kwargs.pop("current_time", None)
        if kwargs:
            raise TypeError(f"Unsupported simglucose update fields: {sorted(kwargs)}")
        if not np.isfinite(delivered_insulin) or float(delivered_insulin) < 0.0:
            raise ValueError("delivered_insulin must be finite and non-negative")
        if not np.isfinite(carb_intake) or float(carb_intake) < 0.0:
            raise ValueError("carb_intake must be finite and non-negative")
        if not np.isfinite(delivered_glucagon) or delivered_glucagon < 0.0:
            raise ValueError("delivered_glucagon_mg must be finite and non-negative")
        if delivered_glucagon > 0.0:
            raise NotImplementedError(
                "The simglucose wrapper does not implement glucagon PK/PD"
            )
        if current_time is not None and not np.isfinite(current_time):
            raise ValueError("current_time must be finite when supplied")
        minutes = int(round(float(time_step)))
        if minutes <= 0 or not np.isclose(float(time_step), float(minutes)):
            raise ValueError("simglucose requires a positive whole-minute time step")
        insulin_rate_u_per_min = float(delivered_insulin) / minutes
        announced_meal_g = float(carb_intake)
        for minute in range(minutes):
            action = PatientAction(
                CHO=announced_meal_g if minute == 0 else 0.0,
                insulin=insulin_rate_u_per_min,
            )
            self.patient.step(action)

        if delivered_insulin > 0.0:
            self._active_insulin_doses.append(
                {"amount": float(delivered_insulin), "age": 0.0}
            )
        for dose in self._active_insulin_doses:
            dose["age"] += float(time_step)
        self._active_insulin_doses = [
            dose
            for dose in self._active_insulin_doses
            if dose["age"] <= self.insulin_action_duration
        ]
        self._insulin_on_board = sum(
            dose["amount"]
            * max(0.0, 1.0 - dose["age"] / self.insulin_action_duration)
            for dose in self._active_insulin_doses
        )

        if announced_meal_g > 0.0:
            self._active_carb_intakes.append(
                {"amount": announced_meal_g, "age": 0.0}
            )
        for meal in self._active_carb_intakes:
            meal["age"] += float(time_step)
        self._active_carb_intakes = [
            meal
            for meal in self._active_carb_intakes
            if meal["age"] <= self.carb_absorption_duration_minutes
        ]
        self._carbs_on_board = sum(
            meal["amount"]
            * max(0.0, 1.0 - meal["age"] / self.carb_absorption_duration_minutes)
            for meal in self._active_carb_intakes
        )
        return self.get_current_glucose()
    
    @property
    def insulin_on_board(self):
        """Get insulin on board."""
        return self._insulin_on_board
    
    @property
    def carbs_on_board(self):
        """Get carbs on board."""
        return self._carbs_on_board
    
    def trigger_event(self, event_type, value):
        """Record unsupported events without pretending they changed physiology."""
        self._last_unsupported_event = {
            "event_type": str(event_type),
            "value": value,
            "applied": False,
            "reason": "not_supported_by_simglucose_adapter",
        }
    
    def get_patient_state(self):
        """Get patient state for logging."""
        return {
            "current_glucose": self.get_current_glucose(),
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "patient_name": self.patient_name,
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
            "requested_initial_glucose_mgdl": self.requested_initial_glucose,
            "initial_glucose_source": "simglucose_native_state",
            "iob_cob_source": "iints_deterministic_adapter_estimate",
            "last_unsupported_event": self._last_unsupported_event,
        }

    def get_ratio_state(self) -> Dict[str, float]:
        return {
            "basal_rate_u_per_hr": self.basal_insulin_rate,
            "isf": self.insulin_sensitivity,
            "icr": self.carb_factor,
            "dia_minutes": self.insulin_action_duration,
        }

    def set_ratio_state(
        self,
        isf: Optional[float] = None,
        icr: Optional[float] = None,
        basal_rate: Optional[float] = None,
        dia_minutes: Optional[float] = None,
    ) -> None:
        if isf is not None:
            self.insulin_sensitivity = float(isf)
        if icr is not None:
            self.carb_factor = float(icr)
        if basal_rate is not None:
            self.basal_insulin_rate = float(basal_rate)
        if dia_minutes is not None:
            self.insulin_action_duration = float(dia_minutes)
