import numpy as np
from typing import Dict, Any, Optional
from .models import CustomPatientModel

try:
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.base import Action
    SIMGLUCOSE_AVAILABLE = True
except ImportError:
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
    def create_patient(patient_type='custom', patient_id=None, initial_glucose=120.0, **kwargs):
        """Create a patient model based on type."""
        if patient_type == 'custom':
            return CustomPatientModel(initial_glucose=initial_glucose, **kwargs)
        elif patient_type == 'simglucose':
            if not SIMGLUCOSE_AVAILABLE:
                print("Warning: Simglucose not available, falling back to custom model")
                return CustomPatientModel(initial_glucose=initial_glucose, **kwargs)
            
            patient_name = patient_id or PatientFactory.SIMGLUCOSE_PATIENTS[0]
            return SimglucosePatientWrapper(patient_name, initial_glucose)
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
                CustomPatientModel(initial_glucose=120, glucose_decay_rate=0.03), # Slow metabolism
                CustomPatientModel(initial_glucose=120, glucose_decay_rate=0.07), # Fast metabolism
            ]
        else:
            # Use FDA-approved virtual patients
            selected_patients = [
                'adolescent#001', 'adolescent#005', 'adult#001', 
                'adult#005', 'child#001', 'child#005'
            ]
            return [SimglucosePatientWrapper(name) for name in selected_patients]

class SimglucosePatientWrapper:
    """Wrapper for simglucose patients to match CustomPatientModel interface."""
    
    def __init__(self, patient_name='adolescent#001', initial_glucose=120.0):
        if not SIMGLUCOSE_AVAILABLE:
            raise ImportError("Simglucose not available")
            
        self.patient = T1DPatient.make(patient_name)
        self.sensor = CGMSensor.make()
        self.pump = InsulinPump.make()
        self.env = T1DSimEnv(patient=self.patient, sensor=self.sensor, pump=self.pump)
        self.patient_name = patient_name
        self.reset()
    
    def reset(self):
        """Reset the simglucose environment."""
        self.env.reset()
    
    def get_current_glucose(self):
        """Get current glucose in mg/dL."""
        return self.env.patient.BG * 18.0156
    
    def update(self, time_step, delivered_insulin, carb_intake=0.0, **kwargs):
        """Update patient state."""
        action = Action(basal=0.0, bolus=delivered_insulin, carb=carb_intake)
        obs, reward, done, info = self.env.step(action)
        self._last_info = info
        return obs[0] * 18.0156
    
    @property
    def insulin_on_board(self):
        """Get insulin on board."""
        if hasattr(self, '_last_info') and 'IOB' in self._last_info:
            return self._last_info['IOB']
        return getattr(self.env.patient, 'IOB', 0.0)
    
    @property
    def carbs_on_board(self):
        """Get carbs on board."""
        if hasattr(self, '_last_info') and 'COB' in self._last_info:
            return self._last_info['COB']
        return getattr(self.env.patient, 'COB', 0.0)
    
    def trigger_event(self, event_type, value):
        """Trigger stress events."""
        print(f"SimglucosePatient {self.patient_name}: {event_type} = {value}")
    
    def get_patient_state(self):
        """Get patient state for logging."""
        return {
            "current_glucose": self.get_current_glucose(),
            "insulin_on_board": self.insulin_on_board,
            "carbs_on_board": self.carbs_on_board,
            "patient_name": self.patient_name
        }