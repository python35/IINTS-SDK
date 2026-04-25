#!/usr/bin/env python3
"""
Industry-Standard PID Controller - IINTS-AF
Simple PID implementation for algorithm comparison
"""

import numpy as np
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput

class PIDController(InsulinAlgorithm):
    """Industry-standard PID controller for glucose management"""
    
    def __init__(self):
        super().__init__()
        
        # PID parameters (tuned for glucose control)
        self.kp = 0.1   # Proportional gain
        self.ki = 0.01  # Integral gain  
        self.kd = 0.05  # Derivative gain
        
        # Controller state
        self.integral = 0
        self.previous_error = 0
        self.target_glucose = 120  # mg/dL target
        self.integral_limit = 300.0  # Anti-windup clamp on accumulated error
        
        # Safety limits
        self.max_insulin = 5.0  # Maximum insulin dose
        self.min_insulin = 0.0  # Minimum insulin dose
        
    def predict_insulin(self, data: AlgorithmInput):
        self.why_log = [] # Clear the log for this prediction cycle

        # Calculate error from target
        error = data.current_glucose - self.target_glucose
        self._log_reason(f"Glucose error from target ({self.target_glucose} mg/dL)", "glucose_level", error, f"Current glucose: {data.current_glucose:.0f} mg/dL")

        # Derivative term (rate of change)
        derivative = error - self.previous_error
        self._log_reason("Derivative term calculated", "control_parameter", derivative)

        # Integral term (accumulated error) with anti-windup protection
        proposed_integral = max(-self.integral_limit, min(self.integral + error, self.integral_limit))
        unsaturated_output = (
            self.kp * error
            + self.ki * proposed_integral
            + self.kd * derivative
        )
        if (unsaturated_output > self.max_insulin and error > 0) or (
            unsaturated_output < self.min_insulin and error < 0
        ):
            self._log_reason(
                "Integral term held to prevent windup while output is saturated",
                "control_parameter",
                self.integral,
            )
        else:
            self.integral = proposed_integral
            self._log_reason("Integral term updated", "control_parameter", self.integral)
        
        # PID formula
        insulin_dose = (self.kp * error + 
                       self.ki * self.integral + 
                       self.kd * derivative)
        self._log_reason("Initial insulin dose calculated by PID", "insulin_calculation", insulin_dose)
        
        # Update previous error for next iteration
        self.previous_error = error
        
        # Apply safety constraints
        original_insulin_dose = insulin_dose
        insulin_dose = max(self.min_insulin, min(insulin_dose, self.max_insulin))
        if insulin_dose != original_insulin_dose:
            self._log_reason(f"Insulin dose adjusted due to safety limits (min: {self.min_insulin}, max: {self.max_insulin})", 
                             "safety_constraint", 
                             f"Original: {original_insulin_dose:.2f}, Adjusted: {insulin_dose:.2f}")
        else:
            self._log_reason("Insulin dose within safety limits", "safety_constraint", insulin_dose)
        
        return {
            'total_insulin_delivered': insulin_dose,
            'bolus_insulin': insulin_dose,
            'basal_insulin': 0
        }
    
    def reset(self):
        """Reset controller state for new patient"""
        super().reset()
        self.integral = 0
        self.previous_error = 0
    
    def get_algorithm_info(self):
        """Return algorithm information"""
        return {
            'name': 'Industry PID Controller',
            'type': 'Classical Control',
            'parameters': {
                'kp': self.kp,
                'ki': self.ki, 
                'kd': self.kd,
                'target_glucose': self.target_glucose
            },
            'description': 'Industry-standard PID controller for glucose regulation'
        }
