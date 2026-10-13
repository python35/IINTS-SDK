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
        
        # PID parameters (tuned for glucose control at the reference interval)
        self.kp = 0.1   # Proportional gain
        self.ki = 0.01  # Integral gain
        self.kd = 0.05  # Derivative gain

        # D3 (sample-time contract): the gains above were tuned for a 5-minute
        # control interval, and the terms used to be accumulated per *call*:
        # the integral grew by one error per step and the derivative was a
        # plain difference. The same gains then produced 24.35 U/h at dt = 1
        # min and 0.55 U/h at dt = 15 min. Both terms are now expressed per
        # reference interval, so the hourly dose is the same at any sample
        # time while the numbers below keep their tuned meaning at dt = 5.
        self.reference_interval_minutes = 5.0
        
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

        # Sample interval of this call, normalised against the tuning point.
        raw_interval = getattr(data, "time_step", None)
        interval = float(self.reference_interval_minutes if raw_interval is None
                         else raw_interval)
        if interval <= 0.0:
            raise ValueError("time_step must be positive for the PID controller")
        interval_ratio = interval / self.reference_interval_minutes

        # Derivative term: error change per reference interval
        derivative = (error - self.previous_error) / interval_ratio
        self._log_reason("Derivative term calculated", "control_parameter", derivative)

        # Integral term (accumulated error) with anti-windup protection
        # SCIENTIFIC UPGRADE: Pharmacokinetic (PK) Feed-Forward Integration
        # We dynamically drain the integral windup based on the actual Insulin On Board (mass balance).
        pk_feed_forward_inhibition = data.insulin_on_board * 2.0  # Tuning parameter for PK dampening

        # Integral and its PK drain both accumulate per reference interval.
        # Trapezoidal accumulation. A left-rectangle rule would leave a
        # residual sample-time dependence (about 6% per hour between dt = 5 and
        # dt = 15); the trapezoid rule removes it exactly for a constant error.
        mean_error = 0.5 * (error + self.previous_error)
        proposed_integral = max(
            -self.integral_limit,
            min(self.integral + (mean_error - pk_feed_forward_inhibition) * interval_ratio,
                self.integral_limit)
        )
        
        unsaturated_output = interval_ratio * (
            self.kp * error
            + self.ki * proposed_integral
            + self.kd * derivative
        )
        step_max_insulin = self.max_insulin * interval_ratio
        step_min_insulin = self.min_insulin * interval_ratio
        if (unsaturated_output > step_max_insulin and error > 0) or (
            unsaturated_output < step_min_insulin and error < 0
        ):
            self._log_reason(
                "Integral term held to prevent windup while output is saturated",
                "control_parameter",
                self.integral,
            )
        else:
            self.integral = proposed_integral
            self._log_reason(f"Integral term updated (dampened by {pk_feed_forward_inhibition:.2f} due to {data.insulin_on_board:.1f}U IOB)", "control_parameter", self.integral)
        
        # PID formula; the result is the dose for this interval, so the
        # per-reference-interval control signal is scaled by the interval.
        insulin_dose = interval_ratio * (self.kp * error +
                                         self.ki * self.integral +
                                         self.kd * derivative)
        self._log_reason("Initial insulin dose calculated by PID", "insulin_calculation", insulin_dose)
        
        # Update previous error for next iteration
        self.previous_error = error
        
        # Apply safety constraints
        original_insulin_dose = insulin_dose
        insulin_dose = max(step_min_insulin, min(insulin_dose, step_max_insulin))
        if insulin_dose != original_insulin_dose:
            self._log_reason(f"Insulin dose adjusted due to safety limits (min: {step_min_insulin}, max: {step_max_insulin})",
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
