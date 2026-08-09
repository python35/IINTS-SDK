import pytest
from iints.core.supervisor import IndependentSupervisor
from iints.core.devices.models import PumpModel
from iints.core.algorithms.pid_controller import PIDController
from iints.api.base_algorithm import AlgorithmInput

def test_supervisor_iob_headroom_limit():
    """
    Test that the safety supervisor enforces its configured IOB headroom.
    """
    supervisor = IndependentSupervisor(max_iob=4.0)
    
    # Under a linear/naive heuristic, if current_iob = 3.5 and proposed = 1.0, 
    # it might block completely or clamp linearly.
    # The new logic explicitly calculates: max_safe_bolus = max(0, max_iob - current_iob)
    # So if IOB is 3.5, the max safe bolus should be clamped precisely to 0.5.
    
    result = supervisor.evaluate_safety(
        current_glucose=160.0,
        proposed_insulin=1.0,
        current_time=0.0,
        current_iob=3.5
    )
    
    assert result["approved_insulin"] == 0.5
    assert "IOB_HEADROOM_LIMIT" in result["safety_reason"]

def test_supervisor_linear_trajectory_guard():
    """
    Test that the supervisor blocks insulin under a linear trend projection.
    """
    supervisor = IndependentSupervisor(severe_hypoglycemia_threshold=54.0)
    
    # Simulate a fast drop: 100 mg/dL dropping at -2.0 mg/dL/min
    supervisor.evaluate_safety(current_glucose=104.0, proposed_insulin=0.0, current_time=0.0)
    supervisor.evaluate_safety(current_glucose=102.0, proposed_insulin=0.0, current_time=1.0)
    
    # At t=2, glucose is 100. Rate is -2.0.
    # 30-min momentum = 100 + (-2.0 * 30) = 40.0 mg/dL
    # 40.0 is below the 54.0 severe hypo threshold, so the trajectory guard fires.
    
    result = supervisor.evaluate_safety(
        current_glucose=100.0,
        proposed_insulin=1.0,
        current_time=2.0
    )
    
    assert result["approved_insulin"] == 0.0
    assert "LINEAR_TRAJECTORY_GUARD" in result["safety_reason"]

def test_pid_pharmacokinetic_feedforward():
    """
    Test that the PID Controller integrates Pharmacokinetic (PK) Feed-Forward
    to prevent integral windup mathematically based on active insulin (IOB).
    """
    pid = PIDController()
    pid.target_glucose = 120.0
    pid.ki = 0.1  # Make Integral gain high for testing
    
    # Setup data with moderate error (so PID doesn't saturate) but ZERO IOB
    data_no_iob = AlgorithmInput(current_glucose=130.0, time_step=5.0, insulin_on_board=0.0)
    result_no_iob = pid.predict_insulin(data_no_iob)
    
    # Store the calculated integral
    integral_no_iob = pid.integral
    
    # Reset PID
    pid.reset()
    
    # Setup data with same error but HUGE IOB
    data_with_iob = AlgorithmInput(current_glucose=130.0, time_step=5.0, insulin_on_board=10.0)
    result_with_iob = pid.predict_insulin(data_with_iob)
    
    # The integral term should be significantly suppressed due to PK feed-forward
    integral_with_iob = pid.integral
    
    assert integral_with_iob < integral_no_iob

def test_pump_microstepper_quantization():
    """
    Test that a quantized pump abstraction emits only whole delivery steps.
    """
    pump = PumpModel(quantization_units=0.05, delivery_noise_std=0.15, seed=42)
    
    # Request 0.075 units. 
    # A true micro-stepper will either round down to 0.05 (1 step) or up to 0.10 (2 steps)
    # A generic step error may alter the number of steps.
    delivery = pump.deliver(requested_units=0.075, time_step_minutes=5.0)
    
    # The final delivered units must be an EXACT multiple of quantization_units (0.05)
    # It cannot be a floating point Gaussian value like 0.081234
    
    # Because of binary float precision, compare to the nearest integer micro-step.
    assert abs(round(delivery.delivered_units / 0.05) * 0.05 - delivery.delivered_units) < 1e-9
