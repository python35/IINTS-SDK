import pytest
import pandas as pd
import numpy as np
from iints.emulation.medtronic_780g import Medtronic780GEmulator
from iints.emulation.tandem_controliq import TandemControlIQEmulator
from iints.emulation.legacy_base import EmulatorDecision, PumpBehavior, PIDParameters, SafetyLimits, SafetyLevel

# Helper function to create a basic emulator state for testing
def setup_emulator_state(emulator):
    emulator.state = {
        'previous_glucose': 120.0,
        'integral_term': 0.0,
        'cumulative_insulin': 0.0,
        'velocity': 0.0,
        'last_delivery_time': 0.0
    }
    emulator._decision_history = []


# --- Medtronic 780G Tests ---

def test_medtronic_780g_auto_correction_high_glucose():
    """
    Test that Medtronic 780G delivers an auto-correction micro-bolus when glucose is high and rising.
    """
    emulator = Medtronic780GEmulator()
    setup_emulator_state(emulator)

    # High glucose, rising, no carbs
    decision = emulator.emulate_decision(glucose=180, velocity=1.0, insulin_on_board=0.5, carbs=0, current_time=5)
    
    assert decision.action == 'deliver'
    assert decision.insulin_delivered > 0.0 # Expect a small micro-bolus
    assert "Auto-correction micro-bolus" in decision.reasoning[0]
    assert "PLGS activated" not in str(decision.reasoning) # Should not activate PLGS

def test_medtronic_780g_plgs_low_glucose():
    """
    Test that Medtronic 780G PLGS suspends insulin when glucose is predicted to be low and falling.
    """
    emulator = Medtronic780GEmulator()
    setup_emulator_state(emulator)
    
    # Glucose falling rapidly, predicted low
    # velocity of -1.5 mg/dL/min over 30 min PLGS window = -45 mg/dL.
    # 70 mg/dL - 45 mg/dL = 25 mg/dL predicted
    decision = emulator.emulate_decision(glucose=70, velocity=-1.5, insulin_on_board=1.0, carbs=0, current_time=5)
    
    assert decision.action == 'suspend_insulin'
    assert decision.insulin_delivered == 0.0
    assert any("PLGS activated" in r or "Rapid fall detected" in r for r in decision.reasoning)

def test_medtronic_780g_max_bolus_respected():
    """
    Test that Medtronic 780G respects the max bolus limit.
    """
    emulator = Medtronic780GEmulator()
    setup_emulator_state(emulator)
    
    # Very high glucose, rising fast, to trigger large bolus
    # Max bolus is 10.0 U by default in Medtronic780GBehavior
    decision = emulator.emulate_decision(glucose=300, velocity=5.0, insulin_on_board=0.0, carbs=0, current_time=5)
    
    assert decision.insulin_delivered <= emulator.behavior.safety_limits.max_bolus + 1e-6 # Allow for float precision

# --- Tandem Control-IQ Tests ---

def test_tandem_controliq_phgs_high_glucose():
    """
    Test that Tandem Control-IQ PHGS increases insulin when glucose is predicted to be high and rising.
    """
    emulator = TandemControlIQEmulator()
    setup_emulator_state(emulator)
    
    # High glucose, rising, predicted high in 60 min
    # velocity of 2.0 mg/dL/min over 60 min prediction = +120 mg/dL.
    # 180 mg/dL + 120 mg/dL = 300 mg/dL predicted
    glucose = 180
    velocity = 2.0
    
    # First, test without PHGS activation (e.g. glucose rising, but not predicted over high limit)
    decision_no_phgs = emulator.emulate_decision(glucose=150, velocity=0.5, insulin_on_board=0.0, carbs=0, current_time=5)
    expected_insulin_no_phgs = decision_no_phgs.insulin_delivered
    
    setup_emulator_state(emulator) # Reset state 
    
    # Now with PHGS activation
    decision_with_phgs = emulator.emulate_decision(glucose=glucose, velocity=velocity, insulin_on_board=0.0, carbs=0, current_time=5)
    
    assert "PHGS activated" in str(decision_with_phgs.reasoning)
    assert decision_with_phgs.insulin_delivered > expected_insulin_no_phgs # Expect more insulin due to PHGS aggressiveness

def test_tandem_controliq_plgs_low_glucose():
    """
    Test that Tandem Control-IQ PLGS suspends insulin when glucose is predicted to be low and falling.
    """
    emulator = TandemControlIQEmulator()
    setup_emulator_state(emulator)
    
    # Glucose falling rapidly, predicted low
    # velocity of -1.5 mg/dL/min over 30 min PLGS window = -45 mg/dL.
    # 70 mg/dL - 45 mg/dL = 25 mg/dL predicted
    decision = emulator.emulate_decision(glucose=70, velocity=-1.5, insulin_on_board=1.0, carbs=0, current_time=5)
    
    assert decision.action == 'suspend_insulin'
    assert decision.insulin_delivered == 0.0
    assert any("PLGS activated" in r or "Rapid fall detected" in r for r in decision.reasoning)

def test_tandem_controliq_range_based_correction():
    """
    Test that Tandem Control-IQ shows different insulin delivery at 112.5 mg/dL vs 160 mg/dL.
    """
    emulator = TandemControlIQEmulator()
    setup_emulator_state(emulator)

    # Test at 112.5 mg/dL (near target)
    decision_at_target = emulator.emulate_decision(glucose=112.5, velocity=0.0, insulin_on_board=0.0, carbs=0, current_time=5)
    
    # Test at 160 mg/dL (elevated, should trigger more correction)
    setup_emulator_state(emulator) # Reset state for independent test
    decision_elevated = emulator.emulate_decision(glucose=160, velocity=0.0, insulin_on_board=0.0, carbs=0, current_time=5)
    
    assert decision_elevated.insulin_delivered > decision_at_target.insulin_delivered, \
        "Control-IQ should deliver more insulin at 160 mg/dL than at 112.5 mg/dL."

def test_tandem_controliq_max_delivery_respected():
    """
    Test that Tandem Control-IQ respects the max_delivery limit for auto delivery.
    """
    emulator = TandemControlIQEmulator()
    setup_emulator_state(emulator)
    
    # Very high glucose, rising fast, to trigger large auto delivery
    # max_delivery is 3.0 U/hr by default in TandemControlIQBehavior.
    # For a 5-min interval, max delivery would be 3.0 * (5/60) = 0.25 U
    decision = emulator.emulate_decision(glucose=300, velocity=5.0, insulin_on_board=0.0, carbs=0, current_time=5)
    
    expected_max_per_interval = emulator.behavior.max_delivery * (emulator.behavior.time_step_minutes / 60.0)
    assert decision.insulin_delivered <= expected_max_per_interval + 1e-6 # Allow for float precision
