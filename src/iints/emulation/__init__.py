from iints.api.base_algorithm import InsulinAlgorithm

"""
IINTS-AF Legacy Emulation Module
Commercial insulin pump emulation for research and comparison.

This module provides emulators for major commercial insulin pumps,
allowing researchers to compare new algorithms against established
commercial systems and identify areas for improvement.

Part of the #WeAreNotWaiting movement for transparent diabetes tech.
"""

from .legacy_base import (
    LegacyEmulator,
    PumpBehavior,
    PIDParameters,
    SafetyLimits,
    SafetyLevel,
    EmulatorDecision
)

from .medtronic_780g import Medtronic780GEmulator, Medtronic780GBehavior

from .tandem_controliq import TandemControlIQEmulator, TandemControlIQBehavior

from .omnipod_5 import Omnipod5Emulator, Omnipod5Behavior

__all__ = [
    # Base classes
    'LegacyEmulator',
    'PumpBehavior',
    'PIDParameters',
    'SafetyLimits',
    'SafetyLevel',
    'EmulatorDecision',
    
    # Emulators
    'Medtronic780GEmulator',
    'Medtronic780GBehavior',
    'TandemControlIQEmulator',
    'TandemControlIQBehavior',
    'Omnipod5Emulator',
    'Omnipod5Behavior',
]

# Quick access to all emulators
EMULATORS = {
    'medtronic_780g': Medtronic780GEmulator,
    'tandem_controliq': TandemControlIQEmulator,
    'omnipod_5': Omnipod5Emulator,
}


def get_emulator(pump_type: str) -> InsulinAlgorithm:
    """
    Get an emulator instance by pump type.
    
    Args:
        pump_type: One of 'medtronic_780g', 'tandem_controliq', 'omnipod_5'
        
    Returns:
        LegacyEmulator instance
        
    Raises:
        ValueError: If pump_type is not recognized
    """
    if pump_type.lower() in EMULATORS:
        return EMULATORS[pump_type.lower()]() # type: ignore
    else:
        raise ValueError(
            f"Unknown pump type: {pump_type}. "
            f"Available: {list(EMULATORS.keys())}"
        )


def list_available_emulators() -> list:
    """List all available pump emulators"""
    return list(EMULATORS.keys())

