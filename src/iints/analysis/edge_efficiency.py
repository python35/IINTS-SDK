from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyEstimate:
    energy_joules: float
    energy_microjoules: float
    decisions_per_day: int
    energy_joules_per_day: float
    energy_millijoules_per_day: float


def estimate_energy_per_decision(power_watts: float, latency_ms: float, decisions_per_day: int = 288) -> EnergyEstimate:
    """
    Estimate energy cost per decision on any device.

    Energy (J) = Power (W) * Time (s)

    decisions_per_day default: 288 (5-minute control loop)
    """
    latency_s = max(latency_ms, 0.0) / 1000.0
    energy_j = max(power_watts, 0.0) * latency_s
    energy_uj = energy_j * 1_000_000.0
    energy_day_j = energy_j * max(int(decisions_per_day), 1)
    return EnergyEstimate(
        energy_joules=energy_j,
        energy_microjoules=energy_uj,
        decisions_per_day=max(int(decisions_per_day), 1),
        energy_joules_per_day=energy_day_j,
        energy_millijoules_per_day=energy_day_j * 1_000.0,
    )
