"""
Population Generator — IINTS-AF
================================
Generates a virtual population of N patients with physiological variation
around a base patient profile.  Each parameter is drawn from a configurable
distribution (truncated normal or log-normal) whose bounds respect the
clinically valid ranges defined in the SDK validation schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from iints.core.patient.profile import PatientProfile


@dataclass
class ParameterDistribution:
    """Distribution specification for a single patient parameter."""

    mean: float
    cv: float  # coefficient of variation (0–1), e.g. 0.20 = 20 %
    distribution: str = "truncated_normal"  # "truncated_normal" or "log_normal"
    lower_bound: float = 0.0
    upper_bound: float = float("inf")


@dataclass
class PopulationConfig:
    """Configuration for virtual population generation."""

    n_patients: int = 100
    base_profile: Optional[PatientProfile] = None
    parameter_distributions: Dict[str, ParameterDistribution] = field(default_factory=dict)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.base_profile is None:
            self.base_profile = PatientProfile()
        if not self.parameter_distributions:
            self.parameter_distributions = _default_distributions(self.base_profile)


def _default_distributions(bp: PatientProfile) -> Dict[str, ParameterDistribution]:
    """Sensible defaults based on published T1D inter-patient variability."""
    return {
        "isf": ParameterDistribution(
            mean=bp.isf, cv=0.20, distribution="log_normal",
            lower_bound=10.0, upper_bound=200.0,
        ),
        "icr": ParameterDistribution(
            mean=bp.icr, cv=0.20, distribution="log_normal",
            lower_bound=3.0, upper_bound=30.0,
        ),
        "basal_rate": ParameterDistribution(
            mean=bp.basal_rate, cv=0.15, distribution="truncated_normal",
            lower_bound=0.1, upper_bound=3.0,
        ),
        "initial_glucose": ParameterDistribution(
            mean=bp.initial_glucose, cv=0.10, distribution="truncated_normal",
            lower_bound=70.0, upper_bound=300.0,
        ),
        "insulin_action_duration": ParameterDistribution(
            mean=bp.insulin_action_duration, cv=0.15, distribution="truncated_normal",
            lower_bound=120.0, upper_bound=600.0,
        ),
        "dawn_phenomenon_strength": ParameterDistribution(
            mean=bp.dawn_phenomenon_strength, cv=0.0, distribution="truncated_normal",
            lower_bound=0.0, upper_bound=30.0,
        ),
    }


class PopulationGenerator:
    """Generates *N* virtual :class:`PatientProfile` instances with
    physiological variation drawn from configurable distributions."""

    def __init__(self, config: PopulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _sample_parameter(self, dist: ParameterDistribution, n: int) -> np.ndarray:
        if dist.cv <= 0 or dist.mean == 0:
            return np.full(n, dist.mean)

        std = dist.mean * dist.cv

        if dist.distribution == "log_normal":
            variance = std ** 2
            mu_ln = np.log(dist.mean ** 2 / np.sqrt(variance + dist.mean ** 2))
            sigma_ln = np.sqrt(np.log(1 + variance / dist.mean ** 2))
            samples = self.rng.lognormal(mu_ln, sigma_ln, size=n)
        else:  # truncated_normal
            samples = self.rng.normal(dist.mean, std, size=n)

        return np.clip(samples, dist.lower_bound, dist.upper_bound)

    def generate(self) -> List[PatientProfile]:
        """Return a list of *n_patients* :class:`PatientProfile` instances."""
        n = self.config.n_patients
        base = self.config.base_profile
        if base is None:
            base = PatientProfile()

        sampled: Dict[str, np.ndarray] = {
            name: self._sample_parameter(dist, n)
            for name, dist in self.config.parameter_distributions.items()
        }

        profiles: List[PatientProfile] = []
        for i in range(n):
            profile = PatientProfile(
                isf=float(sampled["isf"][i]) if "isf" in sampled else base.isf,
                icr=float(sampled["icr"][i]) if "icr" in sampled else base.icr,
                basal_rate=float(sampled["basal_rate"][i]) if "basal_rate" in sampled else base.basal_rate,
                initial_glucose=float(sampled["initial_glucose"][i]) if "initial_glucose" in sampled else base.initial_glucose,
                insulin_action_duration=float(sampled["insulin_action_duration"][i]) if "insulin_action_duration" in sampled else base.insulin_action_duration,
                dawn_phenomenon_strength=float(sampled["dawn_phenomenon_strength"][i]) if "dawn_phenomenon_strength" in sampled else base.dawn_phenomenon_strength,
                # Non-varied parameters carry forward from the base profile
                dawn_start_hour=base.dawn_start_hour,
                dawn_end_hour=base.dawn_end_hour,
                glucose_decay_rate=base.glucose_decay_rate,
                glucose_absorption_rate=base.glucose_absorption_rate,
                insulin_peak_time=base.insulin_peak_time,
                meal_mismatch_epsilon=base.meal_mismatch_epsilon,
            )
            profiles.append(profile)

        return profiles
