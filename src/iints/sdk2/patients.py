"""Module A — Standardized Virtual Patient Engine for IINTS Simulation SDK 2.0.

Provides an interchangeable ``VirtualPatientAdapter`` interface wrapping:
- ``hovorka_equilibrium``: Equilibrium-consistent 19-state Hovorka ODE patient,
- ``iints_ode``: Fast educational/stress-testing ODE model (``CustomPatientModel``),
- ``bergman``: Bergman Minimal Model ODE patient (for cross-simulator validation),
- ``simglucose``: Optional adapter to open-source ``simglucose`` UVA/Padova-derived cohort.

Regulatory note: While the original UVA/Padova simulator was accepted by the FDA in
2008 for specific pre-clinical contexts, neither ``simglucose`` nor IINTS inherits
regulatory status. All adapters here are strictly for research and education.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from iints.core.patient import equilibrium as eq
from iints.core.patient.models import CustomPatientModel
from iints.core.patient.patient_factory import (
    BERGMAN_AVAILABLE,
    SIMGLUCOSE_AVAILABLE,
    BergmanPatientModel,
    SimglucosePatientWrapper,
)
from iints.sdk2.contracts import PatientSpec

__all__ = [
    "VirtualPatientAdapter",
    "SimpleODEPatientAdapter",
    "EquilibriumHovorkaAdapter",
    "BergmanPatientAdapter",
    "ExternalSimglucoseAdapter",
    "create_virtual_patient",
    "available_simulator_backends",
]


@runtime_checkable
class VirtualPatientAdapter(Protocol):
    """Standardized interchangeable interface for any virtual patient simulator."""

    model_family: str
    profile_id: str

    def reset(self) -> None: ...

    def step(
        self,
        dt_minutes: float,
        insulin_units: float,
        carbs_grams: float,
        current_time_minutes: float,
    ) -> float: ...

    @property
    def true_glucose_mgdl(self) -> float: ...

    def metadata(self) -> Dict[str, Any]: ...


class SimpleODEPatientAdapter:
    """Fast, deterministic educational & stress-test patient adapter (``iints_ode``)."""

    model_family = "iints_ode"

    def __init__(self, spec: PatientSpec) -> None:
        self.spec = spec
        self.profile_id = spec.profile
        self._model = CustomPatientModel(
            basal_insulin_rate=spec.basal_u_per_h,
            insulin_sensitivity=spec.insulin_sensitivity_mgdl_per_u,
            carb_factor=spec.carb_factor_g_per_u,
            initial_glucose=spec.initial_glucose_mgdl,
            glucose_absorption_rate=1.0 / max(spec.meal_tau_min, 10.0),
            max_glucose_rate_mgdl_per_min=spec.max_glucose_rate_mgdl_per_min,
        )

    def reset(self) -> None:
        self._model.reset()

    def step(
        self,
        dt_minutes: float,
        insulin_units: float,
        carbs_grams: float,
        current_time_minutes: float,
    ) -> float:
        return float(
            self._model.update(
                float(dt_minutes),
                float(insulin_units),
                float(carbs_grams),
                current_time=float(current_time_minutes),
            )
        )

    @property
    def true_glucose_mgdl(self) -> float:
        return float(self._model.current_glucose)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "profile_id": self.profile_id,
            "basal_u_per_h": self.spec.basal_u_per_h,
            "regulatory_status": "research_only",
        }


class EquilibriumHovorkaAdapter:
    """Nonlinear Hovorka ODE patient initialized at exact fasting equilibrium."""

    model_family = "hovorka_equilibrium"

    def __init__(self, spec: PatientSpec) -> None:
        self.spec = spec
        self.profile_id = spec.profile
        if spec.sigma is not None:
            self.sigma = float(spec.sigma)
            self.operating_glucose = eq.equilibrium_glucose(spec.basal_u_per_h, self.sigma)
        else:
            self.operating_glucose = float(spec.initial_glucose_mgdl)
            self.sigma = eq.basal_scale(spec.basal_u_per_h, self.operating_glucose)
        self._build()

    def _build(self) -> None:
        self._model = eq.make_equilibrium_patient(
            self.spec.basal_u_per_h,
            self.sigma,
            self.operating_glucose,
            max_glucose_rate_mgdl_per_min=self.spec.max_glucose_rate_mgdl_per_min,
        )

    def reset(self) -> None:
        self._build()

    def step(
        self,
        dt_minutes: float,
        insulin_units: float,
        carbs_grams: float,
        current_time_minutes: float,
    ) -> float:
        return float(
            self._model.update(
                float(dt_minutes),
                float(insulin_units),
                float(carbs_grams),
                current_time=float(current_time_minutes),
            )
        )

    @property
    def true_glucose_mgdl(self) -> float:
        return float(self._model.current_glucose)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "profile_id": self.profile_id,
            "basal_u_per_h": self.spec.basal_u_per_h,
            "sigma": round(self.sigma, 6),
            "operating_glucose_mgdl": round(self.operating_glucose, 2),
            "regulatory_status": "research_only",
        }


class BergmanPatientAdapter:
    """Bergman Minimal Model ODE patient for cross-simulator sensitivity studies."""

    model_family = "bergman"

    def __init__(self, spec: PatientSpec) -> None:
        if not BERGMAN_AVAILABLE or BergmanPatientModel is None:
            raise ImportError("BergmanPatientModel requires SciPy.")
        self.spec = spec
        self.profile_id = spec.profile
        self._build()

    def _build(self) -> None:
        assert BergmanPatientModel is not None
        self._model = BergmanPatientModel(
            initial_glucose=self.spec.initial_glucose_mgdl,
            basal_insulin_rate=self.spec.basal_u_per_h,
            insulin_sensitivity=self.spec.insulin_sensitivity_mgdl_per_u,
            carb_factor=self.spec.carb_factor_g_per_u,
            max_glucose_rate_mgdl_per_min=self.spec.max_glucose_rate_mgdl_per_min,
        )

    def reset(self) -> None:
        self._build()

    def step(
        self,
        dt_minutes: float,
        insulin_units: float,
        carbs_grams: float,
        current_time_minutes: float,
    ) -> float:
        return float(
            self._model.update(
                float(dt_minutes),
                float(insulin_units),
                float(carbs_grams),
                current_time=float(current_time_minutes),
            )
        )

    @property
    def true_glucose_mgdl(self) -> float:
        return float(self._model.current_glucose)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "profile_id": self.profile_id,
            "basal_u_per_h": self.spec.basal_u_per_h,
            "regulatory_status": "research_only",
        }


class ExternalSimglucoseAdapter:
    """Optional external simulator adapter wrapping ``simglucose`` when installed."""

    model_family = "simglucose"

    def __init__(self, spec: PatientSpec) -> None:
        if not SIMGLUCOSE_AVAILABLE:
            raise ImportError(
                "External simulator 'simglucose' is not installed in this environment. "
                "Install 'simglucose' to run cross-simulator checks against it."
            )
        self.spec = spec
        self.profile_id = spec.profile if "#" in spec.profile else "adult#001"
        self._build()

    def _build(self) -> None:
        self._model = SimglucosePatientWrapper(
            patient_name=self.profile_id,
            initial_glucose=self.spec.initial_glucose_mgdl,
            basal_insulin_rate=self.spec.basal_u_per_h,
            insulin_sensitivity=self.spec.insulin_sensitivity_mgdl_per_u,
            carb_factor=self.spec.carb_factor_g_per_u,
        )

    def reset(self) -> None:
        self._build()

    def step(
        self,
        dt_minutes: float,
        insulin_units: float,
        carbs_grams: float,
        current_time_minutes: float,
    ) -> float:
        return float(
            self._model.update(
                float(dt_minutes),
                float(insulin_units),
                float(carbs_grams),
                current_time=float(current_time_minutes),
            )
        )

    @property
    def true_glucose_mgdl(self) -> float:
        return float(self._model.current_glucose)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_family": self.model_family,
            "profile_id": self.profile_id,
            "regulatory_status": "open_source_research_adapter_not_fda_cleared",
        }


def available_simulator_backends() -> list[str]:
    backends = ["hovorka_equilibrium", "iints_ode"]
    if BERGMAN_AVAILABLE:
        backends.append("bergman")
    if SIMGLUCOSE_AVAILABLE:
        backends.append("simglucose")
    return backends


def create_virtual_patient(spec: PatientSpec) -> VirtualPatientAdapter:
    if spec.model == "hovorka_equilibrium":
        return EquilibriumHovorkaAdapter(spec)
    if spec.model == "iints_ode":
        return SimpleODEPatientAdapter(spec)
    if spec.model == "bergman":
        return BergmanPatientAdapter(spec)
    if spec.model == "simglucose":
        return ExternalSimglucoseAdapter(spec)
    raise ValueError(f"Unsupported virtual patient model: {spec.model}")
