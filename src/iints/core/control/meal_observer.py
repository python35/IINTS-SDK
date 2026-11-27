from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .linear_plant import LinearPlant


@dataclass(frozen=True)
class MealLinearModel:
    """Hovorka linear plant augmented with a two-compartment meal model."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    scales: np.ndarray
    glucose_mgdl: float
    basal_u_per_h: float

    @classmethod
    def from_plant(
        cls,
        plant: LinearPlant,
        *,
        meal_tau_minutes: float = 60.0,
        bioavailability: float = 0.65,
    ) -> MealLinearModel:
        if not np.isfinite(meal_tau_minutes) or meal_tau_minutes <= 0.0:
            raise ValueError("meal_tau_minutes must be finite and positive")
        if not np.isfinite(bioavailability) or not 0.0 <= bioavailability <= 1.0:
            raise ValueError("bioavailability must be between 0 and 1")
        scales = np.concatenate((plant.scales, np.asarray([50.0, 50.0])))
        A = np.zeros((10, 10), dtype=float)
        A[:8, :8] = plant.A
        tau = float(meal_tau_minutes)
        A[8, 8] = -1.0 / tau
        A[9, 8] = 1.0 / tau
        A[9, 9] = -1.0 / tau
        A[0, 9] = 1000.0 * float(bioavailability) * scales[9] / (tau * scales[0])

        B = np.zeros(10, dtype=float)
        B[:8] = plant.B
        C = np.zeros(10, dtype=float)
        C[:8] = plant.C
        return cls(A, B, C, scales, plant.glucose_mgdl, plant.basal_u_per_h)

    def discretise(
        self,
        step_minutes: float,
        *,
        sensitivity: float = 1.0,
        absorption: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(step_minutes) or step_minutes <= 0.0:
            raise ValueError("step_minutes must be finite and positive")
        if not np.isfinite(sensitivity) or sensitivity <= 0.0:
            raise ValueError("sensitivity must be finite and positive")
        if not np.isfinite(absorption) or absorption <= 0.0:
            raise ValueError("absorption must be finite and positive")
        n = self.A.shape[0]
        augmented = np.zeros((n + 1, n + 1), dtype=float)
        dynamics = self.A.copy()
        dynamics[0:2, 5:8] *= float(sensitivity)
        dynamics[5:8, 4] *= float(sensitivity)
        dynamics[0, 9] *= float(absorption)
        augmented[:n, :n] = dynamics
        augmented[:n, n] = self.B
        discrete = expm(augmented * float(step_minutes))
        return discrete[:n, :n], discrete[:n, n]


class MealObserver:
    """Kalman filter for plant deviations, announced meals, and sensor noise."""

    def __init__(
        self,
        model: MealLinearModel,
        *,
        sensor_sd: float = 6.0,
        meal_noise_g_per_sqrt_min: float = 0.4,
        sensor_tau_minutes: float = 6.0,
        noise_ar1_phi: float = 0.85,
    ) -> None:
        values = (sensor_sd, meal_noise_g_per_sqrt_min, sensor_tau_minutes, noise_ar1_phi)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("observer parameters must be finite")
        if sensor_sd <= 0.0 or meal_noise_g_per_sqrt_min < 0.0 or sensor_tau_minutes <= 0.0:
            raise ValueError("observer noise and time constants are out of range")
        if not 0.0 <= noise_ar1_phi < 1.0:
            raise ValueError("noise_ar1_phi must be in [0, 1)")

        self.model = model
        self.sensor_sd = float(sensor_sd)
        self.sensor_variance = self.sensor_sd**2
        self.sensor_tau_minutes = float(sensor_tau_minutes)
        self.noise_ar1_phi = float(noise_ar1_phi)
        self.sensor_index = model.A.shape[0]
        self.noise_index = self.sensor_index + 1
        self.x = np.zeros(self.noise_index + 1, dtype=float)
        self.P = np.diag(
            np.concatenate((np.ones(self.sensor_index), [1.0, self.sensor_variance]))
        )
        self.Q = np.zeros_like(self.P)
        self.Q[0, 0] = 1e-4
        self.Q[1, 1] = 1e-4
        self.Q[2:4, 2:4] = np.eye(2) * 0.05
        self.Q[4, 4] = 1e-4
        self.Q[5:8, 5:8] = np.eye(3) * 1e-5
        self.Q[8:10, 8:10] = np.eye(2) * (
            float(meal_noise_g_per_sqrt_min) / model.scales[8]
        ) ** 2
        self.Q[self.sensor_index, self.sensor_index] = 1e-4
        self.readout_variance = max(1.0, (self.sensor_sd * 0.2) ** 2)
        self.measurement_C = np.zeros(self.noise_index + 1, dtype=float)
        self.measurement_C[self.sensor_index :] = 1.0
        self.innovation = 0.0

    def transition(self, step_minutes: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(step_minutes) or step_minutes <= 0.0:
            raise ValueError("step_minutes must be finite and positive")
        n = self.sensor_index
        augmented = np.zeros((n + 2, n + 2), dtype=float)
        augmented[:n, :n] = self.model.A
        augmented[:n, n + 1] = self.model.B
        augmented[n, :n] = self.model.C / self.sensor_tau_minutes
        augmented[n, n] = -1.0 / self.sensor_tau_minutes
        discrete = expm(augmented * float(step_minutes))
        A = np.zeros_like(self.P)
        A[: self.noise_index, : self.noise_index] = discrete[: n + 1, : n + 1]
        A[-1, -1] = self.noise_ar1_phi ** (float(step_minutes) / 5.0)
        B = np.zeros_like(self.x)
        B[: self.noise_index] = discrete[: n + 1, n + 1]
        return A, B

    def predict(self, insulin_units: float, step_minutes: float) -> None:
        if not np.isfinite(insulin_units) or not np.isfinite(step_minutes) or step_minutes <= 0.0:
            raise ValueError("insulin_units must be finite and step_minutes positive")
        A, B = self.transition(step_minutes)
        rate = float(insulin_units) * 60.0 / float(step_minutes)
        self.x = A @ self.x + B * (rate - self.model.basal_u_per_h)
        self.P = A @ self.P @ A.T + self.Q * float(step_minutes)
        self.P[self.noise_index, self.noise_index] += self.sensor_variance * (
            1.0 - self.noise_ar1_phi ** (2.0 * float(step_minutes) / 5.0)
        )

    def announce_meal(self, carbs_g: float) -> None:
        if not np.isfinite(carbs_g) or carbs_g < 0.0:
            raise ValueError("carbs_g must be finite and non-negative")
        self.x[8] += float(carbs_g) / self.model.scales[8]

    def observe(self, glucose_mgdl: float) -> float:
        if not np.isfinite(glucose_mgdl):
            raise ValueError("glucose_mgdl must be finite")
        measurement = float(glucose_mgdl) - self.model.glucose_mgdl
        residual = measurement - float(self.measurement_C @ self.x)
        innovation_variance = float(self.measurement_C @ self.P @ self.measurement_C)
        innovation_variance += self.readout_variance
        gain = self.P @ self.measurement_C / max(innovation_variance, 1e-12)
        self.x += gain * residual
        identity = np.eye(len(self.x))
        correction = identity - np.outer(gain, self.measurement_C)
        self.P = correction @ self.P @ correction.T + np.outer(gain, gain) * self.readout_variance
        self.P = 0.5 * (self.P + self.P.T)
        self.innovation = float(residual)
        return self.innovation
