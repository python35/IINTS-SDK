from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from iints.core.patient.hovorka_model import HovorkaPatientModel


@dataclass(frozen=True)
class LinearPlant:
    """Scaled local linearisation of the eight-state Hovorka control plant."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    scales: np.ndarray
    glucose_mgdl: float
    basal_u_per_h: float

    def discretise(self, step_minutes: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(step_minutes) or step_minutes <= 0.0:
            raise ValueError("step_minutes must be finite and positive")
        n = self.A.shape[0]
        augmented = np.zeros((n + 1, n + 1), dtype=float)
        augmented[:n, :n] = self.A
        augmented[:n, n] = self.B
        discrete = expm(augmented * float(step_minutes))
        return discrete[:n, :n], discrete[:n, n]

    def dc_gain_per_u_per_h(self) -> float:
        return float(-self.C @ np.linalg.solve(self.A, self.B))

    def is_stable(self) -> bool:
        return bool(np.all(np.linalg.eigvals(self.A).real < 0.0))


def linearise(patient: HovorkaPatientModel) -> LinearPlant:
    """Linearise a Hovorka patient around its current fasting operating point."""
    state = np.asarray(patient.get_state()["ode_state"], dtype=float)[:8]
    glucose_volume = patient._glucose_volume_dl()
    basal_rate = patient.basal_insulin_rate
    basal_input = basal_rate * 1000.0 / 60.0
    t_max = patient._insulin_tmax_minutes()
    insulin_volume = patient.params.V_I_per_kg * patient.params.body_weight_kg
    basal_insulin = basal_input / (insulin_volume * patient.params.k_e)

    scales = np.asarray(
        [
            glucose_volume * 50.0,
            glucose_volume * 50.0,
            max(basal_input * t_max, 1.0),
            max(basal_input * t_max, 1.0),
            max(basal_insulin, 1.0),
            max(abs(float(state[5])), 1e-5),
            max(abs(float(state[6])), 1e-5),
            max(abs(float(state[7])), 1e-5),
        ],
        dtype=float,
    )
    operating_state = np.asarray(patient.get_state()["ode_state"], dtype=float)

    def derivative(values: np.ndarray, rate_u_per_h: float) -> np.ndarray:
        full_state = operating_state.copy()
        full_state[:8] = values
        result = patient._ode(
            0.0,
            full_state,
            rate_u_per_h * 1000.0 / 60.0,
            0.0,
            0.0,
        )
        return np.asarray(result[:8], dtype=float)

    jacobian = np.empty((8, 8), dtype=float)
    for column in range(8):
        delta = max(abs(float(state[column])) * 1e-5, scales[column] * 1e-5, 1e-8)
        above = state.copy()
        below = state.copy()
        above[column] += delta
        below[column] -= delta
        jacobian[:, column] = (
            derivative(above, basal_rate) - derivative(below, basal_rate)
        ) / (2.0 * delta)

    input_delta = max(abs(basal_rate) * 1e-5, 1e-5)
    input_jacobian = (
        derivative(state, basal_rate + input_delta)
        - derivative(state, basal_rate - input_delta)
    ) / (2.0 * input_delta)

    A = jacobian * scales[None, :] / scales[:, None]
    B = input_jacobian / scales
    C = np.zeros(8, dtype=float)
    C[0] = scales[0] / glucose_volume
    operating_glucose = float(state[0] / glucose_volume)
    return LinearPlant(A, B, C, scales, operating_glucose, basal_rate)
