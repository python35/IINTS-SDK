from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxQP:
    """Strictly convex quadratic objective with independent box constraints."""

    hessian: np.ndarray
    linear: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class BoxQPSolution:
    inputs: np.ndarray
    objective: float
    kkt_residual: float
    iterations: int


def _projected_gradient(gradient: np.ndarray, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    projected = gradient.copy()
    projected[(x <= lower) & (gradient > 0.0)] = 0.0
    projected[(x >= upper) & (gradient < 0.0)] = 0.0
    return projected


def solve_box_qp(problem: BoxQP, *, tolerance: float = 1e-10, max_iterations: int = 20000) -> BoxQPSolution:
    """Minimise ``0.5*x.T*H*x + f.T*x`` subject to ``lower <= x <= upper``."""
    H = np.asarray(problem.hessian, dtype=float)
    f = np.asarray(problem.linear, dtype=float).reshape(-1)
    lower = np.asarray(problem.lower, dtype=float).reshape(-1)
    upper = np.asarray(problem.upper, dtype=float).reshape(-1)
    n = f.size
    if H.shape != (n, n) or lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("QP dimensions do not match")
    if not all(np.all(np.isfinite(value)) for value in (H, f, lower, upper)):
        raise ValueError("QP inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("QP lower bounds must not exceed upper bounds")
    H = 0.5 * (H + H.T)
    if np.min(np.linalg.eigvalsh(H)) <= 0.0:
        raise ValueError("QP hessian must be positive definite")
    if tolerance <= 0.0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")

    try:
        x = np.linalg.solve(H, -f)
    except np.linalg.LinAlgError:
        x = np.zeros(n, dtype=float)
    x = np.clip(x, lower, upper)
    diagonal = np.diag(H)
    if np.any(diagonal <= 0.0):
        raise ValueError("QP hessian must have a positive diagonal")

    for iteration in range(1, max_iterations + 1):
        for index in range(n):
            gradient_i = float(H[index] @ x + f[index])
            x[index] = np.clip(x[index] - gradient_i / diagonal[index], lower[index], upper[index])
        gradient = H @ x + f
        residual = float(np.max(np.abs(_projected_gradient(gradient, x, lower, upper))))
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(f"BoxQP did not converge; KKT residual is {residual:g}")

    objective = 0.5 * float(x @ H @ x) + float(f @ x)
    return BoxQPSolution(x, objective, residual, iteration)
