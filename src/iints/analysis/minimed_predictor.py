"""Hybrid Gray-Box Glucose Predictor & In-Silico Chance-Constrained Insulin Estimator.

Designed for real Medtronic MiniMed 780G / CareLink CSV timelines imported via
:func:`iints.data.importer.import_carelink_timeline` with ``fill_gaps=True``.

Architecture
------------
1. **Equilibrium-Consistent Causal Physics Core (White-Box):**
   Linearised Hovorka 8-state plant (:func:`iints.core.control.linear_plant.linearise`)
   constructed at an exact fasting equilibrium (machine-zero drift, no double
   ISF scaling), augmented with a 2-compartment meal chain, interstitial CGM
   diffusion lag, stationary AR(1) colored sensor noise, and an identified
   closed-loop SmartGuard proportional feedback gain ``k_sg >= 0``:
   ``A_cl = A + k_sg * B @ C`` (positive ``k_sg`` increases insulin when glucose
   is above equilibrium, shifting the slow open-loop pole toward the observed
   closed-loop mean-reversion rate).
2. **Gap-Aware Kalman Filtering:**
   On continuous 5-minute CareLink grids where sensor dropouts have
   ``glucose = NaN``, the observer runs ``predict()`` and ``announce_meal()``
   across gaps while skipping ``observe()``, conserving 100% of boluses/carbs
   delivered during sensor swaps.
3. **Bounded Circadian + Innovation Residual Layer (Gray-Box):**
   Ridge-regularized 24-hour Fourier + state-innovation residual model fitted
   strictly on the earlier temporal training split (with optional rolling window
   to prevent multi-month drift), paired with uncapped conformal 90% uncertainty
   calibration and honest Persistence + AR(2) baselines.
4. **In-Silico Chance-Constrained BoxQP Insulin Estimator:**
   Strictly convex quadratic programme via :func:`iints.core.control.mpc_qp.solve_box_qp`.

Research and educational boundary
---------------------------------
For pre-clinical in-silico evaluation and mathematical modeling only. Never use
outputs from this module to program an insulin pump or make clinical dosing
decisions for a real person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from iints.analysis.error_grid import clarke_zones
from iints.analysis.prediction_accuracy import directional_report
from iints.core.control.linear_plant import linearise
from iints.core.control.meal_observer import MealLinearModel, MealObserver
from iints.core.control.mpc_qp import BoxQP, solve_box_qp
from iints.core.patient.hovorka_model import HovorkaParameters, HovorkaPatientModel
from iints.data.importer import import_carelink_timeline

Z_90_TWO_SIDED = 1.6448536269514722  # Phi^{-1}(0.95) for 90% central interval


def solve_equilibrium_glucose(
    basal_rate_u_per_h: float,
    sensitivity_scale: float,
    *,
    hovorka_params: Optional[HovorkaParameters] = None,
    low_mgdl: float = 82.0,
    high_mgdl: float = 450.0,
) -> float:
    """Solve for the unique fasting equilibrium glucose G* (mg/dL) given
    ``(basal_rate_u_per_h, sensitivity_scale)`` with ``insulin_sensitivity = 50.0``.

    Raises ``ValueError`` if no fasting equilibrium exists in ``[low_mgdl, high_mgdl]``
    instead of silently clipping to an out-of-equilibrium operating point.
    """
    if not math.isfinite(basal_rate_u_per_h) or basal_rate_u_per_h <= 0.0:
        raise ValueError("basal_rate_u_per_h must be finite and positive")
    if not math.isfinite(sensitivity_scale) or sensitivity_scale <= 0.0:
        raise ValueError("sensitivity_scale must be finite and positive")

    probe = HovorkaPatientModel(
        initial_glucose=120.0,
        basal_insulin_rate=basal_rate_u_per_h,
        insulin_sensitivity=50.0,
        insulin_sensitivity_scale=sensitivity_scale,
        hovorka_params=hovorka_params,
        max_glucose_rate_mgdl_per_min=8.0,
    )
    basal_input_mu_min = basal_rate_u_per_h * 1000.0 / 60.0

    def dQ1_at(g_mgdl: float) -> float:
        y = probe._default_ode_state(glucose_mgdl=g_mgdl)
        return float(probe._ode(0.0, y, basal_input_mu_min, 0.0, 0.0)[0])

    r_lo = dQ1_at(low_mgdl)
    r_hi = dQ1_at(high_mgdl)
    if r_lo <= 0.0 or r_hi >= 0.0:
        raise ValueError(
            f"No fasting equilibrium in [{low_mgdl}, {high_mgdl}] mg/dL for "
            f"basal={basal_rate_u_per_h:.3f} U/h, sigma={sensitivity_scale:.4f} "
            f"(dQ1/dt at bounds: {r_lo:.3f}, {r_hi:.3f})"
        )

    lo, hi = float(low_mgdl), float(high_mgdl)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dQ1_at(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_equilibrium_hovorka(
    basal_rate_u_per_h: float,
    sensitivity_scale: float,
    *,
    hovorka_params: Optional[HovorkaParameters] = None,
) -> HovorkaPatientModel:
    """Build a HovorkaPatientModel initialized at its exact fasting equilibrium G*."""
    g_star = solve_equilibrium_glucose(
        basal_rate_u_per_h,
        sensitivity_scale,
        hovorka_params=hovorka_params,
    )
    return HovorkaPatientModel(
        initial_glucose=g_star,
        basal_insulin_rate=basal_rate_u_per_h,
        insulin_sensitivity=50.0,
        insulin_sensitivity_scale=sensitivity_scale,
        hovorka_params=hovorka_params,
        max_glucose_rate_mgdl_per_min=8.0,
    )


def _apply_smartguard_feedback(model: MealLinearModel, k_sg_u_per_h_per_mgdl: float) -> MealLinearModel:
    """Close the unobserved SmartGuard loop on the 10-state linearised plant:
    du_sg = +k_sg * (G - G*) => A_cl = A + k_sg * B @ C.
    """
    if k_sg_u_per_h_per_mgdl <= 0.0:
        return model
    A_cl = model.A + float(k_sg_u_per_h_per_mgdl) * np.outer(model.B, model.C)
    eigs = np.linalg.eigvals(A_cl)
    if float(np.max(eigs.real)) >= -1e-6:
        raise ValueError(f"Closed-loop gain k_sg={k_sg_u_per_h_per_mgdl} destabilizes the plant")
    return replace(model, A=A_cl)


@dataclass(frozen=True)
class PredictorConfig:
    horizons_minutes: Tuple[int, ...] = (30, 60, 120)
    step_minutes: float = 5.0
    train_fraction: float = 0.70
    max_train_days: Optional[float] = 35.0
    target_glucose_mgdl: float = 110.0
    safety_floor_mgdl: float = 70.0
    sensor_sd_mgdl: float = 6.0
    sensor_tau_minutes: float = 6.0
    noise_ar1_phi: float = 0.85
    meal_tau_minutes: float = 60.0
    bioavailability: float = 0.65
    fourier_harmonics: int = 2
    ridge_alpha: float = 25.0
    max_residual_correction_mgdl: float = 60.0
    max_insulin_rate_u_per_h: float = 5.0
    insulin_weight: float = 800.0
    max_gap_minutes: float = 20.0
    warmup_steps: int = 6
    smartguard_gain_grid: Tuple[float, ...] = (0.0, 0.015, 0.03, 0.05, 0.075, 0.10, 0.14)
    process_noise_scale: float = 2.0

    def __post_init__(self) -> None:
        if not self.horizons_minutes or any(h <= 0 for h in self.horizons_minutes):
            raise ValueError("horizons_minutes must contain positive integers")
        if not 0.2 <= self.train_fraction <= 0.9:
            raise ValueError("train_fraction must lie in [0.2, 0.9]")
        if self.step_minutes <= 0.0:
            raise ValueError("step_minutes must be positive")
        if self.ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be positive")


@dataclass
class HorizonEvaluation:
    horizon_minutes: int
    n_samples: int
    persistence_rmse_mgdl: float
    ar2_rmse_mgdl: float
    naive_rmse_mgdl: float
    naive_mae_mgdl: float
    whitebox_rmse_mgdl: float
    whitebox_mae_mgdl: float
    whitebox_clarke_ab_pct: float
    graybox_rmse_mgdl: float
    graybox_mae_mgdl: float
    graybox_bias_mgdl: float
    graybox_clarke_ab_pct: float
    graybox_coverage_90_pct: float
    graybox_directional_accurate_pct: float
    graybox_directional_erroneous_pct: float
    hypo_n: int = 0
    hypo_sensitivity_p05_pct: float = 0.0
    hypo_false_alarm_pct: float = 0.0
    hypo_brier_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Exact Gaussian CDF Phi(z) using math.erfc (no extra scipy.stats dependency)."""
    return np.array([0.5 * math.erfc(-float(v) / math.sqrt(2.0)) for v in z], dtype=float)


def _circadian_features(
    minute_of_day: float,
    glucose_mgdl: float,
    prev_glucose_mgdl: float,
    wb_delta_mgdl: float,
    slope_15m: float,
    innovation_mgdl: float,
    harmonics: int = 2,
) -> np.ndarray:
    """Physics-informed Gray-Box feature vector (strictly causal).

    Includes the Hovorka+Kalman predicted change ``wb_delta_mgdl = y_wb - G_i``
    alongside lag-1 momentum ``G_i - G_{i-1}``, level deviation ``(G_i - 120)/50``,
    24-hour Fourier harmonics, 15-min slope, and Kalman innovation. Because AR(2)
    is a strict linear subset of this basis, Ridge regression optimally blends
    causal physics with autoregressive mean reversion.
    """
    feats: List[float] = [1.0]
    for m in range(1, harmonics + 1):
        angle = 2.0 * math.pi * m * (minute_of_day % 1440.0) / 1440.0
        feats.append(math.sin(angle))
        feats.append(math.cos(angle))
    feats.append((glucose_mgdl - 120.0) / 50.0)
    feats.append(max(-40.0, min(40.0, glucose_mgdl - prev_glucose_mgdl)) / 10.0)
    feats.append(max(-150.0, min(150.0, wb_delta_mgdl)) / 30.0)
    feats.append(max(-4.0, min(4.0, slope_15m)))
    feats.append(max(-5.0, min(5.0, innovation_mgdl / 20.0)))
    return np.asarray(feats, dtype=float)


class MiniMedGrayBoxPredictor:
    """Equilibrium-consistent Hovorka+Kalman White-Box core with SmartGuard feedback,
    Ridge circadian residual, and chance-constrained BoxQP."""

    def __init__(self, config: Optional[PredictorConfig] = None) -> None:
        self.config = config or PredictorConfig()
        self.open_loop_model: Optional[MealLinearModel] = None
        self.model: Optional[MealLinearModel] = None
        self.operating_glucose_mgdl: float = 120.0
        self.basal_rate_u_per_h: float = 1.0
        self.estimated_tdd_u_per_day: float = 50.0
        self.estimated_isf_mgdl_per_u: float = 36.0
        self.identified_smartguard_gain: float = 0.0
        self._residual_weights: Dict[int, np.ndarray] = {}
        self._uncertainty_scales: Dict[int, float] = {}
        self._ar2_weights: Dict[int, np.ndarray] = {}

    def _init_physics_model(self, train_df: pd.DataFrame) -> MealLinearModel:
        basal_candidates = train_df["basal_rate_u_per_hr"].dropna()
        basal_pos = basal_candidates[basal_candidates > 0.0]
        basal_rate = float(basal_pos.median()) if not basal_pos.empty else 1.0
        basal_rate = float(np.clip(basal_rate, 0.2, max(0.5, self.config.max_insulin_rate_u_per_h * 0.85)))

        # Report TDD & 1800-rule ISF for telemetry ONLY — never inject ISF != 50 into
        # HovorkaPatientModel, which breaks the fasting equilibrium (double scaling).
        ts_span_days = max(
            1.0 / 24.0,
            float(train_df["timestamp"].iloc[-1] - train_df["timestamp"].iloc[0]) / 1440.0,
        )
        total_train_insulin = float(train_df["insulin"].fillna(0.0).sum())
        est_tdd = total_train_insulin / ts_span_days if total_train_insulin > 0.0 else basal_rate * 48.0
        est_isf = float(np.clip(1800.0 / max(est_tdd, 10.0), 10.0, 120.0))

        # Operating point safely above Hovorka's 81 mg/dL low-glucose knee
        med_glucose = float(train_df["glucose"].dropna().median())
        op_glucose = float(np.clip(med_glucose, 95.0, 220.0))

        # Exact equilibrium at insulin_sensitivity = 50.0 (drift < 1e-12 mg/dL/min)
        patient = HovorkaPatientModel(
            initial_glucose=op_glucose,
            basal_insulin_rate=basal_rate,
            insulin_sensitivity=50.0,
            max_glucose_rate_mgdl_per_min=8.0,
        )
        plant = linearise(patient)
        meal_model = MealLinearModel.from_plant(
            plant,
            meal_tau_minutes=self.config.meal_tau_minutes,
            bioavailability=self.config.bioavailability,
        )
        self.operating_glucose_mgdl = op_glucose
        self.basal_rate_u_per_h = basal_rate
        self.estimated_tdd_u_per_day = float(est_tdd)
        self.estimated_isf_mgdl_per_u = est_isf
        self.open_loop_model = meal_model
        self.model = meal_model
        self.identified_smartguard_gain = 0.0
        return meal_model

    def _new_observer(self, model: Optional[MealLinearModel] = None) -> MealObserver:
        mdl = model or self.model
        assert mdl is not None
        obs = MealObserver(
            mdl,
            sensor_sd=self.config.sensor_sd_mgdl,
            sensor_tau_minutes=self.config.sensor_tau_minutes,
            noise_ar1_phi=self.config.noise_ar1_phi,
        )
        if self.config.process_noise_scale != 1.0:
            obs.Q = obs.Q * (self.config.process_noise_scale ** 2)
        return obs

    def _rollout_observer(
        self,
        observer: MealObserver,
        horizon_steps: int,
        step_minutes: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate observer state and covariance forward under constant basal."""
        mdl = observer.model
        A, _ = observer.transition(step_minutes)
        C = observer.measurement_C
        x = observer.x.copy()
        P = observer.P.copy()
        means = np.empty(horizon_steps, dtype=float)
        stds = np.empty(horizon_steps, dtype=float)
        for step in range(horizon_steps):
            x = A @ x
            P = A @ P @ A.T + observer.Q * step_minutes
            if observer.noise_index is not None:
                P[observer.noise_index, observer.noise_index] += observer.sensor_variance * (
                    1.0 - observer.noise_ar1_phi ** (2.0 * step_minutes / 5.0)
                )
            means[step] = float(mdl.glucose_mgdl + C @ x)
            var = float(max(1.0, C @ P @ C + observer.readout_variance))
            stds[step] = math.sqrt(var)
        return means, stds

    def _collect_rolling_forecasts(
        self,
        df: pd.DataFrame,
        *,
        apply_residual: bool,
        model_override: Optional[MealLinearModel] = None,
        horizons_override: Optional[Tuple[int, ...]] = None,
        collect_profiles: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        mdl = model_override or self.model
        assert mdl is not None
        cfg = self.config
        dt_nom = cfg.step_minutes
        horizons = horizons_override or cfg.horizons_minutes
        max_h = max(horizons)
        max_steps = int(round(max_h / dt_nom))

        ts_min = df["timestamp"].to_numpy(dtype=float)
        ts_dt = pd.to_datetime(df["timestamp_dt"])
        minutes_of_day = (ts_dt.dt.hour * 60 + ts_dt.dt.minute).to_numpy(dtype=float)
        glucose = df["glucose"].to_numpy(dtype=float)
        carbs = df["carbs"].fillna(0.0).to_numpy(dtype=float)
        insulin = df["insulin"].fillna(0.0).to_numpy(dtype=float)

        buckets: Dict[int, Dict[str, List[Any]]] = {
            h: {
                "idx": [],
                "y_true": [],
                "persistence": [],
                "ar2_x": [],
                "naive": [],
                "wb": [],
                "wb_std": [],
                "gray": [],
                "gray_std": [],
                "features": [],
                "ref_profile": [],
                "gray_profile": [],
            }
            for h in horizons
        }

        observer = self._new_observer(mdl)
        obs_count = 0
        n = len(df)
        last_valid_idx = -1
        prev_valid_idx = -1

        for i in range(n):
            g_i = float(glucose[i])
            is_valid_g = math.isfinite(g_i)

            if i == 0:
                if is_valid_g:
                    observer.observe(g_i)
                    obs_count = 1
                    last_valid_idx = i
                continue

            step_dt = float(ts_min[i] - ts_min[i - 1])
            if step_dt <= 0.0:
                continue

            # If the grid itself jumped across a large un-gridded gap, reset observer
            if step_dt > cfg.max_gap_minutes:
                observer = self._new_observer(mdl)
                obs_count = 0
                prev_valid_idx = -1
                last_valid_idx = -1
                if is_valid_g:
                    observer.observe(g_i)
                    obs_count = 1
                    last_valid_idx = i
                continue

            # Gap-aware Kalman step: ALWAYS predict insulin & announce carbs,
            # even when glucose[i] is NaN during a sensor swap!
            observer.predict(float(max(0.0, insulin[i - 1])), step_dt)
            if carbs[i - 1] > 0.0:
                observer.announce_meal(float(carbs[i - 1]))

            if not is_valid_g:
                continue

            observer.observe(g_i)
            obs_count += 1
            prev_valid_idx = last_valid_idx
            last_valid_idx = i

            if obs_count < cfg.warmup_steps:
                continue

            # 15-min trailing slope over valid observations
            slope_idx = max(0, i - 3)
            slope_15m = 0.0
            if math.isfinite(glucose[slope_idx]):
                span = ts_min[i] - ts_min[slope_idx]
                if span >= 5.0:
                    slope_15m = (g_i - float(glucose[slope_idx])) / span

            g_prev = float(glucose[prev_valid_idx]) if prev_valid_idx >= 0 else g_i
            means, stds = self._rollout_observer(observer, max_steps, dt_nom)

            for h in horizons:
                k_steps = int(round(h / dt_nom))
                target_idx = i + k_steps
                if target_idx >= n or not math.isfinite(glucose[target_idx]):
                    continue
                actual_dt = ts_min[target_idx] - ts_min[i]
                if abs(actual_dt - h) > max(2.5, 0.1 * h):
                    continue

                wb_pred = float(np.clip(means[k_steps - 1], 20.0, 500.0))
                wb_sd = float(stds[k_steps - 1])
                naive_pred = float(np.clip(g_i + slope_15m * h, 20.0, 500.0))

                feat = _circadian_features(
                    minutes_of_day[i],
                    g_i,
                    g_prev,
                    wb_pred - g_i,
                    slope_15m,
                    observer.innovation,
                    harmonics=cfg.fourier_harmonics,
                )

                if apply_residual and h in self._residual_weights:
                    delta_pred = float(feat @ self._residual_weights[h])
                    gray_pred = float(np.clip(g_i + delta_pred, 20.0, 500.0))
                    gray_sd = wb_sd * self._uncertainty_scales.get(h, 1.0)
                else:
                    gray_pred = wb_pred
                    gray_sd = wb_sd

                b = buckets[h]
                b["idx"].append(i)
                b["y_true"].append(float(glucose[target_idx]))
                b["persistence"].append(g_i)
                b["ar2_x"].append(np.array([1.0, g_i, g_prev], dtype=float))
                b["naive"].append(naive_pred)
                b["wb"].append(wb_pred)
                b["wb_std"].append(wb_sd)
                b["gray"].append(gray_pred)
                b["gray_std"].append(gray_sd)
                b["features"].append(feat)

                if collect_profiles and k_steps >= 4:
                    ref_prof = glucose[i : target_idx + 1]
                    if np.all(np.isfinite(ref_prof)):
                        ramp = np.linspace(0.0, gray_pred - wb_pred, k_steps + 1)
                        pred_prof = np.concatenate(([g_i], np.clip(means[:k_steps], 20.0, 500.0))) + ramp
                        b["ref_profile"].append(ref_prof)
                        b["gray_profile"].append(np.clip(pred_prof, 20.0, 500.0))

        return buckets

    def _identify_smartguard_gain(self, fit_df: pd.DataFrame) -> float:
        """Identify the unobserved closed-loop SmartGuard proportional gain k_sg
        strictly on the training split by minimizing 60-min White-Box RMSE.
        """
        assert self.open_loop_model is not None
        grid = self.config.smartguard_gain_grid
        if len(grid) <= 1 or len(fit_df) < 80:
            return float(grid[0]) if grid else 0.0

        # Use the most recent ~2000 rows of fit_df (~7 days) for fast gain identification
        sub_df = fit_df.tail(min(2000, len(fit_df))).reset_index(drop=True)
        h_eval = 60 if 60 in self.config.horizons_minutes else self.config.horizons_minutes[0]
        best_k = 0.0
        best_rmse = float("inf")

        for k_cand in grid:
            try:
                cand_model = _apply_smartguard_feedback(self.open_loop_model, float(k_cand))
            except ValueError:
                continue
            b = self._collect_rolling_forecasts(
                sub_df,
                apply_residual=False,
                model_override=cand_model,
                horizons_override=(h_eval,),
                collect_profiles=False,
            )[h_eval]
            if len(b["y_true"]) < 20:
                continue
            yt = np.asarray(b["y_true"], dtype=float)
            ywb = np.asarray(b["wb"], dtype=float)
            rmse = float(np.sqrt(np.mean((ywb - yt) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = float(k_cand)
        return best_k

    def fit(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit equilibrium physics, closed-loop SmartGuard gain, AR(2) baseline,
        Ridge circadian residual, and uncapped conformal 90% uncertainty scaling.
        """
        if len(train_df) < 40:
            raise ValueError("At least 40 timeline samples are required to fit the predictor")

        # Optional rolling training window (e.g. last 35 days) to track slow physiological drift
        fit_df = train_df.sort_values("timestamp_dt").reset_index(drop=True)
        if self.config.max_train_days is not None and len(fit_df) > 100:
            t_end = float(fit_df["timestamp"].iloc[-1])
            cutoff = t_end - float(self.config.max_train_days) * 1440.0
            recent = fit_df[fit_df["timestamp"] >= cutoff].reset_index(drop=True)
            if len(recent) >= 60:
                fit_df = recent

        self._init_physics_model(fit_df)
        assert self.open_loop_model is not None

        k_sg = self._identify_smartguard_gain(fit_df)
        self.identified_smartguard_gain = k_sg
        self.model = _apply_smartguard_feedback(self.open_loop_model, k_sg)

        train_buckets = self._collect_rolling_forecasts(fit_df, apply_residual=False, collect_profiles=False)

        fit_summary: Dict[str, Any] = {
            "operating_glucose_mgdl": round(self.operating_glucose_mgdl, 2),
            "basal_rate_u_per_h": round(self.basal_rate_u_per_h, 3),
            "estimated_tdd_u_per_day": round(self.estimated_tdd_u_per_day, 2),
            "estimated_isf_mgdl_per_u": round(self.estimated_isf_mgdl_per_u, 2),
            "identified_smartguard_gain": round(self.identified_smartguard_gain, 4),
            "train_window_rows": len(fit_df),
            "horizons": {},
        }
        for h, b in train_buckets.items():
            if len(b["y_true"]) < 15:
                continue
            X = np.vstack(b["features"])
            y_true = np.asarray(b["y_true"], dtype=float)
            y_pers = np.asarray(b["persistence"], dtype=float)
            y_wb = np.asarray(b["wb"], dtype=float)
            y_delta = y_true - y_pers
            wb_sd = np.asarray(b["wb_std"], dtype=float)

            # Honest AR(2) statistical baseline fitted strictly on training split
            X_ar2 = np.vstack(b["ar2_x"])
            w_ar2 = np.linalg.lstsq(X_ar2, y_true, rcond=None)[0]
            self._ar2_weights[h] = w_ar2

            # Closed-form Ridge regression on delta = y_true - G_i (strictly nests AR(2) + physics)
            reg = self.config.ridge_alpha * np.eye(X.shape[1])
            reg[0, 0] = 1e-6
            w = np.linalg.solve(X.T @ X + reg, X.T @ y_delta)
            self._residual_weights[h] = w

            # Conformal calibration of 90% uncertainty scale (high ceiling 10.0 so clip never binds silently)
            pred_delta = X @ w
            abs_err = np.abs(y_delta - pred_delta)
            z_scores = abs_err / np.maximum(wb_sd, 1.0)
            q90 = float(np.quantile(z_scores, 0.90))
            scale = float(np.clip(q90 / Z_90_TWO_SIDED, 0.5, 10.0))
            self._uncertainty_scales[h] = scale
            fit_summary["horizons"][str(h)] = {
                "train_samples": len(b["y_true"]),
                "uncertainty_scale_90": round(scale, 3),
                "mean_train_wb_residual_mgdl": round(float(np.mean(y_true - y_wb)), 2),
                "physics_blend_weight": round(float(w[-3]), 3),
            }
        return fit_summary

    def recommend_in_silico_insulin(
        self,
        observer: MealObserver,
        *,
        horizon_steps: int = 12,
        step_minutes: float = 5.0,
    ) -> Dict[str, Any]:
        """Solve a strictly convex chance-constrained BoxQP for in-silico insulin rate.

        RESEARCH ONLY — never use for real human dosing.
        """
        assert self.open_loop_model is not None
        cfg = self.config
        # Optimize over the causal open-loop plant so the QP commands total rate directly
        A, B = self.open_loop_model.discretise(step_minutes)
        C = self.open_loop_model.C
        x0 = observer.x[:10].copy()

        free = np.empty(horizon_steps, dtype=float)
        impulse = np.empty(horizon_steps, dtype=float)
        Ak = np.eye(10)
        for k in range(horizon_steps):
            impulse[k] = float(C @ (Ak @ B))
            Ak = Ak @ A
            free[k] = float(self.open_loop_model.glucose_mgdl + C @ (Ak @ x0))

        gamma = np.zeros((horizon_steps, horizon_steps), dtype=float)
        for j in range(horizon_steps):
            gamma[j:, j] = impulse[: horizon_steps - j]

        _, stds = self._rollout_observer(observer, horizon_steps, step_minutes)
        scale_60 = self._uncertainty_scales.get(60, 1.0)
        lower_margin = Z_90_TWO_SIDED * scale_60 * stds

        H = 2.0 * (gamma.T @ gamma + cfg.insulin_weight * np.eye(horizon_steps))
        f = 2.0 * (gamma.T @ (free - cfg.target_glucose_mgdl))
        basal = self.open_loop_model.basal_u_per_h
        qp = BoxQP(
            hessian=0.5 * (H + H.T),
            linear=f,
            lower=np.full(horizon_steps, -basal, dtype=float),
            upper=np.full(horizon_steps, cfg.max_insulin_rate_u_per_h - basal, dtype=float),
        )
        sol = solve_box_qp(qp)
        du_opt = sol.inputs
        pred_glucose = free + gamma @ du_opt
        pred_lower_05 = pred_glucose - lower_margin

        rate_u_per_h = float(basal + du_opt[0])
        safety_suspended = False
        if float(np.min(pred_lower_05)) < cfg.safety_floor_mgdl:
            rate_u_per_h = 0.0
            safety_suspended = True

        return {
            "recommended_rate_u_per_h": max(0.0, rate_u_per_h),
            "recommended_step_units": max(0.0, rate_u_per_h * step_minutes / 60.0),
            "safety_floor_suspended": safety_suspended,
            "min_predicted_p05_mgdl": float(np.min(pred_lower_05)),
            "predicted_60m_mgdl": float(pred_glucose[-1]),
            "kkt_residual": float(sol.kkt_residual),
        }

    def evaluate_timeline(self, timeline: pd.DataFrame) -> Dict[str, Any]:
        """Strict temporal train/test evaluation vs Persistence, AR(2), and Naive."""
        ordered = timeline.sort_values("timestamp_dt").reset_index(drop=True)
        split_idx = int(len(ordered) * self.config.train_fraction)
        train_df = ordered.iloc[:split_idx].reset_index(drop=True)
        test_df = ordered.iloc[split_idx:].reset_index(drop=True)

        fit_info = self.fit(train_df)
        test_buckets = self._collect_rolling_forecasts(test_df, apply_residual=True)

        evaluations: List[Dict[str, Any]] = []
        for h in self.config.horizons_minutes:
            b = test_buckets[h]
            if not b["y_true"]:
                continue
            y_true = np.asarray(b["y_true"], dtype=float)
            y_pers = np.asarray(b["persistence"], dtype=float)
            y_naive = np.asarray(b["naive"], dtype=float)
            y_wb = np.asarray(b["wb"], dtype=float)
            y_gray = np.asarray(b["gray"], dtype=float)
            sd_gray = np.asarray(b["gray_std"], dtype=float)

            if h in self._ar2_weights:
                X_ar2 = np.vstack(b["ar2_x"])
                y_ar2 = np.clip(X_ar2 @ self._ar2_weights[h], 20.0, 500.0)
                ar2_rmse = float(np.sqrt(np.mean((y_ar2 - y_true) ** 2)))
            else:
                ar2_rmse = float("nan")

            wb_zones = clarke_zones(y_true, y_wb)
            gray_zones = clarke_zones(y_true, y_gray)
            wb_ab = float(np.mean(np.isin(wb_zones, ["A", "B"])) * 100.0)
            gray_ab = float(np.mean(np.isin(gray_zones, ["A", "B"])) * 100.0)

            lo_90 = y_gray - Z_90_TWO_SIDED * sd_gray
            hi_90 = y_gray + Z_90_TWO_SIDED * sd_gray
            cov_90 = float(np.mean((y_true >= lo_90) & (y_true <= hi_90)) * 100.0)

            # Dedicated hypoglycemia early-warning metrics (< 70 mg/dL)
            is_hypo = y_true < self.config.safety_floor_mgdl
            hypo_n = int(np.sum(is_hypo))
            p05_alarm = lo_90 < self.config.safety_floor_mgdl
            hypo_sens = float(np.mean(p05_alarm[is_hypo]) * 100.0) if hypo_n > 0 else 0.0
            non_hypo = ~is_hypo
            hypo_far = float(np.mean(p05_alarm[non_hypo]) * 100.0) if int(np.sum(non_hypo)) > 0 else 0.0
            p_hypo = _normal_cdf((self.config.safety_floor_mgdl - y_gray) / np.maximum(sd_gray, 1.0))
            hypo_brier = float(np.mean((p_hypo - is_hypo.astype(float)) ** 2))

            dir_acc = float("nan")
            dir_err = float("nan")
            if b["ref_profile"]:
                ref_mat = np.vstack(b["ref_profile"])
                pred_mat = np.vstack(b["gray_profile"])
                drep = directional_report(ref_mat, pred_mat, step_minutes=self.config.step_minutes)
                pcts = drep["overall"]["percentages"]
                dir_acc = float(pcts["accurate"]) if pcts["accurate"] is not None else 0.0
                dir_err = float(pcts["erroneous"]) if pcts["erroneous"] is not None else 0.0

            ev = HorizonEvaluation(
                horizon_minutes=h,
                n_samples=len(y_true),
                persistence_rmse_mgdl=round(float(np.sqrt(np.mean((y_pers - y_true) ** 2))), 2),
                ar2_rmse_mgdl=round(ar2_rmse, 2) if math.isfinite(ar2_rmse) else 0.0,
                naive_rmse_mgdl=round(float(np.sqrt(np.mean((y_naive - y_true) ** 2))), 2),
                naive_mae_mgdl=round(float(np.mean(np.abs(y_naive - y_true))), 2),
                whitebox_rmse_mgdl=round(float(np.sqrt(np.mean((y_wb - y_true) ** 2))), 2),
                whitebox_mae_mgdl=round(float(np.mean(np.abs(y_wb - y_true))), 2),
                whitebox_clarke_ab_pct=round(wb_ab, 2),
                graybox_rmse_mgdl=round(float(np.sqrt(np.mean((y_gray - y_true) ** 2))), 2),
                graybox_mae_mgdl=round(float(np.mean(np.abs(y_gray - y_true))), 2),
                graybox_bias_mgdl=round(float(np.mean(y_gray - y_true)), 2),
                graybox_clarke_ab_pct=round(gray_ab, 2),
                graybox_coverage_90_pct=round(cov_90, 2),
                graybox_directional_accurate_pct=round(dir_acc, 2) if math.isfinite(dir_acc) else 0.0,
                graybox_directional_erroneous_pct=round(dir_err, 2) if math.isfinite(dir_err) else 0.0,
                hypo_n=hypo_n,
                hypo_sensitivity_p05_pct=round(hypo_sens, 1),
                hypo_false_alarm_pct=round(hypo_far, 1),
                hypo_brier_score=round(hypo_brier, 5),
            )
            evaluations.append(ev.to_dict())

        tail_observer = self.reconstruct_observer_at(test_df, anchor_idx=len(test_df) - 1, lookback_minutes=180.0)
        latest_qp = self.recommend_in_silico_insulin(tail_observer)

        return {
            "research_disclaimer": (
                "In-silico research evaluation only; not a medical device and "
                "must never be used for clinical insulin dosing."
            ),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "fit_summary": fit_info,
            "horizon_evaluations": evaluations,
            "latest_in_silico_boxqp": latest_qp,
        }

    def reconstruct_observer_at(
        self,
        timeline: pd.DataFrame,
        anchor_idx: int,
        *,
        lookback_minutes: float = 360.0,
    ) -> MealObserver:
        """Replay real CGM, insulin, and carb history up to ``anchor_idx``
        (gap-aware: skips observe() on NaN glucose while still predicting insulin/carbs).
        """
        assert self.model is not None
        ordered = timeline.sort_values("timestamp_dt").reset_index(drop=True)
        if not 0 <= anchor_idx < len(ordered):
            raise ValueError(f"anchor_idx={anchor_idx} out of bounds for {len(ordered)} rows")
        t_anchor = float(ordered["timestamp"].iloc[anchor_idx])
        start_mask = ordered["timestamp"] >= (t_anchor - lookback_minutes)
        start_idx = int(np.argmax(start_mask.to_numpy()))
        window = ordered.iloc[start_idx : anchor_idx + 1].reset_index(drop=True)

        observer = self._new_observer()
        ts = window["timestamp"].to_numpy(dtype=float)
        gl = window["glucose"].to_numpy(dtype=float)
        ins = window["insulin"].fillna(0.0).to_numpy(dtype=float)
        carbs = window["carbs"].fillna(0.0).to_numpy(dtype=float)

        for idx in range(len(window)):
            g_val = float(gl[idx])
            if idx == 0:
                if math.isfinite(g_val):
                    observer.observe(g_val)
                continue
            dt_step = float(ts[idx] - ts[idx - 1])
            if 0.0 < dt_step <= self.config.max_gap_minutes:
                observer.predict(float(max(0.0, ins[idx - 1])), dt_step)
                if carbs[idx - 1] > 0.0:
                    observer.announce_meal(float(carbs[idx - 1]))
                if math.isfinite(g_val):
                    observer.observe(g_val)
            else:
                observer = self._new_observer()
                if math.isfinite(g_val):
                    observer.observe(g_val)
        return observer

    def what_if_forecast(
        self,
        observer: MealObserver,
        *,
        horizon_minutes: int = 180,
        carbs_g: float = 0.0,
        bolus_u: float = 0.0,
        basal_pct: float = 100.0,
        sensitivity_multiplier: float = 1.0,
        absorption_multiplier: float = 1.0,
        start_minute_of_day: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Forward causal what-if simulation from the reconstructed observer state."""
        assert self.open_loop_model is not None
        if carbs_g < 0.0 or bolus_u < 0.0 or basal_pct < 0.0:
            raise ValueError("carbs_g, bolus_u, and basal_pct must be non-negative")
        if sensitivity_multiplier <= 0.0 or absorption_multiplier <= 0.0:
            raise ValueError("sensitivity_multiplier and absorption_multiplier must be positive")

        dt = self.config.step_minutes
        n_steps = max(1, int(round(horizon_minutes / dt)))
        # Use open-loop plant for explicit what-if basal/bolus interventions
        A_plant, B_plant = self.open_loop_model.discretise(
            dt,
            sensitivity=sensitivity_multiplier,
            absorption=absorption_multiplier,
        )
        C_plant = self.open_loop_model.C
        basal_base = self.open_loop_model.basal_u_per_h
        planned_basal_rate = basal_base * (basal_pct / 100.0)

        x = observer.x[:10].copy()
        if carbs_g > 0.0:
            x[8] += float(carbs_g) / float(self.open_loop_model.scales[8])

        _, step_stds = self._rollout_observer(observer, n_steps, dt)
        unc_scale = self._uncertainty_scales.get(60, 1.0)
        w60 = self._residual_weights.get(60)

        times_min = np.arange(0, n_steps + 1, dtype=float) * dt
        current_g = float(np.clip(self.open_loop_model.glucose_mgdl + C_plant @ x, 20.0, 500.0))
        median = np.empty(n_steps + 1, dtype=float)
        p05 = np.empty(n_steps + 1, dtype=float)
        p95 = np.empty(n_steps + 1, dtype=float)
        rates = np.empty(n_steps, dtype=float)

        median[0] = current_g
        p05[0] = current_g
        p95[0] = current_g

        for k in range(n_steps):
            step_rate = planned_basal_rate + (float(bolus_u) * 60.0 / dt if k == 0 else 0.0)
            rates[k] = step_rate
            du = step_rate - basal_base
            x = A_plant @ x + B_plant * du
            g_pred = float(self.open_loop_model.glucose_mgdl + C_plant @ x)

            if start_minute_of_day is not None and w60 is not None:
                m_day = (start_minute_of_day + (k + 1) * dt) % 1440.0
                ramp = min(1.0, ((k + 1) * dt) / 60.0)
                swing = 0.0
                for m in range(1, self.config.fourier_harmonics + 1):
                    ang = 2.0 * math.pi * m * m_day / 1440.0
                    swing += float(w60[2 * m - 1]) * math.sin(ang) + float(w60[2 * m]) * math.cos(ang)
                g_pred += 0.5 * ramp * swing

            sd = float(step_stds[k]) * unc_scale
            median[k + 1] = float(np.clip(g_pred, 20.0, 500.0))
            p05[k + 1] = float(np.clip(g_pred - Z_90_TWO_SIDED * sd, 20.0, 500.0))
            p95[k + 1] = float(np.clip(g_pred + Z_90_TWO_SIDED * sd, 20.0, 500.0))

        return {
            "times_min": times_min.tolist(),
            "median_mgdl": median.tolist(),
            "p05_mgdl": p05.tolist(),
            "p95_mgdl": p95.tolist(),
            "insulin_rate_u_per_h": rates.tolist(),
            "start_glucose_mgdl": round(current_g, 1),
            "peak_mgdl": round(float(np.max(median)), 1),
            "nadir_mgdl": round(float(np.min(median)), 1),
            "end_mgdl": round(float(median[-1]), 1),
            "min_p05_mgdl": round(float(np.min(p05)), 1),
            "hypo_risk_p05": bool(np.min(p05) < self.config.safety_floor_mgdl),
            "scenario": {
                "carbs_g": float(carbs_g),
                "bolus_u": float(bolus_u),
                "basal_pct": float(basal_pct),
                "sensitivity_multiplier": float(sensitivity_multiplier),
                "absorption_multiplier": float(absorption_multiplier),
            },
        }


def evaluate_carelink_csv(
    csv_path: Union[str, Path],
    config: Optional[PredictorConfig] = None,
) -> Dict[str, Any]:
    """Import a Medtronic CareLink CSV on a gap-preserving 5-min grid and run
    the full out-of-sample Gray-Box benchmark against Persistence and AR(2).
    """
    timeline = import_carelink_timeline(csv_path, fill_gaps=True)
    predictor = MiniMedGrayBoxPredictor(config=config)
    report = predictor.evaluate_timeline(timeline)
    report["source_csv"] = str(csv_path)
    report["total_timeline_rows"] = len(timeline)
    return report
