"""Hybrid Gray-Box Glucose Predictor & In-Silico Chance-Constrained Insulin Estimator.

Designed for real Medtronic MiniMed 780G / CareLink CSV timelines imported via
:func:`iints.data.importer.import_carelink_timeline`.

Architecture
------------
1. **Causal Physics Core (White-Box):**
   Linearised Hovorka 8-state plant (:func:`iints.core.control.linear_plant.linearise`)
   augmented with a 2-compartment meal chain, interstitial CGM diffusion lag, and
   stationary AR(1) colored sensor noise (:class:`iints.core.control.meal_observer.MealObserver`).
   All insulin sensitivity and carbohydrate response dynamics remain strictly causal
   so closed-loop confounding cannot invert the sign of insulin action.
2. **Bounded Circadian + Innovation Residual Layer (Gray-Box):**
   A Ridge-regularized 24-hour Fourier + state-innovation residual model fitted
   strictly on the earlier temporal training split (never shuffled), paired with
   conformal variance scaling to produce calibrated 90% prediction bands
   ``[G_p05, G_p95]``.
3. **In-Silico Chance-Constrained BoxQP Insulin Estimator:**
   Computes the mathematically optimal first-step insulin rate over a 60-minute
   horizon via :func:`iints.core.control.mpc_qp.solve_box_qp` subject to pump
   bounds ``[0, max_rate]`` and a lower-confidence safety floor ``G_p05 >= 70 mg/dL``.

Research and educational boundary
---------------------------------
For pre-clinical in-silico evaluation and mathematical modeling only. Never use
outputs from this module to program an insulin pump or make clinical dosing
decisions for a real person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
from iints.core.patient.hovorka_model import HovorkaPatientModel
from iints.data.importer import import_carelink_timeline

Z_90_TWO_SIDED = 1.6448536269514722  # Phi^{-1}(0.95) for 90% central interval


@dataclass(frozen=True)
class PredictorConfig:
    horizons_minutes: Tuple[int, ...] = (30, 60, 120)
    step_minutes: float = 5.0
    train_fraction: float = 0.70
    target_glucose_mgdl: float = 110.0
    safety_floor_mgdl: float = 70.0
    sensor_sd_mgdl: float = 5.0
    sensor_tau_minutes: float = 5.0
    noise_ar1_phi: float = 0.0
    meal_tau_minutes: float = 40.0
    bioavailability: float = 0.80
    fourier_harmonics: int = 2
    ridge_alpha: float = 25.0
    max_residual_correction_mgdl: float = 60.0
    max_insulin_rate_u_per_h: float = 4.0
    insulin_weight: float = 800.0
    max_gap_minutes: float = 20.0
    warmup_steps: int = 6

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _circadian_features(
    minute_of_day: float,
    glucose_mgdl: float,
    slope_15m: float,
    innovation_mgdl: float,
    harmonics: int = 2,
) -> np.ndarray:
    feats: List[float] = [1.0]
    for m in range(1, harmonics + 1):
        angle = 2.0 * math.pi * m * (minute_of_day % 1440.0) / 1440.0
        feats.append(math.sin(angle))
        feats.append(math.cos(angle))
    feats.append((glucose_mgdl - 120.0) / 50.0)
    feats.append(max(-4.0, min(4.0, slope_15m)))
    feats.append(max(-5.0, min(5.0, innovation_mgdl / 20.0)))
    return np.asarray(feats, dtype=float)


class MiniMedGrayBoxPredictor:
    """Causal Hovorka+Kalman White-Box core with Ridge circadian residual & BoxQP."""

    def __init__(self, config: Optional[PredictorConfig] = None) -> None:
        self.config = config or PredictorConfig()
        self.model: Optional[MealLinearModel] = None
        self.operating_glucose_mgdl: float = 120.0
        self.basal_rate_u_per_h: float = 1.0
        self._residual_weights: Dict[int, np.ndarray] = {}
        self._uncertainty_scales: Dict[int, float] = {}
        self._rollout_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _init_physics_model(self, train_df: pd.DataFrame) -> MealLinearModel:
        basal_candidates = train_df["basal_rate_u_per_hr"].dropna()
        basal_pos = basal_candidates[basal_candidates > 0.0]
        basal_rate = float(basal_pos.median()) if not basal_pos.empty else 1.0
        basal_rate = float(np.clip(basal_rate, 0.2, max(0.5, self.config.max_insulin_rate_u_per_h * 0.75)))

        # Estimate personal TDD & 1800-rule ISF from training split only (zero leakage)
        ts_span_days = max(
            1.0 / 24.0,
            float(train_df["timestamp"].iloc[-1] - train_df["timestamp"].iloc[0]) / 1440.0,
        )
        total_train_insulin = float(train_df["insulin"].fillna(0.0).sum())
        est_tdd = total_train_insulin / ts_span_days if total_train_insulin > 0.0 else basal_rate * 48.0
        est_isf = float(np.clip(1800.0 / max(est_tdd, 10.0), 15.0, 100.0))

        # Operating point must stay safely above Hovorka's 81 mg/dL low-glucose knee
        med_glucose = float(train_df["glucose"].dropna().median())
        op_glucose = float(np.clip(med_glucose, 95.0, 200.0))

        patient = HovorkaPatientModel(
            initial_glucose=op_glucose,
            basal_insulin_rate=basal_rate,
            insulin_sensitivity=est_isf,
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
        self.model = meal_model
        self._rollout_cache.clear()
        return meal_model

    def _new_observer(self) -> MealObserver:
        assert self.model is not None
        return MealObserver(
            self.model,
            sensor_sd=self.config.sensor_sd_mgdl,
            sensor_tau_minutes=self.config.sensor_tau_minutes,
            noise_ar1_phi=self.config.noise_ar1_phi,
        )

    def _rollout_observer(
        self,
        observer: MealObserver,
        horizon_steps: int,
        step_minutes: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate observer state and covariance forward under constant basal."""
        assert self.model is not None
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
            means[step] = float(self.model.glucose_mgdl + C @ x)
            var = float(max(1.0, C @ P @ C + observer.readout_variance))
            stds[step] = math.sqrt(var)
        return means, stds

    def _collect_rolling_forecasts(
        self,
        df: pd.DataFrame,
        *,
        apply_residual: bool,
    ) -> Dict[int, Dict[str, Any]]:
        assert self.model is not None
        cfg = self.config
        dt_nom = cfg.step_minutes
        max_h = max(cfg.horizons_minutes)
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
                "naive": [],
                "wb": [],
                "wb_std": [],
                "gray": [],
                "gray_std": [],
                "features": [],
                "ref_profile": [],
                "gray_profile": [],
            }
            for h in cfg.horizons_minutes
        }

        observer = self._new_observer()
        seg_age = 0
        n = len(df)

        for i in range(n):
            if i == 0:
                observer.observe(glucose[i])
                seg_age = 1
                continue

            step_dt = float(ts_min[i] - ts_min[i - 1])
            if step_dt <= 0.0 or step_dt > cfg.max_gap_minutes:
                observer = self._new_observer()
                observer.observe(glucose[i])
                seg_age = 1
                continue

            observer.predict(float(max(0.0, insulin[i - 1])), step_dt)
            if carbs[i - 1] > 0.0:
                observer.announce_meal(float(carbs[i - 1]))
            observer.observe(glucose[i])
            seg_age += 1

            if seg_age < cfg.warmup_steps:
                continue

            # 15-min trailing slope for naive baseline & feature vector
            slope_idx = max(0, i - 3)
            span = ts_min[i] - ts_min[slope_idx]
            slope_15m = (glucose[i] - glucose[slope_idx]) / span if span >= 5.0 else 0.0

            means, stds = self._rollout_observer(observer, max_steps, dt_nom)
            feat = _circadian_features(
                minutes_of_day[i],
                glucose[i],
                slope_15m,
                observer.innovation,
                harmonics=cfg.fourier_harmonics,
            )

            for h in cfg.horizons_minutes:
                k_steps = int(round(h / dt_nom))
                target_idx = i + k_steps
                if target_idx >= n:
                    continue
                # Ensure target lies on a contiguous timeline (no gap inside horizon)
                actual_dt = ts_min[target_idx] - ts_min[i]
                if abs(actual_dt - h) > max(2.5, 0.1 * h):
                    continue

                wb_pred = float(np.clip(means[k_steps - 1], 20.0, 500.0))
                wb_sd = float(stds[k_steps - 1])
                naive_pred = float(np.clip(glucose[i] + slope_15m * h, 20.0, 500.0))

                if apply_residual and h in self._residual_weights:
                    corr = float(feat @ self._residual_weights[h])
                    corr = float(np.clip(corr, -cfg.max_residual_correction_mgdl, cfg.max_residual_correction_mgdl))
                    gray_pred = float(np.clip(wb_pred + corr, 20.0, 500.0))
                    gray_sd = wb_sd * self._uncertainty_scales.get(h, 1.0)
                else:
                    corr = 0.0
                    gray_pred = wb_pred
                    gray_sd = wb_sd

                b = buckets[h]
                b["idx"].append(i)
                b["y_true"].append(float(glucose[target_idx]))
                b["naive"].append(naive_pred)
                b["wb"].append(wb_pred)
                b["wb_std"].append(wb_sd)
                b["gray"].append(gray_pred)
                b["gray_std"].append(gray_sd)
                b["features"].append(feat)

                # Store full trajectory profiles for rate-aware directional accuracy
                if k_steps >= 4:
                    ref_prof = glucose[i : target_idx + 1]
                    # Linearly ramp the horizon residual correction from 0 at t=0 to corr at t=h
                    ramp = np.linspace(0.0, corr, k_steps + 1)
                    pred_prof = np.concatenate(([glucose[i]], np.clip(means[:k_steps], 20.0, 500.0))) + ramp
                    b["ref_profile"].append(ref_prof)
                    b["gray_profile"].append(np.clip(pred_prof, 20.0, 500.0))

        return buckets

    def fit(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit the causal Hovorka operating point and Ridge circadian residual layer."""
        if len(train_df) < 40:
            raise ValueError("At least 40 timeline samples are required to fit the predictor")
        self._init_physics_model(train_df)
        train_buckets = self._collect_rolling_forecasts(train_df, apply_residual=False)

        fit_summary: Dict[str, Any] = {
            "operating_glucose_mgdl": self.operating_glucose_mgdl,
            "basal_rate_u_per_h": self.basal_rate_u_per_h,
            "estimated_tdd_u_per_day": round(getattr(self, "estimated_tdd_u_per_day", 0.0), 2),
            "estimated_isf_mgdl_per_u": round(getattr(self, "estimated_isf_mgdl_per_u", 50.0), 2),
            "horizons": {},
        }
        for h, b in train_buckets.items():
            if len(b["y_true"]) < 15:
                continue
            X = np.vstack(b["features"])
            y_res = np.asarray(b["y_true"], dtype=float) - np.asarray(b["wb"], dtype=float)
            wb_sd = np.asarray(b["wb_std"], dtype=float)

            # Closed-form Ridge regression (leave intercept unpenalized)
            reg = self.config.ridge_alpha * np.eye(X.shape[1])
            reg[0, 0] = 1e-6
            w = np.linalg.solve(X.T @ X + reg, X.T @ y_res)
            self._residual_weights[h] = w

            # Conformal calibration of 90% uncertainty scale on training residuals
            pred_corr = np.clip(X @ w, -self.config.max_residual_correction_mgdl, self.config.max_residual_correction_mgdl)
            abs_err = np.abs(y_res - pred_corr)
            z_scores = abs_err / np.maximum(wb_sd, 1.0)
            q90 = float(np.quantile(z_scores, 0.90))
            scale = float(np.clip(q90 / Z_90_TWO_SIDED, 0.5, 4.0))
            self._uncertainty_scales[h] = scale
            fit_summary["horizons"][str(h)] = {
                "train_samples": len(b["y_true"]),
                "uncertainty_scale_90": round(scale, 3),
                "mean_train_residual_mgdl": round(float(np.mean(y_res)), 2),
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
        assert self.model is not None
        cfg = self.config
        A, B = self.model.discretise(step_minutes)
        C = self.model.C
        x0 = observer.x[:10].copy()

        # Build free response and Toeplitz impulse matrix Gamma over 10-state plant
        free = np.empty(horizon_steps, dtype=float)
        impulse = np.empty(horizon_steps, dtype=float)
        Ak = np.eye(10)
        for k in range(horizon_steps):
            impulse[k] = float(C @ (Ak @ B))
            Ak = Ak @ A
            free[k] = float(self.model.glucose_mgdl + C @ (Ak @ x0))

        gamma = np.zeros((horizon_steps, horizon_steps), dtype=float)
        for j in range(horizon_steps):
            gamma[j:, j] = impulse[: horizon_steps - j]

        # Propagate uncertainty for the 90% lower safety floor
        _, stds = self._rollout_observer(observer, horizon_steps, step_minutes)
        scale_60 = self._uncertainty_scales.get(60, 1.0)
        lower_margin = Z_90_TWO_SIDED * scale_60 * stds

        # BoxQP: minimize ||free + gamma*du - target||^2 + insulin_weight*||du||^2
        H = 2.0 * (gamma.T @ gamma + cfg.insulin_weight * np.eye(horizon_steps))
        f = 2.0 * (gamma.T @ (free - cfg.target_glucose_mgdl))
        basal = self.model.basal_u_per_h
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

        # Chance-constrained safety guard: if lower 95% band dips below 70 mg/dL, suspend
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
        """Strict temporal train/test evaluation of Naive vs White-Box vs Gray-Box."""
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
            y_naive = np.asarray(b["naive"], dtype=float)
            y_wb = np.asarray(b["wb"], dtype=float)
            y_gray = np.asarray(b["gray"], dtype=float)
            sd_gray = np.asarray(b["gray_std"], dtype=float)

            wb_zones = clarke_zones(y_true, y_wb)
            gray_zones = clarke_zones(y_true, y_gray)
            wb_ab = float(np.mean(np.isin(wb_zones, ["A", "B"])) * 100.0)
            gray_ab = float(np.mean(np.isin(gray_zones, ["A", "B"])) * 100.0)

            lo_90 = y_gray - Z_90_TWO_SIDED * sd_gray
            hi_90 = y_gray + Z_90_TWO_SIDED * sd_gray
            cov_90 = float(np.mean((y_true >= lo_90) & (y_true <= hi_90)) * 100.0)

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
            )
            evaluations.append(ev.to_dict())

        # Compute latest-state in-silico BoxQP insulin estimate on the tail of test_df
        tail_observer = self._new_observer()
        tail_slice = test_df.tail(min(36, len(test_df))).reset_index(drop=True)
        ts_tail = tail_slice["timestamp"].to_numpy(dtype=float)
        for idx in range(len(tail_slice)):
            if idx == 0:
                tail_observer.observe(float(tail_slice["glucose"].iloc[idx]))
                continue
            dt_step = float(ts_tail[idx] - ts_tail[idx - 1])
            if 0.0 < dt_step <= self.config.max_gap_minutes:
                tail_observer.predict(float(max(0.0, tail_slice["insulin"].iloc[idx - 1])), dt_step)
                if float(tail_slice["carbs"].iloc[idx - 1]) > 0.0:
                    tail_observer.announce_meal(float(tail_slice["carbs"].iloc[idx - 1]))
                tail_observer.observe(float(tail_slice["glucose"].iloc[idx]))
            else:
                tail_observer = self._new_observer()
                tail_observer.observe(float(tail_slice["glucose"].iloc[idx]))

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


def evaluate_carelink_csv(
    csv_path: Union[str, Path],
    config: Optional[PredictorConfig] = None,
) -> Dict[str, Any]:
    """Import a Medtronic CareLink CSV and run the full out-of-sample Gray-Box benchmark."""
    timeline = import_carelink_timeline(csv_path)
    predictor = MiniMedGrayBoxPredictor(config=config)
    report = predictor.evaluate_timeline(timeline)
    report["source_csv"] = str(csv_path)
    report["total_timeline_rows"] = len(timeline)
    return report
