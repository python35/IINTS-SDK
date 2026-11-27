"""Module C — Pluggable Algorithm Sandbox & Blind Causal Evaluation Harness.

Guarantees fair comparison across forecasting algorithms:
1. Every algorithm sees ONLY causal past observations ``[0 .. origin]`` sliced
   according to the pre-registered ``MissingDataContract``.
2. Ground-truth columns (``bg_true_mgdl``, ``ig_mgdl``, ``cgm_clean_mgdl``,
   ``carbs_physical_g``) are physically stripped before any forecaster runs,
   making lookahead or truth-leakage impossible.
3. Includes 5 built-in forecasters spanning naive, linear, statistical, ML, and
   mechanistic gray-box paradigms, plus a plugin registry for custom models.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Type

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from iints.analysis import nonlinear_propagator as nlp
from iints.core.control.linear_plant import linearise
from iints.core.control.meal_observer import MealLinearModel, MealObserver
from iints.core.patient import equilibrium as eq
from iints.sdk2.contracts import AlgorithmSpec, EvaluationSpec, MissingDataContract

__all__ = [
    "Z_90",
    "ForecastOutput",
    "GlucoseForecaster",
    "PersistenceForecaster",
    "LinearSavgolForecaster",
    "AR2Forecaster",
    "RidgeWindowMLForecaster",
    "MechanisticKalmanGrayBoxForecaster",
    "register_forecaster",
    "list_forecasters",
    "create_forecaster",
    "evaluate_forecasters_on_trace",
]

Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class ForecastOutput:
    """Point and 90 % interval predictions keyed by horizon in minutes."""

    point_mgdl: Dict[int, float]
    lower_90_mgdl: Dict[int, float]
    upper_90_mgdl: Dict[int, float]


class GlucoseForecaster(ABC):
    """Standardized pluggable forecaster interface for IINTS Simulation SDK 2.0."""

    model_id: str = "abstract"

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        """Optional causal calibration on the pre-evaluation warmup segment."""

    @abstractmethod
    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        """Predict future glucose at ``horizons_minutes`` from strictly past data."""


def _last_valid_and_staleness(series: np.ndarray) -> tuple[float, int]:
    finite_idx = np.flatnonzero(np.isfinite(series))
    if finite_idx.size == 0:
        return 120.0, len(series)
    last_i = int(finite_idx[-1])
    return float(series[last_i]), int(len(series) - 1 - last_i)


def _causal_last_valid_arrays(raw: np.ndarray, default_val: float = 120.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single causal O(N) scan returning (last_val, prev_valid_val, stale_steps) at every index."""
    n = len(raw)
    last_vals = np.empty(n, dtype=float)
    prev_vals = np.empty(n, dtype=float)
    stale = np.empty(n, dtype=int)
    cur = default_val
    prv = default_val
    misses = 0
    for i in range(n):
        if np.isfinite(raw[i]):
            prv = cur if i > 0 else float(raw[i])
            cur = float(raw[i])
            misses = 0
        else:
            misses += 1
        last_vals[i] = cur
        prev_vals[i] = prv
        stale[i] = misses
    return last_vals, prev_vals, stale


class PersistenceForecaster(GlucoseForecaster):
    """Zero-order hold baseline: carries last valid sensor value forward."""

    model_id = "persistence_v1"

    def __init__(self, rw_sd_per_sqrt_5m: float = 6.5) -> None:
        self.rw_sd = float(rw_sd_per_sqrt_5m)

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        g = warmup_history["sensor_mgdl"].to_numpy(dtype=float)
        diffs = np.diff(g)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size >= 10:
            self.rw_sd = float(np.clip(np.std(diffs), 2.5, 25.0))

    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        g = causal_history["sensor_mgdl"].to_numpy(dtype=float)
        last_val, stale_steps = _last_valid_and_staleness(g)
        pt, lo, hi = {}, {}, {}
        for h in horizons_minutes:
            k = max(1, int(round(h / step_minutes))) + stale_steps
            sd = self.rw_sd * float(np.sqrt(k))
            val = float(np.clip(last_val, 20.0, 500.0))
            pt[int(h)] = val
            lo[int(h)] = float(np.clip(val - Z_90 * sd, 20.0, 500.0))
            hi[int(h)] = float(np.clip(val + Z_90 * sd, 20.0, 500.0))
        return ForecastOutput(pt, lo, hi)

    def predict_trace_batch_causal(
        self,
        blind_view: pd.DataFrame,
        eval_origins: np.ndarray,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raw = blind_view["sensor_mgdl"].to_numpy(dtype=float)
        last_vals, _, stale = _causal_last_valid_arrays(raw)
        vals = np.clip(last_vals[eval_origins], 20.0, 500.0)
        st = stale[eval_origins]
        out: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for h in horizons_minutes:
            k = max(1, int(round(h / step_minutes))) + st
            sds = self.rw_sd * np.sqrt(k)
            out[int(h)] = (
                vals,
                np.clip(vals - Z_90 * sds, 20.0, 500.0),
                np.clip(vals + Z_90 * sds, 20.0, 500.0),
            )
        return out


class LinearSavgolForecaster(GlucoseForecaster):
    """Damped Savitzky-Golay causal slope extrapolation baseline."""

    model_id = "savgol_linear_v1"

    def __init__(self, damping_tau_min: float = 45.0) -> None:
        self.damping_tau_min = float(damping_tau_min)
        self._resid_sd: Dict[int, float] = {}

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        g = pd.Series(warmup_history["sensor_mgdl"].to_numpy(dtype=float)).ffill().bfill().to_numpy()
        diffs = np.diff(g)
        base_sd = float(np.std(diffs)) if diffs.size > 5 else 6.0
        for h in (30, 60, 120):
            self._resid_sd[h] = base_sd * float(np.sqrt(max(1.0, h / step_minutes))) * 1.15

    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        raw = causal_history["sensor_mgdl"].to_numpy(dtype=float)
        last_val, stale_steps = _last_valid_and_staleness(raw)
        filled = pd.Series(raw).ffill().bfill().to_numpy()
        if len(filled) >= 7:
            w = min(9, len(filled[-9:]) | 1)
            slope = float(savgol_filter(filled[-9:], w, 2, deriv=1, delta=step_minutes)[-1])
        elif len(filled) >= 2:
            slope = float((filled[-1] - filled[-2]) / step_minutes)
        else:
            slope = 0.0
        slope *= float(np.exp(-stale_steps * step_minutes / self.damping_tau_min))
        slope = float(np.clip(slope, -3.5, 3.5))

        pt, lo, hi = {}, {}, {}
        for h in horizons_minutes:
            eff_h = self.damping_tau_min * (1.0 - float(np.exp(-float(h) / self.damping_tau_min)))
            pred = float(np.clip(last_val + slope * eff_h, 20.0, 500.0))
            sd = self._resid_sd.get(int(h), 18.0) * float(np.sqrt(1.0 + 0.25 * stale_steps))
            pt[int(h)] = pred
            lo[int(h)] = float(np.clip(pred - Z_90 * sd, 20.0, 500.0))
            hi[int(h)] = float(np.clip(pred + Z_90 * sd, 20.0, 500.0))
        return ForecastOutput(pt, lo, hi)


class AR2Forecaster(GlucoseForecaster):
    """Causal stationary AR(2) statistical baseline in difference-reversion form."""

    model_id = "ar2_v1"

    def __init__(self) -> None:
        self.g_anchor = 135.0
        self.reversion = -0.04
        self.momentum = 0.45
        self.innov_sd = 5.5

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        g = warmup_history["sensor_mgdl"].to_numpy(dtype=float)
        fin = g[np.isfinite(g)]
        if fin.size >= 12:
            self.g_anchor = float(np.clip(np.median(fin), 85.0, 220.0))
        ok = np.isfinite(g[2:]) & np.isfinite(g[1:-1]) & np.isfinite(g[:-2])
        if int(ok.sum()) >= 25:
            dy_next = g[2:][ok] - g[1:-1][ok]
            level_dev = g[1:-1][ok] - self.g_anchor
            dy_prev = g[1:-1][ok] - g[:-2][ok]
            design = np.column_stack([level_dev, dy_prev])
            pen = np.diag([400.0, 10.0])
            w = np.linalg.solve(design.T @ design + pen, design.T @ dy_next)
            self.reversion = float(np.clip(w[0], -0.045, 0.0))
            self.momentum = float(np.clip(w[1], -0.35, 0.72))
            resid = dy_next - design @ np.array([self.reversion, self.momentum])
            self.innov_sd = float(np.clip(np.std(resid), 2.0, 20.0))

    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        g = causal_history["sensor_mgdl"].to_numpy(dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(g))
        if finite_idx.size == 0:
            cur, prev, stale = self.g_anchor, self.g_anchor, len(g)
        elif finite_idx.size == 1:
            cur = prev = float(g[finite_idx[-1]])
            stale = int(len(g) - 1 - finite_idx[-1])
        else:
            last_i = int(finite_idx[-1])
            cur = float(g[last_i])
            prev = float(g[last_i - 1]) if np.isfinite(g[last_i - 1]) else cur
            stale = int(len(g) - 1 - last_i)

        max_steps = max(int(round(h / step_minutes)) for h in horizons_minutes) + stale
        path = np.empty(max_steps + 1, dtype=float)
        path[0] = cur
        c_cur, c_prev = cur, prev
        for k in range(1, max_steps + 1):
            delta = self.reversion * (c_cur - self.g_anchor) + self.momentum * (c_cur - c_prev)
            nxt = float(np.clip(c_cur + delta, 20.0, 500.0))
            path[k] = nxt
            c_prev, c_cur = c_cur, nxt

        pt, lo, hi = {}, {}, {}
        for h in horizons_minutes:
            idx = stale + max(1, int(round(h / step_minutes)))
            pred = float(path[idx])
            sd = self.innov_sd * float(np.sqrt(idx)) * 1.25
            pt[int(h)] = pred
            lo[int(h)] = float(np.clip(pred - Z_90 * sd, 20.0, 500.0))
            hi[int(h)] = float(np.clip(pred + Z_90 * sd, 20.0, 500.0))
        return ForecastOutput(pt, lo, hi)

    def predict_trace_batch_causal(
        self,
        blind_view: pd.DataFrame,
        eval_origins: np.ndarray,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raw = blind_view["sensor_mgdl"].to_numpy(dtype=float)
        last_vals, prev_vals, stale_all = _causal_last_valid_arrays(raw, default_val=self.g_anchor)
        c_cur = last_vals[eval_origins].copy()
        c_prev = prev_vals[eval_origins].copy()
        st = stale_all[eval_origins]

        max_h_steps = max(int(round(h / step_minutes)) for h in horizons_minutes)
        max_total = max_h_steps + int(np.max(st))
        paths = np.empty((len(eval_origins), max_total + 1), dtype=float)
        paths[:, 0] = c_cur
        for k in range(1, max_total + 1):
            delta = self.reversion * (c_cur - self.g_anchor) + self.momentum * (c_cur - c_prev)
            nxt = np.clip(c_cur + delta, 20.0, 500.0)
            paths[:, k] = nxt
            c_prev, c_cur = c_cur, nxt

        rows_idx = np.arange(len(eval_origins))
        out: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for h in horizons_minutes:
            idx = st + max(1, int(round(h / step_minutes)))
            preds = paths[rows_idx, idx]
            sds = self.innov_sd * np.sqrt(idx) * 1.25
            out[int(h)] = (
                preds,
                np.clip(preds - Z_90 * sds, 20.0, 500.0),
                np.clip(preds + Z_90 * sds, 20.0, 500.0),
            )
        return out


class RidgeWindowMLForecaster(GlucoseForecaster):
    """Regularized multi-feature ML forecaster using 60-min CGM lags + IOB + COB + staleness."""

    model_id = "ridge_ml_v1"

    def __init__(self, alpha: float = 15.0) -> None:
        self.alpha = float(alpha)
        self._weights: Dict[int, np.ndarray] = {}
        self._conf_sd: Dict[int, float] = {}

    @staticmethod
    def _feature_vector(
        g_filled: np.ndarray,
        insulin: np.ndarray,
        carbs: np.ndarray,
        idx: int,
        stale_steps: int,
    ) -> np.ndarray:
        cur = g_filled[idx]
        l1 = g_filled[max(0, idx - 1)]
        l3 = g_filled[max(0, idx - 3)]
        l6 = g_filled[max(0, idx - 6)]
        l12 = g_filled[max(0, idx - 12)]
        recent_ins = float(np.sum(insulin[max(0, idx - 24) : idx + 1]))
        recent_carbs = float(np.sum(carbs[max(0, idx - 24) : idx + 1]))
        return np.array(
            [
                1.0,
                (cur - 130.0) / 50.0,
                (cur - l1) / 10.0,
                (cur - l3) / 20.0,
                (cur - l6) / 30.0,
                (cur - l12) / 40.0,
                recent_ins / 5.0,
                recent_carbs / 50.0,
                min(stale_steps, 12) / 6.0,
            ],
            dtype=float,
        )

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        raw = warmup_history["sensor_mgdl"].to_numpy(dtype=float)
        g_filled = pd.Series(raw).ffill().bfill().fillna(130.0).to_numpy()
        ins = warmup_history.get("insulin_u", pd.Series(0.0, index=warmup_history.index)).to_numpy(dtype=float)
        carbs = warmup_history.get("carbs_announced_g", pd.Series(0.0, index=warmup_history.index)).to_numpy(dtype=float)

        for h in (30, 60, 120):
            k = max(1, int(round(h / step_minutes)))
            if len(g_filled) <= k + 15:
                continue
            xs, ys = [], []
            for idx in range(12, len(g_filled) - k):
                if not np.isfinite(raw[idx + k]):
                    continue
                stale = 0 if np.isfinite(raw[idx]) else 2
                xs.append(self._feature_vector(g_filled, ins, carbs, idx, stale))
                ys.append(raw[idx + k] - g_filled[idx])
            if len(xs) >= 25:
                design = np.vstack(xs)
                target = np.asarray(ys, dtype=float)
                cut = max(15, int(0.75 * len(design)))
                pen = self.alpha * np.eye(design.shape[1])
                pen[0, 0] = 1e-6
                w = np.linalg.solve(design[:cut].T @ design[:cut] + pen, design[:cut].T @ target[:cut])
                self._weights[h] = w
                if len(design) - cut >= 5:
                    cal_err = np.abs(target[cut:] - design[cut:] @ w)
                    self._conf_sd[h] = float(np.quantile(cal_err, 0.90) / Z_90)
                else:
                    self._conf_sd[h] = float(np.std(target - design @ w))

    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        raw = causal_history["sensor_mgdl"].to_numpy(dtype=float)
        last_val, stale_steps = _last_valid_and_staleness(raw)
        g_filled = pd.Series(raw).ffill().bfill().fillna(last_val).to_numpy()
        ins = causal_history.get("insulin_u", pd.Series(0.0, index=causal_history.index)).to_numpy(dtype=float)
        carbs = causal_history.get("carbs_announced_g", pd.Series(0.0, index=causal_history.index)).to_numpy(dtype=float)
        feat = self._feature_vector(g_filled, ins, carbs, len(g_filled) - 1, stale_steps)

        pt, lo, hi = {}, {}, {}
        for h in horizons_minutes:
            w = self._weights.get(int(h))
            delta = float(feat @ w) if w is not None else 0.0
            pred = float(np.clip(last_val + np.clip(delta, -180.0, 180.0), 20.0, 500.0))
            sd = self._conf_sd.get(int(h), 18.0) * float(np.sqrt(1.0 + 0.2 * stale_steps))
            pt[int(h)] = pred
            lo[int(h)] = float(np.clip(pred - Z_90 * sd, 20.0, 500.0))
            hi[int(h)] = float(np.clip(pred + Z_90 * sd, 20.0, 500.0))
        return ForecastOutput(pt, lo, hi)

    def predict_trace_batch_causal(
        self,
        blind_view: pd.DataFrame,
        eval_origins: np.ndarray,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raw = blind_view["sensor_mgdl"].to_numpy(dtype=float)
        last_vals, _, stale_all = _causal_last_valid_arrays(raw, default_val=130.0)
        ins = blind_view.get("insulin_u", pd.Series(0.0, index=blind_view.index)).to_numpy(dtype=float)
        carbs = blind_view.get("carbs_announced_g", pd.Series(0.0, index=blind_view.index)).to_numpy(dtype=float)

        feats = np.vstack(
            [self._feature_vector(last_vals, ins, carbs, int(o), int(stale_all[o])) for o in eval_origins]
        )
        base = last_vals[eval_origins]
        st = stale_all[eval_origins]
        out: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for h in horizons_minutes:
            w = self._weights.get(int(h))
            deltas = feats @ w if w is not None else np.zeros(len(eval_origins))
            preds = np.clip(base + np.clip(deltas, -180.0, 180.0), 20.0, 500.0)
            sds = self._conf_sd.get(int(h), 18.0) * np.sqrt(1.0 + 0.2 * st)
            out[int(h)] = (
                preds,
                np.clip(preds - Z_90 * sds, 20.0, 500.0),
                np.clip(preds + Z_90 * sds, 20.0, 500.0),
            )
        return out


class MechanisticKalmanGrayBoxForecaster(GlucoseForecaster):
    """Causal Kalman + Nonlinear RK4 Hovorka forecaster resilient to CGM dropouts."""

    model_id = "hovorka_graybox_v2"

    def __init__(self, meal_tau_min: float = 50.0, bioavailability: float = 0.80) -> None:
        self.meal_tau_min = float(meal_tau_min)
        self.bioavailability = float(bioavailability)
        self.basal_u_per_h = 1.5
        self.operating_glucose = 135.0
        self._conf_scale: Dict[int, float] = {30: 1.8, 60: 2.2, 120: 2.6}
        self._bias: Dict[int, float] = {}
        self._setup_plant()

    def _setup_plant(self) -> None:
        sigma = eq.basal_scale(self.basal_u_per_h, self.operating_glucose)
        self.patient = eq.make_equilibrium_patient(self.basal_u_per_h, sigma, self.operating_glucose)
        self.operating_state = np.asarray(self.patient.get_state()["ode_state"], dtype=float)[:8]
        self.plant_params = nlp.from_patient(
            self.patient,
            meal_tau_min=self.meal_tau_min,
            bioavailability=self.bioavailability,
        )
        plant = linearise(self.patient)
        self.linear_model = MealLinearModel.from_plant(
            plant,
            meal_tau_minutes=self.meal_tau_min,
            bioavailability=self.bioavailability,
        )

    def _filter_df(self, df: pd.DataFrame, step_minutes: float) -> tuple[MealObserver, int]:
        obs = MealObserver(
            self.linear_model,
            sensor_sd=6.0,
            meal_noise_g_per_sqrt_min=0.35,
            sensor_tau_minutes=6.0,
            noise_ar1_phi=0.75,
        )
        g = df["sensor_mgdl"].to_numpy(dtype=float)
        ins = df.get("insulin_u", pd.Series(self.basal_u_per_h * step_minutes / 60.0, index=df.index)).to_numpy(dtype=float)
        carbs = df.get("carbs_announced_g", pd.Series(0.0, index=df.index)).to_numpy(dtype=float)
        stale_flag = df.get("is_stale_displayed", pd.Series(False, index=df.index)).to_numpy(dtype=bool)

        misses = 0
        for i in range(len(df)):
            if i > 0:
                obs.predict(float(max(0.0, ins[i - 1])), step_minutes)
                if carbs[i - 1] > 0.0:
                    obs.announce_meal(float(carbs[i - 1]))
            if np.isfinite(g[i]) and not stale_flag[i]:
                obs.observe(float(g[i]))
                misses = 0
            else:
                misses += 1
        return obs, misses

    def fit(self, warmup_history: pd.DataFrame, step_minutes: float = 5.0) -> None:
        if "basal_u" in warmup_history and len(warmup_history) > 0:
            med_basal = float(np.nanmedian(warmup_history["basal_u"].to_numpy(dtype=float)) * 60.0 / step_minutes)
            if np.isfinite(med_basal) and med_basal > 0.05:
                self.basal_u_per_h = float(np.clip(med_basal, 0.2, 6.0))
        g = warmup_history["sensor_mgdl"].to_numpy(dtype=float)
        fin = g[np.isfinite(g)]
        if fin.size >= 12:
            self.operating_glucose = float(np.clip(np.median(fin), 90.0, 220.0))
        self._setup_plant()

    def predict_horizons(
        self,
        causal_history: pd.DataFrame,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> ForecastOutput:
        obs, misses = self._filter_df(causal_history, step_minutes)
        dev = obs.x[:10] * self.linear_model.scales
        phys = np.zeros((1, 10), dtype=float)
        phys[0, :8] = self.operating_state + dev[:8]
        phys[0, 8:] = dev[8:]
        phys = np.maximum(phys, 0.0)

        max_steps = max(1, max(int(round(h / step_minutes)) for h in horizons_minutes))
        rate = np.full((1, max_steps), self.basal_u_per_h, dtype=float)
        zero_carbs = np.zeros((1, max_steps), dtype=float)
        traj = nlp.propagate(phys, rate, zero_carbs, self.plant_params, step_minutes=step_minutes, substeps=2)[0]

        a_mat, _ = obs.transition(step_minutes)
        c_vec = obs.measurement_C
        cov = obs.P.copy()
        sds = np.empty(max_steps, dtype=float)
        for k in range(max_steps):
            cov = a_mat @ cov @ a_mat.T + obs.Q * step_minutes
            sds[k] = float(np.sqrt(max(1.0, c_vec @ cov @ c_vec + obs.readout_variance)))

        pt, lo, hi = {}, {}, {}
        for h in horizons_minutes:
            idx = max(1, int(round(h / step_minutes))) - 1
            pred = float(np.clip(traj[idx] + self._bias.get(int(h), 0.0), 20.0, 500.0))
            scale = self._conf_scale.get(int(h), 2.2) * float(np.sqrt(1.0 + 0.08 * misses))
            sd = sds[idx] * scale
            pt[int(h)] = pred
            lo[int(h)] = float(np.clip(pred - Z_90 * sd, 20.0, 500.0))
            hi[int(h)] = float(np.clip(pred + Z_90 * sd, 20.0, 500.0))
        return ForecastOutput(pt, lo, hi)

    def predict_trace_batch_causal(
        self,
        blind_view: pd.DataFrame,
        eval_origins: np.ndarray,
        horizons_minutes: Sequence[int],
        step_minutes: float = 5.0,
    ) -> Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Single causal Kalman pass + vectorised RK4 rollout across all origins."""
        obs = MealObserver(
            self.linear_model,
            sensor_sd=6.0,
            meal_noise_g_per_sqrt_min=0.35,
            sensor_tau_minutes=6.0,
            noise_ar1_phi=0.75,
        )
        g = blind_view["sensor_mgdl"].to_numpy(dtype=float)
        ins = blind_view.get(
            "insulin_u", pd.Series(self.basal_u_per_h * step_minutes / 60.0, index=blind_view.index)
        ).to_numpy(dtype=float)
        carbs = blind_view.get("carbs_announced_g", pd.Series(0.0, index=blind_view.index)).to_numpy(dtype=float)
        stale_flag = blind_view.get("is_stale_displayed", pd.Series(False, index=blind_view.index)).to_numpy(dtype=bool)

        n = len(blind_view)
        n_state = len(obs.x)
        states = np.empty((n, n_state), dtype=float)
        miss_arr = np.empty(n, dtype=int)
        misses = 0
        for i in range(n):
            if i > 0:
                obs.predict(float(max(0.0, ins[i - 1])), step_minutes)
                if carbs[i - 1] > 0.0:
                    obs.announce_meal(float(carbs[i - 1]))
            if np.isfinite(g[i]) and not stale_flag[i]:
                obs.observe(float(g[i]))
                misses = 0
            else:
                misses += 1
            states[i] = obs.x
            miss_arr[i] = misses

        dev = states[eval_origins][:, :10] * self.linear_model.scales[None, :]
        phys = np.zeros((len(eval_origins), 10), dtype=float)
        phys[:, :8] = self.operating_state[None, :] + dev[:, :8]
        phys[:, 8:] = dev[:, 8:]
        phys = np.maximum(phys, 0.0)

        max_steps = max(1, max(int(round(h / step_minutes)) for h in horizons_minutes))
        rate = np.full((len(eval_origins), max_steps), self.basal_u_per_h, dtype=float)
        zero_carbs = np.zeros((len(eval_origins), max_steps), dtype=float)
        traj = nlp.propagate(phys, rate, zero_carbs, self.plant_params, step_minutes=step_minutes, substeps=2)

        a_mat, _ = obs.transition(step_minutes)
        c_vec = obs.measurement_C
        cov = obs.P.copy()
        base_sds = np.empty(max_steps, dtype=float)
        for k in range(max_steps):
            cov = a_mat @ cov @ a_mat.T + obs.Q * step_minutes
            base_sds[k] = float(np.sqrt(max(1.0, c_vec @ cov @ c_vec + obs.readout_variance)))

        origin_misses = miss_arr[eval_origins]
        out_map: Dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for h in horizons_minutes:
            idx = max(1, int(round(h / step_minutes))) - 1
            preds = np.clip(traj[:, idx] + self._bias.get(int(h), 0.0), 20.0, 500.0)
            scale = self._conf_scale.get(int(h), 2.2) * np.sqrt(1.0 + 0.08 * origin_misses)
            sds = base_sds[idx] * scale
            lo = np.clip(preds - Z_90 * sds, 20.0, 500.0)
            hi = np.clip(preds + Z_90 * sds, 20.0, 500.0)
            out_map[int(h)] = (preds, lo, hi)
        return out_map


_FORECASTER_REGISTRY: Dict[str, Callable[..., GlucoseForecaster]] = {
    PersistenceForecaster.model_id: PersistenceForecaster,
    LinearSavgolForecaster.model_id: LinearSavgolForecaster,
    AR2Forecaster.model_id: AR2Forecaster,
    RidgeWindowMLForecaster.model_id: RidgeWindowMLForecaster,
    MechanisticKalmanGrayBoxForecaster.model_id: MechanisticKalmanGrayBoxForecaster,
    "baseline_v1": PersistenceForecaster,
    "statistical_ar2": AR2Forecaster,
    "ml_ridge": RidgeWindowMLForecaster,
}


def register_forecaster(model_id: str, factory: Callable[..., GlucoseForecaster] | Type[GlucoseForecaster]) -> None:
    _FORECASTER_REGISTRY[str(model_id)] = factory


def list_forecasters() -> List[str]:
    return sorted(_FORECASTER_REGISTRY.keys())


def create_forecaster(spec: AlgorithmSpec) -> GlucoseForecaster:
    if spec.model_id not in _FORECASTER_REGISTRY:
        raise KeyError(
            f"Unknown forecaster '{spec.model_id}'. Available: {', '.join(list_forecasters())}"
        )
    return _FORECASTER_REGISTRY[spec.model_id](**spec.params)


def _build_blind_view(trace: pd.DataFrame, contract: MissingDataContract) -> pd.DataFrame:
    """Construct the strictly sanitized observable stream allowed by ``contract``."""
    sensor = trace["cgm_displayed_mgdl"].to_numpy(dtype=float).copy()
    if contract.impute_before_algorithm == "ffill":
        sensor = pd.Series(sensor).ffill().to_numpy(dtype=float)
    elif not contract.allow_last_valid_observation:
        stale = trace["is_stale_displayed"].to_numpy(dtype=bool)
        sensor = np.where(stale, np.nan, sensor)

    view = pd.DataFrame(
        {
            "t_min": trace["t_min"].to_numpy(dtype=float),
            "minute_of_day": trace["minute_of_day"].to_numpy(dtype=float),
            "sensor_mgdl": sensor,
        }
    )
    if contract.expose_stale_flag_to_algorithm:
        view["is_stale_displayed"] = trace["is_stale_displayed"].to_numpy(dtype=bool)
    if contract.expose_insulin_and_announced_carbs:
        view["basal_u"] = trace["basal_u"].to_numpy(dtype=float)
        view["bolus_u"] = trace["bolus_u"].to_numpy(dtype=float)
        view["insulin_u"] = trace["insulin_u"].to_numpy(dtype=float)
        view["carbs_announced_g"] = trace["carbs_announced_g"].to_numpy(dtype=float)
    return view


def evaluate_forecasters_on_trace(
    trace: pd.DataFrame,
    algorithms: Sequence[AlgorithmSpec],
    evaluation: EvaluationSpec,
    step_minutes: float = 5.0,
) -> pd.DataFrame:
    """Run all ``algorithms`` causally on the exact same simulated sensor trace."""
    contract = evaluation.missing_data_contract
    blind_view = _build_blind_view(trace, contract)
    horizons = evaluation.all_horizons()
    max_h_steps = max(int(round(h / step_minutes)) for h in horizons)
    win_steps = max(12, int(round(contract.history_window_minutes / step_minutes)))

    in_eval = trace["in_eval_window"].to_numpy(dtype=bool)
    warmup_idx = np.flatnonzero(~in_eval)
    eval_origins = np.flatnonzero(in_eval)
    eval_origins = eval_origins[eval_origins + max_h_steps < len(trace)]

    target_col = "bg_true_mgdl" if contract.reference_target == "true_plasma_glucose" else "cgm_clean_mgdl"
    y_ref = trace[target_col].to_numpy(dtype=float)
    eval_hour_arr = trace["eval_hour"].to_numpy(dtype=float)
    is_drop = trace["is_dropout"].to_numpy(dtype=bool)
    is_stale = trace["is_stale_displayed"].to_numpy(dtype=bool)
    rec_steps = trace["steps_since_recovery"].to_numpy(dtype=int)
    rec_limit = int(round(evaluation.recovery_window_minutes / step_minutes))

    frames: List[pd.DataFrame] = []
    for algo_spec in algorithms:
        forecaster = create_forecaster(algo_spec)
        if warmup_idx.size >= 12:
            forecaster.fit(blind_view.iloc[: int(warmup_idx[-1]) + 1], step_minutes=step_minutes)
        else:
            forecaster.fit(blind_view.iloc[: min(36, len(blind_view))], step_minutes=step_minutes)

        batch_fn = getattr(forecaster, "predict_trace_batch_causal", None)
        if callable(batch_fn) and eval_origins.size > 0:
            batch_out = batch_fn(blind_view, eval_origins, horizons, step_minutes=step_minutes)
            for h in horizons:
                k = max(1, int(round(h / step_minutes)))
                truth = y_ref[eval_origins + k]
                preds, los, his = batch_out[int(h)]
                signed = preds - truth
                frames.append(
                    pd.DataFrame(
                        {
                            "algorithm_id": algo_spec.model_id,
                            "origin_idx": eval_origins,
                            "eval_hour": eval_hour_arr[eval_origins],
                            "horizon_min": int(h),
                            "y_true": truth,
                            "y_pred": preds,
                            "lower_90": los,
                            "upper_90": his,
                            "abs_error": np.abs(signed),
                            "sq_error": signed ** 2,
                            "signed_error": signed,
                            "inside_90": (los <= truth) & (truth <= his),
                            "origin_in_dropout": is_drop[eval_origins],
                            "origin_stale_displayed": is_stale[eval_origins],
                            "origin_in_recovery": (rec_steps[eval_origins] >= 0) & (rec_steps[eval_origins] < rec_limit),
                            "steps_since_recovery": rec_steps[eval_origins],
                        }
                    )
                )
            continue

        records: List[Dict[str, object]] = []
        for origin in eval_origins:
            start_i = max(0, int(origin) - win_steps + 1)
            causal_slice = blind_view.iloc[start_i : int(origin) + 1]
            out = forecaster.predict_horizons(causal_slice, horizons, step_minutes=step_minutes)
            for h in horizons:
                k = max(1, int(round(h / step_minutes)))
                truth_v = float(y_ref[int(origin) + k])
                pred_v = float(out.point_mgdl[int(h)])
                lo_v = float(out.lower_90_mgdl[int(h)])
                hi_v = float(out.upper_90_mgdl[int(h)])
                records.append(
                    {
                        "algorithm_id": algo_spec.model_id,
                        "origin_idx": int(origin),
                        "eval_hour": float(eval_hour_arr[int(origin)]),
                        "horizon_min": int(h),
                        "y_true": truth_v,
                        "y_pred": pred_v,
                        "lower_90": lo_v,
                        "upper_90": hi_v,
                        "abs_error": abs(pred_v - truth_v),
                        "sq_error": (pred_v - truth_v) ** 2,
                        "signed_error": pred_v - truth_v,
                        "inside_90": bool(lo_v <= truth_v <= hi_v),
                        "origin_in_dropout": bool(is_drop[int(origin)]),
                        "origin_stale_displayed": bool(is_stale[int(origin)]),
                        "origin_in_recovery": bool(0 <= rec_steps[int(origin)] < rec_limit),
                        "steps_since_recovery": int(rec_steps[int(origin)]),
                    }
                )
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
