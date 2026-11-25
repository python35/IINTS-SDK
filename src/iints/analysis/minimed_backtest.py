"""Rolling-origin falsification harness for the Gray-Box MiniMed 780G predictor.

Combines:
- Equilibrium-consistent Hovorka parameterisation (``equilibrium.py``),
- Linear Kalman filter for causal state reconstruction across CGM gaps,
- Vectorised nonlinear RK4 forward propagation (``nonlinear_propagator.py``),
- Split-conformal uncertainty calibration per horizon and glucose band,
- Strict AR(2)-nested Gray-Box blending (optional, default True) so the Ridge
  layer regresses ``y_true - G_i`` on ``[1, Fourier(24h), G_i, G_i - G_{i-1},
  slope, innovation, (y_wb - G_i), meal_on_board]`` and can shrink noisy
  long-horizon physics variance while preserving 30/60-min physics gains.

Research, education and in-silico evaluation only; not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from iints.analysis import nonlinear_propagator as nlp
from iints.core.control.linear_plant import linearise
from iints.core.control.meal_observer import MealLinearModel, MealObserver
from iints.core.patient import equilibrium as pf

__all__ = [
    "HORIZONS",
    "STEP_MINUTES",
    "Z_90",
    "PhysicsConfig",
    "GrayBoxEngine",
    "smoothed_slope",
    "fit_ar2",
    "ar2_forecast",
    "add_strata",
    "walk_forward",
    "score",
]

HORIZONS: Tuple[int, ...] = (30, 60, 120)
STEP_MINUTES = 5.0
Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class PhysicsConfig:
    """Everything the white-box and gray-box core needs, with sensitivity as one knob."""

    sigma: Optional[float] = None          # None -> implied by the basal rate
    basal_u_per_h: Optional[float] = None  # None -> median delivered rate
    meal_tau_min: float = 40.0
    bioavailability: float = 0.80
    sensor_tau_min: float = 6.0
    noise_ar1_phi: float = 0.75
    sensor_sd: float = 6.0
    meal_noise_g_per_sqrt_min: float = 0.4
    q_scale: float = 1.0
    ridge_alpha: float = 25.0
    fourier_harmonics: int = 2
    max_residual_mgdl: float = 60.0
    warmup_steps: int = 12
    propagation: str = "linear"            # "linear" | "nonlinear"
    feedback_gain: float = 0.0             # U/h per mg/dL, 0 disables closed loop
    feedback_target_mgdl: float = 120.0
    counterregulation: bool = True
    substeps: int = 2
    free_rate_window_steps: int = 12
    calibration_fraction: float = 0.25
    ar2_nested_graybox: bool = True        # True: regress y_true - G_i on AR(2) + (y_wb - G_i) + MOB
    meal_model: str = "hovorka_2comp"      # "hovorka_2comp" | "biphasic" | "glucose_gated"
    fast_carb_fraction: float = 0.35
    fast_meal_tau_min: float = 25.0
    gastric_glucose_gain: float = 0.35

    def label(self) -> str:
        sigma = "auto" if self.sigma is None else f"{self.sigma:.4f}"
        nest = "ar2nest" if self.ar2_nested_graybox else "resid"
        return (
            f"{self.propagation},{nest},{self.meal_model},sigma={sigma},"
            f"tauM={self.meal_tau_min:g},AG={self.bioavailability:g},"
            f"Q={self.q_scale:g},K={self.feedback_gain:g}"
        )


def _delivered_basal_rate(frame: pd.DataFrame) -> float:
    """Median auto-basal rate in U/h, excluding bolus steps."""
    rate = frame["auto_basal_u"].to_numpy(dtype=float) * 60.0 / STEP_MINUTES
    positive = rate[rate > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.clip(np.median(positive), 0.1, 6.0))


class GrayBoxEngine:
    """One physics configuration, fitted and queried on a timeline."""

    def __init__(self, config: PhysicsConfig, timeline: pd.DataFrame) -> None:
        self.config = config
        self.basal = config.basal_u_per_h or _delivered_basal_rate(timeline)
        self.sigma = config.sigma if config.sigma is not None else pf.basal_scale(self.basal, 140.0)
        self.operating_glucose = pf.equilibrium_glucose(self.basal, self.sigma)
        patient = pf.make_equilibrium_patient(self.basal, self.sigma, self.operating_glucose)
        self.patient = patient
        self.operating_state = np.asarray(patient.get_state()["ode_state"], dtype=float)[:8]
        self.plant_params = nlp.from_patient(
            patient,
            meal_tau_min=config.meal_tau_min,
            bioavailability=config.bioavailability,
            counterregulation=config.counterregulation,
            meal_model=config.meal_model,
            fast_carb_fraction=config.fast_carb_fraction,
            fast_meal_tau_min=config.fast_meal_tau_min,
            gastric_glucose_gain=config.gastric_glucose_gain,
        )
        plant = linearise(patient)
        self.model = MealLinearModel.from_plant(
            plant,
            meal_tau_minutes=config.meal_tau_min,
            bioavailability=config.bioavailability,
        )
        self._template = self._new_observer()
        self._precompute()
        self._weights: Dict[int, np.ndarray] = {}
        self._sd_cache: Dict[int, np.ndarray] = {}

    def _new_observer(self) -> MealObserver:
        cfg = self.config
        observer = MealObserver(
            self.model,
            sensor_sd=cfg.sensor_sd,
            meal_noise_g_per_sqrt_min=cfg.meal_noise_g_per_sqrt_min,
            sensor_tau_minutes=cfg.sensor_tau_min,
            noise_ar1_phi=cfg.noise_ar1_phi,
        )
        observer.Q = observer.Q * cfg.q_scale
        return observer

    def _precompute(self) -> None:
        """Precompute ``C A^k``, insulin impulse response, and carb impulse response."""
        observer = self._template
        a_mat, b_vec = observer.transition(STEP_MINUTES)
        c_vec = observer.measurement_C
        self.n_state = len(c_vec)
        steps = int(max(HORIZONS) / STEP_MINUTES)
        self.max_steps = steps
        rows = np.empty((steps, self.n_state))
        insulin_impulse = np.empty(steps)
        carb_impulse = np.empty(steps)
        power = np.eye(self.n_state)
        carb_unit = np.zeros(self.n_state)
        carb_unit[8] = 1.0 / self.model.scales[8]
        for k in range(steps):
            insulin_impulse[k] = float(c_vec @ (power @ b_vec))
            carb_impulse[k] = float(c_vec @ (power @ carb_unit))
            power = a_mat @ power
            rows[k] = c_vec @ power
        self.free_rows = rows
        self.insulin_impulse = insulin_impulse
        self.carb_impulse = carb_impulse
        self.A_step, self.B_step, self.C_vec = a_mat, b_vec, c_vec

    def filter_timeline(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        """One causal pass: predict every step, update only where CGM is finite."""
        observer = self._new_observer()
        glucose = frame["glucose"].to_numpy(dtype=float)
        insulin = frame["insulin_u"].to_numpy(dtype=float)
        carbs = frame["carbs_g"].to_numpy(dtype=float)
        minutes = frame["t_min"].to_numpy(dtype=float)
        n = len(frame)

        states = np.zeros((n, self.n_state))
        innovation = np.zeros(n)
        miss_count = np.zeros(n, dtype=int)
        age = np.zeros(n, dtype=int)

        observed = np.isfinite(glucose)
        misses = 0
        segment_age = 0
        for i in range(n):
            if i > 0:
                step = float(np.clip(minutes[i] - minutes[i - 1], 0.5, 10.0))
                observer.predict(float(max(0.0, insulin[i - 1])), step)
                if carbs[i - 1] > 0.0:
                    observer.announce_meal(float(carbs[i - 1]))
            if observed[i]:
                observer.observe(float(glucose[i]))
                innovation[i] = observer.innovation
                misses = 0
            else:
                misses += 1
            segment_age += 1
            states[i] = observer.x
            miss_count[i] = misses
            age[i] = segment_age
        return {
            "state": states,
            "innovation": innovation,
            "miss_count": miss_count,
            "age": age,
            "observed": observed,
        }

    def forecast_sd(self, misses: int) -> np.ndarray:
        """Converged Kalman forecast SDs after ``misses`` predict-only steps."""
        key = int(min(misses, 36))
        if key in self._sd_cache:
            return self._sd_cache[key]
        observer = self._new_observer()
        a_mat, _ = observer.transition(STEP_MINUTES)
        c_vec = observer.measurement_C
        for _ in range(400):
            observer.predict(self.basal * STEP_MINUTES / 60.0, STEP_MINUTES)
            observer.observe(self.model.glucose_mgdl)
        for _ in range(key):
            observer.predict(self.basal * STEP_MINUTES / 60.0, STEP_MINUTES)
        covariance = observer.P.copy()
        out = np.empty(self.max_steps)
        for k in range(self.max_steps):
            covariance = a_mat @ covariance @ a_mat.T + observer.Q * STEP_MINUTES
            if observer.noise_index is not None:
                j = observer.noise_index
                covariance[j, j] += observer.sensor_variance * (1.0 - observer.noise_ar1_phi ** 2.0)
            out[k] = float(np.sqrt(max(1.0, c_vec @ covariance @ c_vec + observer.readout_variance)))
        self._sd_cache[key] = out
        return out

    def forecasts(
        self,
        frame: pd.DataFrame,
        filtered: Dict[str, np.ndarray],
        origins: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """White-box forecasts, blind (``free``) and input-informed (``informed``)."""
        states = filtered["state"][origins]
        free = states @ self.free_rows.T + self.model.glucose_mgdl

        insulin = frame["insulin_u"].to_numpy(dtype=float)
        carbs = frame["carbs_g"].to_numpy(dtype=float)
        n = len(frame)
        pad_insulin = np.concatenate([insulin, np.zeros(self.max_steps)])
        pad_carbs = np.concatenate([carbs, np.zeros(self.max_steps)])

        deviation = np.empty((len(origins), self.max_steps))
        carb_future = np.empty((len(origins), self.max_steps))
        for j in range(self.max_steps):
            deviation[:, j] = pad_insulin[origins + j] * 60.0 / STEP_MINUTES - self.basal
            carb_future[:, j] = pad_carbs[origins + j]

        window = self.config.free_rate_window_steps
        if window > 0:
            kernel = np.ones(window) / window
            basal_series = frame.get("auto_basal_u", frame["insulin_u"]).to_numpy(dtype=float)
            trailing = np.convolve(basal_series * 60.0 / STEP_MINUTES, kernel)[:n]
            held_rate = trailing[origins]
        else:
            held_rate = np.full(len(origins), self.basal)

        if self.config.propagation == "nonlinear":
            physical = self.physical_states(filtered, origins)
            held = np.repeat(held_rate[:, None], self.max_steps, axis=1)
            zero_carbs = np.zeros((len(origins), self.max_steps))
            actual_rate = deviation + self.basal
            free_nl = nlp.propagate(
                physical,
                held,
                zero_carbs,
                self.plant_params,
                step_minutes=STEP_MINUTES,
                substeps=self.config.substeps,
                feedback_gain=self.config.feedback_gain,
                feedback_target_mgdl=self.config.feedback_target_mgdl,
            )
            informed_nl = nlp.propagate(
                physical,
                actual_rate,
                carb_future,
                self.plant_params,
                step_minutes=STEP_MINUTES,
                substeps=self.config.substeps,
            )
            return {"free": free_nl, "informed": informed_nl}

        offset = np.cumsum(self.insulin_impulse)[None, :] * (held_rate - self.basal)[:, None]
        free = free + offset
        informed = free - offset
        for k in range(1, self.max_steps + 1):
            for j in range(k):
                weight = self.insulin_impulse[k - 1 - j]
                informed[:, k - 1] += weight * deviation[:, j]
                informed[:, k - 1] += self.carb_impulse[k - 1 - j] * carb_future[:, j]
        return {"free": np.clip(free, 20.0, 500.0), "informed": np.clip(informed, 20.0, 500.0)}

    def physical_states(self, filtered: Dict[str, np.ndarray], origins: np.ndarray) -> np.ndarray:
        """Convert scaled Kalman deviations into absolute Hovorka states, shape ``(n, 10)``."""
        deviations = filtered["state"][origins][:, :10] * self.model.scales[None, :]
        physical = np.empty((len(origins), 10))
        physical[:, :8] = self.operating_state[None, :] + deviations[:, :8]
        physical[:, 8:] = deviations[:, 8:]
        return np.maximum(physical, 0.0)

    def features(
        self,
        frame: pd.DataFrame,
        filtered: Dict[str, np.ndarray],
        origins: np.ndarray,
        slope: np.ndarray,
        whitebox_at_horizon: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Design matrix for the Gray-Box Ridge layer.

        When ``ar2_nested_graybox=True``, strictly nests AR(2) (``G_i``, ``G_i - G_{i-1}``),
        the nonlinear Hovorka physics delta ``(y_wb - G_i)/30``, and Kalman meal-on-board
        ``(D1 + D2)/30`` alongside the 24h Fourier harmonics, slope, and innovation.
        """
        minute = frame["minute_of_day"].to_numpy(dtype=float)[origins]
        all_glucose = pd.Series(frame["glucose"].to_numpy(dtype=float)).interpolate(limit_direction="both").to_numpy()
        glucose = all_glucose[origins]
        prev_glucose = all_glucose[np.maximum(origins - 1, 0)]
        innovation = filtered["innovation"][origins]

        columns = [np.ones(len(origins))]
        for m in range(1, self.config.fourier_harmonics + 1):
            angle = 2.0 * np.pi * m * (minute % 1440.0) / 1440.0
            columns += [np.sin(angle), np.cos(angle)]
        columns.append((glucose - 120.0) / 50.0)
        columns.append(np.clip(slope[origins], -4.0, 4.0))
        columns.append(np.clip(innovation / 20.0, -5.0, 5.0))

        if self.config.ar2_nested_graybox:
            columns.append(np.clip((glucose - prev_glucose) / 10.0, -5.0, 5.0))
            if whitebox_at_horizon is not None:
                columns.append(np.clip((whitebox_at_horizon - glucose) / 30.0, -8.0, 8.0))
            else:
                columns.append(np.zeros(len(origins)))
            phys = self.physical_states(filtered, origins)
            mob_g = phys[:, 8] + phys[:, 9]
            columns.append(np.clip(mob_g / 30.0, 0.0, 6.0))
        return np.column_stack(columns)

    def fit_residual(self, horizon: int, design: np.ndarray, target_delta: np.ndarray) -> None:
        penalty = self.config.ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 1e-6
        self._weights[horizon] = np.linalg.solve(design.T @ design + penalty, design.T @ target_delta)

    def apply_residual(self, horizon: int, design: np.ndarray) -> np.ndarray:
        weights = self._weights.get(horizon)
        if weights is None:
            return np.zeros(len(design))
        cap = 220.0 if self.config.ar2_nested_graybox else self.config.max_residual_mgdl
        return np.clip(design @ weights, -cap, cap)


def smoothed_slope(glucose: np.ndarray, window: int = 9, order: int = 2) -> np.ndarray:
    """mg/dL per minute from a Savitzky-Golay derivative."""
    series = pd.Series(glucose).interpolate(limit_direction="both").to_numpy()
    if len(series) < window:
        return np.zeros_like(series)
    return savgol_filter(series, window, order, deriv=1, delta=STEP_MINUTES)


def fit_ar2(glucose: np.ndarray) -> np.ndarray:
    """Least-squares AR(2) fitted on the training window."""
    series = pd.Series(glucose).interpolate(limit_direction="both").to_numpy()
    y = series[2:]
    design = np.column_stack([np.ones(len(y)), series[1:-1], series[:-2]])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def ar2_forecast(
    glucose: np.ndarray,
    origins: np.ndarray,
    coefficients: np.ndarray,
    steps: int,
) -> np.ndarray:
    series = pd.Series(glucose).interpolate(limit_direction="both").to_numpy()
    current = series[origins].astype(float)
    previous = series[np.maximum(origins - 1, 0)].astype(float)
    out = np.empty((len(origins), steps))
    for k in range(steps):
        nxt = coefficients[0] + coefficients[1] * current + coefficients[2] * previous
        out[:, k] = nxt
        previous, current = current, nxt
    return np.clip(out, 20.0, 500.0)


def add_strata(
    table: pd.DataFrame,
    frame: pd.DataFrame,
    origins: np.ndarray,
    bad_days: Iterable,
) -> pd.DataFrame:
    """Attach clinical regime labels so errors can be attributed by stratum."""
    carbs = frame["carbs_g"].to_numpy(dtype=float)
    bolus = frame["bolus_u"].to_numpy(dtype=float)
    minutes = frame["t_min"].to_numpy(dtype=float)

    def minutes_since(flag: np.ndarray) -> np.ndarray:
        last = np.full(len(frame), -np.inf)
        marker = -np.inf
        for i in range(len(frame)):
            if flag[i]:
                marker = minutes[i]
            last[i] = minutes[i] - marker
        return last

    since_meal = minutes_since(carbs > 0.0)[origins]
    since_bolus = minutes_since(bolus > 0.5)[origins]
    table["minutes_since_meal"] = since_meal
    table["minutes_since_bolus"] = since_bolus
    table["post_meal"] = since_meal <= 180.0
    table["post_bolus"] = since_bolus <= 180.0
    table["nocturnal"] = frame["minute_of_day"].to_numpy()[origins] < 360
    table["bad_sensor_day"] = pd.Series(frame["day"].to_numpy()[origins]).isin(set(bad_days)).to_numpy()
    table["target_hypo"] = table["y_true"] < 70.0
    table["target_hyper"] = table["y_true"] > 180.0
    return table


def walk_forward(
    frame: pd.DataFrame,
    config: PhysicsConfig,
    *,
    train_days: Optional[float] = 21.0,
    test_days: float = 7.0,
    initial_train_days: float = 30.0,
    bad_days: Sequence = (),
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Causal rolling-origin walk-forward evaluation with split-conformal calibration."""
    frame = frame.reset_index(drop=True)
    minutes = frame["t_min"].to_numpy(dtype=float)
    glucose = frame["glucose"].to_numpy(dtype=float)
    span_start, span_end = minutes[0], minutes[-1]
    slope = smoothed_slope(glucose)

    engine_prototype = GrayBoxEngine(config, frame)
    filtered = engine_prototype.filter_timeline(frame)
    steps_max = engine_prototype.max_steps

    results: List[pd.DataFrame] = []
    fold = 0
    test_start = span_start + initial_train_days * 1440.0
    while test_start < span_end:
        test_end = min(test_start + test_days * 1440.0, span_end)
        train_start = span_start if train_days is None else max(span_start, test_start - train_days * 1440.0)

        train_mask = (minutes >= train_start) & (minutes < test_start)
        test_mask = (minutes >= test_start) & (minutes < test_end)
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) < 500 or len(test_idx) < 50:
            test_start = test_end
            continue

        engine = GrayBoxEngine(config, frame)
        engine._sd_cache = engine_prototype._sd_cache
        ar_coefficients = fit_ar2(glucose[train_idx])

        def eligible(indices: np.ndarray) -> np.ndarray:
            ok = (filtered["age"][indices] >= config.warmup_steps) & np.isfinite(glucose[indices])
            ok &= indices + steps_max < len(frame)
            return indices[ok]

        all_train = eligible(train_idx)
        cut = int(len(all_train) * (1.0 - config.calibration_fraction))
        train_origins, calib_origins = all_train[:cut], all_train[cut:]
        test_origins = eligible(test_idx)
        if len(train_origins) < 200 or len(test_origins) < 20:
            test_start = test_end
            continue

        train_fc = engine.forecasts(frame, filtered, train_origins)
        test_fc = engine.forecasts(frame, filtered, test_origins)
        calib_fc = engine.forecasts(frame, filtered, calib_origins) if len(calib_origins) > 100 else None

        for horizon in horizons:
            k = int(horizon / STEP_MINUTES)
            train_wb = train_fc["free"][:, k - 1]
            test_wb = test_fc["free"][:, k - 1]
            train_design = engine.features(frame, filtered, train_origins, slope, train_wb)
            test_design = engine.features(frame, filtered, test_origins, slope, test_wb)

            train_target = glucose[train_origins + k]
            train_ok = np.isfinite(train_target)
            if train_ok.sum() > 50:
                if config.ar2_nested_graybox:
                    delta_target = train_target[train_ok] - glucose[train_origins[train_ok]]
                else:
                    delta_target = train_target[train_ok] - train_wb[train_ok]
                engine.fit_residual(horizon, train_design[train_ok], delta_target)

            scale = 1.0
            scale_by_band: Dict[int, float] = {}
            if calib_fc is not None and len(calib_origins) > 100:
                calib_wb = calib_fc["free"][:, k - 1]
                calib_design = engine.features(frame, filtered, calib_origins, slope, calib_wb)
                calib_target = glucose[calib_origins + k]
                ok = np.isfinite(calib_target)
                if ok.sum() > 50:
                    corr_c = engine.apply_residual(horizon, calib_design[ok])
                    base_c = glucose[calib_origins[ok]] if config.ar2_nested_graybox else calib_wb[ok]
                    centre = np.clip(base_c + corr_c, 20.0, 500.0)
                    sd_calib = np.array(
                        [engine.forecast_sd(m)[k - 1] for m in filtered["miss_count"][calib_origins[ok]]]
                    )
                    ratio = np.abs(calib_target[ok] - centre) / np.maximum(sd_calib, 1.0)
                    scale = float(np.quantile(ratio, 0.90) / Z_90)
                    band = np.digitize(glucose[calib_origins[ok]], [100.0, 180.0])
                    for b in (0, 1, 2):
                        sel = band == b
                        if sel.sum() > 40:
                            scale_by_band[b] = float(np.quantile(ratio[sel], 0.90) / Z_90)

            target = glucose[test_origins + k]
            keep = np.isfinite(target)
            if keep.sum() == 0:
                continue
            origins = test_origins[keep]
            white = test_wb[keep]
            informed = test_fc["informed"][keep, k - 1]
            correction = engine.apply_residual(horizon, test_design[keep])
            base_pred = glucose[origins] if config.ar2_nested_graybox else white
            y_gb = np.clip(base_pred + correction, 20.0, 500.0)
            sd = np.array([engine.forecast_sd(m)[k - 1] for m in filtered["miss_count"][origins]])

            table = pd.DataFrame(
                {
                    "fold": fold,
                    "horizon": horizon,
                    "origin_idx": origins,
                    "origin_time": frame["timestamp_dt"].to_numpy()[origins],
                    "y_true": target[keep],
                    "y_persistence": glucose[origins],
                    "y_linear": np.clip(glucose[origins] + slope[origins] * horizon, 20.0, 500.0),
                    "y_ar2": ar2_forecast(glucose, origins, ar_coefficients, k)[:, -1],
                    "y_whitebox": white,
                    "y_whitebox_informed": informed,
                    "y_graybox": y_gb,
                    "sd_whitebox": sd,
                    "glucose_at_origin": glucose[origins],
                    "slope_at_origin": slope[origins],
                    "conformal_scale": scale,
                    "conformal_scale_band": [
                        scale_by_band.get(int(b), scale)
                        for b in np.digitize(glucose[origins], [100.0, 180.0])
                    ],
                }
            )
            results.append(add_strata(table, frame, origins, bad_days))
        fold += 1
        test_start = test_end

    if not results:
        raise RuntimeError("walk_forward produced no folds - check the window sizes")
    out = pd.concat(results, ignore_index=True)
    out.attrs["config"] = config.label()
    return out


def score(
    table: pd.DataFrame,
    columns: Sequence[str] = (
        "y_persistence",
        "y_linear",
        "y_ar2",
        "y_whitebox",
        "y_whitebox_informed",
        "y_graybox",
    ),
    by: Sequence[str] = ("horizon",),
) -> pd.DataFrame:
    """Compute RMSE, MAE, and bias per model, grouped by ``by``."""
    records = []
    for keys, group in table.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for column in columns:
            error = group[column].to_numpy() - group["y_true"].to_numpy()
            records.append(
                {
                    **dict(zip(by, keys)),
                    "model": column.replace("y_", ""),
                    "n": len(group),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                }
            )
    return pd.DataFrame(records)
