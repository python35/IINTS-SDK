import numpy as np
from scipy.optimize import minimize
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel, BergmanParameters

class DigitalTwinCalibrator:
    """
    Fit a limited AdvancedMetabolicModel parameter set to research data.

    The result is a calibrated simulation profile, not a clinically validated
    or uniquely identifiable patient digital twin.
    """
    
    def __init__(
        self,
        real_glucose_array,
        carbs_array,
        insulin_array,
        time_step_min=5.0,
        basal_insulin_rate_u_per_hour=0.8,
        observation_semantics="cgm_observation",
    ):
        self.real_glucose = np.asarray(real_glucose_array, dtype=float).reshape(-1)
        self.carbs = np.asarray(carbs_array, dtype=float).reshape(-1)
        self.insulin = np.asarray(insulin_array, dtype=float).reshape(-1)
        self.time_step = float(time_step_min)
        self.basal_insulin_rate = float(basal_insulin_rate_u_per_hour)
        self.observation_semantics = str(observation_semantics)
        if not (len(self.real_glucose) == len(self.carbs) == len(self.insulin)):
            raise ValueError("glucose, carbs, and insulin arrays must have equal length")
        if len(self.real_glucose) < 24:
            raise ValueError("at least 24 samples are required for calibration")
        if not np.all(np.isfinite(self.real_glucose)):
            raise ValueError("glucose must contain only finite values")
        if not np.all(np.isfinite(self.carbs)) or np.any(self.carbs < 0.0):
            raise ValueError("carbs must contain finite non-negative interval events")
        if not np.all(np.isfinite(self.insulin)) or np.any(self.insulin < 0.0):
            raise ValueError("insulin must contain finite non-negative bolus events")
        if not np.isfinite(self.time_step) or self.time_step <= 0.0:
            raise ValueError("time_step_min must be finite and positive")
        if not np.isfinite(self.basal_insulin_rate) or self.basal_insulin_rate < 0.0:
            raise ValueError("basal_insulin_rate_u_per_hour must be finite and non-negative")
        if self.observation_semantics not in {"cgm_observation", "latent_glucose"}:
            raise ValueError(
                "observation_semantics must be 'cgm_observation' or 'latent_glucose'"
            )
        
    def _simulate_and_score(self, params_array, *, end_index=None, score_start_index=0):
        """
        params_array: [p1_scale, p3_scale, meal_tmax_scale]
        Returns MSE between simulated glucose and real glucose.
        """
        # Ensure parameters are positive
        if any(p <= 0 for p in params_array):
            return 1e6
            
        p1_val = 0.028 * params_array[0]
        p3_val = 5.0e-6 * params_array[1]
        tau_meal = 40.0 * params_array[2]
        
        # Build model with these params
        p = BergmanParameters(p1=p1_val, p3=p3_val, tau_meal=tau_meal)
        model = AdvancedMetabolicModel(
            initial_glucose=float(self.real_glucose[0]),
            basal_insulin_rate=self.basal_insulin_rate,
            bergman_params=p,
        )
        
        stop = len(self.real_glucose) if end_index is None else int(end_index)
        simulated_glucose = [float(self.real_glucose[0])]
        for i in range(1, stop):
            event_index = i - 1
            delivered = (
                float(self.insulin[event_index])
                + self.basal_insulin_rate * self.time_step / 60.0
            )
            g_sim = model.update(
                self.time_step,
                delivered_insulin=delivered,
                carb_intake=float(self.carbs[event_index]),
                current_time=float(event_index) * self.time_step,
            )
            simulated_glucose.append(g_sim)
            
        sim = np.array(simulated_glucose)
        real = np.array(self.real_glucose[:stop])
        
        # Mean Squared Error
        score_start = max(0, int(score_start_index))
        if score_start >= stop:
            raise ValueError("score_start_index must be smaller than the simulated interval")
        mse = np.mean((sim[score_start:] - real[score_start:])**2)
        return float(mse)

    def fit(self):
        """Runs the optimization algorithm."""
        # Initial guess (scales = 1.0)
        initial_guess = [1.0, 1.0, 1.0]
        
        print("Starting bounded research-profile calibration...")
        print("Initial MSE:", self._simulate_and_score(initial_guess))
        
        bounds = [(0.5, 2.0), (0.35, 2.5), (0.5, 2.0)]
        calibration_end = max(16, int(len(self.real_glucose) * 0.8))

        result = minimize(
            lambda values: self._simulate_and_score(values, end_index=calibration_end)
            + 15.0 * float(np.sum(np.log(np.maximum(values, 1e-6)) ** 2)),
            initial_guess,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 200, 'disp': True},
        )
        
        print("\nCalibration Complete!")
        validation_mse = self._simulate_and_score(
            result.x,
            score_start_index=calibration_end,
        )
        nominal_validation_mse = self._simulate_and_score(
            initial_guess,
            score_start_index=calibration_end,
        )
        full_trace_mse = self._simulate_and_score(result.x)
        print("Calibration MSE:", result.fun)
        print("Held-out validation MSE:", validation_mse)
        print("Nominal held-out validation MSE:", nominal_validation_mse)
        print("Optimized Scaling Factors (p1, p3, meal tmax):", result.x)

        warnings = []
        if self.observation_semantics == "cgm_observation":
            warnings.append(
                "Latent model glucose was compared directly with CGM observations; "
                "sensor dynamics may be absorbed into the fitted profile."
            )
        if validation_mse >= nominal_validation_mse:
            warnings.append(
                "Held-out error did not improve; do not promote this fitted profile."
            )
        near_bounds = [
            name
            for name, value, (low, high) in zip(
                ("p1_scale", "p3_scale", "meal_tmax_scale"), result.x, bounds
            )
            if min(abs(float(value) - low), abs(high - float(value)))
            <= 0.02 * (high - low)
        ]
        if near_bounds:
            warnings.append(
                "Optimizer reached or approached bounds: " + ", ".join(near_bounds)
            )
        
        return {
            'p1': 0.028 * result.x[0],
            'p2': 0.025,
            'p3': 5.0e-6 * result.x[1],
            'Gb': 120.0,
            'tau_meal': 40.0 * result.x[2],
            'calibration_mse': float(result.fun),
            'validation_mse': float(validation_mse),
            'nominal_validation_mse': float(nominal_validation_mse),
            'validation_improvement_mse': float(nominal_validation_mse - validation_mse),
            'full_trace_mse': float(full_trace_mse),
            'optimizer_success': bool(result.success),
            'event_alignment': 'events at sample i drive the following interval',
            'observation_semantics': self.observation_semantics,
            'parameters_near_bounds': near_bounds,
            'warnings': warnings,
            'interpretation': (
                'bounded research calibration profile; CGM/event data do not uniquely '
                'identify these parameters and this is not a clinically validated digital twin'
            ),
        }
