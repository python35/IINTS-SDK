from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

from iints.core.patient.hovorka_model import HovorkaPatientModel, HovorkaParameters

logger = logging.getLogger(__name__)

class DigitalTwinCalibrator:
    """
    AI-driven Calibration Engine for creating a personalized Digital Twin.
    It takes historical data (CGM, carbs, insulin) and uses scipy optimization
    (L-BFGS-B) to find the metabolic "Big 5" parameters that minimize the RMSE
    between the simulation and real patient data.
    """

    def __init__(self, base_weight_kg: float = 70.0):
        self.base_weight_kg = base_weight_kg
        self.calibrated_params: Optional[HovorkaParameters] = None

    def fit(self, history: List[Dict[str, Any]]) -> HovorkaParameters:
        """
        Fit the Hovorka model to historical data.
        history format: [{"time_min": 0, "cgm": 110, "carbs": 0, "insulin": 0.0}, ...]
        Assumes uniform time steps for simplicity of this implementation.
        """
        if len(history) < 2:
            raise ValueError("Not enough historical data to calibrate.")

        # Extract vectors
        times = np.array([row["time_min"] for row in history])
        true_cgm = np.array([row["cgm"] for row in history])
        carbs = np.array([row.get("carbs", 0.0) for row in history])
        insulin = np.array([row.get("insulin", 0.0) for row in history])

        dt = times[1] - times[0] if len(times) > 1 else 5.0

        # We only consider valid true_cgm for MSE calculation (ignore NaNs)
        valid_mask = ~np.isnan(true_cgm)
        if not np.any(valid_mask):
            raise ValueError("No valid CGM data found in history.")

        # Optimization bounds and initial guess for the "Big 5"
        # 1. EGP_0_per_kg (Endogenous Glucose Production) [mg/min/kg]
        # 2. S_IT (Insulin Sensitivity) [10^-4 /min / (mU/L)]
        # 3. V_I_per_kg (Insulin Volume) [L/kg]
        # 4. k_e (Insulin clearance) [1/min]
        # 5. k_max (Max Gastric emptying) [1/min]

        # Initial Guess (Normalized to 1.0 for the optimizer, we'll scale inside cost function)
        x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Base nominal values from Hovorka
        nominal_EGP = 2.4
        nominal_S_IT = 51.2
        nominal_V_I = 0.12
        nominal_k_e = 0.138
        nominal_k_max = 0.0558

        # Bounds (allowing 20% to 500% variance for extreme metabolic conditions)
        bounds = [
            (0.2, 3.0), # EGP scaling
            (0.1, 5.0), # S_IT scaling
            (0.5, 2.0), # V_I scaling
            (0.5, 2.0), # k_e scaling
            (0.2, 3.0), # k_max scaling
        ]

        def objective_function(x):
            # Scale parameters back to biological values
            test_EGP = x[0] * nominal_EGP
            test_S_IT = x[1] * nominal_S_IT
            test_V_I = x[2] * nominal_V_I
            test_k_e = x[3] * nominal_k_e
            test_k_max = x[4] * nominal_k_max

            # Setup patient params
            params = HovorkaParameters(
                body_weight_kg=self.base_weight_kg,
                EGP_0_per_kg=test_EGP,
                S_IT=test_S_IT,
                V_I_per_kg=test_V_I,
                k_e=test_k_e,
                k_max=test_k_max
            )

            # Initialize simulator at the first known glucose point
            first_valid_idx = np.argmax(valid_mask)
            init_bg = true_cgm[first_valid_idx]

            model = HovorkaPatientModel(
                initial_glucose=init_bg,
                hovorka_params=params
            )

            sim_cgm = np.zeros_like(true_cgm)
            sim_cgm[:first_valid_idx] = init_bg

            # Simulate the sequence
            for i in range(first_valid_idx, len(times)):
                # Record current BG
                sim_cgm[i] = model.get_current_glucose()

                # Process inputs for this step
                meal_g = carbs[i]
                ins_u = insulin[i]

                # Update
                model.update(
                    time_step=dt,
                    delivered_insulin=ins_u,
                    carb_intake=meal_g,
                    current_time=times[i]
                )

            # Calculate RMSE only on valid points
            diff = sim_cgm[valid_mask] - true_cgm[valid_mask]
            mse = np.mean(diff**2)

            # Add mild regularization (L2 penalty) to prevent extreme parameter wandering
            reg = 0.0 * np.sum((x - 1.0)**2)

            return mse + reg

        # Run Nelder-Mead Optimization (Gradient-free is better for ODEs)
        logger.info("Starting Digital Twin calibration via Nelder-Mead...")
        res = minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp': False, 'xatol': 1e-2, 'fatol': 1e-2, 'maxiter': 30}
        )

        if not res.success:
            logger.warning(f"Optimization may have failed or hit bounds: {res.message}")
        else:
            logger.info("Calibration successful!")

        # Create the final calibrated params
        opt_x = res.x
        final_params = HovorkaParameters(
            body_weight_kg=self.base_weight_kg,
            EGP_0_per_kg=opt_x[0] * nominal_EGP,
            S_IT=opt_x[1] * nominal_S_IT,
            V_I_per_kg=opt_x[2] * nominal_V_I,
            k_e=opt_x[3] * nominal_k_e,
            k_max=opt_x[4] * nominal_k_max
        )

        self.calibrated_params = final_params
        return final_params

    def export_profile(self, filepath: str) -> None:
        """Export the calibrated parameters to JSON for Digital Twin what-if scenarios."""
        if not self.calibrated_params:
            raise ValueError("Model not calibrated yet. Call fit() first.")

        # Convert dataclass to dict via __dict__
        data = {k: v for k, v in self.calibrated_params.__dict__.items()}

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Digital Twin profile saved to {filepath}")
