import numpy as np
from scipy.optimize import minimize
from iints.core.patient.advanced_metabolic_model import AdvancedMetabolicModel, BergmanParameters

class DigitalTwinCalibrator:
    """
    Fits the AdvancedMetabolicModel parameters (p1, p2, p3, Gb) to real 
    patient data (like the OhioT1DM dataset) to create a true Digital Twin.
    """
    
    def __init__(self, real_glucose_array, carbs_array, insulin_array, time_step_min=5.0):
        self.real_glucose = real_glucose_array
        self.carbs = carbs_array
        self.insulin = insulin_array
        self.time_step = time_step_min
        
    def _simulate_and_score(self, params_array):
        """
        params_array: [p1_scale, p2_scale, p3_scale, gb_scale]
        Returns MSE between simulated glucose and real glucose.
        """
        # Ensure parameters are positive
        if any(p <= 0 for p in params_array):
            return 1e6
            
        p1_val = 0.028 * params_array[0]
        p2_val = 0.025 * params_array[1]
        p3_val = 5.0e-6 * params_array[2]
        gb_val = 120.0 * params_array[3]
        
        # Build model with these params
        p = BergmanParameters(p1=p1_val, p2=p2_val, p3=p3_val, Gb=gb_val)
        model = AdvancedMetabolicModel(
            initial_glucose=self.real_glucose[0], 
            bergman_params=p
        )
        
        simulated_glucose = [self.real_glucose[0]]
        
        # Pre-feed if needed
        model.update(1.0, delivered_insulin=0, carb_intake=40.0)
        
        for i in range(1, len(self.real_glucose)):
            g_sim = model.update(
                self.time_step, 
                delivered_insulin=self.insulin[i], 
                carb_intake=self.carbs[i]
            )
            simulated_glucose.append(g_sim)
            
        sim = np.array(simulated_glucose)
        real = np.array(self.real_glucose)
        
        # Mean Squared Error
        mse = np.mean((sim - real)**2)
        return mse

    def fit(self):
        """Runs the optimization algorithm."""
        # Initial guess (scales = 1.0)
        initial_guess = [1.0, 1.0, 1.0, 1.0]
        
        print("Starting Auto-Calibration (Digital Twin Fitter)...")
        print("Initial MSE:", self._simulate_and_score(initial_guess))
        
        bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.5, 3.0)]
        
        result = minimize(
            self._simulate_and_score, 
            initial_guess, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'disp': True}
        )
        
        print("\nCalibration Complete!")
        print("Final MSE:", result.fun)
        print("Optimized Scaling Factors (p1, p2, p3, Gb):", result.x)
        
        return {
            'p1': 0.028 * result.x[0],
            'p2': 0.025 * result.x[1],
            'p3': 5.0e-6 * result.x[2],
            'Gb': 120.0 * result.x[3],
            'final_mse': result.fun
        }
