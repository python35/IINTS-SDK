from typing import Dict, Any, Optional
from iints.api.base_algorithm import InsulinAlgorithm, AlgorithmInput

class FixedBasalBolus(InsulinAlgorithm):
    """
    A simple insulin algorithm that delivers a fixed basal rate and
    a meal bolus based on carbohydrate intake.

    This algorithm is purely rule-based and stateless (or nearly so) for transparency.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        # Default settings, can be overridden by 'settings' dict
        self.default_settings = {
            "fixed_basal_rate": 0.8,  # Units per hour
            "carb_ratio": 10.0,       # Grams of carbs per unit of insulin
        }
        # Merge default settings with provided settings
        self.settings = {**self.default_settings, **self.settings}

        # Validate essential settings
        if not all(k in self.settings for k in ["fixed_basal_rate", "carb_ratio"]):
            raise ValueError("FixedBasalBolus algorithm requires 'fixed_basal_rate' and 'carb_ratio' in settings.")

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        """
        Calculates insulin dose based on fixed basal rate and carb intake.

        Args:
            data (AlgorithmInput): Dataclass containing all input data.

        Returns:
            Dict[str, Any]: Contains 'basal_insulin' and 'bolus_insulin' for the time step.
        """
        basal_rate = self.settings["fixed_basal_rate"]
        if data.basal_rate_u_per_hr is not None:
            basal_rate = float(data.basal_rate_u_per_hr)
        basal_rate_units_per_minute = basal_rate / 60.0
        basal_insulin = basal_rate_units_per_minute * data.time_step

        carb_intake = data.carb_intake
        bolus_insulin = 0.0
        if carb_intake > 0:
            carb_ratio = self.settings["carb_ratio"]
            if data.icr is not None:
                carb_ratio = float(data.icr)
            bolus_insulin = carb_intake / carb_ratio

        return {
            "basal_insulin": basal_insulin,
            "bolus_insulin": bolus_insulin,
            "total_insulin_delivered": basal_insulin + bolus_insulin
        }

    def __str__(self):
        return (f"FixedBasalBolus Algorithm:\n"
                f"  Fixed Basal Rate: {self.settings['fixed_basal_rate']} U/hr\n"
                f"  Carb Ratio (CR): {self.settings['carb_ratio']} g/U")
