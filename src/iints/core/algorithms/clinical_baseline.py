from __future__ import annotations

from typing import Any, Dict, Optional

from iints.api.base_algorithm import AlgorithmInput, AlgorithmMetadata, InsulinAlgorithm


class ClinicalBaselineAlgorithm(InsulinAlgorithm):
    """
    Conservative clinician-style baseline.

    This baseline is intentionally simple and explainable: it follows a steady
    basal plan, adds meal coverage, and only applies partial corrections when
    glucose is clearly high and the immediate hypo risk looks low.
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings)
        self.set_algorithm_metadata(
            AlgorithmMetadata(
                name="Clinical Baseline",
                author="IINTS-AF Team",
                description=(
                    "Conservative clinician-style heuristic baseline with meal coverage, "
                    "partial corrections, and explicit hypo-avoidance logic."
                ),
                algorithm_type="rule_based",
            )
        )
        defaults = {
            "target_glucose": 110.0,
            "correction_threshold": 145.0,
            "fixed_basal_rate": 0.75,
            "carb_ratio": 11.0,
            "insulin_sensitivity_factor": 55.0,
            "max_correction_units": 2.5,
            # Allows meal coverage implied by the bundled 10-15 g/U profiles.
            # The independent supervisor still owns low-glucose and dose gates.
            "max_total_units_per_step": 12.0,
            "hypo_guard_glucose": 90.0,
            "falling_trend_guard": -1.2,
        }
        self.settings = {**defaults, **self.settings}

    def predict_insulin(self, data: AlgorithmInput) -> Dict[str, Any]:
        self.why_log = []

        target = float(self.settings["target_glucose"])
        threshold = float(self.settings["correction_threshold"])
        basal_rate = float(data.basal_rate_u_per_hr or self.settings["fixed_basal_rate"])
        carb_ratio = float(data.icr or self.settings["carb_ratio"])
        isf = float(data.isf or self.settings["insulin_sensitivity_factor"])
        trend = float(data.glucose_trend_mgdl_min or 0.0)
        predicted_30 = float(data.predicted_glucose_30min or data.current_glucose)
        uncertainty = float(data.predicted_glucose_30min_std or 0.0)
        iob = float(data.insulin_on_board)

        basal_units = (basal_rate / 60.0) * data.time_step
        meal_bolus = 0.0
        correction_bolus = 0.0

        if data.current_glucose <= float(self.settings["hypo_guard_glucose"]) or trend <= float(
            self.settings["falling_trend_guard"]
        ):
            basal_units *= 0.25
            self._log_reason(
                "Basal reduced because glucose is low or falling quickly",
                "safety",
                basal_units,
                "Clinician-style hypo guard keeps the baseline conservative when immediate risk is elevated.",
            )
        else:
            self._log_reason(
                "Basal maintained at the programmed rate",
                "basal",
                basal_units,
                f"Delivered {basal_units:.2f} units from {basal_rate:.2f} U/hr.",
            )

        if data.carb_intake > 0:
            meal_bolus = float(data.carb_intake) / carb_ratio
            self._log_reason(
                "Meal coverage added from announced carbohydrates",
                "meal_response",
                meal_bolus,
                f"{float(data.carb_intake):.0f} g / {carb_ratio:.1f} g/U.",
            )

        correction_candidate = max(predicted_30, data.current_glucose)
        correction_signal = correction_candidate >= threshold and iob < 2.5
        if correction_signal:
            raw_correction = (correction_candidate - target) / isf
            conservative_factor = 0.5 if uncertainty <= 15.0 else 0.35
            if trend < 0:
                conservative_factor *= 0.5
            correction_bolus = max(
                0.0,
                min(
                    raw_correction * conservative_factor,
                    float(self.settings["max_correction_units"]),
                ),
            )
            self._log_reason(
                "Partial correction added from clinician-style high-glucose logic",
                "glucose_correction",
                correction_bolus,
                (
                    f"Using the larger of current glucose ({data.current_glucose:.0f}) and predicted "
                    f"30-minute glucose ({predicted_30:.0f}) with IOB {iob:.2f} U."
                ),
            )
        else:
            self._log_reason(
                "No correction bolus because the risk-adjusted threshold was not met",
                "glucose_correction",
                0.0,
                "The baseline only corrects when glucose is clearly high and active insulin is still modest.",
            )

        total = basal_units + meal_bolus + correction_bolus
        capped_total = min(total, float(self.settings["max_total_units_per_step"]))
        if capped_total != total:
            scale = capped_total / total if total > 0 else 0.0
            basal_units *= scale
            meal_bolus *= scale
            correction_bolus *= scale
            total = capped_total
            self._log_reason(
                "Total insulin capped by per-step safety ceiling",
                "safety_constraint",
                total,
                f"Capped at {float(self.settings['max_total_units_per_step']):.2f} U.",
            )
        else:
            total = capped_total

        self._log_reason(
            "Final clinician-style decision ready",
            "final_decision",
            total,
            (
                f"Basal {basal_units:.2f} U, meal {meal_bolus:.2f} U, correction {correction_bolus:.2f} U, "
                f"confidence driven by explicit safety guards."
            ),
        )

        return {
            "total_insulin_delivered": total,
            "basal_insulin": basal_units,
            "meal_bolus": meal_bolus,
            "correction_bolus": correction_bolus,
            "bolus_insulin": meal_bolus + correction_bolus,
        }
