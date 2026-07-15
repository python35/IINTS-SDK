from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Mapping, Sequence

from iints.core.simulator import Simulator
from iints.core.patient.bergman_model import BergmanPatientModel, BergmanParameters
from iints.core.algorithms.pid_controller import PIDController
from iints.core.devices.models import SensorModel, PumpModel
from iints.research.stem_cell_transplant import (
    StemCellTransplantParameters,
    TransplantPlacement,
    run_stem_cell_transplant_simulation,
)


class StemCellOptimizer:
    """
    Research optimizer for stem-cell / islet-graft simulation hypotheses.

    This is a pre-clinical/educational abstraction. ``engraftment_percent`` is
    interpreted as a functional beta-cell graft fraction, not as a clinical
    transplant outcome estimate.
    """

    def __init__(self, duration_minutes: int = 1440, time_step: int = 5):
        self.duration_minutes = duration_minutes
        self.time_step = time_step

    @staticmethod
    def _normalize_meal_schedule(
        meal_schedule: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Accept both simulator event payloads and simple meal dictionaries."""
        events: list[dict[str, Any]] = []
        for meal in meal_schedule:
            if "start_time" in meal and "event_type" in meal:
                events.append(dict(meal))
                continue

            start_time = meal.get("start_time", meal.get("time", meal.get("minute")))
            carbs = meal.get("value", meal.get("carbs", meal.get("carb_grams")))
            if start_time is None or carbs is None:
                raise ValueError(
                    "Meal schedule entries must either be simulator events with "
                    "'start_time'/'event_type' or simple meals with 'time' and 'carbs'."
                )

            events.append(
                {
                    "start_time": int(start_time),
                    "event_type": str(meal.get("event_type", "meal")),
                    "value": float(carbs),
                    "reported_value": meal.get("reported_value"),
                    "absorption_delay_minutes": int(meal.get("absorption_delay_minutes", 0)),
                    "duration": int(meal.get("duration", 0)),
                }
            )
        return events

    @staticmethod
    def _glucose_column(dataframe: pd.DataFrame) -> str:
        for column in ("glucose_actual_mgdl", "glucose", "bg", "CGM"):
            if column in dataframe.columns:
                return column
        if len(dataframe.columns) < 2:
            raise ValueError("Simulation results do not contain a glucose column.")
        return str(dataframe.columns[1])

    def evaluate_graft_configuration(
        self,
        engraftment_percent: float,
        subq_fraction: float,
        immune_decay: float,
        meal_schedule: Sequence[Mapping[str, Any]],
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run one graft hypothesis against a meal schedule.

        ``meal_schedule`` may use full simulator events, for example
        ``{"start_time": 480, "event_type": "meal", "value": 60}``, or the
        shorter research form ``{"time": 480, "carbs": 60}``.
        """
        # Create bergman parameters with stem cell settings
        params = BergmanParameters(
            stem_cell_engraftment_percent=engraftment_percent,
            stem_cell_subq_fraction=subq_fraction,
            immune_rejection_rate=immune_decay,
            gamma=0.005 # Baseline healthy beta cell secretion rate
        )
        
        patient = BergmanPatientModel(
            initial_glucose=100.0,
            bergman_params=params
        )
        
        # We use a dummy controller since the patient is autonomously producing insulin
        controller = PIDController()
        sensor = SensorModel(noise_std=1.0, seed=seed)
        pump = PumpModel()
        
        from iints.validation import build_stress_events
        
        simulator = Simulator(
            patient_model=patient,  # type: ignore
            algorithm=controller,
            time_step=self.time_step,
            seed=seed,
            sensor_model=sensor,
            pump_model=pump,
        )
        events = self._normalize_meal_schedule(meal_schedule)
        for event in build_stress_events(events):
            simulator.add_stress_event(event)

        df, stats = simulator.run_batch(duration_minutes=self.duration_minutes)

        # Calculate TIR
        if not df.empty:
            bg_col = self._glucose_column(df)
            tir_mask = (df[bg_col] >= 70) & (df[bg_col] <= 180)
            tir = tir_mask.mean() * 100.0

            hypo_mask = df[bg_col] < 70
            hypo = hypo_mask.mean() * 100.0

            hyper_mask = df[bg_col] > 180
            hyper = hyper_mask.mean() * 100.0
            max_glucose = float(df[bg_col].max())
            min_glucose = float(df[bg_col].min())
        else:
            tir, hypo, hyper = 0.0, 0.0, 0.0
            max_glucose, min_glucose = 0.0, 0.0

        patient_state = patient.get_patient_state()

        return {
            "tir_percent": float(tir),
            "hypo_percent": float(hypo),
            "hyper_percent": float(hyper),
            "max_glucose": max_glucose,
            "min_glucose": min_glucose,
            "engraftment_percent": float(engraftment_percent),
            "subq_fraction": float(subq_fraction),
            "immune_decay": float(immune_decay),
            "final_graft_mass_fraction": float(patient_state["stem_cell_graft_mass_fraction"]),
            "final_plasma_insulin_mU_L": float(patient_state["plasma_insulin_mU_L"]),
            "simulation_stats": stats,
            "df": df
        }

    def evaluate_transplant_configuration(
        self,
        *,
        placement: TransplantPlacement = "portal",
        initial_cell_mass: float = 1.0,
        initial_maturation_fraction: float = 0.30,
        immunosuppression_effect: float = 0.0,
        encapsulation_effect: float = 0.0,
        meal_schedule: Sequence[Mapping[str, Any]] = (),
        initial_glucose: float = 120.0,
        basal_insulin_units_per_hour: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run the newer multi-compartment transplant simulation.

        This mode tracks immature cells, functional beta-cell mass,
        vascularization, oxygenation, inflammation, adaptive immunity,
        fibrosis, and delayed insulin release. It is still research-only.
        """
        params = StemCellTransplantParameters(
            placement=placement,
            initial_cell_mass=initial_cell_mass,
            initial_maturation_fraction=initial_maturation_fraction,
            immunosuppression_effect=immunosuppression_effect,
            encapsulation_effect=encapsulation_effect,
        )
        df = run_stem_cell_transplant_simulation(
            duration_minutes=self.duration_minutes,
            time_step_minutes=self.time_step,
            initial_glucose=initial_glucose,
            parameters=params,
            meal_schedule=meal_schedule,
            basal_insulin_units_per_hour=basal_insulin_units_per_hour,
        )
        glucose = df["glucose_mgdl"]
        return {
            "tir_percent": float(((glucose >= 70) & (glucose <= 180)).mean() * 100.0),
            "hypo_percent": float((glucose < 70).mean() * 100.0),
            "hyper_percent": float((glucose > 180).mean() * 100.0),
            "max_glucose": float(glucose.max()),
            "min_glucose": float(glucose.min()),
            "placement": placement,
            "initial_cell_mass": float(initial_cell_mass),
            "initial_maturation_fraction": float(initial_maturation_fraction),
            "immunosuppression_effect": float(immunosuppression_effect),
            "encapsulation_effect": float(params.with_placement_defaults().encapsulation_effect),
            "final_functional_mass": float(df["functional_mass"].iloc[-1]),
            "final_vascularization": float(df["vascularization"].iloc[-1]),
            "final_oxygenation": float(df["oxygenation"].iloc[-1]),
            "final_adaptive_immunity": float(df["adaptive_immunity"].iloc[-1]),
            "final_fibrosis": float(df["fibrosis"].iloc[-1]),
            "total_released_insulin_units": float(df["released_insulin_units"].sum()),
            "df": df,
        }
