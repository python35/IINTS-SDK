import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from iints.core.simulator import Simulator
from iints.core.patient.bergman_model import BergmanPatientModel, BergmanParameters
from iints.core.algorithms.pid_controller import PIDController
from iints.core.devices.models import SensorModel, PumpModel

class StemCellOptimizer:
    """
    Research optimization module for stem cell / islet cell graft simulations.
    Runs batch simulations to evaluate graft survival and kinetics for T1D cures.
    """
    
    def __init__(self, duration_minutes: int = 1440, time_step: int = 5):
        self.duration_minutes = duration_minutes
        self.time_step = time_step
        
    def evaluate_graft_configuration(
        self, 
        engraftment_percent: float, 
        subq_fraction: float, 
        immune_decay: float,
        meal_schedule: List[Dict[str, float]],
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs a simulation for a specific stem cell graft configuration.
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
            patient_model=patient,
            algorithm=controller,
            sensor_model=sensor,
            pump_model=pump,
        )
        for event in build_stress_events(meal_schedule):
            simulator.add_stress_event(event)
        
        df, stats = simulator.run_batch(duration_minutes=self.duration_minutes)
        
        # Calculate TIR
        if not df.empty:
            bg_col = "bg" if "bg" in df.columns else "CGM" if "CGM" in df.columns else "glucose" if "glucose" in df.columns else df.columns[1]
            tir_mask = (df[bg_col] >= 70) & (df[bg_col] <= 180)
            tir = tir_mask.mean() * 100.0
            
            hypo_mask = df[bg_col] < 70
            hypo = hypo_mask.mean() * 100.0
            
            hyper_mask = df[bg_col] > 180
            hyper = hyper_mask.mean() * 100.0
        else:
            tir, hypo, hyper = 0.0, 0.0, 0.0
            
        return {
            "tir_percent": tir,
            "hypo_percent": hypo,
            "hyper_percent": hyper,
            "max_glucose": df[bg_col].max() if not df.empty else 0.0,
            "min_glucose": df[bg_col].min() if not df.empty else 0.0,
            "engraftment_percent": engraftment_percent,
            "subq_fraction": subq_fraction,
            "immune_decay": immune_decay,
            "df": df
        }
