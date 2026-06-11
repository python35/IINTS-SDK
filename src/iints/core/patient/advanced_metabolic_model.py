import numpy as np
from typing import Any, Dict, List, Optional
from scipy.integrate import solve_ivp

from .bergman_model import BergmanPatientModel, BergmanParameters

class AdvancedMetabolicModel(BergmanPatientModel):
    """
    Advanced Metabolic Model for IINTS-AF.
    Extends the 13-state Bergman model to a 16-state model including:
    - F: Free Fatty Acids (FFA) (mmol/L)
    - K: Ketone Bodies (mmol/L)
    - Beta: Residual Beta-cell mass fraction (0.0 to 1.0)
    
    Includes lipotoxicity (high FFA causes insulin resistance) and 
    DKA (Ketone production under extreme insulin deficiency).
    """

    def __init__(
        self,
        initial_beta_mass: float = 0.0,  # 0.0 = completely destroyed, 1.0 = healthy
        autoimmune_aggressiveness: float = 7e-6,  # Decay rate of beta cells per min
        initial_ffa: float = 0.4,
        initial_ketones: float = 0.1,
        gamma_healthy: float = 0.005,
        **kwargs,
    ) -> None:
        self.initial_beta_mass = initial_beta_mass
        self.autoimmune_aggressiveness = autoimmune_aggressiveness
        self.initial_ffa = initial_ffa
        self.initial_ketones = initial_ketones
        self.gamma_healthy = gamma_healthy
        
        super().__init__(**kwargs)
        
        # Override the 13-state with 16-state
        # State vector: [G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta]
        self._state = np.array([
            self.initial_glucose,  # 0: G
            0.0,                   # 1: X
            self.params.Ib,        # 2: I
            0.0,                   # 3: Q_sto1
            0.0,                   # 4: Q_sto2
            0.0,                   # 5: Q_gut
            0.0,                   # 6: S1
            0.0,                   # 7: S2
            0.0,                   # 8: Y1
            0.0,                   # 9: Y2
            0.0,                   # 10: Gamma
            0.0,                   # 11: x_gluc
            0.0,                   # 12: HAAF
            self.initial_ffa,      # 13: F (FFA)
            self.initial_ketones,  # 14: K (Ketones)
            self.initial_beta_mass,# 15: Beta
        ], dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self._state = np.array([
            self.initial_glucose, 0.0, self.params.Ib, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, self.initial_ffa, self.initial_ketones, self.initial_beta_mass
        ], dtype=np.float64)

    def get_patient_state(self) -> Dict[str, float]:
        state_dict = super().get_patient_state()
        state_dict.update({
            "plasma_ffa_mmol_L": float(self._state[13]),
            "plasma_ketones_mmol_L": float(self._state[14]),
            "residual_beta_cell_mass": float(self._state[15]),
        })
        return state_dict

    def set_state(self, state: Dict[str, Any]) -> None:
        if "ode_state" in state:
            ode_state = np.array(state["ode_state"], dtype=np.float64)
            if ode_state.size == 13:
                # Upgrade legacy 13-state to 16-state
                ode_state = np.append(ode_state, [self.initial_ffa, self.initial_ketones, self.initial_beta_mass])
            self._state = ode_state
        # Call super for the rest, but temporarly pop ode_state so super doesn't overwrite it
        st_copy = state.copy()
        if "ode_state" in st_copy:
            del st_copy["ode_state"]
        super().set_state(st_copy)

    def _ode(
        self,
        t: float,
        y: np.ndarray,
        u_insulin_mu_per_min: float,
        u_glucagon_pg_per_min: float,
        current_time: float,
    ) -> np.ndarray:
        # Unpack 16 states
        G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L
        V_glucagon = p.V_glucagon_per_kg * p.body_weight_kg

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Exogenous Glucagon Kinetics ---
        dY1_dt = u_glucagon_pg_per_min - Y1 / p.t_max_glucagon
        dY2_dt = Y1 / p.t_max_glucagon - Y2 / p.t_max_glucagon
        U_Gamma = Y2 / p.t_max_glucagon
        dGamma_dt = U_Gamma / V_glucagon - p.k_e_glucagon * Gamma
        dx_gluc_dt = -p.k_a_glucagon * x_gluc + p.S_glucagon * p.k_a_glucagon * Gamma

        # --- Dawn phenomenon ---
        dawn = 0.0
        if self.dawn_phenomenon_strength > 0:
            minutes_in_day = current_time % 1440
            ds = self.dawn_start_hour * 60
            de = self.dawn_end_hour * 60
            if ds <= minutes_in_day <= de:
                dawn = self.dawn_phenomenon_strength / 60.0  # mg/dL/min

        # --- Exercise & Stress Physiologic Impact ---
        exercise_p1_multiplier = 1.0
        exercise_p3_multiplier = 1.0
        exercise_glucose_uptake = 0.0
        if self.is_exercising:
            exercise_p1_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_p3_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_glucose_uptake = self.exercise_intensity * self.exercise_glucose_consumption_rate

        stress_p1_multiplier = 1.0
        stress_p3_multiplier = 1.0
        stress_Gb_multiplier = 1.0
        if self.is_stressed:
            stress_p1_multiplier = 1.0 - 0.2 * self.stress_intensity
            stress_p3_multiplier = 1.0 - 0.7 * self.stress_intensity
            stress_Gb_multiplier = 1.0 + 0.5 * self.stress_intensity

        # --- Endogenous Rescue & HAAF ---
        hypo_delta = max(0.0, 70.0 - G)
        rescue_multiplier = 1.0 + (hypo_delta / 10.0) * (1.0 - HAAF)

        # HAAF Memory Dynamics
        k_haaf_build = 0.005
        k_haaf_decay = 1.0 / (24 * 60)
        dHAAF_dt = k_haaf_build * hypo_delta * (1.0 - HAAF) - k_haaf_decay * HAAF

        # --- NEW: Beta-cell autoimmune decay ---
        dBeta_dt = -self.autoimmune_aggressiveness * Beta

        # --- NEW: FFA & Ketone Dynamics ---
        # F basal = 0.4. Max = 2.0. Insulin sharply suppresses lipolysis.
        l_0 = 0.2
        l_1 = 0.23
        k_f = 0.1
        dF_dt = l_0 * np.exp(-l_1 * I) - k_f * F

        # K basal = 0.1. Max = 5.0. Ketone production driven by high F and very low I.
        k_0 = 0.125
        k_1 = 0.33
        k_2 = 0.05
        dK_dt = k_0 * F * np.exp(-k_1 * I) - k_2 * K

        # --- Lipotoxicity (Insulin Resistance via FFAs) ---
        # Normal F is 0.4. If F rises to 2.0, sensitivity (p3) drops to 0.4 / 2.0 = 20%
        lipotoxicity_factor = 0.4 / max(0.4, F)

        p1_eff = p.p1 * exercise_p1_multiplier * stress_p1_multiplier
        p3_eff = p.p3 * exercise_p3_multiplier * stress_p3_multiplier * lipotoxicity_factor
        
        # In T1D with zero insulin, the liver aggressively produces glucose (EGP).
        # We model this by increasing the basal glucose target Gb_eff exponentially 
        # as insulin drops and FFAs rise (hepatic insulin resistance).
        starvation_factor = np.exp(-0.4 * I) * (max(F, 0.4) / 0.4)
        hepatic_glucose_production_multiplier = 1.0 + 3.0 * starvation_factor

        # Gb is multiplied by stress, rescue adrenaline, exogenous glucagon, and hepatic starvation
        Gb_eff = p.Gb * stress_Gb_multiplier * rescue_multiplier * max(0.0, 1.0 + x_gluc) * hepatic_glucose_production_multiplier

        # --- Physiological Renal Clearance ---
        smooth_threshold_diff = G - 162.0
        softplus_diff = 10.0 * np.log1p(np.exp(smooth_threshold_diff / 10.0))
        F_R = 0.003 * softplus_diff

        # --- dG/dt (INSTABILITY UPGRADE) ---
        # In the original model: dGdt = -(p1_eff + X)*G + p1_eff*Gb_eff + ...
        # This forces G to magically return to Gb_eff (Homeostasis).
        # We decouple this to make it a true T1D unstable model.
        # Basal Endogenous Glucose Production:
        EGP = p1_eff * Gb_eff
        
        # In T1D, Glucose Effectiveness at zero insulin (GEZI) is very low. 
        # We drop the automatic -p1_eff * G tissue uptake and ONLY rely on insulin (X).
        dGdt = -X * G + EGP + Ra + dawn - exercise_glucose_uptake - F_R

        # --- dX/dt ---
        dXdt = -p.p2 * X + p3_eff * max(I - p.Ib, 0.0)

        # --- dS1/dt, dS2/dt (Subcutaneous Insulin Absorption) ---
        dS1dt = u_insulin_mu_per_min - p.k_a * S1
        dS2dt = p.k_a * S1 - p.k_a * S2

        # Rate of appearance of insulin into plasma (mU/min)
        Ra_I = p.k_a * S2

        # --- dI/dt (Including residual beta-cell endogenous secretion) ---
        # Beta is fraction of healthy beta cells.
        endogenous_secretion = Beta * self.gamma_healthy * max(G - p.h, 0.0)
        
        # In T1D, basal endogenous insulin (p.Ib) should be proportional to Beta mass.
        # If Beta = 0, and pump is off, insulin should decay to ZERO, not p.Ib.
        target_Ib = p.Ib * Beta
        
        dIdt = -p.n * (I - target_Ib) + endogenous_secretion + Ra_I / Vi_abs

        # --- Dalla Man Multi-compartment Meal Kinetcs ---
        gastric_emptying_rate = 1.0 / max(float(p.tau_meal), 1.0)
        solid_to_liquid_rate = gastric_emptying_rate * 1.5 
        dQ_sto1_dt = -solid_to_liquid_rate * Q_sto1
        dQ_sto2_dt = solid_to_liquid_rate * Q_sto1 - gastric_emptying_rate * Q_sto2
        dQ_gut_dt = gastric_emptying_rate * Q_sto2 - p.k_abs * Q_gut

        return np.array([
            dGdt, dXdt, dIdt, dQ_sto1_dt, dQ_sto2_dt, dQ_gut_dt, 
            dS1dt, dS2dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, 
            dHAAF_dt, dF_dt, dK_dt, dBeta_dt
        ])
