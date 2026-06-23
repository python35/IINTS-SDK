import numpy as np
from typing import Any, Dict, List, Optional
from scipy.integrate import solve_ivp

from .bergman_model import BergmanPatientModel, BergmanParameters
from .physiology import renal_glucose_clearance_concentration

class AdvancedMetabolicModel(BergmanPatientModel):
    """
    Advanced Metabolic Model for IINTS-AF.
    Extends the 13-state Bergman model to an 18-state model including:
    - F: Free Fatty Acids (FFA) (mmol/L)
    - K: Ketone Bodies (mmol/L)
    - Beta: Residual Beta-cell mass fraction (0.0 to 1.0)
    - Q_fat: Fat stomach pool (grams)
    - Q_prot: Protein stomach pool (grams)
    
    Includes lipotoxicity, DKA, illness, menstrual cycles, and cannula degradation.
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
        
        # --- NEW GAMECHANGER STATES & FLAGS ---
        self.is_ill = False
        self.illness_severity = 0.0  # 0.0 to 1.0
        
        self.menstrual_cycle_active = False
        self.cycle_start_time_minutes = 0.0  # Time offset for the 28-day wave
        
        self.pump_cannula_age_minutes = 0.0  # Tracks lipohypertrophy
        
        super().__init__(**kwargs)
        
        # Override the 13-state with 18-state
        # State vector: [G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta, Q_fat, Q_prot]
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
            0.0,                   # 16: Q_fat
            0.0,                   # 17: Q_prot
        ], dtype=np.float64)

    def reset(self) -> None:
        super().reset()
        self._state = np.array([
            self.initial_glucose, 0.0, self.params.Ib, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, self.initial_ffa, self.initial_ketones, self.initial_beta_mass, 0.0, 0.0
        ], dtype=np.float64)
        self.pump_cannula_age_minutes = 0.0

    def get_patient_state(self) -> Dict[str, float]:
        state_dict = super().get_patient_state()
        state_dict.update({
            "plasma_ffa_mmol_L": float(self._state[13]),
            "plasma_ketones_mmol_L": float(self._state[14]),
            "residual_beta_cell_mass": float(self._state[15]),
            "fat_pool_g": float(self._state[16]),
            "protein_pool_g": float(self._state[17]),
        })
        return state_dict

    def set_state(self, state: Dict[str, Any]) -> None:
        if "ode_state" in state:
            ode_state = np.array(state["ode_state"], dtype=np.float64)
            if ode_state.size == 13:
                # Upgrade legacy 13-state to 18-state
                ode_state = np.append(ode_state, [self.initial_ffa, self.initial_ketones, self.initial_beta_mass, 0.0, 0.0])
            elif ode_state.size == 16:
                # Upgrade legacy 16-state to 18-state
                ode_state = np.append(ode_state, [0.0, 0.0])
            self._state = ode_state
        # Call super for the rest, but temporarly pop ode_state so super doesn't overwrite it
        st_copy = state.copy()
        if "ode_state" in st_copy:
            del st_copy["ode_state"]
        super().set_state(st_copy)
        self.is_ill = bool(state.get("is_ill", self.is_ill))
        self.illness_severity = float(state.get("illness_severity", self.illness_severity))
        self.menstrual_cycle_active = bool(state.get("menstrual_cycle_active", self.menstrual_cycle_active))
        self.cycle_start_time_minutes = float(state.get("cycle_start_time_minutes", self.cycle_start_time_minutes))
        self.pump_cannula_age_minutes = float(state.get("pump_cannula_age_minutes", self.pump_cannula_age_minutes))

    def update(
        self,
        time_step: float = 0.0,
        delivered_insulin: float = 0.0,
        carb_intake: float = 0.0,
        delivered_glucagon_mg: float = 0.0,
        current_time: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """Advance the 18-state metabolic model by ``time_step`` minutes.

        The public signature stays compatible with ``BergmanPatientModel``.
        Advanced-only inputs are accepted as keyword arguments:
        ``fat_intake`` and ``protein_intake``. Backward-compatible aliases are
        accepted for earlier scratch scripts: ``dt_minutes``,
        ``delivered_glucagon``, and ``current_time_minutes``.
        """
        if "dt_minutes" in kwargs:
            time_step = float(kwargs.pop("dt_minutes"))
        if "delivered_glucagon" in kwargs:
            delivered_glucagon_mg = float(kwargs.pop("delivered_glucagon"))
        if "current_time_minutes" in kwargs:
            current_time = float(kwargs.pop("current_time_minutes"))

        fat_intake = float(kwargs.pop("fat_intake", 0.0))
        protein_intake = float(kwargs.pop("protein_intake", 0.0))

        # Add macronutrients and age the cannula before solving this step.
        self._state[16] += max(0.0, fat_intake)
        self._state[17] += max(0.0, protein_intake)
        self.pump_cannula_age_minutes += max(0.0, float(time_step))

        return super().update(
            time_step=time_step,
            delivered_insulin=delivered_insulin,
            carb_intake=carb_intake,
            delivered_glucagon_mg=delivered_glucagon_mg,
            current_time=current_time,
            **kwargs,
        )

    def start_illness(self, severity: float) -> None:
        self.is_ill = True
        self.illness_severity = float(np.clip(severity, 0.0, 1.0))

    def stop_illness(self) -> None:
        self.is_ill = False
        self.illness_severity = 0.0

    def start_menstrual_cycle(self, current_time_minutes: float = 0.0) -> None:
        self.menstrual_cycle_active = True
        self.cycle_start_time_minutes = float(current_time_minutes)

    def stop_menstrual_cycle(self) -> None:
        self.menstrual_cycle_active = False

    def trigger_event(self, event_type: str, value: Any) -> None:
        if event_type == "illness":
            self.start_illness(float(value))
        elif event_type == "illness_end":
            self.stop_illness()
        elif event_type == "menstrual_cycle":
            self.start_menstrual_cycle(float(value or 0.0))
        elif event_type == "menstrual_cycle_end":
            self.stop_menstrual_cycle()
        else:
            super().trigger_event(event_type, value)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state.update({
            "is_ill": self.is_ill,
            "illness_severity": self.illness_severity,
            "menstrual_cycle_active": self.menstrual_cycle_active,
            "cycle_start_time_minutes": self.cycle_start_time_minutes,
            "pump_cannula_age_minutes": self.pump_cannula_age_minutes,
        })
        return state

    def _ode(
        self,
        t: float,
        y: np.ndarray,
        u_insulin_mu_per_min: float,
        u_glucagon_pg_per_min: float,
        current_time: float,
    ) -> np.ndarray:
        # Unpack 18 states
        G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta, Q_fat, Q_prot = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L
        V_glucagon = p.V_glucagon_per_kg * p.body_weight_kg
        
        # --- Fat & Protein Macronutrient Kinetics ---
        # Fat decays slowly with a ~4 h model time constant.
        k_fat = 1.0 / 240.0
        dQ_fat_dt = -k_fat * Q_fat
        
        # Protein decays slowly with a ~5 h model time constant.
        k_prot = 1.0 / 300.0
        dQ_prot_dt = -k_prot * Q_prot
        
        # Protein Gluconeogenesis: ~50% of protein becomes glucose very slowly.
        # R_a_prot adds to EGP. (Convert grams to mg, divide by Vg_abs to get mg/dL/min)
        Ra_prot = 0.5 * k_prot * Q_prot * 1000.0 / Vg_abs

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Exogenous Glucagon Kinetics ---
        dY1_dt = u_glucagon_pg_per_min - Y1 / p.t_max_glucagon
        dY2_dt = Y1 / p.t_max_glucagon - Y2 / p.t_max_glucagon
        U_Gamma = Y2 / p.t_max_glucagon
        dGamma_dt = U_Gamma / V_glucagon - p.k_e_glucagon * Gamma
        dx_gluc_dt = -p.k_a_glucagon * x_gluc + p.S_glucagon * p.k_a_glucagon * Gamma

        # --- Circadian Rhythms & Dawn Phenomenon ---
        t_hours = (current_time / 60.0) % 24
        A_circadian = 0.2 if self.dawn_phenomenon_strength > 0 else 0.0
        circadian_multiplier = 1.0 + A_circadian * np.cos((2 * np.pi / 24) * (t_hours - 5.0))

        # --- Exercise & Stress Physiologic Impact ---
        exercise_p1_multiplier = 1.0
        exercise_p3_multiplier = 1.0
        exercise_glucose_uptake = 0.0
        if self.is_exercising:
            exercise_p1_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_p3_multiplier = 1.0 + 2.0 * self.exercise_intensity
            exercise_glucose_uptake = self.exercise_intensity * 0.005 * G

        stress_p1_multiplier = 1.0
        stress_p3_multiplier = 1.0
        stress_Gb_multiplier = 1.0
        if self.is_stressed:
            stress_p1_multiplier = 1.0 - 0.2 * self.stress_intensity
            stress_p3_multiplier = 1.0 - 0.7 * self.stress_intensity
            stress_Gb_multiplier = 1.0 + 0.5 * self.stress_intensity
            
        # --- Illness / Cytokine Resistance ---
        illness_Gb_multiplier = 1.0 + 0.8 * self.illness_severity if self.is_ill else 1.0
        illness_p3_multiplier = 1.0 - 0.5 * self.illness_severity if self.is_ill else 1.0
        
        # --- Menstrual Cycle Hormonal Drifts ---
        cycle_p3_multiplier = 1.0
        if self.menstrual_cycle_active:
            # 28 days = 40320 minutes
            cycle_time = (current_time - self.cycle_start_time_minutes) % 40320
            # Sine wave peaking in resistance (lowest p3) around day 21 (luteal phase)
            cycle_p3_multiplier = 1.0 - 0.25 * np.sin(2 * np.pi * cycle_time / 40320.0)

        # --- Endogenous Rescue & HAAF ---
        hypo_delta = max(0.0, 70.0 - G)
        rescue_multiplier = 1.0 + (hypo_delta / 10.0) * (1.0 - HAAF)

        # HAAF Memory Dynamics
        k_haaf_build = 0.005
        k_haaf_decay = 1.0 / (24 * 60)
        dHAAF_dt = k_haaf_build * hypo_delta * (1.0 - HAAF) - k_haaf_decay * HAAF

        # --- Beta-cell autoimmune decay ---
        dBeta_dt = -self.autoimmune_aggressiveness * Beta

        # --- FFA & Ketone Dynamics ---
        l_0, l_1, k_f = 0.2, 0.23, 0.1
        dF_dt = l_0 * np.exp(-l_1 * I) - k_f * F

        k_0, k_1, k_2 = 0.125, 0.33, 0.05
        dK_dt = k_0 * F * np.exp(-k_1 * I) - k_2 * K

        # --- Lipotoxicity (Insulin Resistance via FFAs) ---
        lipotoxicity_factor = 0.4 / max(0.4, F)

        p1_eff = p.p1 * exercise_p1_multiplier * stress_p1_multiplier
        # Total p3 is degraded by lipotoxicity, acute illness, and menstrual hormonal drifts
        p3_eff = p.p3 * exercise_p3_multiplier * stress_p3_multiplier * lipotoxicity_factor * illness_p3_multiplier * cycle_p3_multiplier
        
        starvation_factor = np.exp(-0.4 * I) * (max(F, 0.4) / 0.4)
        hepatic_glucose_production_multiplier = 1.0 + 3.0 * starvation_factor

        # Gb is multiplied by stress, rescue adrenaline, exogenous glucagon, hepatic starvation, circadian rhythms, and illness
        Gb_eff = p.Gb * stress_Gb_multiplier * rescue_multiplier * max(0.0, 1.0 + x_gluc) * hepatic_glucose_production_multiplier * circadian_multiplier * illness_Gb_multiplier

        # --- Physiological Renal Clearance ---
        RGC = renal_glucose_clearance_concentration(G, threshold_mgdl=180.0, gain=0.05, splay_mgdl=10.0)

        # --- dG/dt (INSTABILITY UPGRADE) ---
        EGP = p1_eff * Gb_eff
        dGdt = -X * G + EGP + Ra + Ra_prot - exercise_glucose_uptake - RGC

        # --- dX/dt ---
        dXdt = -p.p2 * X + p3_eff * max(I - p.Ib, 0.0)
        
        # --- Cannula Degradation / Lipohypertrophy ---
        # After 48 hours (2880 mins), absorption drops linearly by up to 30%
        cannula_degradation_factor = 1.0 - 0.3 * min(1.0, max(0.0, (self.pump_cannula_age_minutes - 2880) / 2880.0))
        ka_eff = p.k_a * cannula_degradation_factor

        # --- dS1/dt, dS2/dt (Subcutaneous Insulin Absorption) ---
        dS1dt = u_insulin_mu_per_min - ka_eff * S1
        dS2dt = ka_eff * S1 - ka_eff * S2

        # Rate of appearance of insulin into plasma (mU/min)
        Ra_I = ka_eff * S2

        # --- dI/dt (Including residual beta-cell endogenous secretion) ---
        endogenous_secretion = Beta * self.gamma_healthy * max(G - p.h, 0.0)
        target_Ib = p.Ib * Beta
        
        dIdt = -p.n * (I - target_Ib) + endogenous_secretion + Ra_I / Vi_abs

        # --- Dalla Man Multi-compartment Meal Kinetcs (Fat-Delayed) ---
        # Fat massively delays gastric emptying.
        fat_delay_factor = np.exp(-0.02 * Q_fat) 
        gastric_emptying_rate = (1.0 / max(float(p.tau_meal), 1.0)) * fat_delay_factor
        solid_to_liquid_rate = gastric_emptying_rate * 1.5 
        dQ_sto1_dt = -solid_to_liquid_rate * Q_sto1
        dQ_sto2_dt = solid_to_liquid_rate * Q_sto1 - gastric_emptying_rate * Q_sto2
        dQ_gut_dt = gastric_emptying_rate * Q_sto2 - p.k_abs * Q_gut

        return np.array([
            dGdt, dXdt, dIdt, dQ_sto1_dt, dQ_sto2_dt, dQ_gut_dt, 
            dS1dt, dS2dt, dY1_dt, dY2_dt, dGamma_dt, dx_gluc_dt, 
            dHAAF_dt, dF_dt, dK_dt, dBeta_dt, dQ_fat_dt, dQ_prot_dt
        ])
