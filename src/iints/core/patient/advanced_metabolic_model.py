import copy

import numpy as np
from typing import Any, Dict, List, Optional
from scipy.integrate import solve_ivp

from .bergman_model import BergmanPatientModel, BergmanParameters
from .physiology import (
    antecedent_hypoglycemia_memory_derivative,
    counterregulatory_rescue_multiplier,
    dawn_glucose_rate_mgdl_min,
    dawn_insulin_sensitivity_multiplier,
    renal_glucose_clearance_concentration,
    validated_snapshot_bool,
    validated_snapshot_scalar,
)

class AdvancedMetabolicModel(BergmanPatientModel):
    """
    Advanced Metabolic Model for IINTS-AF.
    Extends the Bergman-compatible base state to an 18-state model including:
    - F: Free Fatty Acids (FFA) (mmol/L)
    - K: Ketone Bodies (mmol/L)
    - Beta: Residual Beta-cell mass fraction (0.0 to 1.0)
    - Q_fat: Fat stomach pool (grams)
    - Q_prot: Protein stomach pool (grams)
    
    Adds explicit research stressors for FFA/ketone dynamics, illness,
    menstrual-cycle sensitivity, macronutrients, and cannula aging. These
    extensions are hypothesis-generating abstractions, not validated clinical
    DKA, reproductive-endocrinology, or infusion-set models.
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
        values = {
            "initial_beta_mass": float(initial_beta_mass),
            "autoimmune_aggressiveness": float(autoimmune_aggressiveness),
            "initial_ffa": float(initial_ffa),
            "initial_ketones": float(initial_ketones),
            "gamma_healthy": float(gamma_healthy),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Advanced metabolic parameters must all be finite")
        if not 0.0 <= values["initial_beta_mass"] <= 1.0:
            raise ValueError("initial_beta_mass must be between 0 and 1")
        for name in (
            "autoimmune_aggressiveness",
            "initial_ffa",
            "initial_ketones",
            "gamma_healthy",
        ):
            if values[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")

        self.initial_beta_mass = values["initial_beta_mass"]
        self.autoimmune_aggressiveness = values["autoimmune_aggressiveness"]
        self.initial_ffa = values["initial_ffa"]
        self.initial_ketones = values["initial_ketones"]
        self.gamma_healthy = values["gamma_healthy"]
        
        # Experimental stressor states and flags.
        self.is_ill = False
        self.illness_severity = 0.0  # 0.0 to 1.0
        
        self.menstrual_cycle_active = False
        self.cycle_start_time_minutes = 0.0  # Time offset for the 28-day wave
        
        self.pump_cannula_age_minutes = 0.0  # Tracks lipohypertrophy
        
        super().__init__(**kwargs)

        # Advanced T1D mode has no endogenous basal secretion when Beta=0.
        # Reconcile the configured basal pump rate with the plasma-insulin
        # steady state instead of starting with empty subcutaneous depots.
        self._basal_input_mu_per_min = max(self.basal_insulin_rate, 0.0) * 1000.0 / 60.0
        Vi_abs = self.params.Vi * self.params.body_weight_kg
        if self._basal_input_mu_per_min > 0.0:
            self.params.Ib = self._basal_input_mu_per_min / max(Vi_abs * self.params.n, 1e-9)
        
        # Override the Bergman state with an advanced 18-state vector.
        # Advanced mode uses Beta for residual endogenous secretion and does
        # not include the experimental Bergman M_graft stem-cell state.
        # State vector: [G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta, Q_fat, Q_prot]
        self._state = np.array([
            self.initial_glucose,  # 0: G
            0.0,                   # 1: X
            self.params.Ib,        # 2: I
            0.0,                   # 3: Q_sto1
            0.0,                   # 4: Q_sto2
            0.0,                   # 5: Q_gut
            self._basal_input_mu_per_min / max(self.params.k_a, 1e-9),  # 6: S1
            self._basal_input_mu_per_min / max(self.params.k_a, 1e-9),  # 7: S2
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
        basal_input = getattr(self, "_basal_input_mu_per_min", 0.0)
        basal_depot = basal_input / max(self.params.k_a, 1e-9)
        self._state = np.array([
            self.initial_glucose, 0.0, self.params.Ib, 0.0, 0.0, 0.0, basal_depot, basal_depot,
            0.0, 0.0, 0.0, 0.0, 0.0, self.initial_ffa, self.initial_ketones, self.initial_beta_mass, 0.0, 0.0
        ], dtype=np.float64)
        self.pump_cannula_age_minutes = 0.0

    def get_patient_state(self) -> Dict[str, float]:
        state_dict = super().get_patient_state()
        state_dict.update({
            "stem_cell_graft_mass_fraction": 0.0,
            "stem_cell_engraftment_percent": 0.0,
            "stem_cell_subq_fraction": 0.0,
            "immune_rejection_rate_per_min": 0.0,
            "plasma_ffa_mmol_L": float(self._state[13]),
            "plasma_ketones_mmol_L": float(self._state[14]),
            "residual_beta_cell_mass": float(self._state[15]),
            "fat_pool_g": float(self._state[16]),
            "protein_pool_g": float(self._state[17]),
        })
        return state_dict

    def _bounded_fraction_state_indices(self) -> tuple[int, ...]:
        return (12, 15)  # antecedent-hypoglycemia memory, beta-cell mass

    def set_state(self, state: Dict[str, Any]) -> None:
        loaded_ode_state = False
        if "ode_state" in state:
            ode_state = np.array(state["ode_state"], dtype=np.float64)
            if ode_state.size == 13:
                # Upgrade legacy 13-state to 18-state
                ode_state = np.append(ode_state, [self.initial_ffa, self.initial_ketones, self.initial_beta_mass, 0.0, 0.0])
            elif ode_state.size == 14:
                # Upgrade current Bergman 14-state snapshots by dropping
                # M_graft; advanced mode models residual beta-cell mass via Beta.
                ode_state = np.append(ode_state[:13], [self.initial_ffa, self.initial_ketones, self.initial_beta_mass, 0.0, 0.0])
            elif ode_state.size == 16:
                # Upgrade legacy 16-state to 18-state
                ode_state = np.append(ode_state, [0.0, 0.0])
            if ode_state.size != 18:
                raise ValueError(
                    f"Unsupported advanced metabolic snapshot length {ode_state.size}; "
                    "expected 18"
                )
            if not np.all(np.isfinite(ode_state)):
                raise ValueError("Advanced metabolic snapshot contains non-finite values")
            if ode_state[0] < 0.0 or np.any(ode_state[2:] < 0.0):
                raise ValueError(
                    "Advanced metabolic snapshot contains a negative mass or concentration"
                )
            for index in self._bounded_fraction_state_indices():
                if ode_state[index] > 1.0:
                    raise ValueError(
                        "Advanced metabolic snapshot contains a fraction state above 1"
                    )
            self._state = ode_state
            clearance_ml_min = (
                self.params.glucagon_clearance_ml_kg_min
                * self.params.body_weight_kg
            )
            self._state[10] = (
                self.params.k_e_glucagon * self._state[9]
                / max(clearance_ml_min, 1e-9)
            )
            loaded_ode_state = True
        # Call super for the rest, but temporarly pop ode_state so super doesn't overwrite it
        st_copy = state.copy()
        if "ode_state" in st_copy:
            del st_copy["ode_state"]
        if loaded_ode_state:
            supplied_glucose = state.get("current_glucose")
            restored_glucose = float(self._state[0])
            if supplied_glucose is not None and not np.isclose(
                float(supplied_glucose), restored_glucose, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    "Advanced metabolic snapshot is inconsistent: current_glucose "
                    "does not match ode_state[0]"
                )
            st_copy["current_glucose"] = restored_glucose
        super().set_state(st_copy)
        self.is_ill = validated_snapshot_bool(
            state.get("is_ill", self.is_ill), name="is_ill"
        )
        self.illness_severity = validated_snapshot_scalar(
            state.get("illness_severity", self.illness_severity),
            name="illness_severity",
            minimum=0.0,
            maximum=1.0,
        )
        self.menstrual_cycle_active = validated_snapshot_bool(
            state.get("menstrual_cycle_active", self.menstrual_cycle_active),
            name="menstrual_cycle_active",
        )
        self.cycle_start_time_minutes = validated_snapshot_scalar(
            state.get("cycle_start_time_minutes", self.cycle_start_time_minutes),
            name="cycle_start_time_minutes",
        )
        self.pump_cannula_age_minutes = validated_snapshot_scalar(
            state.get("pump_cannula_age_minutes", self.pump_cannula_age_minutes),
            name="pump_cannula_age_minutes",
            minimum=0.0,
        )

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
        if not np.isfinite(fat_intake) or fat_intake < 0.0:
            raise ValueError("fat_intake must be finite and non-negative")
        if not np.isfinite(protein_intake) or protein_intake < 0.0:
            raise ValueError("protein_intake must be finite and non-negative")
        if not np.isfinite(time_step) or float(time_step) <= 0.0:
            raise ValueError("time_step must be a finite positive number of minutes")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported advanced metabolic update arguments: {names}")

        previous_state = copy.deepcopy(self.get_state())
        try:
            # Add macronutrients and age the cannula before solving this step.
            self._state[16] += fat_intake
            self._state[17] += protein_intake
            self.pump_cannula_age_minutes += float(time_step)

            return super().update(
                time_step=time_step,
                delivered_insulin=delivered_insulin,
                carb_intake=carb_intake,
                delivered_glucagon_mg=delivered_glucagon_mg,
                current_time=current_time,
            )
        except Exception:
            self.set_state(previous_state)
            raise

    def start_illness(self, severity: float) -> None:
        if not np.isfinite(severity) or not 0.0 <= float(severity) <= 1.0:
            raise ValueError("illness severity must be between 0 and 1")
        self.is_ill = True
        self.illness_severity = float(severity)

    def stop_illness(self) -> None:
        self.is_ill = False
        self.illness_severity = 0.0

    def start_menstrual_cycle(self, current_time_minutes: float = 0.0) -> None:
        if not np.isfinite(current_time_minutes):
            raise ValueError("current_time_minutes must be finite")
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
            "state_schema": "advanced_iints_v1_18",
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
        record: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        # `record` matches the superclass signature (BergmanPatientModel._ode)
        # so callers that pass it (e.g. flux_snapshot()) don't crash with a
        # TypeError on this subclass. This 18-state model has no published
        # compartment/flux schema of its own (unlike Bergman/Hovorka), so
        # there is nothing to fill in yet -- `record` is accepted and left
        # empty rather than silently dropped.
        # Unpack 18 states
        G, X, I, Q_sto1, Q_sto2, Q_gut, S1, S2, Y1, Y2, Gamma, x_gluc, HAAF, F, K, Beta, Q_fat, Q_prot = y
        p = self.params

        Vg_abs = p.Vg * p.body_weight_kg   # dL
        Vi_abs = p.Vi * p.body_weight_kg    # L
        glucagon_clearance_ml_min = (
            p.glucagon_clearance_ml_kg_min * p.body_weight_kg
        )
        
        # --- Fat & Protein Macronutrient Kinetics ---
        # Fat decays slowly with a ~4 h model time constant.
        k_fat = 1.0 / 240.0
        dQ_fat_dt = -k_fat * Q_fat
        
        # Protein decays slowly with a ~5 h model time constant.
        k_prot = 1.0 / 300.0
        dQ_prot_dt = -k_prot * Q_prot
        
        # Heuristic delayed protein-to-glucose fraction for stress testing.
        # R_a_prot adds to EGP. (Convert grams to mg, divide by Vg_abs to get mg/dL/min)
        Ra_prot = 0.5 * k_prot * Q_prot * 1000.0 / Vg_abs

        # --- Glucose rate of appearance from gut ---
        Ra = (p.k_abs * Q_gut) / Vg_abs  # mg/dL/min

        # --- Exogenous Glucagon Kinetics ---
        glucagon_k1 = 1.0 / max(p.t_max_glucagon, 1.0)
        glucagon_k2 = max(p.k_e_glucagon, 1e-9)
        dY1_dt = u_glucagon_pg_per_min - glucagon_k1 * Y1
        dY2_dt = glucagon_k1 * Y1 - glucagon_k2 * Y2
        glucagon_concentration = (
            glucagon_k2 * Y2 / max(glucagon_clearance_ml_min, 1e-9)
        )
        dGamma_dt = (
            glucagon_k2 * dY2_dt / max(glucagon_clearance_ml_min, 1e-9)
        )
        glucagon_activation = glucagon_concentration / (
            max(p.glucagon_ec50_pg_ml, 1e-9) + glucagon_concentration
        )
        dx_gluc_dt = p.k_a_glucagon * (
            p.S_glucagon * glucagon_activation - x_gluc
        )

        # --- Phenomenological Dawn Perturbation ---
        dawn_rate = dawn_glucose_rate_mgdl_min(
            current_time,
            peak_strength_mgdl_per_hour=self.dawn_phenomenon_strength,
            start_hour=self.dawn_start_hour,
            end_hour=self.dawn_end_hour,
        )
        # Dawn resistance: a loss of insulin sensitivity over the same window,
        # applied to p3 alongside the existing exercise/stress/illness factors.
        dawn_sens_multiplier = dawn_insulin_sensitivity_multiplier(
            current_time,
            peak_resistance_fraction=self.dawn_insulin_resistance_fraction,
            start_hour=self.dawn_start_hour,
            end_hour=self.dawn_end_hour,
        )

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
            # Lowest sensitivity around day 21 in this explicit research profile.
            luteal_peak_minutes = 21.0 * 24.0 * 60.0
            cycle_p3_multiplier = 1.0 - 0.25 * np.cos(
                2.0 * np.pi * (cycle_time - luteal_peak_minutes) / 40320.0
            )

        # --- Endogenous Rescue & HAAF ---
        rescue_multiplier = counterregulatory_rescue_multiplier(G, HAAF)
        dHAAF_dt = antecedent_hypoglycemia_memory_derivative(G, HAAF)

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
        # Total p3 is degraded by dawn resistance, lipotoxicity, acute illness, and menstrual hormonal drifts
        p3_eff = p.p3 * exercise_p3_multiplier * stress_p3_multiplier * dawn_sens_multiplier * lipotoxicity_factor * illness_p3_multiplier * cycle_p3_multiplier
        
        insulin_deficit = max(0.0, (p.Ib - I) / max(p.Ib, 1e-6))
        ffa_excess = max(0.0, (F - self.initial_ffa) / max(self.initial_ffa, 1e-6))
        ketone_excess = max(
            0.0,
            (K - self.initial_ketones) / max(self.initial_ketones, 0.1),
        )
        # Insulin withdrawal cannot create full starvation physiology in one
        # integration step. FFA and ketone states provide the slower metabolic
        # memory; each contribution saturates to avoid an unbounded EGP jump.
        hepatic_glucose_production_multiplier = 1.0 + (
            0.35 * insulin_deficit
            + 0.75 * ffa_excess / (1.0 + ffa_excess)
            + 0.35 * ketone_excess / (1.0 + ketone_excess)
        )

        # These multipliers are explicit research stressors, not measured
        # hormone concentrations or a clinically identified EGP model.
        Gb_eff = (
            p.Gb
            * stress_Gb_multiplier
            * rescue_multiplier
            * max(0.0, 1.0 + x_gluc)
            * hepatic_glucose_production_multiplier
            * illness_Gb_multiplier
        )

        # --- Physiological Renal Clearance ---
        RGC = renal_glucose_clearance_concentration(G, threshold_mgdl=180.0, gain=0.05, splay_mgdl=10.0)

        # --- dG/dt concentration balance ---
        EGP = p1_eff * Gb_eff
        dGdt = (
            -(p1_eff + X) * G
            + EGP
            + Ra
            + Ra_prot
            + dawn_rate
            - exercise_glucose_uptake
            - RGC
        )

        # --- dX/dt ---
        dXdt = -p.p2 * X + p3_eff * (I - p.Ib)
        
        # --- Heuristic infusion-site ageing stressor ---
        # After 48 hours, absorption drops linearly by up to 30%. This is not
        # a device-specific occlusion or lipohypertrophy probability model.
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

        # Adapted multi-compartment meal chain with a heuristic fat delay.
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
