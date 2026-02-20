"""Tests for the Bergman Minimal Model patient simulation."""
import pytest
import numpy as np

from iints.core.patient.bergman_model import BergmanPatientModel, BergmanParameters


class TestBergmanPatientModel:
    def test_init_default(self):
        model = BergmanPatientModel()
        assert model.current_glucose == 120.0
        assert model.insulin_on_board == 0.0
        assert model.carbs_on_board == 0.0

    def test_reset(self):
        model = BergmanPatientModel(initial_glucose=130.0)
        # Perturb state
        model.update(5.0, 1.0, 30.0)
        assert model.current_glucose != 130.0
        model.reset()
        assert model.current_glucose == 130.0
        assert model.insulin_on_board == 0.0

    def test_glucose_decreases_with_insulin(self):
        model = BergmanPatientModel(initial_glucose=200.0)
        # Give 2U bolus
        model.update(5.0, 2.0, 0.0)
        # Run several more steps
        for _ in range(20):
            model.update(5.0, 0.0, 0.0)
        assert model.current_glucose < 200.0, "Insulin should lower glucose"

    def test_glucose_rises_with_carbs(self):
        model = BergmanPatientModel(initial_glucose=100.0)
        model.update(5.0, 0.0, 50.0)  # 50g carbs
        for _ in range(10):
            model.update(5.0, 0.0, 0.0)
        assert model.current_glucose > 100.0, "Carbs should raise glucose"

    def test_glucose_floor(self):
        model = BergmanPatientModel(initial_glucose=50.0)
        # Massive insulin bolus
        model.update(5.0, 20.0, 0.0)
        for _ in range(100):
            model.update(5.0, 0.0, 0.0)
        assert model.current_glucose >= 20.0, "Glucose must not drop below 20 mg/dL"

    def test_iob_tracks_insulin(self):
        model = BergmanPatientModel()
        model.update(5.0, 3.0, 0.0)
        assert model.insulin_on_board > 0.0

    def test_iob_decays(self):
        model = BergmanPatientModel()
        model.update(5.0, 3.0, 0.0)
        iob_after_bolus = model.insulin_on_board
        # Run without insulin for DIA
        for _ in range(60):
            model.update(5.0, 0.0, 0.0)
        assert model.insulin_on_board < iob_after_bolus

    def test_cob_tracks_carbs(self):
        model = BergmanPatientModel()
        model.update(5.0, 0.0, 40.0)
        assert model.carbs_on_board > 0.0

    def test_exercise_lowers_glucose(self):
        model_ex = BergmanPatientModel(initial_glucose=150.0)
        model_no = BergmanPatientModel(initial_glucose=150.0)
        model_ex.start_exercise(0.7)
        for _ in range(20):
            model_ex.update(5.0, 0.0, 0.0)
            model_no.update(5.0, 0.0, 0.0)
        model_ex.stop_exercise()
        assert model_ex.current_glucose < model_no.current_glucose

    def test_meal_mismatch(self):
        model_a = BergmanPatientModel(initial_glucose=100.0, meal_mismatch_epsilon=0.7)
        model_b = BergmanPatientModel(initial_glucose=100.0, meal_mismatch_epsilon=1.3)
        model_a.update(5.0, 0.0, 50.0)
        model_b.update(5.0, 0.0, 50.0)
        for _ in range(20):
            model_a.update(5.0, 0.0, 0.0)
            model_b.update(5.0, 0.0, 0.0)
        # Higher epsilon -> more carbs -> higher glucose
        assert model_b.current_glucose > model_a.current_glucose

    def test_interface_compatibility(self):
        """Bergman should expose the same methods as CustomPatientModel."""
        model = BergmanPatientModel()
        # All interface methods must exist and return correct types
        assert isinstance(model.get_current_glucose(), float)
        assert isinstance(model.get_patient_state(), dict)
        assert isinstance(model.get_ratio_state(), dict)
        model.set_ratio_state(isf=60.0, icr=12.0)
        assert model.insulin_sensitivity == 60.0
        assert model.carb_factor == 12.0
        state = model.get_state()
        assert "ode_state" in state
        model.set_state(state)
        model.trigger_event("test", 1)  # should not raise

    def test_custom_bergman_params(self):
        params = BergmanParameters(p1=0.03, Gb=110.0)
        model = BergmanPatientModel(initial_glucose=110.0, bergman_params=params)
        assert model.params.p1 == 0.03
        assert model.params.Gb == 110.0

    def test_patient_state_extra_fields(self):
        model = BergmanPatientModel()
        state = model.get_patient_state()
        assert "plasma_insulin_mU_L" in state
        assert "remote_insulin_action" in state
        assert "gut_glucose_mg" in state

    def test_simulation_12h_stable(self):
        """A 12h simulation without events should not crash or diverge."""
        model = BergmanPatientModel(initial_glucose=120.0)
        for step in range(144):  # 12h at 5-min steps
            g = model.update(5.0, 0.0, 0.0, current_time=float(step * 5))
            assert 20.0 <= g <= 400.0, f"Glucose out of range at step {step}: {g}"
