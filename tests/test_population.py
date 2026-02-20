"""Tests for the Monte Carlo population testing module."""
import pytest
import numpy as np

from iints.core.patient.profile import PatientProfile
from iints.population.generator import (
    ParameterDistribution,
    PopulationConfig,
    PopulationGenerator,
)
from iints.population.runner import (
    PopulationResult,
    _compute_aggregate_metrics,
    _compute_aggregate_safety,
)


# ---------------------------------------------------------------------------
# PopulationGenerator
# ---------------------------------------------------------------------------

class TestPopulationGenerator:
    def test_default_config_generates_correct_count(self):
        config = PopulationConfig(n_patients=10, seed=42)
        gen = PopulationGenerator(config)
        profiles = gen.generate()
        assert len(profiles) == 10

    def test_all_profiles_are_patient_profiles(self):
        config = PopulationConfig(n_patients=5, seed=42)
        gen = PopulationGenerator(config)
        for p in gen.generate():
            assert isinstance(p, PatientProfile)

    def test_reproducible_with_seed(self):
        cfg_a = PopulationConfig(n_patients=20, seed=123)
        cfg_b = PopulationConfig(n_patients=20, seed=123)
        profiles_a = PopulationGenerator(cfg_a).generate()
        profiles_b = PopulationGenerator(cfg_b).generate()
        for a, b in zip(profiles_a, profiles_b):
            assert a.isf == b.isf
            assert a.icr == b.icr
            assert a.basal_rate == b.basal_rate

    def test_variation_exists(self):
        config = PopulationConfig(n_patients=50, seed=42)
        profiles = PopulationGenerator(config).generate()
        isf_values = [p.isf for p in profiles]
        assert max(isf_values) != min(isf_values), "ISF should vary across population"

    def test_bounds_respected(self):
        config = PopulationConfig(n_patients=200, seed=42)
        profiles = PopulationGenerator(config).generate()
        for p in profiles:
            assert 10.0 <= p.isf <= 200.0
            assert 3.0 <= p.icr <= 30.0
            assert 0.1 <= p.basal_rate <= 3.0
            assert 70.0 <= p.initial_glucose <= 300.0
            assert 120.0 <= p.insulin_action_duration <= 600.0

    def test_custom_cv_override(self):
        config = PopulationConfig(n_patients=100, seed=42)
        config.parameter_distributions["isf"].cv = 0.50
        profiles = PopulationGenerator(config).generate()
        isf_values = [p.isf for p in profiles]
        # With 50 % CV the spread should be large
        assert np.std(isf_values) > 5.0

    def test_zero_cv_produces_constant(self):
        config = PopulationConfig(n_patients=10, seed=42)
        config.parameter_distributions["isf"].cv = 0.0
        profiles = PopulationGenerator(config).generate()
        isf_values = [p.isf for p in profiles]
        assert all(v == isf_values[0] for v in isf_values)

    def test_non_varied_params_carry_forward(self):
        base = PatientProfile(dawn_start_hour=5.5, glucose_decay_rate=0.07)
        config = PopulationConfig(n_patients=5, base_profile=base, seed=42)
        profiles = PopulationGenerator(config).generate()
        for p in profiles:
            assert p.dawn_start_hour == 5.5
            assert p.glucose_decay_rate == 0.07

    def test_log_normal_distribution(self):
        dist = ParameterDistribution(mean=50.0, cv=0.20, distribution="log_normal", lower_bound=10.0, upper_bound=200.0)
        config = PopulationConfig(n_patients=500, seed=42)
        config.parameter_distributions["isf"] = dist
        profiles = PopulationGenerator(config).generate()
        isf_values = [p.isf for p in profiles]
        # Log-normal is right-skewed: median < mean
        assert np.median(isf_values) <= np.mean(isf_values) + 5


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

class TestAggregation:
    def _make_df(self, n=50):
        import pandas as pd
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "tir_70_180": rng.normal(65, 10, n),
            "tir_below_70": rng.normal(5, 2, n),
            "tir_below_54": rng.normal(1, 0.5, n),
            "mean_glucose": rng.normal(140, 20, n),
            "cv": rng.normal(30, 5, n),
            "gmi": rng.normal(7.0, 0.5, n),
            "safety_index_score": rng.normal(80, 10, n),
            "safety_grade": rng.choice(["A", "B", "C"], n),
            "terminated_early": rng.choice([True, False], n, p=[0.05, 0.95]),
        })

    def test_aggregate_metrics_has_ci(self):
        df = self._make_df()
        agg = _compute_aggregate_metrics(df)
        assert "tir_70_180" in agg
        assert "ci_lower" in agg["tir_70_180"]
        assert "ci_upper" in agg["tir_70_180"]
        assert agg["tir_70_180"]["ci_lower"] <= agg["tir_70_180"]["mean"]
        assert agg["tir_70_180"]["ci_upper"] >= agg["tir_70_180"]["mean"]

    def test_aggregate_safety_has_grade_distribution(self):
        df = self._make_df()
        agg = _compute_aggregate_safety(df)
        assert "safety_index" in agg
        assert "grade_distribution" in agg
        total_grades = sum(agg["grade_distribution"].values())
        assert total_grades == len(df)

    def test_early_termination_rate(self):
        df = self._make_df()
        agg = _compute_aggregate_safety(df)
        assert 0.0 <= agg["early_termination_rate"] <= 1.0
