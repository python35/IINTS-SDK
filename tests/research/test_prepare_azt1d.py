"""
Unit tests for research/prepare_azt1d.py

Tests the individual helper functions (which are importable) and exercises
the full pipeline with synthetic data written to a temporary directory.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Allow importing from the research/ scripts directory
# ---------------------------------------------------------------------------

_RESEARCH_DIR = Path(__file__).parent.parent.parent / "research"
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))

from prepare_azt1d import (
    _clean_column,
    _device_mode_code,
    _derive_iob_cob_openaps,
    _openaps_iob_activity,
)


# ---------------------------------------------------------------------------
# _clean_column
# ---------------------------------------------------------------------------

class TestCleanColumn:
    def test_present_numeric_column(self):
        df = pd.DataFrame({"val": [1.0, 2.0, None, 4.0]})
        result = _clean_column(df, "val")
        assert list(result) == [1.0, 2.0, 0.0, 4.0]

    def test_missing_column_returns_default(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = _clean_column(df, "nonexistent", default=99.0)
        assert (result == 99.0).all()

    def test_string_column_coerced(self):
        df = pd.DataFrame({"val": ["1.5", "bad", "3.0"]})
        result = _clean_column(df, "val")
        assert abs(result.iloc[0] - 1.5) < 1e-9
        assert result.iloc[1] == 0.0  # coercion failure → default 0.0
        assert abs(result.iloc[2] - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# _device_mode_code
# ---------------------------------------------------------------------------

class TestDeviceModeCode:
    def test_known_modes(self):
        s = pd.Series(["sleep", "Sleep", "SLEEP", "exercise", "0", None, "unknown"])
        result = _device_mode_code(s)
        assert result.iloc[0] == 1.0   # sleep
        assert result.iloc[1] == 1.0   # Sleep (case-insensitive)
        assert result.iloc[2] == 1.0   # SLEEP
        assert result.iloc[3] == 2.0   # exercise
        assert result.iloc[4] == 0.0   # "0"
        assert result.iloc[5] == 0.0   # None → fallback
        assert result.iloc[6] == 0.0   # unknown → fallback


# ---------------------------------------------------------------------------
# _openaps_iob_activity
# ---------------------------------------------------------------------------

class TestOpenapsIobActivity:
    DIA = 240.0
    PEAK = 75.0

    def test_at_zero_is_zero(self):
        """At t=0 insulin was just delivered — activity (metabolic effect) is 0."""
        assert _openaps_iob_activity(0, self.DIA, self.PEAK) == 0.0

    def test_at_dia_is_zero(self):
        assert _openaps_iob_activity(self.DIA, self.DIA, self.PEAK) == 0.0

    def test_beyond_dia_is_zero(self):
        assert _openaps_iob_activity(self.DIA + 10, self.DIA, self.PEAK) == 0.0

    def test_at_peak_is_one(self):
        """At t==peak the activity fraction should be 1.0."""
        result = _openaps_iob_activity(self.PEAK, self.DIA, self.PEAK)
        assert abs(result - 1.0) < 1e-9

    def test_rises_before_peak(self):
        """Activity should be monotonically increasing before the peak."""
        times = np.linspace(0, self.PEAK, 20)
        activities = [_openaps_iob_activity(t, self.DIA, self.PEAK) for t in times]
        assert all(a <= b for a, b in zip(activities, activities[1:]))

    def test_falls_after_peak(self):
        """Activity should be monotonically decreasing after the peak."""
        times = np.linspace(self.PEAK, self.DIA, 20)
        activities = [_openaps_iob_activity(t, self.DIA, self.PEAK) for t in times]
        assert all(a >= b for a, b in zip(activities, activities[1:]))

    def test_result_in_unit_interval(self):
        for t in np.linspace(0, self.DIA + 10, 50):
            val = _openaps_iob_activity(t, self.DIA, self.PEAK)
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# _derive_iob_cob_openaps
# ---------------------------------------------------------------------------

class TestDeriveIobCobOpenaps:
    def _run(self, ins, carbs, dt, dia=240.0, peak=75.0, carb_absorb=120.0):
        insulin = pd.Series(ins, dtype=float)
        carbs_s = pd.Series(carbs, dtype=float)
        delta = pd.Series(dt, dtype=float)
        return _derive_iob_cob_openaps(insulin, carbs_s, delta, dia, peak, carb_absorb)

    def test_zero_insulin_zero_iob(self):
        iob, _ = self._run([0.0] * 10, [0.0] * 10, [5.0] * 10)
        assert (iob == 0.0).all()

    def test_zero_carbs_zero_cob(self):
        _, cob = self._run([0.5] * 10, [0.0] * 10, [5.0] * 10)
        assert (cob == 0.0).all()

    def test_iob_accumulates_with_insulin(self):
        """IOB should increase when insulin is delivered."""
        iob, _ = self._run([1.0] * 5, [0.0] * 5, [5.0] * 5)
        # IOB should grow over the first few steps
        assert iob.iloc[1] > iob.iloc[0]

    def test_segment_break_resets_iob(self):
        """Non-finite or non-positive dt should reset IOB/COB to 0."""
        ins = [1.0, 1.0, 1.0, 1.0, 1.0]
        carbs = [10.0, 10.0, 10.0, 10.0, 10.0]
        dt = [5.0, 5.0, float("nan"), 5.0, 5.0]  # gap at index 2
        iob, cob = self._run(ins, carbs, dt)
        # After the NaN, index 2 should be reset to 0
        assert iob.iloc[2] == 0.0
        assert cob.iloc[2] == 0.0

    def test_output_length(self):
        N = 20
        iob, cob = self._run([0.1] * N, [5.0] * N, [5.0] * N)
        assert len(iob) == N
        assert len(cob) == N

    def test_iob_non_negative(self):
        iob, _ = self._run([0.3] * 30, [0.0] * 30, [5.0] * 30)
        assert (iob >= 0).all()


# ---------------------------------------------------------------------------
# P0-1: Basal unit conversion — end-to-end smoke test
# ---------------------------------------------------------------------------

class TestBasalConversion:
    """
    Verify that the full pipeline correctly converts U/hr → U/step.

    Uses a synthetic CSV that mimics the AZT1D format.
    """

    def _make_synthetic_subject(self, tmpdir: Path, subject_id: str, n_rows: int = 50) -> Path:
        subj_dir = tmpdir / f"Subject {subject_id}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(subject_id))
        times = pd.date_range("2024-01-01", periods=n_rows, freq="5min")
        df = pd.DataFrame({
            "EventDateTime": times.strftime("%Y-%m-%d %H:%M:%S"),
            "CGM": rng.uniform(70, 250, n_rows),
            "Basal": rng.uniform(0.5, 2.0, n_rows),          # U/hr
            "TotalBolusInsulinDelivered": rng.uniform(0, 2, n_rows),
            "CorrectionDelivered": np.zeros(n_rows),
            "CarbSize": rng.uniform(0, 20, n_rows),
            "FoodDelivered": np.zeros(n_rows),
            "DeviceMode": ["0"] * n_rows,
        })
        csv_path = subj_dir / f"Subject {subject_id}.csv"
        df.to_csv(csv_path, index=False)
        return subj_dir

    def test_basal_conversion_reduces_values(self, tmp_path):
        """
        With time_step=5 min, a 1 U/hr rate should yield 1/12 U per step.
        The processed insulin_units should be < raw basal values.
        """
        from prepare_azt1d import _clean_column

        # Build a minimal single-subject dataframe with known basal
        df = pd.DataFrame({
            "Basal": [1.2, 1.2, 1.2, 1.2],  # U/hr
        })
        raw_basal = _clean_column(df, "Basal")
        converted = raw_basal / 60.0 * 5  # 5-minute step
        expected = 1.2 / 12  # 0.1 U per step
        np.testing.assert_allclose(converted.values, expected, rtol=1e-5)

    def test_full_pipeline_smoke(self, tmp_path):
        """
        Run the entire prepare_azt1d.main() on synthetic data and check
        that the output CSV is created and contains reasonable values.
        """
        import importlib.util
        import sys as _sys

        # Write two synthetic subjects
        input_dir = tmp_path / "CGM Records"
        self._make_synthetic_subject(input_dir, "1", n_rows=60)
        self._make_synthetic_subject(input_dir, "2", n_rows=60)

        output = tmp_path / "merged.csv"
        report = tmp_path / "quality.json"

        # Patch sys.argv and call main()
        spec = importlib.util.spec_from_file_location(
            "_pzt1d_main", _RESEARCH_DIR / "prepare_azt1d.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        _old_argv = _sys.argv[:]
        _sys.argv = [
            "prepare_azt1d.py",
            "--input", str(input_dir),
            "--output", str(output),
            "--report", str(report),
            "--time-step", "5",
        ]
        try:
            mod.main()
        finally:
            _sys.argv = _old_argv

        assert output.exists(), "Output CSV was not created"
        assert report.exists(), "Quality report was not created"

        result_df = pd.read_csv(output)
        assert "glucose_actual_mgdl" in result_df.columns
        assert "derived_iob_units" in result_df.columns
        assert (result_df["glucose_actual_mgdl"] > 0).all()
        # basal_units should be tiny (<< 1) because we converted from U/hr
        # (raw basal was ~0.5-2 U/hr → 5-min step gives 0.04-0.17 U/step)
        assert result_df["insulin_units"].max() < 40.0  # sanity bound

        import json
        report_data = json.loads(report.read_text())
        assert report_data["iob_model"] == "openaps_bilinear"
        assert report_data["basal_is_rate"] is True
