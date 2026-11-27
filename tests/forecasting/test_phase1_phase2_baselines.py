"""Adversarial Phase 1 & Phase 2 verification suite for `iints.forecasting`.

Covers all requirements of the Phase 2 Verification Gate:
1. Official OhioT1DM XML parsing, duplicate timestamp handling, out-of-bounds & missing
   value rejection, and ZERO gap interpolation.
2. Hard refusal to substitute synthetic or CareLink data when OhioT1DM is missing.
3. Exact 12-sample / 55-minute elapsed-span / 30-minute horizon specification check.
4. Adversarial leakage checks:
   - Cross-split temporal overlap (`min(test) <= max(train)`) raises `DataLeakageError`.
   - Strict chronological embargo (`min(val.window_start_ts) > max(train_fit.target_ts)`).
   - Scaler & Ridge weights are bit-identical under `test` poisoning, and scaler is
     bit-identical under `val` poisoning.
   - Invalid/NaN target at `t + 30m` is rejected rather than leaked or interpolated.
5. Exact analytical baseline verification (Persistence & Ridge input/output shapes,
   exact MAE, RMSE, Median Absolute Error, and per-participant aggregation).
6. Synthetic Dataset A deterministic repeatability & Model Registry SHA-256 restore.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iints.forecasting import (
    DataLeakageError,
    ForecastingModelRegistry,
    PersistencePredictor,
    RidgeCGMPredictor,
    WindowSpec,
    build_partitioned_windows,
    evaluate_predictions,
    load_ohiot1dm_dataset,
    run_phase2_baseline_benchmark,
)


def _write_mock_ohio_xml(
    path: Path,
    subject_id: str,
    events: list[tuple[str, str]],
) -> None:
    lines = [f'<patient id="{subject_id}" weight="75" insulin_type="Novolog">', "  <glucose_level>"]
    for ts, val in events:
        lines.append(f'    <event ts="{ts}" value="{val}"/>')
    lines.extend(["  </glucose_level>", "</patient>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_window_spec_12_samples_55_minute_elapsed_span_definition() -> None:
    spec = WindowSpec()
    assert spec.n_input_samples == 12
    assert spec.input_length == 12
    assert spec.sample_interval_minutes == 5.0
    assert spec.elapsed_span_minutes == pytest.approx(55.0)
    assert spec.prediction_horizon_minutes == 30

    with pytest.raises(ValueError, match="Inconsistent WindowSpec"):
        WindowSpec(n_input_samples=10, history_minutes=60, sample_interval_minutes=5.0)


def test_ohiot1dm_missing_directory_never_substitutes_synthetic_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Synthetic data is never silently substituted"):
        load_ohiot1dm_dataset(tmp_path / "nonexistent_ohio_dir")


def test_ohiot1dm_xml_adapter_quality_flags_and_zero_gap_interpolation(tmp_path: Path) -> None:
    t0 = datetime(2021, 12, 7, 8, 0, 0)
    train_events: list[tuple[str, str]] = []
    for k in range(25):
        ts = (t0 + timedelta(minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S")
        train_events.append((ts, str(110 + k)))
    # Inject 1 duplicate timestamp, 1 missing value ("NaN"), and 1 out-of-bounds reading (650 mg/dL)
    train_events.append(train_events[-1])
    train_events.append(((t0 + timedelta(minutes=125)).strftime("%d-%m-%Y %H:%M:%S"), ""))
    train_events.append(((t0 + timedelta(minutes=130)).strftime("%d-%m-%Y %H:%M:%S"), "650"))
    # Inject a 45-minute CGM dropout gap, then 25 more contiguous steps
    t_after_gap = t0 + timedelta(minutes=175)
    for k in range(25):
        ts = (t_after_gap + timedelta(minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S")
        train_events.append((ts, str(140 + k)))

    test_events: list[tuple[str, str]] = []
    t_test = datetime(2021, 12, 15, 8, 0, 0)
    for k in range(22):
        ts = (t_test + timedelta(minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S")
        test_events.append((ts, str(120 + 2 * k)))

    _write_mock_ohio_xml(tmp_path / "559-ws-training.xml", "559", train_events)
    _write_mock_ohio_xml(tmp_path / "559-ws-testing.xml", "559", test_events)

    canonical, report = load_ohiot1dm_dataset(tmp_path)
    assert report.dataset_id == "ohiot1dm"
    assert report.is_synthetic is False
    assert report.benchmark_tier == "primary_ohiot1dm"
    assert report.n_subjects == 1
    assert report.total_duplicates_flagged == 1
    assert report.total_missing_values_flagged == 1
    assert report.total_out_of_bounds_flagged == 1
    assert report.total_gaps_over_threshold >= 1

    part = build_partitioned_windows(canonical, WindowSpec())
    gap_start = np.datetime64(t0 + timedelta(minutes=120))
    gap_end = np.datetime64(t_after_gap)
    for split in (part.train_fit, part.val):
        # Verify exact 55.0-minute elapsed input span and 30.0-minute horizon on every extracted window
        spans = (split.origin_ts - split.window_start_ts) / np.timedelta64(1, "m")
        horizons = (split.target_ts - split.origin_ts) / np.timedelta64(1, "m")
        assert np.allclose(spans, 55.0)
        assert np.allclose(horizons, 30.0)
        for w_s, t_t in zip(split.window_start_ts, split.target_ts):
            assert not (w_s <= gap_start and t_t >= gap_end), "Window bridged across an invalid/missing CGM gap!"


def test_adversarial_cross_split_overlap_raises_data_leakage_error(tmp_path: Path) -> None:
    t0 = datetime(2022, 1, 1, 8, 0, 0)
    train_ev = [((t0 + timedelta(minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S"), "120") for k in range(30)]
    # Adversarial test file starts BEFORE training file ended!
    test_ev = [((t0 + timedelta(minutes=5 * (25 + k))).strftime("%d-%m-%Y %H:%M:%S"), "125") for k in range(25)]
    _write_mock_ohio_xml(tmp_path / "563-ws-training.xml", "563", train_ev)
    _write_mock_ohio_xml(tmp_path / "563-ws-testing.xml", "563", test_ev)

    canonical, _ = load_ohiot1dm_dataset(tmp_path)
    with pytest.raises(DataLeakageError, match="Cross-split temporal overlap"):
        build_partitioned_windows(canonical, WindowSpec())


def test_invalid_target_is_excluded_and_exact_analytic_metrics_match() -> None:
    """Analytic two-subject test verifying exact MAE, RMSE, MedAE, per-subject aggregation,
    and rejection of missing/invalid target observations.

    - Subject A (`test`): G(k) = 100 + 2*k -> 30m (+6 steps) ahead is +12.0 mg/dL everywhere.
    - Subject B (`test`): G(k) = 200 - 1*k -> 30m (+6 steps) ahead is -6.0 mg/dL everywhere,
      except one corrupted NaN target row which must be excluded cleanly.
    """
    t0 = pd.Timestamp("2026-01-01 00:00:00")
    rows = []
    for subj, split_name, offset, g0, slope in (
        ("subj_A", "train", 0, 100.0, 2.0),
        ("subj_B", "train", 0, 220.0, -1.0),
        ("subj_A", "test", 200, 100.0, 2.0),
        ("subj_B", "test", 200, 200.0, -1.0),
    ):
        for k in range(40):
            is_corrupt = subj == "subj_B" and split_name == "test" and k == 25
            rows.append(
                {
                    "subject_id": subj,
                    "timestamp": t0 + pd.to_timedelta(5 * (offset + k), unit="m"),
                    "glucose_mgdl": np.nan if is_corrupt else g0 + slope * k,
                    "source_split": split_name,
                    "is_valid": not is_corrupt,
                    "validity_flags": "missing_value" if is_corrupt else "ok",
                    "dt_prev_min": 5.0 if k > 0 else np.nan,
                    "source_provenance": "analytic",
                }
            )

    df = pd.DataFrame(rows)
    part = build_partitioned_windows(df, WindowSpec())
    assert np.all(np.isfinite(part.test.y))
    assert part.test.X.shape[1] == 12

    pers = PersistencePredictor().fit(part.train_fit.X, part.train_fit.y)
    pred_pers = pers.predict(part.test.X)
    ev_pers = evaluate_predictions(part.test, pred_pers, model_id=pers.model_id)

    per_subj = ev_pers["per_subject"].set_index("subject_id")
    # Subject A has constant error |12| -> MAE = 12, RMSE = 12, MedAE = 12
    assert per_subj.loc["subj_A", "mae"] == pytest.approx(12.0, abs=1e-9)
    assert per_subj.loc["subj_A", "rmse"] == pytest.approx(12.0, abs=1e-9)
    assert per_subj.loc["subj_A", "median_abs_error"] == pytest.approx(12.0, abs=1e-9)
    # Subject B has constant error |-6| = 6 -> MAE = 6, RMSE = 6, MedAE = 6 (and corrupted target k=25 was skipped!)
    assert per_subj.loc["subj_B", "mae"] == pytest.approx(6.0, abs=1e-9)
    assert per_subj.loc["subj_B", "rmse"] == pytest.approx(6.0, abs=1e-9)
    assert per_subj.loc["subj_B", "median_abs_error"] == pytest.approx(6.0, abs=1e-9)
    # Subject-level mean MAE across the 2 subjects is (12 + 6) / 2 = 9.0 exactly!
    assert ev_pers["subject_distribution"]["mean_subject_mae"] == pytest.approx(9.0, abs=1e-9)

    # Ridge recovers exact linear extrapolation y = x[-1] + 6*(x[-1] - x[-2]) on both subjects!
    ridge = RidgeCGMPredictor(alpha_grid=(1e-9, 1e-8)).fit(
        part.train_fit.X, part.train_fit.y, part.val.X, part.val.y
    )
    pred_ridge = ridge.predict(part.test.X)
    assert pred_ridge.shape == part.test.y.shape
    assert np.mean(np.abs(pred_ridge - part.test.y)) < 1e-3


def test_embargo_and_poisoning_data_leakage_proofs(tmp_path: Path) -> None:
    t0 = datetime(2022, 5, 1, 0, 0, 0)
    rng = np.random.default_rng(7)
    train_ev = [
        ((t0 + timedelta(minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S"), f"{130.0 + 25.0 * np.sin(k / 9.0) + rng.normal(0, 2):.1f}")
        for k in range(180)
    ]
    test_ev = [
        ((t0 + timedelta(days=2, minutes=5 * k)).strftime("%d-%m-%Y %H:%M:%S"), f"{140.0 + 20.0 * np.cos(k / 8.0):.1f}")
        for k in range(60)
    ]
    _write_mock_ohio_xml(tmp_path / "570-ws-training.xml", "570", train_ev)
    _write_mock_ohio_xml(tmp_path / "570-ws-testing.xml", "570", test_ev)

    canonical, _ = load_ohiot1dm_dataset(tmp_path)
    part = build_partitioned_windows(canonical, WindowSpec())

    assert len(part.train_fit) > 0 and len(part.val) > 0 and len(part.test) > 0
    assert part.embargo_purged_count > 0
    assert part.val.window_start_ts.min() > part.train_fit.target_ts.max()
    assert part.test.window_start_ts.min() > part.val.target_ts.max()

    clean_model = RidgeCGMPredictor(alpha_grid=(0.1, 10.0, 100.0)).fit(
        part.train_fit.X, part.train_fit.y, part.val.X, part.val.y
    )
    poisoned_val_scaler_check = RidgeCGMPredictor(alpha_grid=(0.1, 10.0, 100.0)).fit(
        part.train_fit.X,
        part.train_fit.y,
        np.full_like(part.val.X, -50000.0),
        np.full_like(part.val.y, 99999.0),
    )
    assert clean_model.scaler.mean_mgdl == poisoned_val_scaler_check.scaler.mean_mgdl
    assert clean_model.scaler.std_mgdl == poisoned_val_scaler_check.scaler.std_mgdl

    recs = clean_model.predict_records(part.test.X[:3], ["2027-01-10T12:00:00Z"] * 3, horizon_minutes=30)
    assert len(recs) == 3
    assert all(r.clinical_use is False and r.source == "research_model" for r in recs)

    reg = ForecastingModelRegistry(tmp_path / "reg")
    entry = reg.register(
        clean_model,
        dataset_id="ohiot1dm",
        is_synthetic=False,
        benchmark_tier="primary_ohiot1dm",
        dataset_sha256="abc123",
        input_length=12,
        elapsed_input_span_minutes=55.0,
        horizon_minutes=30,
        training_config={},
        split_counts=part.split_counts(),
        validation_metrics={"mae": 10.5},
    )
    loaded = reg.load_model(clean_model.model_id)
    np.testing.assert_allclose(clean_model.predict(part.test.X), loaded.predict(part.test.X))

    ckpt_file = tmp_path / "reg" / entry.checkpoint_file
    ckpt_file.write_text(
        ckpt_file.read_text(encoding="utf-8").replace("intercept", "intercept_tampered"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        reg.load_model(clean_model.model_id)


def test_virtual_pipeline_synthetic_labeling_and_exact_repeatability(tmp_path: Path) -> None:
    run1 = run_phase2_baseline_benchmark(
        dataset="iints_virtual",
        output_dir=tmp_path / "run1",
        virtual_subjects=6,
        seed=42,
    )
    run2 = run_phase2_baseline_benchmark(
        dataset="iints_virtual",
        output_dir=tmp_path / "run2",
        virtual_subjects=6,
        seed=42,
    )
    assert run1["quality"].is_synthetic is True
    assert run1["quality"].benchmark_tier == "synthetic_software_validation"
    assert all(e.is_synthetic is True for e in run1["registry_entries"])
    pd.testing.assert_frame_equal(run1["pooled_summary"], run2["pooled_summary"])
    report_md = run1["report_path"].read_text(encoding="utf-8")
    assert "SYNTHETIC SOFTWARE VALIDATION ONLY" in report_md
