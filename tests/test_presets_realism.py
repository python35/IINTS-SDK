from __future__ import annotations

from iints.core.algorithms.clinical_baseline import ClinicalBaselineAlgorithm
from iints.data.realism_validator import validate_realism_dataset
from iints.highlevel import run_simulation
from iints.presets import get_preset


def test_quickstart_preset_contains_early_meal_excursion() -> None:
    preset = get_preset("quickstart_meal")
    meals = [
        event
        for event in preset["scenario"]["stress_events"]
        if event["event_type"] == "meal"
    ]

    assert preset["duration_minutes"] == 180
    assert meals
    assert meals[0]["start_time"] <= 30
    assert any(event["reported_value"] < event["value"] for event in meals)


def test_free_living_preset_includes_realistic_day_features() -> None:
    preset = get_preset("free_living_t1d")
    events = preset["scenario"]["stress_events"]
    meals = [event for event in events if event["event_type"] == "meal"]

    assert preset["duration_minutes"] == 1440
    assert preset["patient_config"] == "reference_free_living_t1d"
    assert len(meals) >= 3
    assert any(event["reported_value"] < event["value"] for event in meals)
    assert any(event["event_type"] == "exercise" for event in events)


def test_free_living_preset_is_realistic_across_multiple_seeds(tmp_path) -> None:
    preset = get_preset("free_living_t1d")
    verdicts: list[str] = []
    for seed in (1, 42, 99):
        outputs = run_simulation(
            algorithm=ClinicalBaselineAlgorithm(),
            scenario=preset["scenario"],
            patient_config=preset["patient_config"],
            duration_minutes=preset["duration_minutes"],
            time_step=preset["time_step_minutes"],
            seed=seed,
            output_dir=tmp_path / f"free_living_seed_{seed}",
            compare_baselines=False,
            export_audit=False,
            generate_report=False,
        )
        standard_df = outputs["results"].rename(
            columns={
                "time_minutes": "timestamp",
                "glucose_actual_mgdl": "glucose",
                "carb_intake_grams": "carbs",
                "delivered_insulin_units": "insulin",
            }
        )[["timestamp", "glucose", "carbs", "insulin"]]
        report = validate_realism_dataset(standard_df, reference="free_living_t1d")
        verdicts.append(report.verdict)

    assert verdicts.count("likely_realistic") >= 2


def test_reference_day_preset_stays_inside_real_data_envelope(tmp_path) -> None:
    preset = get_preset("realistic_reference_day")
    for seed in (1, 42, 99):
        outputs = run_simulation(
            algorithm=ClinicalBaselineAlgorithm(),
            scenario=preset["scenario"],
            patient_config=preset["patient_config"],
            duration_minutes=preset["duration_minutes"],
            time_step=preset["time_step_minutes"],
            seed=seed,
            output_dir=tmp_path / f"reference_day_realism_seed_{seed}",
            compare_baselines=False,
            export_audit=False,
            generate_report=False,
        )
        standard_df = outputs["results"].rename(
            columns={
                "time_minutes": "timestamp",
                "glucose_actual_mgdl": "glucose",
                "carb_intake_grams": "carbs",
                "delivered_insulin_units": "insulin",
            }
        )[["timestamp", "glucose", "carbs", "insulin"]]

        report = validate_realism_dataset(standard_df, reference="free_living_t1d")

        assert report.verdict == "likely_realistic"
        statuses = {check.code: check.status for check in report.checks}
        assert statuses["quality_basics"] == "passed"
        assert statuses["meal_response"] == "passed"


def test_default_baseline_preset_uses_realistic_patient_profile(tmp_path) -> None:
    preset = get_preset("baseline_t1d")

    assert preset["patient_config"] == "reference_free_living_t1d"

    outputs = run_simulation(
        algorithm=ClinicalBaselineAlgorithm(),
        scenario=preset["scenario"],
        patient_config=preset["patient_config"],
        duration_minutes=preset["duration_minutes"],
        time_step=preset["time_step_minutes"],
        seed=42,
        output_dir=tmp_path / "baseline_t1d",
        compare_baselines=False,
        export_audit=False,
        generate_report=False,
    )
    standard_df = outputs["results"].rename(
        columns={
            "time_minutes": "timestamp",
            "glucose_actual_mgdl": "glucose",
            "carb_intake_grams": "carbs",
            "delivered_insulin_units": "insulin",
        }
    )[["timestamp", "glucose", "carbs", "insulin"]]

    report = validate_realism_dataset(standard_df, reference="free_living_t1d")

    assert report.verdict == "likely_realistic"


def test_dataset_specific_reference_presets_target_their_own_envelopes(tmp_path) -> None:
    cases = [
        ("realistic_azt1d_day", "azt1d_daily"),
        ("realistic_hupa_ucm_day", "hupa_ucm_daily"),
    ]
    for preset_name, reference in cases:
        preset = get_preset(preset_name)
        outputs = run_simulation(
            algorithm=ClinicalBaselineAlgorithm(),
            scenario=preset["scenario"],
            patient_config=preset["patient_config"],
            duration_minutes=preset["duration_minutes"],
            time_step=preset["time_step_minutes"],
            seed=42,
            output_dir=tmp_path / preset_name,
            compare_baselines=False,
            export_audit=False,
            generate_report=False,
        )
        standard_df = outputs["results"].rename(
            columns={
                "time_minutes": "timestamp",
                "glucose_actual_mgdl": "glucose",
                "carb_intake_grams": "carbs",
                "delivered_insulin_units": "insulin",
            }
        )[["timestamp", "glucose", "carbs", "insulin"]]

        report = validate_realism_dataset(standard_df, reference=reference)

        assert report.verdict in {"likely_realistic", "needs_review"}
