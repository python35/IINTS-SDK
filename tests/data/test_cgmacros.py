from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from iints.cli.cli import app
from iints.data.cgmacros import (
    CGMacrosImportResult,
    extract_cgmacros_meal_episodes,
    import_cgmacros_dataset,
    parse_cgmacros_bio,
    parse_cgmacros_subject_timeseries,
)

runner = CliRunner()


@pytest.fixture
def mock_cgmacros_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "raw_cgmacros"
    data_dir.mkdir(parents=True)

    # 1. Create mock bio.csv
    bio_content = (
        "subject_id,diabetes_status,age,gender,bmi,hba1c_pct,fasting_glucose,fasting_insulin\n"
        "1,healthy,32,female,22.4,5.2,88,6.5\n"
        "2,prediabetes,45,male,28.1,5.9,105,12.0\n"
        "3,t2d,58,male,31.5,7.1,140,18.2\n"
    )
    (data_dir / "bio.csv").write_text(bio_content, encoding="utf-8")

    # 2. Create mock CGMacros-01.csv
    rows_s1 = [
        "time,dexcom,libre,mets,heart_rate,carbs,protein,fat,fiber,calories,meal_type,proportion_eaten",
    ]
    # Generate 150 minutes of time-series data with 1 meal at t=30
    for t in range(150):
        cgm_dex = 90.0 + (t - 30) * 0.8 if t >= 30 else 90.0
        cgm_lib = 88.0 + (t - 30) * 0.75 if t >= 30 else 88.0
        if t == 30:
            carbs, protein, fat, fiber, cal, m_type = 45.0, 15.0, 10.0, 5.0, 330.0, "breakfast"
        else:
            carbs, protein, fat, fiber, cal, m_type = 0.0, 0.0, 0.0, 0.0, 0.0, "none"
        rows_s1.append(f"2025-01-01 08:{t:02d}:00,{cgm_dex},{cgm_lib},1.2,65,{carbs},{protein},{fat},{fiber},{cal},{m_type},1.0")

    (data_dir / "CGMacros-01.csv").write_text("\n".join(rows_s1), encoding="utf-8")

    # 3. Create mock CGMacros-02.csv
    rows_s2 = [
        "time,dexcom,libre,mets,heart_rate,carbs,protein,fat,fiber,calories,meal_type,proportion_eaten",
    ]
    for t in range(150):
        cgm_dex = 110.0 + (t - 20) * 1.1 if t >= 20 else 110.0
        cgm_lib = 108.0 + (t - 20) * 1.05 if t >= 20 else 108.0
        if t == 20:
            carbs, protein, fat, fiber, cal, m_type = 60.0, 20.0, 18.0, 8.0, 480.0, "lunch"
        else:
            carbs, protein, fat, fiber, cal, m_type = 0.0, 0.0, 0.0, 0.0, 0.0, "none"
        rows_s2.append(f"2025-01-01 12:{t:02d}:00,{cgm_dex},{cgm_lib},1.4,72,{carbs},{protein},{fat},{fiber},{cal},{m_type},1.0")

    (data_dir / "CGMacros-02.csv").write_text("\n".join(rows_s2), encoding="utf-8")

    return data_dir


def test_parse_cgmacros_bio(mock_cgmacros_dir: Path):
    bio_dict = parse_cgmacros_bio(mock_cgmacros_dir / "bio.csv")
    assert len(bio_dict) == 3
    assert "CGMacros-01" in bio_dict
    assert bio_dict["CGMacros-01"].diabetes_status == "healthy"
    assert bio_dict["CGMacros-01"].bmi == 22.4
    assert bio_dict["CGMacros-02"].diabetes_status == "prediabetes"
    assert bio_dict["CGMacros-03"].diabetes_status == "t2d"


def test_parse_cgmacros_subject_timeseries(mock_cgmacros_dir: Path):
    df = parse_cgmacros_subject_timeseries(mock_cgmacros_dir / "CGMacros-01.csv")
    assert len(df) == 150
    assert "glucose_dexcom" in df.columns
    assert "glucose_libre" in df.columns
    assert "carbs_g" in df.columns
    assert df["subject_id"].iloc[0] == "CGMacros-01"


def test_extract_cgmacros_meal_episodes(mock_cgmacros_dir: Path):
    df = parse_cgmacros_subject_timeseries(mock_cgmacros_dir / "CGMacros-01.csv")
    meals = extract_cgmacros_meal_episodes(df, min_carbs_g=10.0, horizon_minutes=120, step_minutes=5)
    assert len(meals) == 1
    meal = meals[0]
    assert meal.meal_type == "breakfast"
    assert meal.carbs_g == 45.0
    assert meal.protein_g == 15.0
    assert meal.fat_g == 10.0
    assert len(meal.post_meal_glucose_dexcom_120) == 25  # t0 to t120 at 5-min steps


def test_import_cgmacros_dataset(mock_cgmacros_dir: Path, tmp_path: Path):
    pytest.importorskip("pyarrow", reason="cgmacros_timeseries.parquet output requires pyarrow")
    out_dir = tmp_path / "standardized_cgmacros"
    res = import_cgmacros_dataset(mock_cgmacros_dir, out_dir)

    assert res.subject_count == 2
    assert res.meal_count == 2
    assert res.time_series_rows == 300
    assert (out_dir / "cgmacros_timeseries.parquet").is_file()
    assert (out_dir / "cgmacros_meals.csv").is_file()
    assert (out_dir / "cgmacros_subjects.csv").is_file()
    assert (out_dir / "cgmacros_manifest.json").is_file()

    meals_df = pd.read_csv(out_dir / "cgmacros_meals.csv")
    assert len(meals_df) == 2
    assert "dexcom_t60" in meals_df.columns
    assert "libre_t60" in meals_df.columns


def test_cli_import_cgmacros(mock_cgmacros_dir: Path, tmp_path: Path):
    pytest.importorskip("pyarrow", reason="cgmacros_timeseries.parquet output requires pyarrow")
    out_dir = tmp_path / "cli_standardized"
    result = runner.invoke(
        app,
        [
            "data",
            "import-cgmacros",
            "--input-dir",
            str(mock_cgmacros_dir),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert "CGMacros Ingestion Completed Successfully" in result.output
    assert (out_dir / "cgmacros_meals.csv").is_file()
