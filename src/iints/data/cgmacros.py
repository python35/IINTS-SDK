from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CGMacrosSubjectBio:
    """Demographics, anthropometrics, and blood biomarkers for one CGMacros participant."""

    subject_id: str
    diabetes_status: str  # 'healthy', 'prediabetes', 't2d'
    age: float | None
    gender: str | None
    bmi: float | None
    hba1c_pct: float | None
    fasting_glucose_mgdl: float | None
    fasting_insulin_u_per_ml: float | None
    triglycerides_mgdl: float | None
    cholesterol_mgdl: float | None
    hdl_mgdl: float | None
    ldl_mgdl: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CGMacrosMealEvent:
    """One quantified meal event with exact macronutrient composition."""

    meal_id: str
    subject_id: str
    timestamp: str
    time_minutes: float
    meal_type: str  # 'breakfast', 'lunch', 'dinner', 'snack'
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float
    calories_kcal: float
    proportion_eaten: float
    pre_meal_glucose_dexcom: float | None
    pre_meal_glucose_libre: float | None
    post_meal_glucose_dexcom_120: tuple[float | None, ...]  # 24 steps at 5-min intervals
    post_meal_glucose_libre_120: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["post_meal_glucose_dexcom_120"] = list(self.post_meal_glucose_dexcom_120)
        data["post_meal_glucose_libre_120"] = list(self.post_meal_glucose_libre_120)
        return data


@dataclass(frozen=True)
class CGMacrosImportResult:
    """Summary of the standardized CGMacros dataset ingestion."""

    output_dir: Path
    subject_count: int
    meal_count: int
    time_series_rows: int
    status_distribution: Mapping[str, int]
    dexcom_measurements: int
    libre_measurements: int
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _find_col(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    norm_map = {_normalize_name(c): c for c in columns}
    for cand in candidates:
        norm_cand = _normalize_name(cand)
        if norm_cand in norm_map:
            return norm_map[norm_cand]
    return None


def parse_cgmacros_bio(bio_path: Path | str) -> dict[str, CGMacrosSubjectBio]:
    """
    Parse CGMacros bio.csv containing clinical screening and subject metadata.
    """
    path = Path(bio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CGMacros bio file not found: {path}")

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    id_col = _find_col(df.columns, ["subject_id", "subject", "participant", "id", "id_number", "record_id"])
    if not id_col:
        raise ValueError(f"could not identify subject ID column in bio file; available: {list(df.columns)}")

    status_col = _find_col(df.columns, ["diabetes_status", "status", "group", "diagnosis", "condition", "cohort"])
    age_col = _find_col(df.columns, ["age", "years"])
    gender_col = _find_col(df.columns, ["gender", "sex"])
    bmi_col = _find_col(df.columns, ["bmi", "body_mass_index"])
    hba1c_col = _find_col(df.columns, ["hba1c", "hba1c_pct", "a1c"])
    fast_glu_col = _find_col(df.columns, ["fasting_glucose", "fpg", "fasting_bg", "fasting_glucose_mgdl"])
    fast_ins_col = _find_col(df.columns, ["fasting_insulin", "insulin_fasting", "fasting_insulin_u_per_ml"])
    tri_col = _find_col(df.columns, ["triglycerides", "triglyceride", "tg"])
    chol_col = _find_col(df.columns, ["cholesterol", "total_cholesterol", "chol"])
    hdl_col = _find_col(df.columns, ["hdl", "hdl_cholesterol"])
    ldl_col = _find_col(df.columns, ["ldl", "ldl_cholesterol"])

    subjects: dict[str, CGMacrosSubjectBio] = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col]).strip()
        if not raw_id or raw_id == "nan":
            continue
        # Standardize subject id e.g. '1' -> 'CGMacros-01' or 'CGMacros-1'
        num_match = re.search(r"\d+", raw_id)
        subj_id = f"CGMacros-{int(num_match.group(0)):02d}" if num_match else raw_id

        # Determine status
        raw_status = str(row.get(status_col, "")).lower().strip() if status_col else ""
        hba1c_val = pd.to_numeric(row.get(hba1c_col, np.nan), errors="coerce")
        if "t2d" in raw_status or "diabetes" in raw_status and "pre" not in raw_status or (pd.notna(hba1c_val) and hba1c_val >= 6.5):
            status = "t2d"
        elif "pre" in raw_status or (pd.notna(hba1c_val) and 5.7 <= hba1c_val < 6.5):
            status = "prediabetes"
        else:
            status = "healthy"

        def _get_float(col: str | None) -> float | None:
            if not col:
                return None
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            return float(val) if pd.notna(val) and np.isfinite(val) else None

        subjects[subj_id] = CGMacrosSubjectBio(
            subject_id=subj_id,
            diabetes_status=status,
            age=_get_float(age_col),
            gender=str(row[gender_col]).strip().lower() if gender_col and pd.notna(row[gender_col]) else None,
            bmi=_get_float(bmi_col),
            hba1c_pct=float(hba1c_val) if pd.notna(hba1c_val) and np.isfinite(hba1c_val) else None,
            fasting_glucose_mgdl=_get_float(fast_glu_col),
            fasting_insulin_u_per_ml=_get_float(fast_ins_col),
            triglycerides_mgdl=_get_float(tri_col),
            cholesterol_mgdl=_get_float(chol_col),
            hdl_mgdl=_get_float(hdl_col),
            ldl_mgdl=_get_float(ldl_col),
        )

    return subjects


def parse_cgmacros_subject_timeseries(
    file_path: Path | str,
    subject_id: str | None = None,
) -> pd.DataFrame:
    """
    Parse a single CGMacros-#.csv file containing dual CGM and meal logs.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CGMacros file not found: {path}")

    if subject_id is None:
        num_match = re.search(r"(\d+)", path.stem)
        subject_id = f"CGMacros-{int(num_match.group(1)):02d}" if num_match else path.stem

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)

    time_col = _find_col(df.columns, ["time", "timestamp", "datetime", "date_time", "date", "time_minutes", "time_min"])
    dex_col = _find_col(df.columns, ["dexcom", "dexcom_g6", "cgm_dexcom", "glucose_dexcom", "dexcom_mgdl", "dexcom_glucose"])
    libre_col = _find_col(df.columns, ["libre", "freestyle_libre", "cgm_libre", "glucose_libre", "libre_mgdl", "libre_glucose"])
    met_col = _find_col(df.columns, ["mets", "fitbit_mets", "met", "activity_mets"])
    hr_col = _find_col(df.columns, ["heart_rate", "hr", "fitbit_hr", "pulse"])

    carb_col = _find_col(df.columns, ["carbs", "carb_g", "carbohydrates", "carbs_grams", "carbohydrate_g", "carbs_g"])
    protein_col = _find_col(df.columns, ["protein", "protein_g", "protein_grams"])
    fat_col = _find_col(df.columns, ["fat", "fat_g", "fat_grams", "total_fat"])
    fiber_col = _find_col(df.columns, ["fiber", "fiber_g", "dietary_fiber"])
    cal_col = _find_col(df.columns, ["calories", "calories_kcal", "kcal", "energy_kcal"])
    meal_type_col = _find_col(df.columns, ["meal_type", "meal", "type"])
    prop_col = _find_col(df.columns, ["proportion_eaten", "proportion", "amount_eaten"])

    out_rows: list[dict[str, Any]] = []
    t_min = 0.0
    for idx, row in df.iterrows():
        raw_time = str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else f"t_{idx}"
        
        dex_val = pd.to_numeric(row.get(dex_col, np.nan), errors="coerce") if dex_col else np.nan
        libre_val = pd.to_numeric(row.get(libre_col, np.nan), errors="coerce") if libre_col else np.nan
        met_val = pd.to_numeric(row.get(met_col, np.nan), errors="coerce") if met_col else np.nan
        hr_val = pd.to_numeric(row.get(hr_col, np.nan), errors="coerce") if hr_col else np.nan

        carb_val = pd.to_numeric(row.get(carb_col, np.nan), errors="coerce") if carb_col else np.nan
        protein_val = pd.to_numeric(row.get(protein_col, np.nan), errors="coerce") if protein_col else np.nan
        fat_val = pd.to_numeric(row.get(fat_col, np.nan), errors="coerce") if fat_col else np.nan
        fiber_val = pd.to_numeric(row.get(fiber_col, np.nan), errors="coerce") if fiber_col else np.nan
        cal_val = pd.to_numeric(row.get(cal_col, np.nan), errors="coerce") if cal_col else np.nan
        prop_val = pd.to_numeric(row.get(prop_col, 1.0), errors="coerce") if prop_col else 1.0

        out_rows.append({
            "subject_id": subject_id,
            "timestamp": raw_time,
            "time_minutes": t_min,
            "glucose_dexcom": float(dex_val) if pd.notna(dex_val) and dex_val > 0 else np.nan,
            "glucose_libre": float(libre_val) if pd.notna(libre_val) and libre_val > 0 else np.nan,
            "mets": float(met_val) if pd.notna(met_val) else np.nan,
            "heart_rate": float(hr_val) if pd.notna(hr_val) else np.nan,
            "carbs_g": float(carb_val) if pd.notna(carb_val) and carb_val > 0 else 0.0,
            "protein_g": float(protein_val) if pd.notna(protein_val) and protein_val > 0 else 0.0,
            "fat_g": float(fat_val) if pd.notna(fat_val) and fat_val > 0 else 0.0,
            "fiber_g": float(fiber_val) if pd.notna(fiber_val) and fiber_val > 0 else 0.0,
            "calories_kcal": float(cal_val) if pd.notna(cal_val) and cal_val > 0 else 0.0,
            "meal_type": str(row.get(meal_type_col, "none")).lower().strip() if meal_type_col and pd.notna(row[meal_type_col]) else "none",
            "proportion_eaten": float(prop_val) if pd.notna(prop_val) else 1.0,
        })
        t_min += 1.0  # CGMacros standardized interpolation step (1 minute)

    return pd.DataFrame(out_rows)


def extract_cgmacros_meal_episodes(
    timeseries_df: pd.DataFrame,
    min_carbs_g: float = 5.0,
    horizon_minutes: int = 120,
    step_minutes: int = 5,
) -> list[CGMacrosMealEvent]:
    """
    Extract discrete meal episodes and pair them with 2-hour postprandial glucose curves.
    """
    meals: list[CGMacrosMealEvent] = []
    n_steps = horizon_minutes // step_minutes

    meal_rows = timeseries_df[timeseries_df["carbs_g"] >= min_carbs_g].copy()
    for m_idx, (row_idx, m_row) in enumerate(meal_rows.iterrows()):
        subject_id = str(m_row["subject_id"])
        meal_t = float(m_row["time_minutes"])
        timestamp = str(m_row["timestamp"])
        meal_type = str(m_row["meal_type"]) if m_row["meal_type"] != "none" else "meal"

        # Look up 2-hour postprandial window
        post_window = timeseries_df[
            (timeseries_df["time_minutes"] >= meal_t)
            & (timeseries_df["time_minutes"] <= meal_t + horizon_minutes)
        ]

        dex_vals: list[float | None] = []
        libre_vals: list[float | None] = []
        for step in range(n_steps + 1):
            target_t = meal_t + step * step_minutes
            closest = post_window.iloc[(post_window["time_minutes"] - target_t).abs().argsort()[:1]]
            if not closest.empty:
                d_val = closest["glucose_dexcom"].values[0]
                l_val = closest["glucose_libre"].values[0]
                dex_vals.append(float(d_val) if pd.notna(d_val) else None)
                libre_vals.append(float(l_val) if pd.notna(l_val) else None)
            else:
                dex_vals.append(None)
                libre_vals.append(None)

        pre_dex = dex_vals[0] if dex_vals else None
        pre_libre = libre_vals[0] if libre_vals else None

        meals.append(
            CGMacrosMealEvent(
                meal_id=f"{subject_id}_meal_{m_idx + 1:03d}",
                subject_id=subject_id,
                timestamp=timestamp,
                time_minutes=meal_t,
                meal_type=meal_type,
                carbs_g=float(m_row["carbs_g"]),
                protein_g=float(m_row["protein_g"]),
                fat_g=float(m_row["fat_g"]),
                fiber_g=float(m_row["fiber_g"]),
                calories_kcal=float(m_row["calories_kcal"]),
                proportion_eaten=float(m_row["proportion_eaten"]),
                pre_meal_glucose_dexcom=pre_dex,
                pre_meal_glucose_libre=pre_libre,
                post_meal_glucose_dexcom_120=tuple(dex_vals),
                post_meal_glucose_libre_120=tuple(libre_vals),
            )
        )

    return meals


def import_cgmacros_dataset(
    data_dir: Path | str,
    output_dir: Path | str,
    *,
    bio_filename: str = "bio.csv",
) -> CGMacrosImportResult:
    """
    Ingest the complete CGMacros clinical dataset and export standardized tables.
    """
    in_dir = Path(data_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bio_path = in_dir / bio_filename
    if not bio_path.is_file():
        # Search for bio in subdirectory
        bio_matches = list(in_dir.glob("**/bio.csv")) + list(in_dir.glob("**/*bio*.csv"))
        bio_path = bio_matches[0] if bio_matches else bio_path

    bio_dict: dict[str, CGMacrosSubjectBio] = {}
    if bio_path.is_file():
        bio_dict = parse_cgmacros_bio(bio_path)

    # Find subject files
    raw_files = (
        list(in_dir.glob("CGMacros-*.csv"))
        + list(in_dir.glob("CGMacros_*.csv"))
        + list(in_dir.glob("**/CGMacros-*.csv"))
        + list(in_dir.glob("**/CGMacros_*.csv"))
    )
    subject_files = sorted(list({p.resolve() for p in raw_files if p.is_file()}))

    all_timeseries: list[pd.DataFrame] = []
    all_meals: list[CGMacrosMealEvent] = []

    for s_file in subject_files:
        ts_df = parse_cgmacros_subject_timeseries(s_file)
        if not ts_df.empty:
            all_timeseries.append(ts_df)
            meals = extract_cgmacros_meal_episodes(ts_df)
            all_meals.extend(meals)

    if not all_timeseries:
        raise ValueError(f"no valid CGMacros subject files found in {in_dir}")

    combined_ts = pd.concat(all_timeseries, ignore_index=True)

    # Export standardized time-series
    ts_csv = out_dir / "cgmacros_timeseries.parquet"
    combined_ts.to_parquet(ts_csv, index=False)

    # Export standardized meals
    meal_records = [m.to_dict() for m in all_meals]
    meals_df = pd.DataFrame(meal_records)
    if not meals_df.empty and "post_meal_glucose_dexcom_120" in meals_df.columns:
        # Expand postprandial steps into tabular columns
        for step_i in range(25):
            t_min = step_i * 5
            meals_df[f"dexcom_t{t_min}"] = meals_df["post_meal_glucose_dexcom_120"].apply(
                lambda v: v[step_i] if isinstance(v, (list, tuple)) and len(v) > step_i else np.nan
            )
            meals_df[f"libre_t{t_min}"] = meals_df["post_meal_glucose_libre_120"].apply(
                lambda v: v[step_i] if isinstance(v, (list, tuple)) and len(v) > step_i else np.nan
            )

    meals_csv = out_dir / "cgmacros_meals.csv"
    meals_df.to_csv(meals_csv, index=False)

    # Export subjects table
    if bio_dict:
        subjects_df = pd.DataFrame([s.to_dict() for s in bio_dict.values()])
    else:
        unique_subjs = combined_ts["subject_id"].unique()
        subjects_df = pd.DataFrame([{"subject_id": s, "diabetes_status": "unknown"} for s in unique_subjs])
    subjects_csv = out_dir / "cgmacros_subjects.csv"
    subjects_df.to_csv(subjects_csv, index=False)

    status_counts: dict[str, int] = (
        {str(k): int(v) for k, v in subjects_df["diabetes_status"].value_counts().items()}
        if "diabetes_status" in subjects_df.columns
        else {}
    )

    manifest = {
        "dataset_name": "CGMacros",
        "source_directory": str(in_dir),
        "source_files": [f.name for f in subject_files],
        "subject_count": int(combined_ts["subject_id"].nunique()),
        "total_meals_extracted": len(all_meals),
        "time_series_rows": len(combined_ts),
        "status_distribution": status_counts,
        "dexcom_valid_readings": int(combined_ts["glucose_dexcom"].notna().sum()),
        "libre_valid_readings": int(combined_ts["glucose_libre"].notna().sum()),
        "files": {
            "timeseries": str(ts_csv),
            "meals": str(meals_csv),
            "subjects": str(subjects_csv),
        },
    }

    manifest_path = out_dir / "cgmacros_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return CGMacrosImportResult(
        output_dir=out_dir,
        subject_count=int(combined_ts["subject_id"].nunique()),
        meal_count=len(all_meals),
        time_series_rows=len(combined_ts),
        status_distribution=status_counts,
        dexcom_measurements=int(combined_ts["glucose_dexcom"].notna().sum()),
        libre_measurements=int(combined_ts["glucose_libre"].notna().sum()),
        manifest_path=manifest_path,
    )


__all__ = [
    "CGMacrosSubjectBio",
    "CGMacrosMealEvent",
    "CGMacrosImportResult",
    "parse_cgmacros_bio",
    "parse_cgmacros_subject_timeseries",
    "extract_cgmacros_meal_episodes",
    "import_cgmacros_dataset",
]
