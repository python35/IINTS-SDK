"""Step 1 — OhioT1DM & Research CGM Dataset Adapter + Explicit Quality Control.

Supports:
1. Official OhioT1DM 2018 & 2020 XML files (``*-ws-training.xml`` / ``*-ws-testing.xml``)
   parsed safely with ``defusedxml.ElementTree`` — preserving the official train/test
   partition and NEVER substituting synthetic data if files are absent.
2. Explicitly labeled ``iints_virtual`` cohorts (Dataset A — software & controlled
   fault verification only).
3. Explicitly labeled ``carelink`` local exports for personal retrospective testing.

Canonical DataFrame columns emitted:
- ``subject_id`` (str)
- ``timestamp`` (pd.Timestamp, UTC-naive or consistent local clock)
- ``glucose_mgdl`` (float)
- ``source_split`` ('train' | 'test')
- ``is_valid`` (bool)
- ``validity_flags`` (str: 'ok' or '+'-joined flags such as 'duplicate_timestamp',
  'out_of_bounds', 'missing_value', 'gap_after_prev')
- ``dt_prev_min`` (float)
- ``source_provenance`` (str)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple, Union

from defusedxml import ElementTree as DET
import numpy as np
import pandas as pd

from iints.forecasting.schemas import BenchmarkTier, DataQualityReport, SubjectQualitySummary
from iints.sdk2.contracts import PatientSpec, ScenarioSpec, SensorFaultSpec
from iints.sdk2.reproducibility import derive_seed, sha256_file
from iints.sdk2.sensors import apply_sensor_fault_model, simulate_patient_trajectory

__all__ = [
    "OHIOT1DM_2018_IDS",
    "OHIOT1DM_2020_IDS",
    "OHIOT1DM_ALL_IDS",
    "MIN_VALID_GLUCOSE_MGDL",
    "MAX_VALID_GLUCOSE_MGDL",
    "GAP_THRESHOLD_MINUTES",
    "load_ohiot1dm_dataset",
    "load_virtual_benchmark_dataset",
    "load_carelink_benchmark_dataset",
    "audit_canonical_dataset",
]

OHIOT1DM_2018_IDS = ("559", "563", "570", "575", "588", "591")
OHIOT1DM_2020_IDS = ("540", "544", "552", "567", "584", "596")
OHIOT1DM_ALL_IDS = OHIOT1DM_2018_IDS + OHIOT1DM_2020_IDS

MIN_VALID_GLUCOSE_MGDL = 20.0
MAX_VALID_GLUCOSE_MGDL = 600.0
GAP_THRESHOLD_MINUTES = 6.5


def _infer_split_from_filename(path: Path) -> Optional[str]:
    low = path.name.lower()
    if "train" in low:
        return "train"
    if "test" in low:
        return "test"
    parent_low = path.parent.name.lower()
    if "train" in parent_low:
        return "train"
    if "test" in parent_low:
        return "test"
    return None


def _parse_single_ohio_xml(xml_path: Path, split_override: Optional[str] = None) -> pd.DataFrame:
    split = split_override or _infer_split_from_filename(xml_path)
    if split not in {"train", "test"}:
        raise ValueError(
            f"Cannot determine official 'train'/'test' split for OhioT1DM XML '{xml_path.name}'. "
            "Keep official filenames (*-ws-training.xml / *-ws-testing.xml) or train/test folders."
        )

    tree = DET.parse(str(xml_path))
    root = tree.getroot()
    if root is None:
        raise ValueError(f"Empty XML root in {xml_path}")

    subject_id = str(root.attrib.get("id") or "").strip()
    if not subject_id:
        match = re.match(r"^(\d{3})", xml_path.name)
        subject_id = match.group(1) if match else xml_path.stem

    glucose_node = root.find("glucose_level")
    if glucose_node is None:
        raise ValueError(f"OhioT1DM XML '{xml_path.name}' has no <glucose_level> section.")

    file_hash = sha256_file(xml_path)[:16]
    provenance = f"ohiot1dm_xml:{xml_path.name}:sha256={file_hash}"

    timestamps: List[Optional[str]] = []
    values: List[Optional[str]] = []
    for ev in glucose_node.findall("event"):
        timestamps.append(ev.attrib.get("ts"))
        values.append(ev.attrib.get("value"))

    if not timestamps:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "timestamp",
                "glucose_mgdl",
                "source_split",
                "is_valid",
                "validity_flags",
                "dt_prev_min",
                "source_provenance",
            ]
        )

    raw_df = pd.DataFrame(
        {
            "subject_id": subject_id,
            "raw_ts": timestamps,
            "raw_val": values,
            "source_split": split,
            "source_provenance": provenance,
        }
    )
    return _finalize_canonical_stream(raw_df, dayfirst_ohio=True)


def _finalize_canonical_stream(raw_df: pd.DataFrame, *, dayfirst_ohio: bool = False) -> pd.DataFrame:
    """Validate timestamps, sort stably, flag duplicates/outliers/gaps without interpolating."""
    out_frames: List[pd.DataFrame] = []
    for (_, _), grp in raw_df.groupby(["subject_id", "source_split"], sort=False):
        g = grp.copy().reset_index(drop=True)
        if dayfirst_ohio:
            ts = pd.to_datetime(g["raw_ts"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
            if ts.isna().any():
                fallback = pd.to_datetime(g["raw_ts"], dayfirst=True, errors="coerce")
                ts = ts.fillna(fallback)
        else:
            ts = pd.to_datetime(g["raw_ts"], errors="coerce")

        val = pd.to_numeric(g["raw_val"], errors="coerce").to_numpy(dtype=float)
        g["timestamp"] = ts
        g["glucose_mgdl"] = val
        g["_orig_order"] = np.arange(len(g))
        # Sort by timestamp (NaT last) preserving file order for ties so duplicates are deterministic
        g = g.sort_values(["timestamp", "_orig_order"], na_position="last").reset_index(drop=True)

        ts_series = g["timestamp"]
        gluc = g["glucose_mgdl"].to_numpy(dtype=float)
        dup_mask = ts_series.notna() & ts_series.duplicated(keep="first")
        missing_ts = ts_series.isna().to_numpy(dtype=bool)
        missing_val = ~np.isfinite(gluc)
        oob_val = np.isfinite(gluc) & ((gluc < MIN_VALID_GLUCOSE_MGDL) | (gluc > MAX_VALID_GLUCOSE_MGDL))

        is_valid = ~(dup_mask.to_numpy(dtype=bool) | missing_ts | missing_val | oob_val)

        # Compute dt_prev_min strictly between consecutive VALID observations (no interpolation!)
        dt_prev = np.full(len(g), np.nan, dtype=float)
        valid_indices = np.flatnonzero(is_valid)
        if valid_indices.size > 1:
            valid_times = ts_series.iloc[valid_indices].to_numpy(dtype="datetime64[ns]")
            diffs_min = (valid_times[1:] - valid_times[:-1]) / np.timedelta64(1, "m")
            dt_prev[valid_indices[1:]] = diffs_min

        gap_mask = np.isfinite(dt_prev) & (dt_prev > GAP_THRESHOLD_MINUTES)

        flags: List[str] = []
        for idx in range(len(g)):
            row_flags: List[str] = []
            if missing_ts[idx]:
                row_flags.append("invalid_timestamp")
            if dup_mask.iloc[idx]:
                row_flags.append("duplicate_timestamp")
            if missing_val[idx]:
                row_flags.append("missing_value")
            if oob_val[idx]:
                row_flags.append("out_of_bounds")
            if gap_mask[idx]:
                row_flags.append("gap_after_prev")
            flags.append("ok" if not row_flags else "+".join(row_flags))

        out_frames.append(
            pd.DataFrame(
                {
                    "subject_id": g["subject_id"].astype(str),
                    "timestamp": g["timestamp"],
                    "glucose_mgdl": g["glucose_mgdl"].astype(float),
                    "source_split": g["source_split"].astype(str),
                    "is_valid": is_valid,
                    "validity_flags": flags,
                    "dt_prev_min": dt_prev,
                    "source_provenance": g["source_provenance"].astype(str),
                }
            )
        )

    if not out_frames:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "timestamp",
                "glucose_mgdl",
                "source_split",
                "is_valid",
                "validity_flags",
                "dt_prev_min",
                "source_provenance",
            ]
        )
    return pd.concat(out_frames, ignore_index=True)


def load_ohiot1dm_dataset(root_dir: Union[str, Path]) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Load official OhioT1DM XML files from ``root_dir`` without synthetic substitution."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"OhioT1DM directory '{root}' does not exist. Request official research access at "
            "https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html and "
            "place the *-ws-training.xml and *-ws-testing.xml files inside this directory. "
            "Synthetic data is never silently substituted for OhioT1DM."
        )

    xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No OhioT1DM .xml files found under '{root}'. Expected official files such as "
            "'559-ws-training.xml' and '559-ws-testing.xml'."
        )

    frames = [_parse_single_ohio_xml(p) for p in xml_files]
    canonical = pd.concat(frames, ignore_index=True)
    report = audit_canonical_dataset(
        canonical,
        dataset_id="ohiot1dm",
        is_synthetic=False,
        benchmark_tier="primary_ohiot1dm",
        source_provenance=f"local_ohiot1dm_dir:{root.resolve()} ({len(xml_files)} XML files)",
    )
    return canonical, report


def load_virtual_benchmark_dataset(
    *,
    n_subjects: int = 12,
    train_days: float = 10.0,
    test_days: float = 4.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Generate Dataset A (`iints_virtual`) explicitly labeled as simulated verification data."""
    from iints.research.scenarios import generate_virtual_cohort

    cohort: List[PatientSpec] = generate_virtual_cohort(n_profiles=n_subjects, seed=seed, unseen_fraction=0.25)
    sensor = SensorFaultSpec(
        model="cgm_v1",
        condition_id="virtual_clean_with_dropouts",
        noise_sd_mgdl=6.0,
        noise_ar1_phi=0.75,
        dropout_minutes=15,
        dropout_at_hour=9.0,
        recurring_dropout_every_hours=18.0,
        display_mode="nan_blank",
    )
    total_hours = (train_days + test_days) * 24.0
    scenario = ScenarioSpec(
        id="virtual_multi_day",
        duration_hours=total_hours,
        warmup_hours=2.0,
        step_minutes=5.0,
        meal_schedule="standard_day",
    )

    raw_chunks: List[pd.DataFrame] = []
    for pat in cohort:
        p_seed = derive_seed(seed, "virtual_ohio_like", pat.profile)
        s_seed = derive_seed(seed, "virtual_sensor", pat.profile)
        traj = simulate_patient_trajectory(pat, scenario, seed=p_seed)
        trace = apply_sensor_fault_model(traj, sensor, seed=s_seed)
        eval_trace = trace[trace["in_eval_window"]].reset_index(drop=True)
        cut_min = float(eval_trace["t_min"].iloc[0]) + train_days * 1440.0

        for split_name, mask in (
            ("train", eval_trace["t_min"] < cut_min),
            ("test", eval_trace["t_min"] >= cut_min),
        ):
            sub = eval_trace.loc[mask]
            # Drop missing sensor points just like real OhioT1DM XML omits dropout events
            observed = sub[sub["cgm_displayed_mgdl"].notna()]
            raw_chunks.append(
                pd.DataFrame(
                    {
                        "subject_id": pat.profile,
                        "raw_ts": observed["timestamp_dt"].astype(str).to_numpy(),
                        "raw_val": observed["cgm_displayed_mgdl"].round(1).to_numpy(),
                        "source_split": split_name,
                        "source_provenance": f"iints_virtual:seed={seed}:profile={pat.profile}",
                    }
                )
            )

    canonical = _finalize_canonical_stream(pd.concat(raw_chunks, ignore_index=True), dayfirst_ohio=False)
    report = audit_canonical_dataset(
        canonical,
        dataset_id="iints_virtual",
        is_synthetic=True,
        benchmark_tier="synthetic_software_validation",
        source_provenance=f"SYNTHETIC IINTS Virtual Patient Cohort (N={n_subjects}, seed={seed}, NOT human clinical data)",
    )
    return canonical, report


def load_carelink_benchmark_dataset(
    csv_path: Union[str, Path],
    *,
    test_fraction: float = 0.20,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Load a local CareLink CSV into the canonical schema with a chronological train/test cut."""
    from iints.analysis.carelink_sections import build_timeline, read_sections

    p = Path(csv_path)
    tl = build_timeline(read_sections(p), fill_gaps=False)
    obs = tl[tl["glucose"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    cut_idx = int(len(obs) * (1.0 - test_fraction))
    file_hash = sha256_file(p)[:16]
    prov = f"carelink_csv:{p.name}:sha256={file_hash}"

    raw_df = pd.DataFrame(
        {
            "subject_id": "carelink_subject_01",
            "raw_ts": obs["timestamp_dt"].astype(str).to_numpy(),
            "raw_val": obs["glucose"].to_numpy(dtype=float),
            "source_split": np.where(np.arange(len(obs)) < cut_idx, "train", "test"),
            "source_provenance": prov,
        }
    )
    canonical = _finalize_canonical_stream(raw_df, dayfirst_ohio=False)
    report = audit_canonical_dataset(
        canonical,
        dataset_id="carelink",
        is_synthetic=False,
        benchmark_tier="personal_exploratory_only",
        source_provenance=prov,
    )
    return canonical, report


def audit_canonical_dataset(
    canonical: pd.DataFrame,
    *,
    dataset_id: str,
    source_provenance: str,
    is_synthetic: bool = False,
    benchmark_tier: BenchmarkTier = "primary_ohiot1dm",
) -> DataQualityReport:
    """Summarize per-subject and cohort-wide data quality without modifying measurements."""
    summaries: List[SubjectQualitySummary] = []
    for (subj, split), grp in canonical.groupby(["subject_id", "source_split"], sort=True):
        flags = grp["validity_flags"].astype(str)
        dups = int(flags.str.contains("duplicate_timestamp").sum())
        oob = int(flags.str.contains("out_of_bounds").sum())
        miss = int(flags.str.contains("missing_value").sum())
        dt = grp.loc[grp["is_valid"], "dt_prev_min"].dropna().to_numpy(dtype=float)
        gaps = int(np.sum(dt > GAP_THRESHOLD_MINUTES)) if dt.size else 0
        max_gap = float(np.max(dt)) if dt.size else 0.0
        med_dt = float(np.median(dt)) if dt.size else 0.0
        valid_ts = grp.loc[grp["is_valid"], "timestamp"]
        summaries.append(
            SubjectQualitySummary(
                subject_id=str(subj),
                source_split=split,  # type: ignore[arg-type]
                raw_rows=int(len(grp)),
                valid_rows=int(grp["is_valid"].sum()),
                duplicate_timestamps=dups,
                out_of_bounds_rows=oob,
                missing_value_rows=miss,
                gaps_over_threshold=gaps,
                max_gap_minutes=round(max_gap, 2),
                median_interval_minutes=round(med_dt, 2),
                start_timestamp=str(valid_ts.iloc[0]) if not valid_ts.empty else None,
                end_timestamp=str(valid_ts.iloc[-1]) if not valid_ts.empty else None,
            )
        )

    return DataQualityReport(
        dataset_id=dataset_id,
        is_synthetic=bool(is_synthetic),
        benchmark_tier=benchmark_tier,
        source_provenance=source_provenance,
        n_subjects=int(canonical["subject_id"].nunique()) if not canonical.empty else 0,
        total_raw_rows=int(len(canonical)),
        total_valid_rows=int(canonical["is_valid"].sum()) if not canonical.empty else 0,
        total_duplicates_flagged=sum(s.duplicate_timestamps for s in summaries),
        total_out_of_bounds_flagged=sum(s.out_of_bounds_rows for s in summaries),
        total_missing_values_flagged=sum(s.missing_value_rows for s in summaries),
        total_gaps_over_threshold=sum(s.gaps_over_threshold for s in summaries),
        subjects=summaries,
    )
