from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iints.research.cgm_jepa import (
    CGMJEPAEncoder,
    extract_cgm_jepa_embeddings,
    load_cgm_jepa_model,
)


@dataclass(frozen=True)
class SimulationJEPAEmbeddingResult:
    """Exportable latent representation of a 24h simulation run using CGM-JEPA."""

    source_path: str
    duration_minutes: float
    glucose_mean_mgdl: float
    glucose_std_mgdl: float
    tir_70_180_pct: float
    embedding_dim: int
    embedding_vector: tuple[float, ...]  # 96 dimensions
    patch_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["embedding_vector"] = list(self.embedding_vector)
        return data


def prepare_cgm_jepa_window(
    df: pd.DataFrame,
    glucose_col: str | None = None,
    time_col: str | None = None,
    target_steps: int = 288,
    step_minutes: float = 5.0,
) -> np.ndarray:
    """
    Format and resample arbitrary simulation or CGM time-series into exact 288-point grid.
    """
    candidates_glu = ["glucose_actual_mgdl", "cgm_mgdl", "glucose", "sensor_glucose_mgdl", "glucose_dexcom", "bg"]
    candidates_time = ["time_minutes", "time", "timestamp_minutes", "minutes", "t"]

    col_g = glucose_col
    if col_g is None:
        for c in candidates_glu:
            if c in df.columns:
                col_g = c
                break
    if col_g is None or col_g not in df.columns:
        raise ValueError(f"could not find glucose column in dataframe; available: {list(df.columns)}")

    col_t = time_col
    if col_t is None:
        for c in candidates_time:
            if c in df.columns:
                col_t = c
                break

    raw_glucose = pd.to_numeric(df[col_g], errors="coerce").values
    if col_t and col_t in df.columns:
        raw_t = pd.to_numeric(df[col_t], errors="coerce").values
    else:
        raw_t = np.arange(len(raw_glucose)) * step_minutes

    valid = np.isfinite(raw_glucose) & np.isfinite(raw_t)
    if not np.any(valid):
        return np.full(target_steps, 135.0, dtype=np.float32)

    t_valid = raw_t[valid]
    g_valid = raw_glucose[valid]

    # Target 5-minute grid for 24 hours: [0, 5, 10, ..., 1435]
    t_target = np.linspace(0, (target_steps - 1) * step_minutes, target_steps)
    interp_glucose = np.interp(t_target, t_valid, g_valid, left=g_valid[0], right=g_valid[-1])

    return np.asarray(interp_glucose, dtype=np.float32)


def bridge_simulation_to_jepa(
    simulation_input: Path | str | pd.DataFrame,
    output_dir: Path | str | None = None,
    model: CGMJEPAEncoder | None = None,
    device: str = "cpu",
) -> SimulationJEPAEmbeddingResult:
    """
    Bridge an IINTS-AF simulation result into a 96-dimensional CGM-JEPA representation.
    """
    if isinstance(simulation_input, pd.DataFrame):
        df = simulation_input
        src_path = "in_memory_dataframe"
    else:
        p = Path(simulation_input).expanduser().resolve()
        src_path = str(p)
        if p.is_dir():
            # Check standard simulation output locations
            if (p / "results.csv").is_file():
                df = pd.read_csv(p / "results.csv")
            elif (p / "raw" / "steps.csv").is_file():
                df = pd.read_csv(p / "raw" / "steps.csv")
            else:
                csvs = list(p.glob("*.csv"))
                if not csvs:
                    raise FileNotFoundError(f"no results.csv or steps.csv found in simulation directory: {p}")
                df = pd.read_csv(csvs[0])
        elif p.is_file():
            df = pd.read_csv(p)
        else:
            raise FileNotFoundError(f"simulation input path not found: {p}")

    window_288 = prepare_cgm_jepa_window(df)
    emb = extract_cgm_jepa_embeddings(window_288, model=model, device=device)

    # Compute descriptive summary metrics
    g_mean = float(np.mean(window_288))
    g_std = float(np.std(window_288))
    tir_pct = float(np.mean((window_288 >= 70.0) & (window_288 <= 180.0)) * 100.0)

    res = SimulationJEPAEmbeddingResult(
        source_path=src_path,
        duration_minutes=288 * 5.0,
        glucose_mean_mgdl=round(g_mean, 2),
        glucose_std_mgdl=round(g_std, 2),
        tir_70_180_pct=round(tir_pct, 2),
        embedding_dim=len(emb),
        embedding_vector=tuple(float(v) for v in emb),
        patch_count=24,
    )

    if output_dir is not None:
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        # Export JSON representation
        (out_path / "cgm_jepa_embedding.json").write_text(
            json.dumps(res.to_dict(), indent=2), encoding="utf-8"
        )
        # Export Tabular embedding
        emb_df = pd.DataFrame([res.to_dict()])
        emb_df.to_csv(out_path / "cgm_jepa_embedding.csv", index=False)

    return res


__all__ = [
    "SimulationJEPAEmbeddingResult",
    "prepare_cgm_jepa_window",
    "bridge_simulation_to_jepa",
]
