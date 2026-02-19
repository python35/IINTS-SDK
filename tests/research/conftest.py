"""
Shared fixtures and configuration for research pipeline tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Make the research scripts importable during tests
_RESEARCH_DIR = Path(__file__).parent.parent.parent / "research"
if str(_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DIR))


@pytest.fixture
def simple_cgm_df() -> pd.DataFrame:
    """A minimal single-subject CGM dataframe for quick tests."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "subject_id": "A",
        "glucose_actual_mgdl": (120.0 + rng.normal(0, 15, n)).clip(40, 400).astype(np.float32),
        "insulin_units": rng.uniform(0, 0.5, n).astype(np.float32),
        "carb_grams": rng.uniform(0, 5, n).astype(np.float32),
        "segment": 0,
    })


@pytest.fixture
def multi_subject_df() -> pd.DataFrame:
    """A multi-subject CGM dataframe with 5 subjects, 60 rows each."""
    rng = np.random.default_rng(7)
    frames = []
    for sid in "ABCDE":
        n = 60
        g = (120.0 + rng.normal(0, 15, n).cumsum() / 5).clip(40, 400)
        frames.append(pd.DataFrame({
            "subject_id": sid,
            "glucose_actual_mgdl": g.astype(np.float32),
            "insulin_units": rng.uniform(0, 0.5, n).astype(np.float32),
            "carb_grams": rng.uniform(0, 5, n).astype(np.float32),
            "segment": 0,
        }))
    return pd.concat(frames, ignore_index=True)
