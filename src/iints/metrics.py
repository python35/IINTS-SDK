from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd

from iints.analysis.clinical_metrics import ClinicalMetricsCalculator


_calculator = ClinicalMetricsCalculator()


def calculate_gmi(glucose: pd.Series) -> float:
    return _calculator.calculate_gmi(glucose)


def calculate_cv(glucose: pd.Series) -> float:
    return _calculator.calculate_cv(glucose)


def calculate_lbgi(glucose: pd.Series) -> float:
    return _calculator.calculate_lbgi(glucose)


def calculate_hbgi(glucose: pd.Series) -> float:
    return _calculator.calculate_hbgi(glucose)


def calculate_tir(glucose: pd.Series, low: float = 70, high: float = 180) -> float:
    return _calculator.calculate_tir(glucose, low, high)


def calculate_full_metrics(glucose: pd.Series, duration_hours: Optional[float] = None):
    return _calculator.calculate(glucose=glucose, duration_hours=duration_hours)
