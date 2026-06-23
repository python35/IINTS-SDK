#!/usr/bin/env python3
"""Compatibility exports for clinical metrics.

The implementation lives in :mod:`iints.core.clinical_metrics` so lower
architecture layers can use glucose metrics without importing reporting or
analysis code. Existing imports from ``iints.analysis.clinical_metrics`` remain
supported.
"""

from iints.core.clinical_metrics import ClinicalMetricsCalculator, ClinicalMetricsResult

__all__ = ["ClinicalMetricsCalculator", "ClinicalMetricsResult"]


def demo_clinical_metrics() -> None:
    from iints.core.clinical_metrics import demo_clinical_metrics as _demo

    _demo()


if __name__ == "__main__":
    demo_clinical_metrics()
