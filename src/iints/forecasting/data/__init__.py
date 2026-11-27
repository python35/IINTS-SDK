"""Data adapters, train-only preprocessing, and embargoed windowing for `iints.forecasting`."""

from iints.forecasting.data.ohio_adapter import (
    GAP_THRESHOLD_MINUTES,
    MAX_VALID_GLUCOSE_MGDL,
    MIN_VALID_GLUCOSE_MGDL,
    OHIOT1DM_2018_IDS,
    OHIOT1DM_2020_IDS,
    OHIOT1DM_ALL_IDS,
    audit_canonical_dataset,
    load_carelink_benchmark_dataset,
    load_ohiot1dm_dataset,
    load_virtual_benchmark_dataset,
)
from iints.forecasting.data.preprocessing import TrainOnlyStandardScaler
from iints.forecasting.data.windows import (
    DataLeakageError,
    PartitionedWindowDataset,
    SplitWindows,
    build_partitioned_windows,
)

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
    "TrainOnlyStandardScaler",
    "SplitWindows",
    "PartitionedWindowDataset",
    "build_partitioned_windows",
]
