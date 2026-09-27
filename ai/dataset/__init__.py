"""Dataset assembly and chronological time-series splitting package.

Provides deterministic joins between point-in-time features and forward-looking labels,
leakage-safe temporal splits with forward purge and embargo handling, and comprehensive
integrity verification.
"""

from __future__ import annotations

from ai.dataset.assembly import (
    AssembledDataset,
    DataLeakageError,
    assemble_dataset,
    infer_horizon_from_target,
    is_label_or_future_column,
)
from ai.dataset.splits import (
    DatasetSplits,
    SplitConfig,
    SplitPartition,
    split_dataset,
)
from ai.dataset.validation import (
    ValidationReport,
    validate_dataset_splits,
)

__all__ = [
    "AssembledDataset",
    "DataLeakageError",
    "DatasetSplits",
    "SplitConfig",
    "SplitPartition",
    "ValidationReport",
    "assemble_dataset",
    "infer_horizon_from_target",
    "is_label_or_future_column",
    "split_dataset",
    "validate_dataset_splits",
]
