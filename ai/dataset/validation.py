"""Comprehensive validation engine for dataset assembly and temporal splits.

Verifies strict temporal ordering, disjoint timestamp sets, feature-target separation,
absence of NaNs, and mathematical correctness of purge/embargo boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai.dataset.assembly import DataLeakageError
from ai.dataset.splits import DatasetSplits


@dataclass(frozen=True)
class ValidationReport:
    """Detailed audit report of dataset splitting integrity.

    Attributes
    ----------
    is_valid : bool
        True if all leakage and integrity checks passed.
    checks : dict[str, bool]
        Individual check statuses.
    details : dict[str, Any]
        Quantitative diagnostic measurements.
    errors : list[str]
        List of failure descriptions if any.
    """

    is_valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


def validate_dataset_splits(
    splits: DatasetSplits,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Perform comprehensive integrity and leakage audit on temporal dataset splits.

    Parameters
    ----------
    splits : DatasetSplits
        The partitioned dataset object to validate.
    raise_on_error : bool, default True
        If True, raises DataLeakageError or ValueError upon any validation violation.

    Returns
    -------
    ValidationReport
        Audit report containing check outcomes and error diagnostics.

    Raises
    ------
    DataLeakageError
        If temporal leakage, label leakage into X, or overlapping timestamps occur.
    ValueError
        If missing data, NaNs, or empty partitions are detected.
    """
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    # 1. Non-empty partitions
    checks["partitions_non_empty"] = (
        len(splits.train) > 0 and len(splits.val) > 0 and len(splits.test) > 0
    )
    if not checks["partitions_non_empty"]:
        errors.append(
            f"Empty partition detected: train={len(splits.train)}, "
            f"val={len(splits.val)}, test={len(splits.test)}"
        )

    # 2. Target not present in feature matrix
    in_train_X = splits.target_name in splits.train_X.columns
    in_val_X = splits.target_name in splits.val_X.columns
    in_test_X = splits.target_name in splits.test_X.columns
    checks["target_not_in_features"] = not (in_train_X or in_val_X or in_test_X)
    if not checks["target_not_in_features"]:
        errors.append(
            f"CRITICAL LEAKAGE: Target '{splits.target_name}' is present inside feature matrix X!"
        )

    # 3. Label/future columns not in feature matrix
    future_in_X = [
        c for c in splits.feature_names if c.startswith("future_") or c.startswith("direction_")
    ]
    checks["no_label_columns_in_features"] = len(future_in_X) == 0
    if not checks["no_label_columns_in_features"]:
        errors.append(
            f"CRITICAL LEAKAGE: Forward label columns found in feature matrix X: {future_in_X}"
        )

    # 4. Target timestamps match feature timestamps
    train_align = len(splits.train_X) == len(splits.train_y) and len(splits.train_X) == len(
        splits.train_timestamps
    )
    val_align = len(splits.val_X) == len(splits.val_y) and len(splits.val_X) == len(
        splits.val_timestamps
    )
    test_align = len(splits.test_X) == len(splits.test_y) and len(splits.test_X) == len(
        splits.test_timestamps
    )
    checks["target_feature_alignment"] = train_align and val_align and test_align
    if not checks["target_feature_alignment"]:
        errors.append("Row count mismatch between features, target, and timestamps within splits.")

    # 5. Zero NaNs in features
    train_x_nans = int(splits.train_X.isna().sum().sum())
    val_x_nans = int(splits.val_X.isna().sum().sum())
    test_x_nans = int(splits.test_X.isna().sum().sum())
    total_x_nans = train_x_nans + val_x_nans + test_x_nans
    checks["no_nans_in_features"] = total_x_nans == 0
    details["feature_nans"] = {
        "train": train_x_nans,
        "val": val_x_nans,
        "test": test_x_nans,
    }
    if not checks["no_nans_in_features"]:
        errors.append(f"NaN values detected in feature matrices: total={total_x_nans}")

    # 6. Zero NaNs in targets
    train_y_nans = int(splits.train_y.isna().sum())
    val_y_nans = int(splits.val_y.isna().sum())
    test_y_nans = int(splits.test_y.isna().sum())
    total_y_nans = train_y_nans + val_y_nans + test_y_nans
    checks["no_nans_in_targets"] = total_y_nans == 0
    details["target_nans"] = {
        "train": train_y_nans,
        "val": val_y_nans,
        "test": test_y_nans,
    }
    if not checks["no_nans_in_targets"]:
        errors.append(f"NaN values detected in target vectors: total={total_y_nans}")

    # 7. Monotonically increasing timestamps within each split
    train_mono = bool(splits.train_timestamps.is_monotonic_increasing)
    val_mono = bool(splits.val_timestamps.is_monotonic_increasing)
    test_mono = bool(splits.test_timestamps.is_monotonic_increasing)
    checks["timestamps_monotonic"] = train_mono and val_mono and test_mono
    if not checks["timestamps_monotonic"]:
        errors.append(
            f"Timestamps are not monotonically increasing: train={train_mono}, "
            f"val={val_mono}, test={test_mono}"
        )

    # 8. Strict temporal ordering across splits
    train_end_ts = splits.train.end_timestamp
    val_start_ts = splits.val.start_timestamp
    val_end_ts = splits.val.end_timestamp
    test_start_ts = splits.test.start_timestamp

    order_tv = train_end_ts < val_start_ts
    order_vt = val_end_ts < test_start_ts
    checks["temporal_ordering"] = order_tv and order_vt
    details["temporal_boundaries"] = {
        "train_end": str(train_end_ts),
        "val_start": str(val_start_ts),
        "val_end": str(val_end_ts),
        "test_start": str(test_start_ts),
    }
    if not checks["temporal_ordering"]:
        errors.append(
            f"Temporal inversion across splits: train_end={train_end_ts} >= "
            f"val_start={val_start_ts} ({order_tv}) or val_end={val_end_ts} >= "
            f"test_start={test_start_ts} ({order_vt})"
        )

    # 9. No timestamp overlap (pairwise disjoint)
    set_train = set(splits.train_timestamps)
    set_val = set(splits.val_timestamps)
    set_test = set(splits.test_timestamps)

    overlap_tv = len(set_train.intersection(set_val))
    overlap_vt = len(set_val.intersection(set_test))
    overlap_tt = len(set_train.intersection(set_test))
    total_overlap = overlap_tv + overlap_vt + overlap_tt
    checks["no_timestamp_overlap"] = total_overlap == 0
    details["timestamp_overlaps"] = {
        "train_val": overlap_tv,
        "val_test": overlap_vt,
        "train_test": overlap_tt,
    }
    if not checks["no_timestamp_overlap"]:
        errors.append(
            f"CRITICAL LEAKAGE: Overlapping timestamps between splits: "
            f"train/val={overlap_tv}, val/test={overlap_vt}, train/test={overlap_tt}"
        )

    # 10. Purge and embargo boundary index gap verification
    # Gap between train end and val start in usable indices:
    gap_tv_indices = splits.val.start_index - splits.train.end_index
    expected_min_tv = splits.purge_bars + splits.embargo_bars
    checks["train_val_gap_sufficient"] = gap_tv_indices >= expected_min_tv
    if not checks["train_val_gap_sufficient"]:
        errors.append(
            f"Train-Val index gap ({gap_tv_indices} bars) is less than expected "
            f"purge + embargo ({expected_min_tv} bars)."
        )

    # Gap between val end and test start in usable indices:
    gap_vt_indices = splits.test.start_index - splits.val.end_index
    expected_min_vt = splits.purge_bars + splits.embargo_bars
    checks["val_test_gap_sufficient"] = gap_vt_indices >= expected_min_vt
    if not checks["val_test_gap_sufficient"]:
        errors.append(
            f"Val-Test index gap ({gap_vt_indices} bars) is less than expected "
            f"purge + embargo ({expected_min_vt} bars)."
        )

    is_valid = len(errors) == 0

    report = ValidationReport(
        is_valid=is_valid,
        checks=checks,
        details=details,
        errors=errors,
    )

    if not is_valid and raise_on_error:
        error_msg = "; ".join(errors)
        if any("CRITICAL LEAKAGE" in e for e in errors):
            raise DataLeakageError(f"Dataset split validation failed: {error_msg}")
        raise ValueError(f"Dataset split validation failed: {error_msg}")

    return report
