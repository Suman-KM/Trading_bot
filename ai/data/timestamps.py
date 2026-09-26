"""Timestamp processing and chronological integrity validation.

Handles conversion of MT5 Unix epoch seconds into timezone-aware UTC timestamps,
verifies strict chronological monotonicity, and checks for duplicate timestamps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TimestampValidationResult:
    """Result of timestamp sequence validation."""

    is_valid: bool
    is_strictly_monotonic: bool
    duplicate_count: int
    row_count: int
    earliest_epoch: int | None
    latest_epoch: int | None
    earliest_utc: str | None
    latest_utc: str | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary representation."""
        return asdict(self)


def validate_timestamps(df: pd.DataFrame, time_col: str = "time") -> TimestampValidationResult:
    """Validate that the timestamp sequence is strictly chronological and duplicate-free.

    Parameters
    ----------
    df : pd.DataFrame
        Market data DataFrame containing the timestamp column.
    time_col : str, default "time"
        Name of the epoch timestamp column.

    Returns
    -------
    TimestampValidationResult
        Detailed report on monotonicity, duplicates, bounds, and errors.
    """
    errors: list[str] = []

    if df.empty:
        return TimestampValidationResult(
            is_valid=False,
            is_strictly_monotonic=False,
            duplicate_count=0,
            row_count=0,
            earliest_epoch=None,
            latest_epoch=None,
            earliest_utc=None,
            latest_utc=None,
            errors=["DataFrame is empty."],
        )

    if time_col not in df.columns:
        return TimestampValidationResult(
            is_valid=False,
            is_strictly_monotonic=False,
            duplicate_count=0,
            row_count=len(df),
            earliest_epoch=None,
            latest_epoch=None,
            earliest_utc=None,
            latest_utc=None,
            errors=[f"Timestamp column '{time_col}' missing from DataFrame."],
        )

    series = df[time_col]

    # Validate that all values are positive integers
    invalid_mask = series.isna() | (series <= 0)
    if invalid_mask.any():
        errors.append(f"Found {invalid_mask.sum()} non-positive or null epoch timestamps.")

    # Check for duplicate timestamps
    duplicate_count = int(series.duplicated().sum())
    if duplicate_count > 0:
        errors.append(f"Found {duplicate_count} duplicate timestamps.")

    # Check strict chronological ordering (each row > previous row)
    if len(series) > 1:
        diffs = series.diff().iloc[1:]
        is_strictly_monotonic = bool((diffs > 0).all())
        if not is_strictly_monotonic:
            violations = int((diffs <= 0).sum())
            errors.append(
                f"Timestamps are not strictly chronological: {violations} ordering violations."
            )
    else:
        is_strictly_monotonic = True

    earliest_epoch = int(series.min()) if not series.empty else None
    latest_epoch = int(series.max()) if not series.empty else None

    earliest_utc: str | None = None
    latest_utc: str | None = None
    if earliest_epoch is not None and earliest_epoch > 0:
        earliest_utc = datetime.fromtimestamp(earliest_epoch, tz=UTC).isoformat()
    if latest_epoch is not None and latest_epoch > 0:
        latest_utc = datetime.fromtimestamp(latest_epoch, tz=UTC).isoformat()

    is_valid = is_strictly_monotonic and (duplicate_count == 0) and (len(errors) == 0)

    return TimestampValidationResult(
        is_valid=is_valid,
        is_strictly_monotonic=is_strictly_monotonic,
        duplicate_count=duplicate_count,
        row_count=len(df),
        earliest_epoch=earliest_epoch,
        latest_epoch=latest_epoch,
        earliest_utc=earliest_utc,
        latest_utc=latest_utc,
        errors=errors,
    )


def add_utc_timestamp(
    df: pd.DataFrame,
    time_col: str = "time",
    target_col: str = "timestamp",
) -> pd.DataFrame:
    """Convert Unix epoch seconds into a timezone-aware UTC datetime column.

    Preserves the original epoch time column while adding an explicit
    timezone-aware UTC datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Input market data DataFrame.
    time_col : str, default "time"
        Name of the Unix epoch seconds column.
    target_col : str, default "timestamp"
        Name of the new UTC timestamp column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the timezone-aware UTC timestamp column inserted.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")

    df_copy = df.copy()
    # Explicit conversion to timezone-aware UTC datetime
    utc_series = pd.to_datetime(df_copy[time_col], unit="s", utc=True)

    # Insert target_col immediately after time_col if possible
    time_idx = df_copy.columns.get_loc(time_col)
    df_copy.insert(time_idx + 1, target_col, utc_series)

    return df_copy
