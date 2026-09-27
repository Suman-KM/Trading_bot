"""Chronological time-series splitting with purge and embargo handling.

Implements strict temporal train/validation/test partitioning designed for financial
time series with forward-looking labels, preventing lookahead leakage across boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ai.dataset.assembly import AssembledDataset


@dataclass(frozen=True)
class SplitConfig:
    """Configuration specification for chronological splitting.

    Attributes
    ----------
    train_ratio : float, default 0.70
        Proportion of usable rows allocated to training.
    val_ratio : float, default 0.15
        Proportion of usable rows allocated to validation.
    test_ratio : float, default 0.15
        Proportion of usable rows allocated to testing.
    purge_bars : int | None, default None
        Number of bars to purge before each boundary. If None, defaults to the
        prediction horizon H.
    embargo_bars : int, default 0
        Number of bars to skip after each boundary as an embargo buffer.
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int | None = None
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.5f}")
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split ratios must be strictly positive (> 0).")
        if self.purge_bars is not None and self.purge_bars < 0:
            raise ValueError(f"purge_bars must be >= 0, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")


@dataclass(frozen=True)
class SplitPartition:
    """Individual partition slice containing features, targets, and metadata.

    Attributes
    ----------
    name : str
        Partition name ('train', 'val', or 'test').
    X : pd.DataFrame
        Point-in-time feature matrix.
    y : pd.Series
        Target vector.
    timestamps : pd.Series
        Observation timestamps.
    start_timestamp : pd.Timestamp
        First timestamp in partition.
    end_timestamp : pd.Timestamp
        Last timestamp in partition.
    row_count : int
        Number of observations in partition.
    pct_of_usable : float
        Percentage of total usable observations.
    start_index : int
        Start index in the usable dataset.
    end_index : int
        End index in the usable dataset (exclusive).
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    row_count: int
    pct_of_usable: float
    start_index: int
    end_index: int

    def __len__(self) -> int:
        return self.row_count

    def to_dict(self) -> dict[str, Any]:
        """Convert partition summary to dictionary."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "pct_of_usable": round(self.pct_of_usable, 2),
            "start_timestamp": str(self.start_timestamp),
            "end_timestamp": str(self.end_timestamp),
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


@dataclass(frozen=True)
class DatasetSplits:
    """Container for chronological Train, Validation, and Test partitions.

    Attributes
    ----------
    train : SplitPartition
        Training partition.
    val : SplitPartition
        Validation partition.
    test : SplitPartition
        Test partition.
    target_name : str
        Target column name.
    horizon_bars : int
        Horizon in bars.
    feature_names : list[str]
        Input feature column names.
    total_assembled_rows : int
        Total rows in assembled dataframe (including boundary label NaNs).
    total_usable_rows : int
        Total usable rows (with complete features AND non-NaN target).
    purge_bars : int
        Number of bars purged at boundaries.
    embargo_bars : int
        Number of bars embargoed after boundaries.
    purged_train_bars : int
        Number of bars purged from training partition.
    purged_val_bars : int
        Number of bars purged from validation partition.
    embargoed_val_bars : int
        Number of bars embargoed from start of validation.
    embargoed_test_bars : int
        Number of bars embargoed from start of test.
    """

    train: SplitPartition
    val: SplitPartition
    test: SplitPartition

    target_name: str
    horizon_bars: int
    feature_names: list[str]
    total_assembled_rows: int
    total_usable_rows: int

    purge_bars: int
    embargo_bars: int
    purged_train_bars: int
    purged_val_bars: int
    embargoed_val_bars: int
    embargoed_test_bars: int

    @property
    def train_X(self) -> pd.DataFrame:
        return self.train.X

    @property
    def train_y(self) -> pd.Series:
        return self.train.y

    @property
    def train_timestamps(self) -> pd.Series:
        return self.train.timestamps

    @property
    def val_X(self) -> pd.DataFrame:
        return self.val.X

    @property
    def val_y(self) -> pd.Series:
        return self.val.y

    @property
    def val_timestamps(self) -> pd.Series:
        return self.val.timestamps

    @property
    def test_X(self) -> pd.DataFrame:
        return self.test.X

    @property
    def test_y(self) -> pd.Series:
        return self.test.y

    @property
    def test_timestamps(self) -> pd.Series:
        return self.test.timestamps

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert split details to machine-readable summary."""
        return {
            "target_name": self.target_name,
            "horizon_bars": self.horizon_bars,
            "feature_count": len(self.feature_names),
            "total_assembled_rows": self.total_assembled_rows,
            "total_usable_rows": self.total_usable_rows,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "purged_train_bars": self.purged_train_bars,
            "purged_val_bars": self.purged_val_bars,
            "embargoed_val_bars": self.embargoed_val_bars,
            "embargoed_test_bars": self.embargoed_test_bars,
            "train": self.train.to_dict(),
            "validation": self.val.to_dict(),
            "test": self.test.to_dict(),
        }


def split_dataset(
    assembled: AssembledDataset,
    config: SplitConfig | None = None,
) -> DatasetSplits:
    """Split an assembled dataset into chronological Train, Validation, and Test sets.

    Applies deterministic forward-purge to prevent lookahead leakage from overlapping
    horizon labels, and optional embargo to eliminate short-term autoregressive feature memory.

    Parameters
    ----------
    assembled : AssembledDataset
        The assembled point-in-time dataset container.
    config : SplitConfig | None, optional
        Splitting parameters. Defaults to 70% Train / 15% Val / 15% Test with
        purge_bars=horizon_bars and embargo_bars=0.

    Returns
    -------
    DatasetSplits
        Validated dataset partitions and boundary metadata.

    Raises
    ------
    ValueError
        If usable rows are insufficient for the requested split and purge parameters.
    """
    if config is None:
        config = SplitConfig()

    effective_purge = config.purge_bars if config.purge_bars is not None else assembled.horizon_bars
    effective_embargo = config.embargo_bars

    # Extract usable dataframe (strictly non-NaN target and complete features)
    df_usable = assembled.usable_df.reset_index(drop=True)
    N = len(df_usable)

    if N < 100:
        raise ValueError(f"Insufficient usable observations for splitting: N={N}")

    # Calculate baseline raw boundaries
    raw_train_end = int(N * config.train_ratio)
    raw_val_end = raw_train_end + int(N * config.val_ratio)
    raw_test_end = N

    # Apply Purge & Embargo
    # 1. Train partition:
    # Observations whose forward label reaches into validation (t + H >= raw_train_end)
    # must be purged. Train ends at raw_train_end - effective_purge.
    train_start = 0
    train_end = raw_train_end - effective_purge
    if train_end <= train_start:
        raise ValueError(
            f"Purge of {effective_purge} bars eliminates entire training partition "
            f"(raw_train_end={raw_train_end})."
        )

    # 2. Validation partition:
    # Starts after optional embargo buffer (raw_train_end + effective_embargo).
    # Observations whose forward label reaches into test (t + H >= raw_val_end)
    # must be purged. Val ends at raw_val_end - effective_purge.
    val_start = raw_train_end + effective_embargo
    val_end = raw_val_end - effective_purge
    if val_end <= val_start:
        raise ValueError(
            f"Purge ({effective_purge} bars) and embargo ({effective_embargo} bars) "
            f"eliminate entire validation partition (val_start={val_start}, val_end={val_end})."
        )

    # 3. Test partition:
    # Starts after optional embargo buffer (raw_val_end + effective_embargo).
    # Ends at raw_test_end.
    test_start = raw_val_end + effective_embargo
    test_end = raw_test_end
    if test_end <= test_start:
        raise ValueError(
            f"Embargo ({effective_embargo} bars) eliminates entire test partition "
            f"(test_start={test_start}, test_end={test_end})."
        )

    # Extract slices
    train_slice = df_usable.iloc[train_start:train_end]
    val_slice = df_usable.iloc[val_start:val_end]
    test_slice = df_usable.iloc[test_start:test_end]

    ts_col = assembled.timestamp_col
    features = assembled.feature_names
    target = assembled.target_name

    train_partition = SplitPartition(
        name="train",
        X=train_slice[features].copy().reset_index(drop=True),
        y=train_slice[target].copy().reset_index(drop=True),
        timestamps=train_slice[ts_col].copy().reset_index(drop=True),
        start_timestamp=train_slice[ts_col].iloc[0],
        end_timestamp=train_slice[ts_col].iloc[-1],
        row_count=len(train_slice),
        pct_of_usable=(len(train_slice) / N) * 100.0,
        start_index=train_start,
        end_index=train_end,
    )

    val_partition = SplitPartition(
        name="val",
        X=val_slice[features].copy().reset_index(drop=True),
        y=val_slice[target].copy().reset_index(drop=True),
        timestamps=val_slice[ts_col].copy().reset_index(drop=True),
        start_timestamp=val_slice[ts_col].iloc[0],
        end_timestamp=val_slice[ts_col].iloc[-1],
        row_count=len(val_slice),
        pct_of_usable=(len(val_slice) / N) * 100.0,
        start_index=val_start,
        end_index=val_end,
    )

    test_partition = SplitPartition(
        name="test",
        X=test_slice[features].copy().reset_index(drop=True),
        y=test_slice[target].copy().reset_index(drop=True),
        timestamps=test_slice[ts_col].copy().reset_index(drop=True),
        start_timestamp=test_slice[ts_col].iloc[0],
        end_timestamp=test_slice[ts_col].iloc[-1],
        row_count=len(test_slice),
        pct_of_usable=(len(test_slice) / N) * 100.0,
        start_index=test_start,
        end_index=test_end,
    )

    return DatasetSplits(
        train=train_partition,
        val=val_partition,
        test=test_partition,
        target_name=target,
        horizon_bars=assembled.horizon_bars,
        feature_names=features,
        total_assembled_rows=len(assembled.df),
        total_usable_rows=N,
        purge_bars=effective_purge,
        embargo_bars=effective_embargo,
        purged_train_bars=effective_purge,
        purged_val_bars=effective_purge,
        embargoed_val_bars=effective_embargo,
        embargoed_test_bars=effective_embargo,
    )
