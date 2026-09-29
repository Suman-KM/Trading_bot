"""Swing dataset aggregation, validation, and chronological partitioning engine.

Aggregates canonical EURUSD M15 historical bars into completed H4 (4-hour) and
D1 (Daily) bars, verifying strict point-in-time timestamp monotonicity, valid OHLC
geometries, and chronological train/validation/test holdout preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Standard chronological cutoffs matching M15 partitions
DEFAULT_TRAIN_END = pd.Timestamp("2025-07-14 04:30:00+00:00")
DEFAULT_VAL_START = pd.Timestamp("2025-07-14 05:45:00+00:00")
DEFAULT_VAL_END = pd.Timestamp("2026-02-19 10:45:00+00:00")
DEFAULT_TEST_START = pd.Timestamp("2026-02-19 12:00:00+00:00")


@dataclass(frozen=True)
class SwingPartition:
    """Individual swing partition containing features, targets, and timestamps."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    row_count: int

    def __len__(self) -> int:
        return self.row_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "start_timestamp": str(self.start_timestamp),
            "end_timestamp": str(self.end_timestamp),
        }


@dataclass(frozen=True)
class SwingSplits:
    """Container for chronological Train, Validation, and Test swing partitions."""

    train: SwingPartition
    val: SwingPartition
    test: SwingPartition
    timeframe: str
    horizon_bars: int


def aggregate_m15_to_h4(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed M15 candles into 4-Hour (H4) bars.

    Parameters
    ----------
    df_m15 : pd.DataFrame
        DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread'.

    Returns
    -------
    pd.DataFrame
        Aggregated H4 DataFrame with completed bar timestamps.
    """
    df_idx = df_m15.copy()
    if not isinstance(df_idx.index, pd.DatetimeIndex):
        df_idx["timestamp"] = pd.to_datetime(df_idx["timestamp"], utc=True)
        df_idx = df_idx.set_index("timestamp")

    agg_rules: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "median",
    }
    if "time" in df_idx.columns:
        agg_rules["time"] = "first"

    h4 = df_idx.resample("4h", origin="start_day").agg(agg_rules)
    h4 = h4.dropna(subset=["open"]).reset_index()
    return h4


def aggregate_m15_to_d1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed M15 candles into Daily (D1) bars (00:00 to 24:00 UTC).

    Parameters
    ----------
    df_m15 : pd.DataFrame
        DataFrame with 'timestamp', 'open', 'high', 'low', 'close', 'tick_volume', 'spread'.

    Returns
    -------
    pd.DataFrame
        Aggregated D1 DataFrame with completed bar timestamps.
    """
    df_idx = df_m15.copy()
    if not isinstance(df_idx.index, pd.DatetimeIndex):
        df_idx["timestamp"] = pd.to_datetime(df_idx["timestamp"], utc=True)
        df_idx = df_idx.set_index("timestamp")

    agg_rules: dict[str, Any] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "spread": "median",
    }
    if "time" in df_idx.columns:
        agg_rules["time"] = "first"

    d1 = df_idx.resample("1D", origin="start_day").agg(agg_rules)
    d1 = d1.dropna(subset=["open"]).reset_index()
    return d1


def validate_swing_data(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Validate data integrity, monotonicity, and OHLC consistency of swing dataset."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    is_monotonic = bool(ts.is_monotonic_increasing)
    has_duplicates = bool(ts.duplicated().any())

    # OHLC geometry checks
    high_valid = bool(((df["high"] >= df["open"]) & (df["high"] >= df["close"])).all())
    low_valid = bool(((df["low"] <= df["open"]) & (df["low"] <= df["close"])).all())
    positive_prices = bool(
        ((df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)).all()
    )

    has_nans = bool(df[["open", "high", "low", "close"]].isna().any().any())
    has_infs = bool(np.isinf(df[["open", "high", "low", "close"]].to_numpy()).any())

    # Detect inter-bar gaps
    time_deltas = ts.diff().dropna()
    expected_hours = 4.0 if timeframe == "H4" else 24.0
    gap_count = int((time_deltas > pd.Timedelta(hours=expected_hours * 1.5)).sum())

    report = {
        "timeframe": timeframe,
        "total_bars": len(df),
        "start_timestamp": str(ts.iloc[0]) if len(ts) > 0 else "None",
        "end_timestamp": str(ts.iloc[-1]) if len(ts) > 0 else "None",
        "is_monotonic_increasing": is_monotonic,
        "has_duplicates": has_duplicates,
        "valid_high_geometry": high_valid,
        "valid_low_geometry": low_valid,
        "positive_prices": positive_prices,
        "has_nans": has_nans,
        "has_infs": has_infs,
        "weekend_and_session_gaps": gap_count,
        "all_valid": bool(
            is_monotonic
            and not has_duplicates
            and high_valid
            and low_valid
            and positive_prices
            and not has_nans
            and not has_infs
        ),
    }
    return report


def split_swing_data(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    timeframe: str,
    horizon_bars: int,
    train_end_ts: pd.Timestamp = DEFAULT_TRAIN_END,
    val_start_ts: pd.Timestamp = DEFAULT_VAL_START,
    val_end_ts: pd.Timestamp = DEFAULT_VAL_END,
    test_start_ts: pd.Timestamp = DEFAULT_TEST_START,
) -> SwingSplits:
    """Split swing features and targets chronologically preserving purge and test holdout."""
    ts = pd.to_datetime(timestamps, utc=True)

    # Valid rows: non-NaN target and non-NaN features
    valid_mask = y.notna() & ~X.isna().any(axis=1)

    # Train partition: up to train_end_ts minus purge window
    train_raw_mask = (ts <= train_end_ts) & valid_mask
    train_indices = np.where(train_raw_mask)[0]
    # Purge last horizon_bars from training partition to prevent lookahead
    if len(train_indices) > horizon_bars:
        train_indices = train_indices[:-horizon_bars]

    # Val partition: between val_start_ts and val_end_ts minus purge window
    val_raw_mask = (ts >= val_start_ts) & (ts <= val_end_ts) & valid_mask
    val_indices = np.where(val_raw_mask)[0]
    if len(val_indices) > horizon_bars:
        val_indices = val_indices[:-horizon_bars]

    # Test partition: strictly from test_start_ts onward (LOCKED / UNTOUCHED)
    test_raw_mask = (ts >= test_start_ts) & valid_mask
    test_indices = np.where(test_raw_mask)[0]

    def make_partition(name: str, idxs: np.ndarray) -> SwingPartition:
        sub_X = X.iloc[idxs].reset_index(drop=True)
        sub_y = y.iloc[idxs].reset_index(drop=True)
        sub_ts = ts.iloc[idxs].reset_index(drop=True)
        return SwingPartition(
            name=name,
            X=sub_X,
            y=sub_y,
            timestamps=sub_ts,
            start_timestamp=sub_ts.iloc[0] if len(sub_ts) > 0 else pd.NaT,
            end_timestamp=sub_ts.iloc[-1] if len(sub_ts) > 0 else pd.NaT,
            row_count=len(sub_X),
        )

    return SwingSplits(
        train=make_partition("train", train_indices),
        val=make_partition("val", val_indices),
        test=make_partition("test", test_indices),
        timeframe=timeframe,
        horizon_bars=horizon_bars,
    )
