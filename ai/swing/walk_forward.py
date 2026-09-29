"""Chronological walk-forward / rolling-origin evaluation generator for swing models.

Enforces strict temporal ordering, expanding training windows, forward-looking
target horizon purging (at minimum target horizon H bars removed at every split),
and complete exclusion of locked out-of-sample test partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Canonical validation cutoff matching Phase 11/12/15 boundaries
DEFAULT_RESEARCH_CUTOFF_TS = pd.Timestamp("2026-02-19 10:45:00+00:00")
# Phase 11/15 Test start boundary (PERMANENTLY LOCKED)
LOCKED_TEST_START_TS = pd.Timestamp("2026-02-19 12:00:00+00:00")


@dataclass(frozen=True)
class WalkForwardFold:
    """Individual chronological walk-forward fold metadata and index pointers."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start_ts: pd.Timestamp
    train_end_ts: pd.Timestamp
    val_start_ts: pd.Timestamp
    val_end_ts: pd.Timestamp
    purge_gap_bars: int
    train_sample_count: int
    val_sample_count: int
    train_class_dist: dict[str, int]
    val_class_dist: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_idx": self.fold_idx,
            "train_sample_count": self.train_sample_count,
            "val_sample_count": self.val_sample_count,
            "train_start_ts": str(self.train_start_ts),
            "train_end_ts": str(self.train_end_ts),
            "val_start_ts": str(self.val_start_ts),
            "val_end_ts": str(self.val_end_ts),
            "purge_gap_bars": self.purge_gap_bars,
            "train_class_dist": self.train_class_dist,
            "val_class_dist": self.val_class_dist,
        }


def generate_chronological_walk_forward_folds(
    h4_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    initial_train_bars: int = 2524,
    n_folds: int = 5,
    purge_bars: int = 8,
    research_end_ts: pd.Timestamp = DEFAULT_RESEARCH_CUTOFF_TS,
) -> list[WalkForwardFold]:
    """Generate strictly chronological expanding-window walk-forward folds.

    Parameters
    ----------
    h4_df : pd.DataFrame
        H4 OHLCV DataFrame with monotonic 'timestamp'.
    features_df : pd.DataFrame
        Computed swing features corresponding to h4_df.
    target_series : pd.Series
        Target labels (e.g. direction_vol_8) corresponding to h4_df.
    initial_train_bars : int, default 2524
        Number of raw H4 bars in the initial training warm-up window (~1.5 years).
    n_folds : int, default 5
        Number of forward chronological validation folds.
    purge_bars : int, default 8
        Horizon in bars to purge between train and validation to prevent target overlap.
    research_end_ts : pd.Timestamp, default DEFAULT_RESEARCH_CUTOFF_TS
        Strict upper timestamp cutoff. Any row after this timestamp is locked test data.

    Returns
    -------
    list[WalkForwardFold]
        List of WalkForwardFold objects.
    """
    ts = pd.to_datetime(h4_df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("h4_df timestamps must be strictly monotonic increasing.")

    # Governance check: strictly exclude locked test partition
    research_mask = ts <= research_end_ts
    if (ts > LOCKED_TEST_START_TS).any() and (ts[research_mask] > LOCKED_TEST_START_TS).any():
        raise PermissionError("Attempted to access locked test partition data.")

    h4_res = h4_df[research_mask].copy().reset_index(drop=True)
    y_res = target_series.loc[research_mask].reset_index(drop=True)
    X_res = features_df.loc[research_mask].reset_index(drop=True)
    ts_res = ts[research_mask].reset_index(drop=True)

    n_raw = len(h4_res)
    if initial_train_bars >= n_raw:
        raise ValueError(
            f"initial_train_bars ({initial_train_bars}) must be less than research bars ({n_raw})."
        )

    remaining_bars = n_raw - initial_train_bars
    fold_size = remaining_bars // n_folds
    if fold_size <= purge_bars:
        raise ValueError(f"Fold size ({fold_size}) must exceed purge window ({purge_bars}).")

    folds: list[WalkForwardFold] = []

    for f in range(n_folds):
        val_start = initial_train_bars + f * fold_size
        val_end = val_start + fold_size if f < n_folds - 1 else n_raw

        # Expanding training window: from 0 up to (val_start - purge_bars)
        train_end = val_start - purge_bars
        # In validation, purge last purge_bars so forward targets do not look past val_end
        val_eval_end = val_end - purge_bars

        tr_raw_indices = np.arange(0, train_end)
        va_raw_indices = np.arange(val_start, val_eval_end)

        # Filter valid samples: non-NaN target and non-NaN features
        tr_valid_mask = y_res.iloc[tr_raw_indices].notna() & ~X_res.iloc[tr_raw_indices].isna().any(
            axis=1
        )
        va_valid_mask = y_res.iloc[va_raw_indices].notna() & ~X_res.iloc[va_raw_indices].isna().any(
            axis=1
        )

        tr_indices = tr_raw_indices[tr_valid_mask]
        va_indices = va_raw_indices[va_valid_mask]

        if len(tr_indices) == 0 or len(va_indices) == 0:
            raise ValueError(f"Fold {f + 1} has zero valid samples.")

        t_tr_start = ts_res.iloc[tr_indices[0]]
        t_tr_end = ts_res.iloc[tr_indices[-1]]
        t_va_start = ts_res.iloc[va_indices[0]]
        t_va_end = ts_res.iloc[va_indices[-1]]

        # Ensure no temporal overlap: t_tr_end < t_va_start
        assert t_tr_end < t_va_start, f"Fold {f + 1}: Temporal overlap detected!"

        # Target class distributions
        tr_y_fold = y_res.iloc[tr_indices]
        va_y_fold = y_res.iloc[va_indices]

        tr_dist = {
            "long": int((tr_y_fold == 1.0).sum()),
            "short": int((tr_y_fold == -1.0).sum()),
        }
        va_dist = {
            "long": int((va_y_fold == 1.0).sum()),
            "short": int((va_y_fold == -1.0).sum()),
        }

        fold = WalkForwardFold(
            fold_idx=f + 1,
            train_indices=tr_indices,
            val_indices=va_indices,
            train_start_ts=t_tr_start,
            train_end_ts=t_tr_end,
            val_start_ts=t_va_start,
            val_end_ts=t_va_end,
            purge_gap_bars=purge_bars,
            train_sample_count=len(tr_indices),
            val_sample_count=len(va_indices),
            train_class_dist=tr_dist,
            val_class_dist=va_dist,
        )
        folds.append(fold)

    return folds
