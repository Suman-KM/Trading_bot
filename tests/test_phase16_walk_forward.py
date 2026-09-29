"""Unit tests for Phase 16 chronological walk-forward fold generation and validation.

Verifies:
- Monotonic timestamps and chronological ordering
- Exact purge gap enforcement (8 H4 bars between train and val)
- No train/val overlap
- Expanding window progression
- Complete exclusion of locked test partitions
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai.dataset.swing import aggregate_m15_to_h4
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.walk_forward import (
    DEFAULT_RESEARCH_CUTOFF_TS,
    LOCKED_TEST_START_TS,
    generate_chronological_walk_forward_folds,
)


@pytest.fixture(scope="module")
def h4_research_data():
    """Load canonical M15 data and construct H4 research features/targets."""
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)
    feats = compute_swing_features(h4_df, timeframe="H4")
    targets = compute_swing_targets(h4_df, horizons=[8], timeframe="H4")
    target_vol_8 = targets["direction_vol_8"]

    ts_h4 = pd.to_datetime(h4_df["timestamp"], utc=True)
    res_mask = ts_h4 <= DEFAULT_RESEARCH_CUTOFF_TS

    h4_res = h4_df[res_mask].copy().reset_index(drop=True)
    feats_res = feats.loc[res_mask].reset_index(drop=True)
    target_res = target_vol_8.loc[res_mask].reset_index(drop=True)

    return h4_res, feats_res, target_res


def test_walk_forward_fold_count_and_types(h4_research_data):
    """Test that 5 valid chronological folds are generated with correct sample counts."""
    h4_res, feats_res, target_res = h4_research_data
    folds = generate_chronological_walk_forward_folds(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        initial_train_bars=2524,
        n_folds=5,
        purge_bars=8,
    )

    assert len(folds) == 5
    for f in folds:
        assert f.train_sample_count > 1000
        assert f.val_sample_count > 200
        assert f.purge_gap_bars == 8


def test_walk_forward_chronological_progression(h4_research_data):
    """Test that training windows expand chronologically and validation windows advance."""
    h4_res, feats_res, target_res = h4_research_data
    folds = generate_chronological_walk_forward_folds(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        initial_train_bars=2524,
        n_folds=5,
        purge_bars=8,
    )

    for i in range(len(folds) - 1):
        f_curr = folds[i]
        f_next = folds[i + 1]

        # Training set expands
        assert f_curr.train_sample_count < f_next.train_sample_count
        assert f_curr.train_end_ts < f_next.train_end_ts

        # Validation window moves strictly forward
        assert f_curr.val_start_ts < f_next.val_start_ts
        assert f_curr.val_end_ts <= f_next.val_start_ts


def test_walk_forward_purge_gap_and_no_overlap(h4_research_data):
    """Test that each fold enforces at least 8 bars gap between train and validation."""
    h4_res, feats_res, target_res = h4_research_data
    folds = generate_chronological_walk_forward_folds(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        initial_train_bars=2524,
        n_folds=5,
        purge_bars=8,
    )

    for f in folds:
        # No index overlap
        overlap = set(f.train_indices).intersection(set(f.val_indices))
        assert len(overlap) == 0

        # Temporal separation: train_end must precede val_start
        assert f.train_end_ts < f.val_start_ts

        # Index separation in raw H4 bars is at least purge_bars
        max_tr_idx = max(f.train_indices)
        min_va_idx = min(f.val_indices)
        assert min_va_idx - max_tr_idx >= f.purge_gap_bars


def test_locked_test_partitions_never_included(h4_research_data):
    """Governance test: verify walk-forward never reaches locked test timestamps."""
    h4_res, feats_res, target_res = h4_research_data
    folds = generate_chronological_walk_forward_folds(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        initial_train_bars=2524,
        n_folds=5,
        purge_bars=8,
    )

    for f in folds:
        assert f.train_end_ts < LOCKED_TEST_START_TS
        assert f.val_end_ts < LOCKED_TEST_START_TS
        assert f.val_end_ts <= DEFAULT_RESEARCH_CUTOFF_TS
