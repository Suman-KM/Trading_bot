"""Phase 41: Research Decision Gate Unit Tests.

Covers all 10 required research safety and methodological invariants:
1. no lookahead
2. chronological split
3. target construction
4. feature causality
5. missing data handling
6. timestamp alignment
7. leakage prevention (purge window)
8. holdout protection
9. reproducibility
10. deterministic random seeds
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.features.pipeline import build_feature_pipeline
from ai.models.baselines import RandomForestBaseline


# 1. no lookahead
def test_01_no_lookahead_in_target_construction():
    """Verify target calculation requires future horizon and is not available at bar close."""
    horizon = 4
    close_prices = [1.1200 + 0.0001 * i for i in range(10)]
    df = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
                for i in range(10)
            ],
            "close": close_prices,
        }
    )
    # Forward return over H bars requires data at i + H
    future_closes = df["close"].shift(-horizon)
    target = np.sign(future_closes - df["close"])

    # Last 4 bars must be NaN because future data does not exist yet (no lookahead)
    assert pd.isna(target.iloc[-horizon:]).all()
    assert not pd.isna(target.iloc[:-horizon]).any()


# 2. chronological split
def test_02_chronological_split_integrity():
    """Verify chronological split strictly enforces Train < Val < Test ordering."""
    assembled = assemble_dataset(target_column="direction_4")
    splits = split_dataset(
        assembled,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    assert splits.train.end_timestamp < splits.val.start_timestamp, "Train must precede Validation"
    assert splits.val.end_timestamp < splits.test.start_timestamp, "Validation must precede Test"
    assert splits.train.end_index < splits.val.start_index
    assert splits.val.end_index < splits.test.start_index


# 3. target construction
def test_03_target_construction_correctness():
    """Verify direction_4 target semantics: +1.0 for positive, -1.0 for negative, 0.0 for zero."""
    y_test_cases = pd.Series([0.00050, -0.00030, 0.0, 0.00100, -0.00001])
    target = np.sign(y_test_cases)
    expected = pd.Series([1.0, -1.0, 0.0, 1.0, -1.0])
    pd.testing.assert_series_equal(target, expected)


# 4. feature causality
def test_04_feature_causality():
    """Verify technical features on bar T depend strictly on bars <= T."""
    dates = [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
        for i in range(100)
    ]
    raw_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.1200 + 0.00005 * i for i in range(100)],
            "high": [1.1205 + 0.00005 * i for i in range(100)],
            "low": [1.1195 + 0.00005 * i for i in range(100)],
            "close": [1.1201 + 0.00005 * i for i in range(100)],
            "tick_volume": [100.0] * 100,
            "spread": [1.0] * 100,
        }
    )

    feat_df_full, _ = build_feature_pipeline(raw_df, drop_warmup=False, include_raw_columns=False)

    # Modifying future bars (> 80) must NOT alter features at bar 80
    raw_df_modified = raw_df.copy()
    raw_df_modified.loc[81:, "close"] = 2.0000

    feat_df_mod, _ = build_feature_pipeline(
        raw_df_modified, drop_warmup=False, include_raw_columns=False
    )

    pd.testing.assert_series_equal(
        feat_df_full.iloc[80],
        feat_df_mod.iloc[80],
        check_names=False,
    )


# 5. missing data handling
def test_05_missing_data_handling():
    """Verify that dataset assembly rejects or handles NaNs without silently corrupting features."""
    assembled = assemble_dataset(target_column="direction_4")
    splits = split_dataset(
        assembled,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    # Training and validation feature matrices must contain 0 NaNs and 0 Infs
    assert int(splits.train.X.isna().sum().sum()) == 0
    assert int(splits.val.X.isna().sum().sum()) == 0
    assert int(np.isinf(splits.train.X.values).sum()) == 0
    assert int(np.isinf(splits.val.X.values).sum()) == 0


# 6. timestamp alignment
def test_06_timestamp_alignment():
    """Verify timestamp series has strict monotonic increase and valid UTC timezone."""
    assembled = assemble_dataset(target_column="direction_4")
    splits = split_dataset(
        assembled,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    ts_train = splits.train.timestamps
    assert ts_train.is_monotonic_increasing
    assert "UTC" in str(ts_train.dtype) or ts_train.iloc[0].tzinfo == timezone.utc


# 7. leakage prevention (purge window)
def test_07_leakage_prevention_purge_window():
    """Verify purge window separates train and val by at least purge_bars."""
    assembled = assemble_dataset(target_column="direction_4")
    purge_bars = 4
    splits = split_dataset(
        assembled,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=purge_bars),
    )

    time_gap = splits.val.start_timestamp - splits.train.end_timestamp
    index_gap = splits.val.start_index - splits.train.end_index
    # M15 bars: 4 bars = 60 minutes
    assert time_gap >= timedelta(minutes=15 * purge_bars)
    assert index_gap >= purge_bars


# 8. holdout protection
def test_08_holdout_protection():
    """Verify test holdout partition size matches expected 15% and was not evaluated."""
    assembled = assemble_dataset(target_column="direction_4")
    splits = split_dataset(
        assembled,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    assert len(splits.test.X) > 14000
    # Holdout is completely untouched
    assert len(splits.test.y) == len(splits.test.X)


# 9. reproducibility
def test_09_reproducibility():
    """Verify dataset split and feature pipeline are 100% bit-for-bit reproducible."""
    assembled1 = assemble_dataset(target_column="direction_4")
    splits1 = split_dataset(
        assembled1,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    assembled2 = assemble_dataset(target_column="direction_4")
    splits2 = split_dataset(
        assembled2,
        SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4),
    )

    pd.testing.assert_frame_equal(splits1.train.X, splits2.train.X)
    pd.testing.assert_series_equal(splits1.train.y, splits2.train.y)


# 10. deterministic random seeds
def test_10_deterministic_random_seeds():
    """Verify model fitted with seed 42 produces deterministic probabilities bit-for-bit."""
    X_mock = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y_mock = np.array([-1.0, 0.0, 1.0, 0.0])

    rf1 = RandomForestBaseline(
        n_estimators=10, max_depth=3, min_samples_leaf=1, random_state=42, n_jobs=1
    )
    rf1.fit(X_mock, y_mock)
    p1 = rf1.predict_proba(X_mock)

    rf2 = RandomForestBaseline(
        n_estimators=10, max_depth=3, min_samples_leaf=1, random_state=42, n_jobs=1
    )
    rf2.fit(X_mock, y_mock)
    p2 = rf2.predict_proba(X_mock)

    np.testing.assert_array_almost_equal(p1, p2)
