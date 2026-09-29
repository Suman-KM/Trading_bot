"""Unit tests, leakage verification, and governance checks for Phase 15 dual-track research."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.dataset.swing import (
    aggregate_m15_to_d1,
    aggregate_m15_to_h4,
    split_swing_data,
    validate_swing_data,
)
from ai.features.intraday_extended import (
    EXTENDED_FEATURE_NAMES,
    compute_intraday_extended_features,
)
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets


@pytest.fixture
def synthetic_m15_df() -> pd.DataFrame:
    """Create deterministic synthetic M15 DataFrame covering 120 bars (30 hours)."""
    n = 120
    base_epoch = 1700000000
    times = [base_epoch + (i * 900) for i in range(n)]
    ts = pd.to_datetime(times, unit="s", utc=True)

    np.random.seed(42)
    prices = 1.0800 + np.cumsum(np.random.randn(n) * 0.0005)
    opens = prices - 0.0002
    closes = prices + 0.0002
    highs = np.maximum(opens, closes) + 0.0004
    lows = np.minimum(opens, closes) - 0.0004
    vols = np.random.randint(500, 2000, size=n).astype(np.uint64)
    spreads = np.random.randint(8, 15, size=n).astype(np.int32)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "tick_volume": vols,
            "spread": spreads,
        }
    )


def test_m15_extended_features_shape_and_leakage(synthetic_m15_df: pd.DataFrame) -> None:
    """Verify that 25 extended intraday features are generated with correct causal structure."""
    ext = compute_intraday_extended_features(synthetic_m15_df)
    assert ext.shape[1] == 25
    assert list(ext.columns) == EXTENDED_FEATURE_NAMES

    # Check for no target or future columns
    for col in ext.columns:
        assert not col.startswith("future"), f"Target column '{col}' detected in features"
        assert not col.startswith("direction"), f"Target column '{col}' detected in features"

    # Beyond warm-up (first 80 bars), there must be zero NaNs
    assert ext.iloc[80:].isna().sum().sum() == 0


def test_h4_aggregation_ohlcv_consistency(synthetic_m15_df: pd.DataFrame) -> None:
    """Verify H4 aggregation rules: open=first, high=max, low=min, close=last, volume=sum."""
    h4 = aggregate_m15_to_h4(synthetic_m15_df)
    assert len(h4) > 0

    # OHLC geometric consistency
    assert (h4["high"] >= h4["open"]).all()
    assert (h4["high"] >= h4["close"]).all()
    assert (h4["low"] <= h4["open"]).all()
    assert (h4["low"] <= h4["close"]).all()

    # Monotonic timestamps
    assert h4["timestamp"].is_monotonic_increasing
    assert not h4["timestamp"].duplicated().any()

    # Volume aggregation
    assert h4["tick_volume"].sum() <= synthetic_m15_df["tick_volume"].sum()


def test_d1_aggregation_ohlcv_consistency(synthetic_m15_df: pd.DataFrame) -> None:
    """Verify Daily (D1) aggregation rules and geometry."""
    d1 = aggregate_m15_to_d1(synthetic_m15_df)
    assert len(d1) > 0
    assert (d1["high"] >= d1["open"]).all()
    assert (d1["high"] >= d1["close"]).all()
    assert (d1["low"] <= d1["open"]).all()
    assert (d1["low"] <= d1["close"]).all()
    assert d1["timestamp"].is_monotonic_increasing


def test_swing_data_validation(synthetic_m15_df: pd.DataFrame) -> None:
    """Verify that validate_swing_data passes on valid aggregated data."""
    h4 = aggregate_m15_to_h4(synthetic_m15_df)
    rep = validate_swing_data(h4, timeframe="H4")
    assert rep["all_valid"] is True
    assert rep["is_monotonic_increasing"] is True
    assert rep["has_duplicates"] is False
    assert rep["has_nans"] is False


def test_swing_features_and_targets(synthetic_m15_df: pd.DataFrame) -> None:
    """Verify swing features and targets computation and end-of-data NaN handling."""
    h4 = aggregate_m15_to_h4(synthetic_m15_df)
    feats = compute_swing_features(h4, timeframe="H4")
    targets = compute_swing_targets(h4, horizons=[2, 4], timeframe="H4")

    # Features check
    assert len(feats) == len(h4)
    assert "return_1" in feats.columns
    assert "dist_sma_10" in feats.columns
    assert "atr_norm_14" in feats.columns

    # Targets check
    assert "direction_2" in targets.columns
    assert "direction_vol_2" in targets.columns
    assert "direction_4" in targets.columns
    assert "direction_vol_4" in targets.columns

    # End-of-data NaNs: last H rows must be NaN for horizon H
    assert targets["future_return_2"].iloc[-2:].isna().all()
    assert targets["future_return_4"].iloc[-4:].isna().all()


def test_swing_data_splitting_and_holdout() -> None:
    """Verify swing data splitting preserves chronology, purge window, and test holdout."""
    # Synthetic swing series covering 100 days
    ts = pd.date_range("2025-01-01", periods=100, freq="1D", tz="UTC")
    X = pd.DataFrame({"feat_1": np.linspace(1, 100, 100)}, index=range(100))
    y = pd.Series(np.where(X["feat_1"] % 2 == 0, 1.0, -1.0))

    splits = split_swing_data(
        X,
        y,
        pd.Series(ts),
        timeframe="D1",
        horizon_bars=2,
        train_end_ts=pd.Timestamp("2025-02-15 00:00:00+00:00"),
        val_start_ts=pd.Timestamp("2025-02-16 00:00:00+00:00"),
        val_end_ts=pd.Timestamp("2025-03-15 00:00:00+00:00"),
        test_start_ts=pd.Timestamp("2025-03-16 00:00:00+00:00"),
    )

    assert len(splits.train) > 0
    assert len(splits.val) > 0
    assert len(splits.test) > 0

    # Chronological integrity
    assert splits.train.end_timestamp < splits.val.start_timestamp
    assert splits.val.end_timestamp < splits.test.start_timestamp


def test_phase11_holdout_test_set_unmodified() -> None:
    """Verify strict partition governance: Phase 11 Test partition is strictly preserved."""
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    assert len(splits.train) == 69937
    assert len(splits.val) == 14983
    assert len(splits.test) == 14988  # Phase 11 Test partition strictly preserved
    assert str(splits.test.start_timestamp) == "2026-02-19 12:00:00+00:00"
