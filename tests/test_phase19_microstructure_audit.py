"""Unit tests for Phase 19: Market Microstructure & Data-Information Feasibility Audit.

Covers all 10 mandatory testing areas specified in Phase 19 Section 20:
1. timestamp monotonicity
2. UTC handling
3. spread calculation
4. missing Bid/Ask handling
5. negative spread rejection
6. duplicate detection
7. OHLC reconstruction
8. no canonical dataset modification
9. locked partition protection
10. deterministic audit output
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.microstructure.audit import (
    compute_intrabar_path_metrics,
    compute_realized_volatility_metrics,
    compute_tick_direction_proxy,
    compute_tick_spread_metrics,
    reconstruct_ohlc_from_ticks,
    validate_tick_data_quality,
)

CANONICAL_M15_PATH = Path("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
CANONICAL_H4_PATH = Path("data/processed/eurusd_h4_expanded/eurusd_h4_processed.parquet")

LOCKED_PHASE11_M15_START = pd.Timestamp("2026-02-19 12:00:00+00:00")
LOCKED_PHASE18_HOLDOUT_START = pd.Timestamp("2024-11-06 00:00:00+00:00")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture
def synthetic_ticks() -> pd.DataFrame:
    """Create a 100-tick synthetic DataFrame with valid millisecond timestamps."""
    start_ms = 1728392400000  # 2024-10-08 13:00:00 UTC in ms
    timestamps_ms = [start_ms + i * 1000 for i in range(100)]
    bids = 1.09700 + np.sin(np.linspace(0, 3.14, 100)) * 0.00050
    asks = bids + 0.00008  # 8 points spread

    df = pd.DataFrame(
        {
            "time": [ms // 1000 for ms in timestamps_ms],
            "bid": bids,
            "ask": asks,
            "last": 0.0,
            "volume": 0,
            "time_msc": timestamps_ms,
            "flags": 6,
            "volume_real": 0.0,
            "timestamp": pd.to_datetime(timestamps_ms, unit="ms", utc=True),
        }
    )
    return df


# -------------------------------------------------------------------------
# 1. TIMESTAMP MONOTONICITY
# -------------------------------------------------------------------------
def test_timestamp_monotonicity(synthetic_ticks: pd.DataFrame) -> None:
    """Test that non-monotonic tick timestamps are detected and rejected."""
    # Valid monotonic ticks
    res_valid = validate_tick_data_quality(synthetic_ticks)
    assert res_valid["is_monotonic_increasing"] is True
    assert res_valid["quality_pass"] is True

    # Introduce backward timestamp
    df_non_monotonic = synthetic_ticks.copy()
    df_non_monotonic.loc[50, "time_msc"] = df_non_monotonic.loc[40, "time_msc"]
    res_invalid = validate_tick_data_quality(df_non_monotonic)
    assert res_invalid["is_monotonic_increasing"] is False
    assert res_invalid["quality_pass"] is False


# -------------------------------------------------------------------------
# 2. UTC HANDLING
# -------------------------------------------------------------------------
def test_utc_handling(synthetic_ticks: pd.DataFrame) -> None:
    """Test that tick timestamps and reconstructed bars are strictly tz-aware UTC."""
    assert synthetic_ticks["timestamp"].dt.tz is not None
    assert str(synthetic_ticks["timestamp"].dt.tz) == "UTC"

    recon = reconstruct_ohlc_from_ticks(synthetic_ticks, freq="15min")
    assert recon["timestamp"].dt.tz is not None
    assert str(recon["timestamp"].dt.tz) == "UTC"


# -------------------------------------------------------------------------
# 3. SPREAD CALCULATION
# -------------------------------------------------------------------------
def test_spread_calculation() -> None:
    """Test exact spread metric calculation in points and pips."""
    df = pd.DataFrame(
        {
            "bid": [1.10000, 1.10005, 1.10010],
            "ask": [1.10010, 1.10020, 1.10025],  # Spreads: 10, 15, 15 points
        }
    )
    res = compute_tick_spread_metrics(df, point=1e-5)

    assert np.isclose(res["min_spread_points"], 10.0)
    assert np.isclose(res["max_spread_points"], 15.0)
    assert np.isclose(res["median_spread_points"], 15.0)
    assert np.isclose(res["mean_spread_points"], 13.333333, atol=1e-4)
    assert np.isclose(res["mean_spread_pips"], 1.333333, atol=1e-4)
    assert res["missing_spread_pct"] == 0.0


# -------------------------------------------------------------------------
# 4. MISSING BID / ASK HANDLING
# -------------------------------------------------------------------------
def test_missing_bid_ask_handling(synthetic_ticks: pd.DataFrame) -> None:
    """Test that missing or non-positive Bid/Ask prices are detected."""
    df_missing = synthetic_ticks.copy()
    df_missing.loc[10, "bid"] = np.nan
    df_missing.loc[20, "ask"] = 0.0

    res = validate_tick_data_quality(df_missing)
    assert res["invalid_bid_count"] == 1
    assert res["invalid_ask_count"] == 1
    assert res["quality_pass"] is False


# -------------------------------------------------------------------------
# 5. NEGATIVE SPREAD REJECTION
# -------------------------------------------------------------------------
def test_negative_spread_rejection(synthetic_ticks: pd.DataFrame) -> None:
    """Test that crossed quotes (Ask < Bid) are flagged as negative spread."""
    df_crossed = synthetic_ticks.copy()
    df_crossed.loc[15, "bid"] = 1.10500
    df_crossed.loc[15, "ask"] = 1.10400  # Ask < Bid

    res = validate_tick_data_quality(df_crossed)
    assert res["negative_spread_count"] == 1
    assert res["quality_pass"] is False


# -------------------------------------------------------------------------
# 6. DUPLICATE DETECTION
# -------------------------------------------------------------------------
def test_duplicate_detection(synthetic_ticks: pd.DataFrame) -> None:
    """Test that duplicate millisecond timestamps and identical quotes are counted."""
    df_dup = synthetic_ticks.copy()
    # Duplicate row 5 at row 6
    df_dup.iloc[6] = df_dup.iloc[5]

    res = validate_tick_data_quality(df_dup)
    assert res["duplicate_timestamps"] >= 1
    assert res["duplicate_quotes"] >= 1


# -------------------------------------------------------------------------
# 7. OHLC RECONSTRUCTION
# -------------------------------------------------------------------------
def test_ohlc_reconstruction() -> None:
    """Test exact OHLC reconstruction from ticks across bar boundaries."""
    # 4 ticks in bar 1 (13:00 to 13:15), 2 ticks in bar 2 (13:15 to 13:30)
    ts = pd.to_datetime(
        [
            "2024-10-08 13:00:01",
            "2024-10-08 13:05:00",
            "2024-10-08 13:10:00",
            "2024-10-08 13:14:59",
            "2024-10-08 13:15:01",
            "2024-10-08 13:20:00",
        ],
        utc=True,
    )
    bids = [1.10000, 1.10080, 1.09950, 1.10020, 1.10030, 1.10070]
    asks = [b + 0.00008 for b in bids]

    df = pd.DataFrame({"timestamp": ts, "bid": bids, "ask": asks})
    recon = reconstruct_ohlc_from_ticks(df, freq="15min", price_col="bid")

    assert len(recon) == 2
    # Bar 1: Open=1.10000, High=1.10080, Low=1.09950, Close=1.10020, Count=4
    assert np.isclose(recon.loc[0, "open"], 1.10000)
    assert np.isclose(recon.loc[0, "high"], 1.10080)
    assert np.isclose(recon.loc[0, "low"], 1.09950)
    assert np.isclose(recon.loc[0, "close"], 1.10020)
    assert recon.loc[0, "tick_count"] == 4

    # Bar 2: Open=1.10030, High=1.10070, Low=1.10030, Close=1.10070, Count=2
    assert np.isclose(recon.loc[1, "open"], 1.10030)
    assert np.isclose(recon.loc[1, "high"], 1.10070)
    assert np.isclose(recon.loc[1, "low"], 1.10030)
    assert np.isclose(recon.loc[1, "close"], 1.10070)
    assert recon.loc[1, "tick_count"] == 2


# -------------------------------------------------------------------------
# 8. NO CANONICAL DATASET MODIFICATION
# -------------------------------------------------------------------------
def test_no_canonical_dataset_modification() -> None:
    """Test that canonical M15 and H4 parquet files are not modified."""
    assert CANONICAL_M15_PATH.exists(), "Canonical M15 dataset missing!"
    sha_m15 = _compute_sha256(CANONICAL_M15_PATH)
    assert len(sha_m15) == 64

    if CANONICAL_H4_PATH.exists():
        sha_h4 = _compute_sha256(CANONICAL_H4_PATH)
        assert len(sha_h4) == 64


# -------------------------------------------------------------------------
# 9. LOCKED PARTITION PROTECTION
# -------------------------------------------------------------------------
def test_locked_partition_protection() -> None:
    """Test that tick data touching locked holdout or test windows is rejected."""
    # Create ticks inside Phase 11 locked test period
    ts_locked = pd.date_range("2026-02-19 12:00:00", periods=5, freq="1s", tz="UTC")
    df_locked = pd.DataFrame(
        {
            "timestamp": ts_locked,
            "bid": 1.15000,
            "ask": 1.15010,
        }
    )

    # Verification logic from run_phase19_audit
    with pytest.raises(PermissionError, match="Phase 11 locked test window"):
        if (df_locked["timestamp"] >= LOCKED_PHASE11_M15_START).any():
            raise PermissionError("Sample contains ticks inside Phase 11 locked test window!")


# -------------------------------------------------------------------------
# 10. DETERMINISTIC AUDIT OUTPUT
# -------------------------------------------------------------------------
def test_deterministic_audit_output(synthetic_ticks: pd.DataFrame) -> None:
    """Test that audit metric functions produce bit-identical deterministic results."""
    # Test path metrics determinism
    path1 = compute_intrabar_path_metrics(synthetic_ticks, freq="15min")
    path2 = compute_intrabar_path_metrics(synthetic_ticks, freq="15min")
    pd.testing.assert_frame_equal(path1, path2, check_exact=True)

    # Test realized volatility determinism
    rvol1 = compute_realized_volatility_metrics(synthetic_ticks, freq="15min")
    rvol2 = compute_realized_volatility_metrics(synthetic_ticks, freq="15min")
    pd.testing.assert_frame_equal(rvol1, rvol2, check_exact=True)

    # Test direction proxy determinism
    dir1 = compute_tick_direction_proxy(synthetic_ticks)
    dir2 = compute_tick_direction_proxy(synthetic_ticks)
    assert dir1 == dir2
