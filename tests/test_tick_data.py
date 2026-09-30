"""Unit tests for Phase 20: Tick Data Structures and Quality Validation.

Verifies:
1. Timestamp monotonicity enforcement
2. Price sanity (finite, positive)
3. Non-negative spread enforcement (no crossed quotes)
4. Phase 11 test partition boundary protection
5. Slicing and binary search retrieval (get_slice_indices, get_first_tick, get_ticks_in_window)
6. Raw slice extraction
7. Quality auditing with validate_tick_quality
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai.backtest.tick_data import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    TickDataRepository,
    TickQuote,
    validate_tick_quality,
)


@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Sample chronological UTC timestamps within Phase 12 validation partition."""
    base_ms = 1752500000000  # Mid 2025 (~July 2025)
    return [
        datetime.fromtimestamp((base_ms + i * 1000) / 1000.0, tz=timezone.utc) for i in range(10)
    ]


@pytest.fixture
def valid_repository(sample_timestamps: list[datetime]) -> TickDataRepository:
    """Create a valid in-memory TickDataRepository with 10 ticks."""
    time_msc = np.array([int(dt.timestamp() * 1000) for dt in sample_timestamps], dtype=np.int64)
    bids = np.array([1.08500 + i * 0.00010 for i in range(10)], dtype=np.float64)
    asks = bids + 0.00008  # 8 points spread
    flags = np.zeros(10, dtype=np.uint32)

    return TickDataRepository(
        time_msc=time_msc,
        bids=bids,
        asks=asks,
        flags=flags,
        point_value=1e-05,
    )


def test_tick_quote_properties():
    """Verify TickQuote dataclass fields and calculated properties."""
    dt = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    tq = TickQuote(
        time_msc=int(dt.timestamp() * 1000),
        timestamp=dt,
        bid=1.08500,
        ask=1.08510,
        spread_points=10.0,
        flags=0,
    )
    assert tq.bid == 1.08500
    assert tq.ask == 1.08510
    assert pytest.approx(tq.mid, rel=1e-6) == 1.08505
    assert pytest.approx(tq.spread_price, rel=1e-6) == 0.00010


def test_repository_length_mismatch():
    """Arrays of unequal length must raise ValueError."""
    time_msc = np.array([1000, 2000], dtype=np.int64)
    bids = np.array([1.08500], dtype=np.float64)
    asks = np.array([1.08510], dtype=np.float64)

    with pytest.raises(ValueError, match="identical length"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_monotonicity_violation():
    """Non-monotonic timestamps must raise ValueError."""
    time_msc = np.array([1000, 3000, 2000], dtype=np.int64)
    bids = np.array([1.08, 1.08, 1.08], dtype=np.float64)
    asks = np.array([1.09, 1.09, 1.09], dtype=np.float64)

    with pytest.raises(ValueError, match="monotonicity violation"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_non_finite_quotes():
    """NaN or Inf quotes must raise ValueError."""
    time_msc = np.array([1000, 2000], dtype=np.int64)
    bids = np.array([1.08, np.nan], dtype=np.float64)
    asks = np.array([1.09, 1.09], dtype=np.float64)

    with pytest.raises(ValueError, match="Non-finite bid or ask"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_non_positive_quotes():
    """Quotes <= 0 must raise ValueError."""
    time_msc = np.array([1000, 2000], dtype=np.int64)
    bids = np.array([1.08, 0.0], dtype=np.float64)
    asks = np.array([1.09, 1.09], dtype=np.float64)

    with pytest.raises(ValueError, match="Non-positive bid or ask"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_crossed_quotes():
    """Crossed quotes (ask < bid) must raise ValueError."""
    time_msc = np.array([1000, 2000], dtype=np.int64)
    bids = np.array([1.08500, 1.08600], dtype=np.float64)
    asks = np.array([1.08510, 1.08550], dtype=np.float64)  # 1.08550 < 1.08600

    with pytest.raises(ValueError, match="Negative spread"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_phase11_test_lock_violation():
    """Data crossing into Phase 11 test partition must raise critical governance error."""
    lock_ms = int(PHASE11_TEST_LOCK_TIMESTAMP.timestamp() * 1000)
    time_msc = np.array([lock_ms + 1000], dtype=np.int64)
    bids = np.array([1.08500], dtype=np.float64)
    asks = np.array([1.08510], dtype=np.float64)

    with pytest.raises(ValueError, match="CRITICAL GOVERNANCE VIOLATION"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_repository_slicing(
    valid_repository: TickDataRepository,
    sample_timestamps: list[datetime],
):
    """Verify binary search slice indexing and retrieval."""
    assert len(valid_repository) == 10

    # Slice middle 4 ticks: index 3 to 6 (timestamps 3 to 6 inclusive)
    t_start = sample_timestamps[3]
    t_end = sample_timestamps[6]

    idx_start, idx_end = valid_repository.get_slice_indices(t_start, t_end)
    assert idx_start == 3
    assert idx_end == 7  # searchsorted right includes timestamp 6

    quotes = valid_repository.get_ticks_in_window(t_start, t_end)
    assert len(quotes) == 4
    assert quotes[0].time_msc == int(t_start.timestamp() * 1000)
    assert quotes[-1].time_msc == int(t_end.timestamp() * 1000)

    # Raw slice
    msc_raw, bids_raw, asks_raw = valid_repository.get_raw_slice(t_start, t_end)
    assert len(msc_raw) == 4
    assert len(bids_raw) == 4
    assert len(asks_raw) == 4
    assert msc_raw[0] == int(t_start.timestamp() * 1000)


def test_repository_first_tick_at_or_after(
    valid_repository: TickDataRepository,
    sample_timestamps: list[datetime],
):
    """Verify get_first_tick_at_or_after precision."""
    # Exact match
    first = valid_repository.get_first_tick_at_or_after(sample_timestamps[2])
    assert first is not None
    assert first.time_msc == int(sample_timestamps[2].timestamp() * 1000)

    # Offset between tick 2 and 3 (+500ms)
    offset_ts = datetime.fromtimestamp(
        sample_timestamps[2].timestamp() + 0.5,
        tz=timezone.utc,
    )
    first_after = valid_repository.get_first_tick_at_or_after(offset_ts)
    assert first_after is not None
    assert first_after.time_msc == int(sample_timestamps[3].timestamp() * 1000)

    # Beyond the end of data
    beyond_ts = datetime.fromtimestamp(
        sample_timestamps[-1].timestamp() + 10.0,
        tz=timezone.utc,
    )
    assert valid_repository.get_first_tick_at_or_after(beyond_ts) is None


def test_validate_tick_quality():
    """Verify validate_tick_quality audit results."""
    # 1. Valid data
    df_valid = pd.DataFrame(
        {
            "time_msc": [1752500000000, 1752500001000, 1752500002000],
            "bid": [1.08500, 1.08510, 1.08505],
            "ask": [1.08508, 1.08518, 1.08513],
        }
    )
    quality = validate_tick_quality(df_valid)
    assert quality["status"] == "PASS"
    assert quality["tick_count"] == 3
    assert quality["is_monotonic"] is True
    assert quality["negative_spreads"] == 0
    assert quality["phase11_test_lock_respected"] is True
    assert quality["spread_metrics"]["mean_points"] == pytest.approx(8.0, rel=1e-2)

    # 2. Negative spread
    df_crossed = pd.DataFrame(
        {
            "time_msc": [1752500000000, 1752500001000],
            "bid": [1.08500, 1.08520],
            "ask": [1.08508, 1.08510],  # crossed
        }
    )
    quality_crossed = validate_tick_quality(df_crossed)
    assert quality_crossed["status"] == "FAIL"
    assert quality_crossed["negative_spreads"] == 1

    # 3. Test set leakage
    lock_ms = int(PHASE11_TEST_LOCK_TIMESTAMP.timestamp() * 1000)
    df_leak = pd.DataFrame(
        {
            "time_msc": [lock_ms - 1000, lock_ms + 1000],
            "bid": [1.08500, 1.08510],
            "ask": [1.08508, 1.08518],
        }
    )
    quality_leak = validate_tick_quality(df_leak)
    assert quality_leak["status"] == "FAIL"
    assert quality_leak["phase11_test_lock_respected"] is False
