"""Unit tests for Phase 21.1: Tick Data Integrity Repair & Affected Result Revalidation.

Verifies:
1. Query inside valid coverage succeeds.
2. Query inside a gap fails safely (returns None).
3. Query before earliest tick fails (returns None).
4. Query after latest tick fails (returns None).
5. Query cannot jump across a gap to a future interval.
6. Explicit regression test: July 25 2025 query cannot return August 1 2025 quote.
7. Exit window coverage lookup cannot cross a gap.
8. Candle fallback cannot manufacture synthetic tick executions.
9. Test-period timestamps (>= 2026-02-19 10:45:00 UTC) are strictly rejected.
10. Tick repository and gap detection remain strictly deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai.backtest.models import BacktestConfig, TradeExitReason
from ai.backtest.tick_data import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    TickDataRepository,
)
from ai.backtest.tick_engine import TickBacktestEngine


class DummyStrategy:
    """Lightweight strategy interface for unit testing."""

    def __init__(self, symbol: str = "EURUSD") -> None:
        self.symbol = symbol
        self.classes = np.array([-1.0, 0.0, 1.0])
        self.short_idx = 0
        self.neutral_idx = 1
        self.long_idx = 2
        self.model_version = "test_phase21_1"
        self.timeframe = "M15"
        self.feature_version = "v1"


def test_query_inside_valid_coverage_succeeds():
    """1. Query inside valid coverage interval succeeds and returns the nearest tick."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    time_msc = np.array([t0_ms, t0_ms + 1000, t0_ms + 2000], dtype=np.int64)
    bids = np.array([1.17000, 1.17010, 1.17020], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Query at t0 + 500ms -> should return tick at t0 + 1000ms
    q_time = t0 + timedelta(milliseconds=500)
    quote = repo.get_first_tick_at_or_after(q_time)

    assert quote is not None
    assert quote.time_msc == t0_ms + 1000
    assert pytest.approx(quote.bid, rel=1e-6) == 1.17010
    assert pytest.approx(quote.ask, rel=1e-6) == 1.17020


def test_query_inside_gap_fails_safely():
    """2. Query inside an unpopulated gap fails safely and returns None."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    # Interval 1: 10:00:00 to 10:05:00
    # Gap: 10:05:00 to 11:00:00 (55 min > 5 min max_allowed_gap_ms)
    # Interval 2: 11:00:00 to 11:05:00
    t_int1 = [t0_ms, t0_ms + 60_000, t0_ms + 300_000]
    t_int2 = [t0_ms + 3600_000, t0_ms + 3900_000]
    time_msc = np.array(t_int1 + t_int2, dtype=np.int64)
    bids = np.full(len(time_msc), 1.17000)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    assert len(repo.gaps) == 1

    # Query at 10:30:00 (strictly inside the gap)
    q_time = t0 + timedelta(minutes=30)
    quote = repo.get_first_tick_at_or_after(q_time)

    assert quote is None


def test_query_before_earliest_tick_fails():
    """3. Query before earliest tick fails and returns None."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    time_msc = np.array([t0_ms, t0_ms + 1000], dtype=np.int64)
    bids = np.array([1.17000, 1.17010], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Query 1 hour before earliest tick
    q_time = t0 - timedelta(hours=1)
    quote = repo.get_first_tick_at_or_after(q_time)

    assert quote is None


def test_query_after_latest_tick_fails():
    """4. Query after latest tick fails and returns None."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    time_msc = np.array([t0_ms, t0_ms + 1000], dtype=np.int64)
    bids = np.array([1.17000, 1.17010], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Query 1 hour after latest tick
    q_time = t0 + timedelta(hours=1)
    quote = repo.get_first_tick_at_or_after(q_time)

    assert quote is None


def test_query_cannot_jump_across_gap():
    """5. Query cannot jump across a gap to future ticks."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    # End of interval 1 at 10:05:00, resume at 11:00:00
    time_msc = np.array([t0_ms + 300_000, t0_ms + 3600_000], dtype=np.int64)
    bids = np.array([1.17000, 1.14000], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Query at 10:06:00 (1 minute into the gap)
    q_time = t0 + timedelta(minutes=6)
    quote = repo.get_first_tick_at_or_after(q_time)

    # Must NOT jump to the 11:00:00 quote (1.14000)
    assert quote is None


def test_regression_july_25_future_tick_substitution_prevented():
    """6. Explicit regression test: July 25 2025 query cannot substitute August 1 2025 quote."""
    # Recreate the exact Phase 20 gap scenario:
    # Last tick: 2025-07-24 15:29:59 UTC @ 1.17450
    # Next tick: 2025-08-01 14:00:00 UTC @ 1.14120
    t_july = datetime(2025, 7, 24, 15, 29, 59, tzinfo=timezone.utc)
    t_aug = datetime(2025, 8, 1, 14, 0, 0, tzinfo=timezone.utc)

    time_msc = np.array(
        [int(t_july.timestamp() * 1000), int(t_aug.timestamp() * 1000)],
        dtype=np.int64,
    )
    bids = np.array([1.17450, 1.14120], dtype=np.float64)
    asks = np.array([1.17460, 1.14130], dtype=np.float64)

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Trade 5/6 signal was on July 25 2025 at 04:30 UTC
    query_july25 = datetime(2025, 7, 25, 4, 30, 0, tzinfo=timezone.utc)
    quote = repo.get_first_tick_at_or_after(query_july25)

    # CRITICAL INVARIANT: Must NOT return August 1 quote (1.14120)!
    assert quote is None


def test_exit_lookup_cannot_cross_gap():
    """7. Exit window coverage lookup cannot cross an unpopulated gap."""
    t0 = datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    # Ticks from 10:00 to 10:10, then gap until 11:00
    time_msc = np.array([t0_ms, t0_ms + 600_000, t0_ms + 3600_000], dtype=np.int64)
    bids = np.array([1.17000, 1.17010, 1.17050], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)

    # Window covering 10:00 to 10:30 crosses the gap starting at 10:10
    w_start = t0
    w_end = t0 + timedelta(minutes=30)

    assert not repo.is_window_covered(w_start, w_end)


def test_candle_fallback_cannot_manufacture_tick_execution():
    """8. Uncovered entry or holding window produces DATA_UNAVAILABLE, not synthetic fills."""
    start_dt = datetime(2025, 7, 25, 4, 0, 0, tzinfo=timezone.utc)
    # 4 bars: Bar 0 has BUY signal at 04:00. Bar 1 entry is at 04:15.
    rows = []
    for i in range(4):
        rows.append(
            {
                "timestamp": start_dt + timedelta(minutes=15 * i),
                "open": 1.17400,
                "high": 1.17800,  # Would hit TP under candle fallback!
                "low": 1.17100,  # Would hit SL under candle fallback!
                "close": 1.17500,
                "spread": 4.0,
                "atr_14": 0.00100,
            }
        )
    df_market = pd.DataFrame(rows)
    df_features = pd.DataFrame({"dummy": np.ones(4)})

    probs = np.zeros((4, 3))
    probs[0] = [0.1, 0.1, 0.8]  # BUY signal at bar 0

    # Ticks are completely empty for this date window (gap)
    # Ticks exist only on July 20 (before) and August 1 (after)
    t_before = int(datetime(2025, 7, 20, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    t_after = int(datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    time_msc = np.array([t_before, t_after], dtype=np.int64)
    bids = np.array([1.17000, 1.14000], dtype=np.float64)
    asks = bids + 0.00010

    repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(confidence_threshold=0.60)
    engine = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)

    # Must NOT manufacture synthetic TP or SL executions!
    assert len(result.unavailable_trades) >= 1
    assert result.unavailable_trades[0]["status"] == "DATA_UNAVAILABLE"

    # All recorded trades in ledger must be marked DATA_UNAVAILABLE with 0 P&L
    for t in result.trade_ledger:
        assert t.exit_reason == TradeExitReason.DATA_UNAVAILABLE
        assert t.net_pnl == 0.0


def test_test_partition_timestamps_rejected():
    """9. Test partition timestamps (>= 2026-02-19 10:45:00 UTC) are strictly rejected."""
    # Attempt to initialize tick repository with timestamps in the locked test partition
    locked_dt = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(minutes=15)
    time_msc = np.array([int(locked_dt.timestamp() * 1000)], dtype=np.int64)
    bids = np.array([1.18000], dtype=np.float64)
    asks = np.array([1.18010], dtype=np.float64)

    with pytest.raises(ValueError, match="Phase 11 test partition boundary"):
        TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)


def test_tick_repository_determinism():
    """10. Tick repository and gap detection remain strictly deterministic."""
    t0 = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)
    time_msc = np.array([t0_ms, t0_ms + 1000, t0_ms + 600_000, t0_ms + 601_000], dtype=np.int64)
    bids = np.array([1.15000, 1.15010, 1.15050, 1.15060], dtype=np.float64)
    asks = bids + 0.00010

    repo1 = TickDataRepository(time_msc=time_msc.copy(), bids=bids.copy(), asks=asks.copy())
    repo2 = TickDataRepository(time_msc=time_msc.copy(), bids=bids.copy(), asks=asks.copy())

    assert len(repo1.coverage_intervals) == len(repo2.coverage_intervals)
    assert len(repo1.gaps) == len(repo2.gaps)
    assert repo1.gaps[0].duration_seconds == repo2.gaps[0].duration_seconds

    # Identical query outputs
    q_time = t0 + timedelta(milliseconds=500)
    quote1 = repo1.get_first_tick_at_or_after(q_time)
    quote2 = repo2.get_first_tick_at_or_after(q_time)

    assert quote1 is not None and quote2 is not None
    assert quote1.time_msc == quote2.time_msc
    assert quote1.bid == quote2.bid
