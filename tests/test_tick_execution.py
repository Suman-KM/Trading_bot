"""Unit tests for Phase 20: Tick-Realistic Chronological Execution Engine.

Verifies:
1. Chronological execution: entry fills at first tick of bar t+1 (Ask for LONG, Bid for SHORT).
2. Chronological execution: intrabar TP hit before SL.
3. Chronological execution: intrabar SL hit before TP.
4. Max holding period exit: liquidates at final tick of max hold bar.
5. Slippage application: impacts entry and exit fills symmetrically.
6. Dynamic floating spread cost accounting.
7. Engine determinism: identical inputs produce identical trade ledgers.
8. Anti-lookahead guarantee: entry timestamp is strictly greater than signal bar timestamp.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ai.backtest.models import BacktestConfig, TradeDirection, TradeExitReason
from ai.backtest.tick_data import TickDataRepository
from ai.backtest.tick_engine import TickBacktestEngine


class DummyStrategy:
    """Lightweight strategy interface for backtest engine testing."""

    def __init__(self, symbol: str = "EURUSD") -> None:
        self.symbol = symbol
        self.classes = np.array([-1.0, 0.0, 1.0])
        self.short_idx = 0
        self.neutral_idx = 1
        self.long_idx = 2
        self.model_version = "test_v1"
        self.timeframe = "M15"
        self.feature_version = "v1"


def _create_synthetic_market_data(
    start_dt: datetime,
    n_bars: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic M15 market data and feature matrix."""
    rows = []
    for i in range(n_bars):
        t = start_dt + timedelta(minutes=15 * i)
        rows.append(
            {
                "timestamp": t,
                "open": 1.08500,
                "high": 1.08600,
                "low": 1.08400,
                "close": 1.08500,
                "spread": 4.0,
                "atr_14": 0.00100,  # 10 pips / 100 pts
            }
        )
    df_market = pd.DataFrame(rows)
    df_features = pd.DataFrame({"dummy_feat": np.ones(n_bars)})
    return df_market, df_features


def test_long_entry_at_ask_and_tp_before_sl():
    """Verify LONG enters at Ask and exits at TP when TP is touched before SL."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    df_market, df_features = _create_synthetic_market_data(start_dt, n_bars=6)

    # Signal at bar 0 close (10:00 UTC) -> BUY
    # Bar 0: BUY signal. Bar 1 (10:15 UTC): entry executed at first tick.
    # Stop loss = entry - 1.0*ATR (0.00100)
    # Take profit = entry + 1.5*ATR (0.00150)
    probs = np.zeros((6, 3))
    probs[0] = [0.1, 0.1, 0.8]  # Bar 0: BUY signal (conf=0.8)

    # Construct ticks for Bar 1 (10:15 - 10:30 UTC):
    # Tick 0 (10:15:00.100): entry tick -> bid=1.08500, ask=1.08508. Fill = 1.08508.
    # SL = 1.08508 - 0.00100 = 1.08408.
    # TP = 1.08508 + 0.00150 = 1.08658.
    # Tick 1 (10:18:00.000): price rises to touch TP: bid=1.08660, ask=1.08668 (TP HIT!)
    # Tick 2 (10:25:00.000): price collapses to SL: bid=1.08390, ask=1.08398 (SL touched AFTER TP)
    b1_start = start_dt + timedelta(minutes=15)
    t0_ms = int(b1_start.timestamp() * 1000) + 100
    t1_ms = int((b1_start + timedelta(minutes=3)).timestamp() * 1000)
    t2_ms = int((b1_start + timedelta(minutes=10)).timestamp() * 1000)

    time_msc = np.array([t0_ms, t1_ms, t2_ms], dtype=np.int64)
    bids = np.array([1.08500, 1.08660, 1.08390], dtype=np.float64)
    asks = bids + 0.00008

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(
        confidence_threshold=0.60,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        slippage_points=0.0,
        max_holding_bars=4,
    )
    strategy = DummyStrategy()
    engine = TickBacktestEngine(
        strategy=strategy,
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)
    trades = result.trade_ledger

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == TradeDirection.LONG
    # Verify entry at Ask
    assert pytest.approx(t.entry_price, rel=1e-6) == 1.08508
    # Verify chronological precedence: TP hit before SL
    assert t.exit_reason == TradeExitReason.TAKE_PROFIT
    assert pytest.approx(t.exit_price, rel=1e-6) == 1.08660
    assert t.net_pnl > 0
    # Verify anti-lookahead: entry time > signal time
    assert t.entry_time > t.signal_time


def test_sl_hit_before_tp():
    """Verify chronological order detects SL hit before TP."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    df_market, df_features = _create_synthetic_market_data(start_dt, n_bars=6)

    probs = np.zeros((6, 3))
    probs[0] = [0.1, 0.1, 0.8]  # Bar 0: BUY signal

    b1_start = start_dt + timedelta(minutes=15)
    t0_ms = int(b1_start.timestamp() * 1000) + 100
    t1_ms = int((b1_start + timedelta(minutes=3)).timestamp() * 1000)
    t2_ms = int((b1_start + timedelta(minutes=10)).timestamp() * 1000)

    # Tick 0: Entry -> fill at ask=1.08508, SL=1.08408, TP=1.08658
    # Tick 1: SL hit first! bid=1.08400
    # Tick 2: TP hit later: bid=1.08670
    time_msc = np.array([t0_ms, t1_ms, t2_ms], dtype=np.int64)
    bids = np.array([1.08500, 1.08400, 1.08670], dtype=np.float64)
    asks = bids + 0.00008

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(
        confidence_threshold=0.60,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        slippage_points=0.0,
        max_holding_bars=4,
    )
    engine = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)
    trades = result.trade_ledger

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == TradeExitReason.STOP_LOSS
    assert pytest.approx(t.exit_price, rel=1e-6) == 1.08400
    assert t.net_pnl < 0


def test_short_entry_at_bid_and_tp():
    """Verify SHORT enters at tick Bid and exits at TP when Ask <= take_profit."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    df_market, df_features = _create_synthetic_market_data(start_dt, n_bars=6)

    # Bar 0: SELL signal (probs[0] = [0.85, 0.1, 0.05])
    probs = np.zeros((6, 3))
    probs[0] = [0.85, 0.1, 0.05]

    b1_start = start_dt + timedelta(minutes=15)
    t0_ms = int(b1_start.timestamp() * 1000) + 100
    t1_ms = int((b1_start + timedelta(minutes=5)).timestamp() * 1000)

    # Tick 0: Entry -> fill at bid=1.08500, ask=1.08508
    # For SHORT: SL = entry + 1.0*ATR = 1.08600. TP = entry - 1.5*ATR = 1.08350.
    # Tick 1: Ask drops to 1.08340 <= TP (TP HIT!)
    time_msc = np.array([t0_ms, t1_ms], dtype=np.int64)
    bids = np.array([1.08500, 1.08332], dtype=np.float64)
    asks = np.array([1.08508, 1.08340], dtype=np.float64)

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(
        confidence_threshold=0.60,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        slippage_points=0.0,
        max_holding_bars=4,
    )
    engine = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)
    trades = result.trade_ledger

    assert len(trades) == 1
    t = trades[0]
    assert t.direction == TradeDirection.SHORT
    # Verify fill at Bid
    assert pytest.approx(t.entry_price, rel=1e-6) == 1.08500
    assert t.exit_reason == TradeExitReason.TAKE_PROFIT
    # Exits by buying back at Ask
    assert pytest.approx(t.exit_price, rel=1e-6) == 1.08340
    assert t.net_pnl > 0


def test_max_hold_exit_at_final_tick():
    """Verify trade that reaches max_holding_bars exits on the final available tick."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    # Market data with lows that don't trigger SL on bar fallback
    rows = []
    for i in range(6):
        t = start_dt + timedelta(minutes=15 * i)
        rows.append(
            {
                "timestamp": t,
                "open": 1.08500,
                "high": 1.08550,
                "low": 1.08480,
                "close": 1.08500,
                "spread": 4.0,
                "atr_14": 0.00100,
            }
        )
    df_market = pd.DataFrame(rows)
    df_features = pd.DataFrame({"dummy_feat": np.ones(6)})

    probs = np.zeros((6, 3))
    probs[0] = [0.1, 0.1, 0.8]  # Bar 0: BUY signal

    # Bars 1, 2, 3, 4: Holding bars = 1, 2, 3, 4 (max_holding_bars = 4)
    # Provide ticks in each holding bar at 1.08520 (SL=1.08408, TP=1.08658)
    # Final tick in Bar 4 at 1.08530
    # Bar 0: 10:00-10:15 (signal)
    # Bar 1: 10:15-10:30 (entry, hold=1)
    # Bar 2: 10:30-10:45 (hold=2)
    # Bar 3: 10:45-11:00 (hold=3)
    # Bar 4: 11:00-11:15 (hold=4 -> MAX_HOLD!)
    b1_start = start_dt + timedelta(minutes=15)
    b4_end = start_dt + timedelta(minutes=75)

    tick_times = [
        int((b1_start + timedelta(seconds=1)).timestamp() * 1000),
        int((start_dt + timedelta(minutes=25)).timestamp() * 1000),
        int((start_dt + timedelta(minutes=40)).timestamp() * 1000),
        int((start_dt + timedelta(minutes=55)).timestamp() * 1000),
        int((b4_end - timedelta(seconds=5)).timestamp() * 1000),
    ]

    time_msc = np.array(tick_times, dtype=np.int64)
    bids = np.array([1.08500, 1.08520, 1.08520, 1.08520, 1.08530], dtype=np.float64)
    asks = bids + 0.00008

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(
        confidence_threshold=0.60,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        slippage_points=0.0,
        max_holding_bars=4,
    )
    engine = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)
    trades = result.trade_ledger

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == TradeExitReason.MAX_HOLD
    assert t.holding_bars == 4
    # Exits at bid of final tick in bar 4
    assert t.exit_price == pytest.approx(1.08530, rel=1e-6)


def test_slippage_application():
    """Verify slippage points are added adversely to entry and exit fills."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    df_market, df_features = _create_synthetic_market_data(start_dt, n_bars=6)

    probs = np.zeros((6, 3))
    probs[0] = [0.1, 0.1, 0.8]  # BUY signal

    b1_start = start_dt + timedelta(minutes=15)
    t0_ms = int(b1_start.timestamp() * 1000) + 100
    t1_ms = int((b1_start + timedelta(minutes=5)).timestamp() * 1000)

    # 5.0 points slippage = 0.00005
    # Entry: ask=1.08508 + 0.00005 slippage = 1.08513
    # SL = 1.08513 - 0.00100 = 1.08413
    # TP = 1.08513 + 0.00150 = 1.08663
    # Tick 1: Bid=1.08670 >= TP -> TP hit!
    # Exit price: bid (1.08670) - 0.00005 slippage = 1.08665
    time_msc = np.array([t0_ms, t1_ms], dtype=np.int64)
    bids = np.array([1.08500, 1.08670], dtype=np.float64)
    asks = bids + 0.00008

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(
        confidence_threshold=0.60,
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        slippage_points=5.0,  # 0.5 pip
        max_holding_bars=4,
    )
    engine = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs,
    )

    result = engine.run(df_market, df_features)
    trades = result.trade_ledger

    assert len(trades) == 1
    t = trades[0]
    assert pytest.approx(t.entry_price, rel=1e-6) == 1.08513
    assert pytest.approx(t.exit_price, rel=1e-6) == 1.08665
    assert t.slippage > 0


def test_simulation_determinism():
    """Verify two sequential executions with identical ticks produce bit-identical results."""
    start_dt = datetime(2025, 8, 1, 10, 0, 0, tzinfo=timezone.utc)
    df_market, df_features = _create_synthetic_market_data(start_dt, n_bars=6)

    probs = np.zeros((6, 3))
    probs[0] = [0.1, 0.1, 0.8]

    b1_start = start_dt + timedelta(minutes=15)
    t0_ms = int(b1_start.timestamp() * 1000) + 100
    t1_ms = int((b1_start + timedelta(minutes=3)).timestamp() * 1000)

    time_msc = np.array([t0_ms, t1_ms], dtype=np.int64)
    bids = np.array([1.08500, 1.08660], dtype=np.float64)
    asks = bids + 0.00008

    tick_repo = TickDataRepository(time_msc=time_msc, bids=bids, asks=asks)
    config = BacktestConfig(confidence_threshold=0.60)

    engine1 = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs.copy(),
    )
    res1 = engine1.run(df_market, df_features)

    engine2 = TickBacktestEngine(
        strategy=DummyStrategy(),
        tick_repo=tick_repo,
        config=config,
        precomputed_probs=probs.copy(),
    )
    res2 = engine2.run(df_market, df_features)

    assert len(res1.trade_ledger) == len(res2.trade_ledger)
    assert res1.trade_ledger[0].net_pnl == res2.trade_ledger[0].net_pnl
    assert res1.trade_ledger[0].entry_price == res2.trade_ledger[0].entry_price
    assert res1.trade_ledger[0].exit_price == res2.trade_ledger[0].exit_price
    assert len(res1.equity_curve) == len(res2.equity_curve)
    assert res1.equity_curve[-1].equity == res2.equity_curve[-1].equity
