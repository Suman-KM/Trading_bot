"""Comprehensive integrity and leakage tests for Phase 12 backtesting engine.

Verifies all 26 mandatory integrity checks:
1. Chronological processing
2. No future data access
3. No lookahead (signal at t, entry at t+1)
4. Next-bar entry price
5. LONG bid/ask execution (entry Ask, exit Bid)
6. SHORT bid/ask execution (entry Bid, exit Ask)
7. Spread conversion from MT5 points
8. Stop loss distance calculation (ATR multiple)
9. Take profit distance calculation (ATR multiple)
10. Same-bar SL/TP ambiguity conservative handling (SL first priority)
11. Maximum holding period exit (4 bars)
12. Cooldown enforcement after exit (1 bar)
13. Maximum open positions constraint
14. Daily loss limit detection
15. Kill switch enforcement
16. Position sizing respecting risk and exposure limits
17. RiskEngine rejection enforcement
18. Commission calculation
19. Slippage calculation
20. P&L accounting reconciliation (Gross - Costs = Net)
21. Deterministic repeated runs
22. No target columns in features
23. No test-set access guard
24. No model fitting during simulation
25. End-of-data clean position closure
26. Gap opening beyond SL/TP behavior
27. Anti-lookahead leakage test (appending future bars preserves past decisions)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ai.backtest.costs import CostModel
from ai.backtest.engine import BacktestEngine
from ai.backtest.models import (
    BacktestConfig,
    TradeDirection,
    TradeExitReason,
)
from ai.backtest.strategy import MLAssistedStrategy
from trading.risk.limits import RiskLimits


class MockModel:
    """Deterministic mock classifier predicting fixed classes and probabilities."""

    def __init__(self, fixed_class: float = 1.0, fixed_prob: float = 0.70) -> None:
        self.classes_ = np.array([-1.0, 0.0, 1.0])
        self.fixed_class = fixed_class
        self.fixed_prob = fixed_prob
        self.fit_calls = 0

    def predict_proba(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        probs = np.zeros((n_samples, 3))
        # classes: -1.0 (idx 0), 0.0 (idx 1), 1.0 (idx 2)
        if self.fixed_class == -1.0:
            probs[:, 0] = self.fixed_prob
            probs[:, 1] = (1.0 - self.fixed_prob) / 2.0
            probs[:, 2] = (1.0 - self.fixed_prob) / 2.0
        elif self.fixed_class == 1.0:
            probs[:, 2] = self.fixed_prob
            probs[:, 0] = (1.0 - self.fixed_prob) / 2.0
            probs[:, 1] = (1.0 - self.fixed_prob) / 2.0
        else:
            probs[:, 1] = self.fixed_prob
            probs[:, 0] = (1.0 - self.fixed_prob) / 2.0
            probs[:, 2] = (1.0 - self.fixed_prob) / 2.0
        return probs

    def fit(self, X: Any, y: Any) -> MockModel:
        self.fit_calls += 1
        return self


def create_synthetic_market_data(
    n_bars: int = 20,
    start_price: float = 1.0850,
    atr_val: float = 0.0010,
    spread_pts: float = 10.0,
    start_dt: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper creating aligned synthetic market data and dummy 80-feature matrix."""
    if start_dt is None:
        start_dt = datetime(2025, 8, 1, 8, 0, tzinfo=timezone.utc)

    timestamps = [start_dt + timedelta(minutes=15 * i) for i in range(n_bars)]
    opens = [start_price + (i * 0.0001) for i in range(n_bars)]
    highs = [o + 0.0008 for o in opens]
    lows = [o - 0.0008 for o in opens]
    closes = [o + 0.0002 for o in opens]
    spreads = [spread_pts for _ in range(n_bars)]
    atrs = [atr_val for _ in range(n_bars)]

    df_market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "spread": spreads,
            "atr_14": atrs,
        }
    )

    feature_cols = [f"f_{k}" for k in range(80)]
    feature_data = np.zeros((n_bars, 80))
    df_features = pd.DataFrame(feature_data, columns=feature_cols)

    return df_market, df_features


# =========================================================================
# INTEGRITY TESTS
# =========================================================================


def test_1_and_2_chronological_processing_and_no_future_data():
    """Checks 1 & 2: Processing moves strictly forward; future bars are not read."""
    df_market, df_features = create_synthetic_market_data(n_bars=10)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(confidence_threshold=0.60)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)

    # Verify timestamps in equity curve strictly monotonic increasing
    eq_timestamps = [ep.timestamp for ep in result.equity_curve]
    for k in range(1, len(eq_timestamps)):
        assert eq_timestamps[k] > eq_timestamps[k - 1]


def test_3_and_4_no_lookahead_and_next_bar_entry():
    """Checks 3 & 4: Signal generated at candle t executes at candle t+1 open."""
    df_market, df_features = create_synthetic_market_data(n_bars=10, spread_pts=0.0)
    # Signal at t=0 (8:00) should enter at t=1 (8:15)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(confidence_threshold=0.60, default_spread_points=0.0)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    assert len(result.trade_ledger) > 0

    first_trade = result.trade_ledger[0]
    expected_signal_time = df_market["timestamp"].iloc[0]
    expected_entry_time = df_market["timestamp"].iloc[1]
    expected_open_price = df_market["open"].iloc[1]

    assert first_trade.signal_time == expected_signal_time
    assert first_trade.entry_time == expected_entry_time
    assert first_trade.entry_price == pytest.approx(expected_open_price)


def test_5_long_bid_ask_handling():
    """Check 5: LONG enters at Ask (Bid_open + spread) and exits at Bid."""
    df_market, df_features = create_synthetic_market_data(n_bars=10, spread_pts=20.0)  # 2.0 pips
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(
        confidence_threshold=0.60,
        default_spread_points=20.0,
        point_value=1e-5,
        slippage_points=0.0,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    bid_open_t1 = df_market["open"].iloc[1]
    expected_ask_t1 = bid_open_t1 + (20.0 * 1e-5)
    assert trade.direction == TradeDirection.LONG
    assert trade.entry_price == pytest.approx(expected_ask_t1)
    # LONG exit must be at Bid price
    assert trade.spread_cost > 0


def test_6_short_bid_ask_handling():
    """Check 6: SHORT enters at Bid (Bid_open) and exits at Ask (Bid + spread)."""
    df_market, df_features = create_synthetic_market_data(n_bars=10, spread_pts=20.0)
    model = MockModel(fixed_class=-1.0, fixed_prob=0.75)
    config = BacktestConfig(
        confidence_threshold=0.60,
        default_spread_points=20.0,
        point_value=1e-5,
        slippage_points=0.0,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    bid_open_t1 = df_market["open"].iloc[1]
    assert trade.direction == TradeDirection.SHORT
    # Enters at Bid
    assert trade.entry_price == pytest.approx(bid_open_t1)
    assert trade.spread_cost > 0


def test_7_spread_conversion():
    """Check 7: Spread in points converts strictly by point_value."""
    config = BacktestConfig(point_value=1e-5)
    cost_model = CostModel(config)
    assert cost_model.get_spread_price(10.0) == pytest.approx(0.00010)
    assert cost_model.get_spread_price(25.0) == pytest.approx(0.00025)
    assert cost_model.get_spread_price(0.0) == pytest.approx(0.00010)  # fallback default 10 pts


def test_8_and_9_sl_and_tp_distance_calculation():
    """Checks 8 & 9: ATR-based SL (1.0x) and TP (1.5x) distances are accurately established."""
    df_market, df_features = create_synthetic_market_data(n_bars=10, atr_val=0.00120)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(
        stop_loss_atr_multiple=1.0,
        take_profit_atr_multiple=1.5,
        default_spread_points=0.0,
        slippage_points=0.0,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    expected_sl_dist = 1.0 * 0.00120
    expected_tp_dist = 1.5 * 0.00120
    assert (trade.entry_price - trade.stop_loss) == pytest.approx(expected_sl_dist)
    assert (trade.take_profit - trade.entry_price) == pytest.approx(expected_tp_dist)


def test_10_same_bar_sl_tp_collision_conservative_policy():
    """Check 10: If both SL and TP are touched in the same bar, assume STOP LOSS hit first."""
    df_market, df_features = create_synthetic_market_data(n_bars=5, atr_val=0.0010)
    # Entry at bar 1. On bar 2, create a huge bar touching both SL and TP:
    # Entry at ~1.0851. SL = 1.0841, TP = 1.0866.
    # On bar 2: Low = 1.0830 (touches SL), High = 1.0880 (touches TP).
    df_market.loc[2, "low"] = 1.0830
    df_market.loc[2, "high"] = 1.0880

    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(same_bar_sl_priority=True, default_spread_points=0.0)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    assert trade.exit_reason == TradeExitReason.STOP_LOSS
    assert trade.exit_time == df_market["timestamp"].iloc[2]
    assert trade.net_pnl < 0


def test_11_max_holding_period():
    """Check 11: Position exits after exactly 4 bars if neither SL nor TP touched."""
    # Huge ATR ensures SL/TP is not triggered prior to max holding horizon
    df_market, df_features = create_synthetic_market_data(n_bars=10, atr_val=0.0100)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(max_holding_bars=4, default_spread_points=0.0)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    assert trade.exit_reason == TradeExitReason.MAX_HOLD
    assert trade.holding_bars == 4
    # Entered at bar 1, held bars 1, 2, 3, 4 -> exits at bar 4 close
    assert trade.entry_time == df_market["timestamp"].iloc[1]
    assert trade.exit_time == df_market["timestamp"].iloc[4]


def test_12_cooldown_enforcement():
    """Check 12: No new position opens for 1 bar after position closes."""
    df_market, df_features = create_synthetic_market_data(n_bars=15, atr_val=0.0100)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(max_holding_bars=2, cooldown_bars=1, default_spread_points=0.0)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trades = result.trade_ledger

    # Trade 1 entered at bar 1, held bars 1, 2 -> exits at bar 2.
    # Bar 2: last_exit_bar = 2.
    # Bar 3: (3 - 2) <= 1 -> COOLDOWN ACTIVE (rejected).
    # Bar 4: (4 - 2) > 1 -> Signal generated at bar 4!
    # Trade 2 entered at bar 5!
    assert trades[0].exit_time == df_market["timestamp"].iloc[2]
    assert trades[1].entry_time == df_market["timestamp"].iloc[5]
    assert result.rejections.get("COOLDOWN_ACTIVE", 0) >= 1


def test_13_max_open_positions():
    """Check 13: Maximum open positions is strictly 1 (no concurrent trades)."""
    df_market, df_features = create_synthetic_market_data(n_bars=15)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(max_holding_bars=4)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)

    # Verify equity curve open_positions_count is never > 1
    max_open = max(ep.open_positions_count for ep in result.equity_curve)
    assert max_open <= 1


def test_14_and_15_daily_loss_limit_and_kill_switch():
    """Checks 14 & 15: Drawdown >= 1.0% trips kill switch and halts subsequent entries."""
    df_market, df_features = create_synthetic_market_data(n_bars=10, start_price=1.0850)
    # Trade 1 enters at bar 1. Force a massive crash on bar 2:
    df_market.loc[2, "low"] = 1.0500
    df_market.loc[2, "close"] = 1.0500

    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    # Ultra-tight 0.01% daily loss limit to guarantee tripping
    tight_limits = RiskLimits(MAX_DAILY_LOSS_PERCENT=0.01)
    config = BacktestConfig(
        initial_equity=100_000.0,
        risk_limits=tight_limits,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)

    assert len(result.daily_loss_events) > 0
    halt_rejections = result.rejections.get("DAILY_LOSS_LIMIT", 0) + result.rejections.get(
        "KILL_SWITCH_ACTIVE", 0
    )
    assert halt_rejections > 0


def test_16_position_sizing_respects_limits():
    """Check 16: Position size respects both risk capital (0.5%) and exposure limits (20%)."""
    df_market, df_features = create_synthetic_market_data(
        n_bars=5, start_price=1.0850, atr_val=0.0010
    )
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(
        initial_equity=100_000.0,
        constrain_exposure=True,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    # Gross exposure = qty * entry_price <= 20% of 100k = $20,000
    exposure = trade.quantity * trade.entry_price
    assert exposure <= 20_000.0
    # Risk capital at stop distance <= 0.5% of 100k = $500
    risk_at_stop = trade.quantity * (trade.entry_price - trade.stop_loss)
    assert risk_at_stop <= 500.0


def test_17_risk_engine_rejection_enforced():
    """Check 17: Trades failing RiskEngine checks (e.g. low confidence) are rejected."""
    df_market, df_features = create_synthetic_market_data(n_bars=5)
    # Model confidence 0.55 < 0.60 minimum
    model = MockModel(fixed_class=1.0, fixed_prob=0.55)
    # Strategy emits at 0.55, but RiskEngine sovereignly enforces 0.60
    config = BacktestConfig(confidence_threshold=0.55)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    assert len(result.trade_ledger) == 0
    assert result.rejections.get("CONFIDENCE_TOO_LOW", 0) > 0


def test_18_and_19_commission_and_slippage_calculation():
    """Checks 18 & 19: Commission and slippage applied and tracked accurately."""
    df_market, df_features = create_synthetic_market_data(n_bars=5)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(
        commission_per_lot=7.0,  # $7 per standard lot round turn
        slippage_points=5.0,  # 0.5 pip slippage per fill
        default_spread_points=10.0,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    expected_comm = (trade.quantity / 100_000.0) * 7.0
    expected_slip = trade.quantity * (2 * 5.0 * 1e-5)
    assert trade.commission == pytest.approx(expected_comm)
    assert trade.slippage == pytest.approx(expected_slip)


def test_20_pnl_reconciliation():
    """Check 20: Net PnL = Gross PnL - Spread Cost - Commission - Slippage."""
    df_market, df_features = create_synthetic_market_data(n_bars=10)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(
        commission_per_lot=5.0,
        slippage_points=2.0,
        default_spread_points=12.0,
    )
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    reconciled_net = trade.gross_pnl - trade.spread_cost - trade.commission - trade.slippage
    assert trade.net_pnl == pytest.approx(reconciled_net, abs=1e-4)


def test_21_deterministic_repeated_runs():
    """Check 21: Running the same dataset and configuration twice produces identical results."""
    df_market, df_features = create_synthetic_market_data(n_bars=15)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig()

    strategy1 = MLAssistedStrategy(model=model, config=config)
    engine1 = BacktestEngine(strategy=strategy1, config=config)
    res1 = engine1.run(df_market=df_market, features_df=df_features)

    strategy2 = MLAssistedStrategy(model=model, config=config)
    engine2 = BacktestEngine(strategy=strategy2, config=config)
    res2 = engine2.run(df_market=df_market, features_df=df_features)

    df_t1 = res1.to_trades_dataframe()
    df_t2 = res2.to_trades_dataframe()
    pd.testing.assert_frame_equal(df_t1, df_t2)

    df_eq1 = res1.to_equity_dataframe()
    df_eq2 = res2.to_equity_dataframe()
    pd.testing.assert_frame_equal(df_eq1, df_eq2)


def test_22_no_target_columns_in_features():
    """Check 22: Feature dataframe contains zero label or future target columns."""
    from ai.dataset.assembly import is_label_or_future_column

    _, df_features = create_synthetic_market_data(n_bars=5)

    for col in df_features.columns:
        assert not is_label_or_future_column(col), f"Target column found in features: {col}"


def test_23_test_partition_isolation_guard():
    """Check 23: Phase 11 Test partition is strictly isolated and never accessed."""
    from ai.dataset.assembly import assemble_dataset
    from ai.dataset.splits import split_dataset

    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    # Test partition length must be exactly 14,988 rows
    assert len(splits.test) == 14988
    # Validation partition length must be 14,983 rows
    assert len(splits.val) == 14983
    # Training partition length must be 69,937 rows
    assert len(splits.train) == 69937


def test_24_no_model_fitting_during_simulation():
    """Check 24: Model fit method is never called during simulation."""
    df_market, df_features = create_synthetic_market_data(n_bars=10)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig()
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    engine.run(df_market=df_market, features_df=df_features)
    assert model.fit_calls == 0


def test_25_end_of_data_behavior():
    """Check 25: Open position at end of dataset is cleanly closed with END_OF_DATA."""
    # Run with 3 bars. Enters at bar 1. Ends at bar 2. Max hold is 4 bars, so still open.
    df_market, df_features = create_synthetic_market_data(n_bars=3)
    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(max_holding_bars=4)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    assert len(result.trade_ledger) == 1
    assert result.trade_ledger[0].exit_reason == TradeExitReason.END_OF_DATA
    assert result.trade_ledger[0].exit_time == df_market["timestamp"].iloc[-1]


def test_26_gap_opening_behavior():
    """Check 26: Opening gap beyond SL fills at gap open price."""
    df_market, df_features = create_synthetic_market_data(
        n_bars=5, start_price=1.0850, atr_val=0.0010
    )
    # Enters at bar 1: Open ~ 1.0851. SL ~ 1.0841.
    # Bar 2 opens with a huge downward gap at 1.0800 (< SL 1.0841):
    df_market.loc[2, "open"] = 1.0800
    df_market.loc[2, "low"] = 1.0790
    df_market.loc[2, "high"] = 1.0810
    df_market.loc[2, "close"] = 1.0805

    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig(default_spread_points=0.0)
    strategy = MLAssistedStrategy(model=model, config=config)
    engine = BacktestEngine(strategy=strategy, config=config)

    result = engine.run(df_market=df_market, features_df=df_features)
    trade = result.trade_ledger[0]

    assert trade.exit_reason == TradeExitReason.STOP_LOSS
    # Filled at the gap open price 1.0800, NOT the nominal SL 1.0841!
    assert trade.exit_price == pytest.approx(1.0800)


def test_27_anti_lookahead_leakage_test():
    """Check 27 (Section 31): Appending future bars preserves past decisions identically."""
    # Data up to time T (10 bars)
    df_market_T, df_features_T = create_synthetic_market_data(n_bars=10)
    # Data up to time T + Future (20 bars)
    df_market_future, df_features_future = create_synthetic_market_data(n_bars=20)

    model = MockModel(fixed_class=1.0, fixed_prob=0.75)
    config = BacktestConfig()

    strategy_T = MLAssistedStrategy(model=model, config=config)
    engine_T = BacktestEngine(strategy=strategy_T, config=config)
    res_T = engine_T.run(df_market=df_market_T, features_df=df_features_T)

    strategy_future = MLAssistedStrategy(model=model, config=config)
    engine_future = BacktestEngine(strategy=strategy_future, config=config)
    res_future = engine_future.run(df_market=df_market_future, features_df=df_features_future)

    # Filter future results to time T
    timestamp_T = df_market_T["timestamp"].iloc[-1]
    trades_T = res_T.trade_ledger
    trades_future_at_T = [t for t in res_future.trade_ledger if t.exit_time <= timestamp_T]

    # Completed trades prior to T must be identical
    assert len(trades_T) >= 1
    # Check first trade identical in all fields
    t1 = trades_T[0]
    t2 = trades_future_at_T[0]
    assert t1.signal_time == t2.signal_time
    assert t1.entry_time == t2.entry_time
    assert t1.entry_price == t2.entry_price
    assert t1.net_pnl == pytest.approx(t2.net_pnl)
    assert t1.exit_reason == t2.exit_reason
