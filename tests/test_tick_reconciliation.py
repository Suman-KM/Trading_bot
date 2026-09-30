"""Unit tests for Phase 20: Candle vs Tick Trade Reconciliation.

Verifies:
1. Empty input dataframe handling.
2. Trade matching on signal timestamp and trade direction.
3. Accurate categorization into:
   - IDENTICAL
   - EXIT_EVENT_DIFF
   - MAX_HOLD_TIMING_DIFF
   - ENTRY_PRICE_DIFF
   - EXIT_PRICE_DIFF
   - SPREAD_COST_DIFF
   - UNMATCHED_TRADE
4. Reconciliation summary statistics (total PnL, differences, percentiles).
"""

from __future__ import annotations

import pandas as pd

from ai.backtest.reconciliation import reconcile_candle_vs_tick


def _make_trade_row(
    trade_id: int,
    signal_time: str,
    direction: str = "LONG",
    entry_price: float = 1.08500,
    exit_price: float = 1.08650,
    exit_reason: str = "TAKE_PROFIT",
    net_pnl: float = 150.0,
    holding_bars: int = 2,
) -> dict:
    return {
        "trade_id": trade_id,
        "signal_time": signal_time,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "net_pnl": net_pnl,
        "holding_bars": holding_bars,
    }


def test_reconcile_empty_inputs():
    """Verify empty inputs return empty dataframe and zeroed summary."""
    c_empty = pd.DataFrame()
    t_empty = pd.DataFrame()

    rec_df, summary = reconcile_candle_vs_tick(c_empty, t_empty)
    assert rec_df.empty
    assert summary["total_candle_trades"] == 0
    assert summary["total_tick_trades"] == 0
    assert summary["matched_trades"] == 0


def test_reconcile_unmatched_trade():
    """Verify unmatched trades (present in one engine only) are categorized as UNMATCHED_TRADE."""
    c_df = pd.DataFrame([_make_trade_row(1, "2025-08-01 10:00:00", "LONG")])
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 12:00:00", "LONG")]
    )  # Different signal time

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 2
    assert summary["matched_trades"] == 0
    assert summary["unmatched_trades"] == 2
    assert set(rec_df["classification"]) == {"UNMATCHED_TRADE"}


def test_reconcile_identical_trade():
    """Verify identical trades with matching prices and exits are categorized as IDENTICAL."""
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["matched_trades"] == 1
    assert summary["classifications"]["IDENTICAL"] == 1
    assert summary["net_pnl_difference"] == 0.0


def test_reconcile_exit_event_diff():
    """Verify trades with different exit reasons are categorized as EXIT_EVENT_DIFF."""
    # Candle was STOP_LOSS (-$100), Tick was TAKE_PROFIT (+$150)
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08400, "STOP_LOSS", -100.00)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["classifications"]["EXIT_EVENT_DIFF"] == 1
    assert rec_df.iloc[0]["pnl_diff"] == 250.00
    assert summary["net_pnl_difference"] == 250.00


def test_reconcile_max_hold_timing_diff():
    """Verify MAX_HOLD trades with different exit price/bars are MAX_HOLD_TIMING_DIFF."""
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08520, "MAX_HOLD", 20.00, 4)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08540, "MAX_HOLD", 40.00, 4)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["classifications"]["MAX_HOLD_TIMING_DIFF"] == 1


def test_reconcile_entry_price_diff():
    """Verify trades where entry differs by > 2 pts are ENTRY_PRICE_DIFF."""
    # Entry differs by 3 pts (1.08500 vs 1.08503), exit identical (1.08650)
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08503, 1.08650, "TAKE_PROFIT", 147.00)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["classifications"]["ENTRY_PRICE_DIFF"] == 1


def test_reconcile_exit_price_diff():
    """Verify trades where exit differs by > 2 pts are categorized as EXIT_PRICE_DIFF."""
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08655, "TAKE_PROFIT", 155.00)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["classifications"]["EXIT_PRICE_DIFF"] == 1


def test_reconcile_spread_cost_diff():
    """Verify minor PnL differences from floating spread are SPREAD_COST_DIFF."""
    # Prices within 1 pt, but PnL differs by $1.50 due to floating spread
    c_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 150.00)]
    )
    t_df = pd.DataFrame(
        [_make_trade_row(1, "2025-08-01 10:00:00", "LONG", 1.08500, 1.08650, "TAKE_PROFIT", 148.50)]
    )

    rec_df, summary = reconcile_candle_vs_tick(c_df, t_df)
    assert len(rec_df) == 1
    assert summary["classifications"]["SPREAD_COST_DIFF"] == 1
    assert summary["mean_pnl_diff_per_trade"] == -1.50
