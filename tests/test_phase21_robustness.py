"""Unit tests for Phase 21: Tick-Realistic Strategy Validation & Execution Robustness.

Verifies:
1. P&L concentration and percentile calculations.
2. Sign-change sensitivity when removing top 1, 3, 5 trades.
3. Chronological block partitioning (5 equal blocks).
4. Monthly performance grouping.
5. Exit event and positive trade source breakdown.
6. Intrabar collision and divergence categorization.
7. Rollover and high-spread regime classification.
8. Reproducible bootstrap distribution resampling (random_state=42).
9. Strict holdout partition boundary rejection (Phase 11 lock).
10. Integrity and consistency of the Phase 21 trade ledger artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ai.backtest.robustness import (
    PHASE11_TEST_LOCK_TIMESTAMP,
    VALIDATION_END_TIMESTAMP,
    VALIDATION_START_TIMESTAMP,
    calculate_pnl_concentration,
    compute_bootstrap_distribution,
    compute_collision_breakdown,
    compute_exit_event_analysis,
    compute_monthly_breakdown,
    compute_rollover_analysis,
    compute_temporal_blocks,
    verify_test_partition_rejection,
)


def _make_dummy_ledger() -> pd.DataFrame:
    """Create synthetic trade ledger for isolated unit tests."""
    rows: list[dict[str, Any]] = [
        {
            "trade_id": 1,
            "signal_time": "2025-07-20T10:00:00+00:00",
            "entry_time": "2025-07-20T10:15:00+00:00",
            "exit_time": "2025-07-20T10:30:00+00:00",
            "direction": "LONG",
            "net_pnl": 500.0,
            "entry_spread_pts": 5.0,
            "exit_spread_pts": 6.0,
            "exit_reason": "TAKE_PROFIT",
            "stop_loss": 1.1500,
            "take_profit": 1.1600,
        },
        {
            "trade_id": 2,
            "signal_time": "2025-08-15T12:00:00+00:00",
            "entry_time": "2025-08-15T12:15:00+00:00",
            "exit_time": "2025-08-15T12:45:00+00:00",
            "direction": "SHORT",
            "net_pnl": 300.0,
            "entry_spread_pts": 4.0,
            "exit_spread_pts": 5.0,
            "exit_reason": "TAKE_PROFIT",
            "stop_loss": 1.1700,
            "take_profit": 1.1550,
        },
        {
            "trade_id": 3,
            "signal_time": "2025-09-10T14:00:00+00:00",
            "entry_time": "2025-09-10T14:15:00+00:00",
            "exit_time": "2025-09-10T15:15:00+00:00",
            "direction": "LONG",
            "net_pnl": -200.0,
            "entry_spread_pts": 25.0,  # High spread
            "exit_spread_pts": 8.0,
            "exit_reason": "STOP_LOSS",
            "stop_loss": 1.1550,
            "take_profit": 1.1650,
        },
        {
            "trade_id": 4,
            "signal_time": "2025-10-05T22:00:00+00:00",  # Rollover window
            "entry_time": "2025-10-05T22:15:00+00:00",
            "exit_time": "2025-10-05T23:00:00+00:00",
            "direction": "SHORT",
            "net_pnl": -100.0,
            "entry_spread_pts": 30.0,
            "exit_spread_pts": 15.0,
            "exit_reason": "STOP_LOSS",
            "stop_loss": 1.1650,
            "take_profit": 1.1500,
        },
        {
            "trade_id": 5,
            "signal_time": "2025-12-01T08:00:00+00:00",
            "entry_time": "2025-12-01T08:15:00+00:00",
            "exit_time": "2025-12-01T09:15:00+00:00",
            "direction": "LONG",
            "net_pnl": 50.0,
            "entry_spread_pts": 5.0,
            "exit_spread_pts": 5.0,
            "exit_reason": "MAX_HOLD",
            "stop_loss": 1.1520,
            "take_profit": 1.1620,
        },
    ]
    return pd.DataFrame(rows)


def test_calculate_pnl_concentration():
    """Verify concentration metrics calculation and sign change logic."""
    df = _make_dummy_ledger()
    # PnLs: 500, 300, 50, -100, -200. Total = 550.
    res = calculate_pnl_concentration(df)

    assert res["trade_count"] == 5
    assert res["total_net_pnl"] == 550.0
    assert res["mean_trade_pnl"] == 110.0
    assert res["median_trade_pnl"] == 50.0
    assert res["top_1_contribution_usd"] == 500.0
    assert round(res["top_1_pct_of_total"], 2) == round((500.0 / 550.0) * 100.0, 2)
    assert res["top_3_contribution_usd"] == 850.0  # 500 + 300 + 50
    # Removing top 1: 550 - 500 = +50 (no sign change)
    assert res["pnl_without_top_1"] == 50.0
    assert res["sign_change_removing_top_1"] is False
    # Removing top 3: 550 - 850 = -300 (sign change: positive to negative)
    assert res["pnl_without_top_3"] == -300.0
    assert res["sign_change_removing_top_3"] is True


def test_compute_temporal_blocks():
    """Verify chronological partitioning into 5 blocks."""
    df = _make_dummy_ledger()
    start = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, 0, 0, 0, tzinfo=timezone.utc)
    blocks = compute_temporal_blocks(df, n_blocks=5, start_dt=start, end_dt=end)

    assert len(blocks) == 5
    total_trades = sum(b["trades"] for b in blocks)
    assert total_trades == 5
    total_pnl = sum(b["net_pnl"] for b in blocks)
    assert round(total_pnl, 2) == 550.0


def test_compute_monthly_breakdown():
    """Verify calendar month grouping."""
    df = _make_dummy_ledger()
    months = compute_monthly_breakdown(df)

    # Months: 2025-07, 2025-08, 2025-09, 2025-10, 2025-12
    assert len(months) == 5
    month_names = [m["month"] for m in months]
    assert month_names == ["2025-07", "2025-08", "2025-09", "2025-10", "2025-12"]
    assert months[0]["net_pnl"] == 500.0


def test_compute_exit_event_analysis():
    """Verify breakdown by exit reason and positive trade sources."""
    df = _make_dummy_ledger()
    res = compute_exit_event_analysis(df)

    by_reason = res["by_exit_reason"]
    assert by_reason["TAKE_PROFIT"]["count"] == 2
    assert by_reason["TAKE_PROFIT"]["total_pnl"] == 800.0
    assert by_reason["STOP_LOSS"]["count"] == 2
    assert by_reason["STOP_LOSS"]["total_pnl"] == -300.0
    assert by_reason["MAX_HOLD"]["count"] == 1
    assert by_reason["MAX_HOLD"]["total_pnl"] == 50.0

    sources = res["positive_trades_sources"]
    assert sources["total_positive_trades"] == 3  # 500, 300, 50
    assert sources["from_tp"] == 2
    assert sources["from_max_hold"] == 1


def test_compute_collision_breakdown():
    """Verify divergence categorization."""
    rec_df = pd.DataFrame(
        [
            {
                "candle_exit_reason": "STOP_LOSS",
                "tick_exit_reason": "TAKE_PROFIT",
                "pnl_diff": 100.0,
            },
            {
                "candle_exit_reason": "MAX_HOLD",
                "tick_exit_reason": "TAKE_PROFIT",
                "pnl_diff": 250.0,
            },
            {
                "candle_exit_reason": "TAKE_PROFIT",
                "tick_exit_reason": "STOP_LOSS",
                "pnl_diff": -80.0,
            },
            {
                "candle_exit_reason": "MAX_HOLD",
                "tick_exit_reason": "MAX_HOLD",
                "pnl_diff": 10.0,
            },
        ]
    )
    res = compute_collision_breakdown(rec_df)
    assert res["total_divergence_usd"] == 280.0  # 100 + 250 - 80 + 10
    cats = {c["category"]: c for c in res["categories"]}
    assert cats["Candle SL -> Tick TP"]["trade_count"] == 1
    assert cats["Candle MAX_HOLD -> Tick TP"]["trade_count"] == 1
    assert cats["Candle TP -> Tick SL"]["trade_count"] == 1
    assert cats["Other / Same Event Diff"]["trade_count"] == 1


def test_compute_rollover_analysis():
    """Verify categorization by rollover window and high-spread regimes."""
    df = _make_dummy_ledger()
    # Threshold 20 pts:
    # Trade 3: entry_spread = 25 (Category A)
    # Trade 4: entry_time = 22:00 UTC (Category A)
    # Trades 1, 2, 5: Category C
    res = compute_rollover_analysis(df, high_spread_threshold_pts=20.0)

    assert res["A_entry_near_high_spread"]["trade_count"] == 2
    assert res["A_entry_near_high_spread"]["total_net_pnl"] == -300.0  # -200 + -100
    assert res["C_neither"]["trade_count"] == 3
    assert res["C_neither"]["total_net_pnl"] == 850.0  # 500 + 300 + 50


def test_compute_bootstrap_distribution():
    """Verify bootstrap resampling determinism with random_state=42."""
    df = _make_dummy_ledger()
    boot1 = compute_bootstrap_distribution(df, n_iterations=1000, random_state=42)
    boot2 = compute_bootstrap_distribution(df, n_iterations=1000, random_state=42)

    assert boot1["mean_trade_pnl"]["point_estimate"] == 110.0
    assert boot1["mean_trade_pnl"]["ci_95_lower"] == boot2["mean_trade_pnl"]["ci_95_lower"]
    assert boot1["mean_trade_pnl"]["ci_95_upper"] == boot2["mean_trade_pnl"]["ci_95_upper"]
    assert boot1["prob_total_pnl_positive_pct"] == boot2["prob_total_pnl_positive_pct"]


def test_verify_test_partition_rejection():
    """Verify strict ValueError when evaluating locked holdout data."""
    from datetime import timedelta

    valid_ts = datetime(2025, 12, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Should pass without error
    assert verify_test_partition_rejection(valid_ts) is True

    # Data after lock timestamp must raise ValueError
    after_lock_ts = PHASE11_TEST_LOCK_TIMESTAMP + timedelta(seconds=1)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(after_lock_ts)

    future_ts = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="CRITICAL TEST PARTITION VIOLATION"):
        verify_test_partition_rejection(future_ts)


def test_phase21_artifacts_consistency():
    """Verify that generated Phase 21 report artifacts exist and match reproducible metrics."""
    reports_dir = Path("reports")
    ledger_path = reports_dir / "phase21_trade_ledger.csv"
    if not ledger_path.exists():
        pytest.skip("reports/phase21_trade_ledger.csv not generated yet")

    ledger_df = pd.read_csv(ledger_path)
    assert len(ledger_df) == 60
    assert abs(ledger_df["net_pnl"].sum() - 1545.43) < 0.10

    # Verify concentration matches exactly
    conc = calculate_pnl_concentration(ledger_df)
    assert conc["top_1_contribution_usd"] == 561.06
    assert conc["top_5_contribution_usd"] == 2069.83
    assert conc["sign_change_removing_top_5"] is True
    assert conc["pnl_without_top_5"] == -524.40

    # Verify temporal blocks
    blocks = compute_temporal_blocks(
        ledger_df,
        n_blocks=5,
        start_dt=VALIDATION_START_TIMESTAMP,
        end_dt=VALIDATION_END_TIMESTAMP,
    )
    assert len(blocks) == 5
    assert blocks[0]["net_pnl"] == 1707.30
    assert blocks[1]["net_pnl"] == -143.32
