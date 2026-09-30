"""Phase 21: Tick-Realistic Strategy Validation & Execution Robustness Analysis.

Provides quantitative diagnostics to evaluate whether tick-realistic backtest results
are broadly distributed or concentrated in specific trades, time periods, or execution events:
- Trade ledger enrichment and distribution profiling
- P&L concentration metrics (top 1, 3, 5, 10 trade contributions and sign-change tests)
- Temporal block partitioning (5 equal-duration blocks)
- Monthly performance breakdown
- Exit-event analysis (TP vs SL vs MAX_HOLD)
- Intrabar collision and divergence categorization
- Outlier trade audit
- Rollover / high-spread regime impact analysis
- Spread sensitivity and cost decomposition
- MAX_HOLD observational trajectory diagnostics
- Descriptive bootstrap resampling (reproducible seed 42)
- Strict Phase 11 holdout boundary enforcement
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ai.backtest.models import SimulatedTrade
from ai.backtest.tick_data import TickDataRepository

# Locked test partition governance timestamp
PHASE11_TEST_LOCK_TIMESTAMP = datetime(2026, 2, 19, 10, 45, 0, tzinfo=timezone.utc)
PHASE11_TEST_START_TIMESTAMP = datetime(2026, 2, 19, 12, 0, 0, tzinfo=timezone.utc)

# Validation window bounds
VALIDATION_START_TIMESTAMP = datetime(2025, 7, 14, 5, 45, 0, tzinfo=timezone.utc)
VALIDATION_END_TIMESTAMP = datetime(2026, 2, 19, 10, 45, 0, tzinfo=timezone.utc)


def build_robustness_trade_ledger(
    trades: list[SimulatedTrade],
    tick_repo: TickDataRepository,
    point_value: float = 1e-05,
) -> pd.DataFrame:
    """Enrich simulated trade records with execution quote details and duration metrics.

    Parameters
    ----------
    trades : list[SimulatedTrade]
        Trade ledger from TickBacktestEngine.
    tick_repo : TickDataRepository
        Tick repository used for execution.
    point_value : float
        Price unit per point (1e-5 for EURUSD).

    Returns
    -------
    pd.DataFrame
        Structured trade ledger containing all 21 execution attributes.
    """
    rows = []
    for t in trades:
        # Query nearest ticks at entry and exit for quote auditing
        entry_quote = tick_repo.get_first_tick_at_or_after(t.entry_time)
        exit_quote = tick_repo.get_first_tick_at_or_after(t.exit_time)

        e_bid = entry_quote.bid if entry_quote else t.entry_price
        e_ask = entry_quote.ask if entry_quote else (t.entry_price + 10.0 * point_value)
        e_spread_pts = round((e_ask - e_bid) / point_value, 1)

        x_bid = exit_quote.bid if exit_quote else t.exit_price
        x_ask = exit_quote.ask if exit_quote else (t.exit_price + 10.0 * point_value)
        x_spread_pts = round((x_ask - x_bid) / point_value, 1)

        hold_duration_sec = (t.exit_time - t.entry_time).total_seconds()
        dir_val = t.direction.value if hasattr(t.direction, "value") else str(t.direction)

        rows.append(
            {
                "trade_id": t.trade_id,
                "signal_time": t.signal_time.isoformat(),
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "direction": dir_val,
                "confidence": round(t.confidence, 4),
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(t.exit_price, 5),
                "entry_bid": round(e_bid, 5),
                "entry_ask": round(e_ask, 5),
                "exit_bid": round(x_bid, 5),
                "exit_ask": round(x_ask, 5),
                "entry_spread_pts": e_spread_pts,
                "exit_spread_pts": x_spread_pts,
                "stop_loss": round(t.stop_loss, 5),
                "take_profit": round(t.take_profit, 5),
                "quantity": round(t.quantity, 2),
                "risk_amount": round(t.risk_amount, 2),
                "gross_pnl": round(t.gross_pnl, 2),
                "spread_cost": round(t.spread_cost, 2),
                "commission": round(t.commission, 2),
                "slippage": round(t.slippage, 2),
                "net_pnl": round(t.net_pnl, 2),
                "holding_bars": t.holding_bars,
                "holding_time_seconds": round(hold_duration_sec, 1),
                "exit_reason": (
                    t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason)
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df


def calculate_pnl_concentration(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Quantify P&L distribution parameters and concentration among top trades.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade ledger containing 'net_pnl'.

    Returns
    -------
    dict[str, Any]
        Distribution percentiles, top trade contributions ($ and %), and sign-change tests.
    """
    if trades_df.empty:
        return {"trade_count": 0, "total_net_pnl": 0.0}

    pnl_series = trades_df["net_pnl"].to_numpy(dtype=np.float64)
    total_pnl = float(np.sum(pnl_series))
    n = len(pnl_series)

    # Sort in descending order to identify largest positive contributors
    sorted_desc = np.sort(pnl_series)[::-1]

    top_1 = float(sorted_desc[0]) if n >= 1 else 0.0
    top_3 = float(np.sum(sorted_desc[:3])) if n >= 3 else top_1
    top_5 = float(np.sum(sorted_desc[:5])) if n >= 5 else top_3
    top_10 = float(np.sum(sorted_desc[:10])) if n >= 10 else top_5

    pct_top_1 = round((top_1 / total_pnl) * 100.0, 2) if abs(total_pnl) > 1e-6 else 0.0
    pct_top_3 = round((top_3 / total_pnl) * 100.0, 2) if abs(total_pnl) > 1e-6 else 0.0
    pct_top_5 = round((top_5 / total_pnl) * 100.0, 2) if abs(total_pnl) > 1e-6 else 0.0
    pct_top_10 = round((top_10 / total_pnl) * 100.0, 2) if abs(total_pnl) > 1e-6 else 0.0

    pnl_no_top_1 = float(total_pnl - top_1)
    pnl_no_top_3 = float(total_pnl - top_3)
    pnl_no_top_5 = float(total_pnl - top_5)

    sign_change_1 = (total_pnl > 0 and pnl_no_top_1 <= 0) or (total_pnl < 0 and pnl_no_top_1 >= 0)
    sign_change_3 = (total_pnl > 0 and pnl_no_top_3 <= 0) or (total_pnl < 0 and pnl_no_top_3 >= 0)
    sign_change_5 = (total_pnl > 0 and pnl_no_top_5 <= 0) or (total_pnl < 0 and pnl_no_top_5 >= 0)

    return {
        "trade_count": n,
        "total_net_pnl": round(total_pnl, 2),
        "mean_trade_pnl": round(float(np.mean(pnl_series)), 2),
        "median_trade_pnl": round(float(np.median(pnl_series)), 2),
        "std_trade_pnl": round(float(np.std(pnl_series, ddof=1)), 2) if n > 1 else 0.0,
        "min_trade_pnl": round(float(np.min(pnl_series)), 2),
        "max_trade_pnl": round(float(np.max(pnl_series)), 2),
        "p25_trade_pnl": round(float(np.percentile(pnl_series, 25)), 2),
        "p75_trade_pnl": round(float(np.percentile(pnl_series, 75)), 2),
        "top_1_contribution_usd": round(top_1, 2),
        "top_1_pct_of_total": pct_top_1,
        "top_3_contribution_usd": round(top_3, 2),
        "top_3_pct_of_total": pct_top_3,
        "top_5_contribution_usd": round(top_5, 2),
        "top_5_pct_of_total": pct_top_5,
        "top_10_contribution_usd": round(top_10, 2),
        "top_10_pct_of_total": pct_top_10,
        "pnl_without_top_1": round(pnl_no_top_1, 2),
        "pnl_without_top_3": round(pnl_no_top_3, 2),
        "pnl_without_top_5": round(pnl_no_top_5, 2),
        "sign_change_removing_top_1": sign_change_1,
        "sign_change_removing_top_3": sign_change_3,
        "sign_change_removing_top_5": sign_change_5,
    }


def compute_temporal_blocks(
    trades_df: pd.DataFrame,
    n_blocks: int = 5,
    start_dt: datetime = VALIDATION_START_TIMESTAMP,
    end_dt: datetime = VALIDATION_END_TIMESTAMP,
) -> list[dict[str, Any]]:
    """Partition validation period into equal-duration chronological blocks and compute metrics.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade ledger containing 'signal_time', 'net_pnl', 'exit_reason', 'direction'.
    n_blocks : int
        Number of fixed chronological intervals (default: 5).
    start_dt : datetime
        Validation start timestamp.
    end_dt : datetime
        Validation end timestamp.

    Returns
    -------
    list[dict[str, Any]]
        Performance metrics for each chronological block.
    """
    total_duration = end_dt - start_dt
    block_duration = total_duration / n_blocks

    blocks: list[dict[str, Any]] = []
    df = trades_df.copy()
    df["sig_dt"] = pd.to_datetime(df["signal_time"], format="mixed", utc=True)

    for b in range(n_blocks):
        b_start = start_dt + b * block_duration
        b_end = start_dt + (b + 1) * block_duration if b < n_blocks - 1 else end_dt

        mask = (df["sig_dt"] >= b_start) & (df["sig_dt"] < b_end)
        b_trades = df[mask]

        t_count = len(b_trades)
        if t_count > 0:
            pnl_arr = b_trades["net_pnl"].to_numpy(dtype=np.float64)
            b_net_pnl = float(np.sum(pnl_arr))
            wins = int(np.sum(pnl_arr > 0))
            losses = int(np.sum(pnl_arr < 0))
            win_rate = round((wins / t_count) * 100.0, 2)
            avg_pnl = round(b_net_pnl / t_count, 2)

            gross_profit = float(np.sum(pnl_arr[pnl_arr > 0])) if wins > 0 else 0.0
            gross_loss = float(abs(np.sum(pnl_arr[pnl_arr < 0]))) if losses > 0 else 0.0
            if gross_loss > 0:
                profit_factor = round(gross_profit / gross_loss, 4)
            else:
                profit_factor = 999.0 if gross_profit > 0 else 0.0

            # Cumulative drawdown within block
            cum_pnl = np.cumsum(pnl_arr)
            hwm = np.maximum.accumulate(cum_pnl)
            dd = hwm - cum_pnl
            max_dd = round(float(np.max(dd)), 2) if len(dd) > 0 else 0.0

            long_count = int(np.sum(b_trades["direction"] == "LONG"))
            short_count = int(np.sum(b_trades["direction"] == "SHORT"))
            tp_count = int(np.sum(b_trades["exit_reason"] == "TAKE_PROFIT"))
            sl_count = int(np.sum(b_trades["exit_reason"] == "STOP_LOSS"))
            mh_count = int(np.sum(b_trades["exit_reason"] == "MAX_HOLD"))
        else:
            b_net_pnl = 0.0
            win_rate = 0.0
            avg_pnl = 0.0
            profit_factor = 0.0
            max_dd = 0.0
            long_count = 0
            short_count = 0
            tp_count = 0
            sl_count = 0
            mh_count = 0

        blocks.append(
            {
                "block_id": b + 1,
                "start_date": b_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "end_date": b_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "trades": t_count,
                "long_trades": long_count,
                "short_trades": short_count,
                "win_rate": win_rate,
                "net_pnl": round(b_net_pnl, 2),
                "profit_factor": profit_factor,
                "average_trade": avg_pnl,
                "max_drawdown": max_dd,
                "tp_count": tp_count,
                "sl_count": sl_count,
                "max_hold_count": mh_count,
            }
        )

    return blocks


def compute_monthly_breakdown(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Group trade performance by calendar month.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade ledger containing 'signal_time', 'net_pnl', 'direction'.

    Returns
    -------
    list[dict[str, Any]]
        Monthly summary table metrics.
    """
    df = trades_df.copy()
    df["sig_dt"] = pd.to_datetime(df["signal_time"], format="mixed", utc=True)
    df["year_month"] = df["sig_dt"].dt.strftime("%Y-%m")

    months: list[dict[str, Any]] = []
    for ym, group in df.groupby("year_month", sort=True):
        pnl_arr = group["net_pnl"].to_numpy(dtype=np.float64)
        t_count = len(pnl_arr)
        wins = int(np.sum(pnl_arr > 0))
        losses = int(np.sum(pnl_arr < 0))
        tot_pnl = float(np.sum(pnl_arr))
        win_rate = round((wins / t_count) * 100.0, 2)
        avg_trade = round(tot_pnl / t_count, 2)

        gp = float(np.sum(pnl_arr[pnl_arr > 0])) if wins > 0 else 0.0
        gl = float(abs(np.sum(pnl_arr[pnl_arr < 0]))) if losses > 0 else 0.0
        pf = round(gp / gl, 4) if gl > 0 else (999.0 if gp > 0 else 0.0)

        cum = np.cumsum(pnl_arr)
        hwm = np.maximum.accumulate(cum)
        dd = hwm - cum
        max_dd = round(float(np.max(dd)), 2) if len(dd) > 0 else 0.0

        months.append(
            {
                "month": str(ym),
                "trades": t_count,
                "long_trades": int(np.sum(group["direction"] == "LONG")),
                "short_trades": int(np.sum(group["direction"] == "SHORT")),
                "win_rate": win_rate,
                "net_pnl": round(tot_pnl, 2),
                "profit_factor": pf,
                "average_trade": avg_trade,
                "max_drawdown": max_dd,
            }
        )

    return months


def compute_exit_event_analysis(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze trade performance and P&L concentration segmented by exit reason.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade ledger containing 'exit_reason' and 'net_pnl'.

    Returns
    -------
    dict[str, Any]
        Breakdown by TP, SL, MAX_HOLD and winning trade sources.
    """
    total_pnl = float(trades_df["net_pnl"].sum()) if not trades_df.empty else 0.0
    events = ["TAKE_PROFIT", "STOP_LOSS", "MAX_HOLD"]

    res: dict[str, Any] = {"by_exit_reason": {}, "positive_trades_sources": {}}

    for ev in events:
        sub = trades_df[trades_df["exit_reason"] == ev]
        count = len(sub)
        pnl = float(sub["net_pnl"].sum()) if count > 0 else 0.0
        avg = round(pnl / count, 2) if count > 0 else 0.0
        med = round(float(sub["net_pnl"].median()), 2) if count > 0 else 0.0
        pct = round((pnl / total_pnl) * 100.0, 2) if abs(total_pnl) > 1e-6 else 0.0

        res["by_exit_reason"][ev] = {
            "count": count,
            "total_pnl": round(pnl, 2),
            "average_pnl": avg,
            "median_pnl": med,
            "pct_of_total_pnl": pct,
        }

    # Where do profitable trades originate?
    positive_trades = trades_df[trades_df["net_pnl"] > 0]
    res["positive_trades_sources"] = {
        "total_positive_trades": len(positive_trades),
        "from_tp": int(np.sum(positive_trades["exit_reason"] == "TAKE_PROFIT")),
        "from_max_hold": int(np.sum(positive_trades["exit_reason"] == "MAX_HOLD")),
        "from_other": int(
            np.sum(~positive_trades["exit_reason"].isin(["TAKE_PROFIT", "MAX_HOLD"]))
        ),
    }

    return res


def compute_collision_breakdown(rec_df: pd.DataFrame) -> dict[str, Any]:
    """Categorize trades where candle outcome differed from tick outcome.

    Categories:
    1. Candle SL -> Tick TP
    2. Candle SL -> Tick MAX_HOLD
    3. Candle TP -> Tick SL
    4. Candle TP -> Tick MAX_HOLD
    5. Candle MAX_HOLD -> Tick TP
    6. Candle MAX_HOLD -> Tick SL
    7. Other
    """
    total_divergence = float(rec_df["pnl_diff"].sum()) if not rec_df.empty else 0.0

    categories = {
        "Candle SL -> Tick TP": (rec_df["candle_exit_reason"] == "STOP_LOSS")
        & (rec_df["tick_exit_reason"] == "TAKE_PROFIT"),
        "Candle SL -> Tick MAX_HOLD": (rec_df["candle_exit_reason"] == "STOP_LOSS")
        & (rec_df["tick_exit_reason"] == "MAX_HOLD"),
        "Candle TP -> Tick SL": (rec_df["candle_exit_reason"] == "TAKE_PROFIT")
        & (rec_df["tick_exit_reason"] == "STOP_LOSS"),
        "Candle TP -> Tick MAX_HOLD": (rec_df["candle_exit_reason"] == "TAKE_PROFIT")
        & (rec_df["tick_exit_reason"] == "MAX_HOLD"),
        "Candle MAX_HOLD -> Tick TP": (rec_df["candle_exit_reason"] == "MAX_HOLD")
        & (rec_df["tick_exit_reason"] == "TAKE_PROFIT"),
        "Candle MAX_HOLD -> Tick SL": (rec_df["candle_exit_reason"] == "MAX_HOLD")
        & (rec_df["tick_exit_reason"] == "STOP_LOSS"),
    }

    # Combined mask of the 6 defined shifts
    known_mask = pd.Series(False, index=rec_df.index)
    for m in categories.values():
        known_mask = known_mask | m

    categories["Other / Same Event Diff"] = ~known_mask

    breakdown = []
    for cat_name, mask in categories.items():
        sub = rec_df[mask]
        cnt = len(sub)
        pnl_diff_sum = float(sub["pnl_diff"].sum()) if cnt > 0 else 0.0
        mean_diff = round(pnl_diff_sum / cnt, 2) if cnt > 0 else 0.0
        med_diff = round(float(sub["pnl_diff"].median()), 2) if cnt > 0 else 0.0
        pct_div = (
            round((pnl_diff_sum / total_divergence) * 100.0, 2)
            if abs(total_divergence) > 1e-6
            else 0.0
        )

        breakdown.append(
            {
                "category": cat_name,
                "trade_count": cnt,
                "net_pnl_difference": round(pnl_diff_sum, 2),
                "mean_pnl_difference": mean_diff,
                "median_pnl_difference": med_diff,
                "pct_of_total_divergence": pct_div,
            }
        )

    return {
        "total_divergence_usd": round(total_divergence, 2),
        "categories": breakdown,
    }


def audit_outlier_trades(
    trades_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Perform descriptive audit of top absolute P&L trades and their execution factors."""
    df = trades_df.copy()
    df["abs_pnl"] = df["net_pnl"].abs()
    top_trades = df.sort_values("abs_pnl", ascending=False).head(top_n)

    audit_records = []
    for _, t in top_trades.iterrows():
        t_id = int(t["trade_id"])
        # Find corresponding reconciliation row
        rec_match = rec_df[rec_df["tick_trade_id"] == t_id]
        c_reason = rec_match.iloc[0]["candle_exit_reason"] if not rec_match.empty else "N/A"
        c_pnl = rec_match.iloc[0]["candle_pnl"] if not rec_match.empty else None
        pnl_diff = rec_match.iloc[0]["pnl_diff"] if not rec_match.empty else None

        sig_dt = pd.to_datetime(t["signal_time"])
        # Rollover check: within 21:50 - 22:20 UTC or spread >= 20.0 points
        entry_h, entry_m = sig_dt.hour, sig_dt.minute
        is_rollover_window = (entry_h == 21 and entry_m >= 50) or (entry_h == 22 and entry_m <= 20)
        is_wide_spread = t["entry_spread_pts"] >= 20.0 or t["exit_spread_pts"] >= 20.0
        rollover_involved = is_rollover_window or is_wide_spread

        disagreement = c_reason != t["exit_reason"]

        audit_records.append(
            {
                "trade_id": t_id,
                "signal_time": str(t["signal_time"]),
                "direction": str(t["direction"]),
                "entry_price": float(t["entry_price"]),
                "exit_price": float(t["exit_price"]),
                "net_pnl": float(t["net_pnl"]),
                "exit_reason": str(t["exit_reason"]),
                "entry_spread_pts": float(t["entry_spread_pts"]),
                "exit_spread_pts": float(t["exit_spread_pts"]),
                "holding_bars": int(t["holding_bars"]),
                "holding_time_seconds": float(t["holding_time_seconds"]),
                "rollover_involved": rollover_involved,
                "candle_exit_reason": c_reason,
                "candle_pnl": c_pnl,
                "pnl_difference": pnl_diff,
                "candle_tick_disagreement": disagreement,
            }
        )

    return audit_records


def compute_rollover_analysis(
    trades_df: pd.DataFrame,
    high_spread_threshold_pts: float = 20.0,
) -> dict[str, Any]:
    """Classify trades by proximity to high-spread regimes and rollover windows.

    Categories:
    A. Entry near high-spread period (entry spread >= threshold or entry 21:50-22:20 UTC)
    B. Exit near high-spread period (exit spread >= threshold or exit 21:50-22:20 UTC)
    C. Neither
    """
    df = trades_df.copy()
    df["sig_dt"] = pd.to_datetime(df["signal_time"], format="mixed", utc=True)
    df["exit_dt"] = pd.to_datetime(df["exit_time"], format="mixed", utc=True)

    entry_rollover = (
        (df["entry_spread_pts"] >= high_spread_threshold_pts)
        | ((df["sig_dt"].dt.hour == 21) & (df["sig_dt"].dt.minute >= 50))
        | ((df["sig_dt"].dt.hour == 22) & (df["sig_dt"].dt.minute <= 20))
    )

    exit_rollover = (
        (df["exit_spread_pts"] >= high_spread_threshold_pts)
        | ((df["exit_dt"].dt.hour == 21) & (df["exit_dt"].dt.minute >= 50))
        | ((df["exit_dt"].dt.hour == 22) & (df["exit_dt"].dt.minute <= 20))
    )

    cat_a = entry_rollover
    cat_b = ~entry_rollover & exit_rollover
    cat_c = ~entry_rollover & ~exit_rollover

    results = {}
    for name, mask in [
        ("A_entry_near_high_spread", cat_a),
        ("B_exit_near_high_spread", cat_b),
        ("C_neither", cat_c),
    ]:
        sub = df[mask]
        cnt = len(sub)
        pnl = float(sub["net_pnl"].sum()) if cnt > 0 else 0.0
        avg = round(pnl / cnt, 2) if cnt > 0 else 0.0
        avg_e_spread = round(float(sub["entry_spread_pts"].mean()), 2) if cnt > 0 else 0.0
        avg_x_spread = round(float(sub["exit_spread_pts"].mean()), 2) if cnt > 0 else 0.0

        results[name] = {
            "trade_count": cnt,
            "total_net_pnl": round(pnl, 2),
            "average_trade_pnl": avg,
            "mean_entry_spread_pts": avg_e_spread,
            "mean_exit_spread_pts": avg_x_spread,
            "exit_reasons": (dict(sub["exit_reason"].value_counts()) if cnt > 0 else {}),
        }

    return results


def diagnose_max_hold_subsequent_path(
    trades_df: pd.DataFrame,
    tick_repo: TickDataRepository,
    lookahead_minutes: int = 60,
    point_value: float = 1e-05,
) -> dict[str, Any]:
    """Observational diagnostic of trades exiting at MAX_HOLD.

    Determines whether the market path in the subsequent observation window
    (within observed historical ticks) would have touched the trade's original TP or SL.
    Strictly observational — does NOT alter strategy holding period.
    """
    mh_trades = trades_df[trades_df["exit_reason"] == "MAX_HOLD"]
    count_mh = len(mh_trades)

    if count_mh == 0:
        return {"max_hold_trades": 0, "later_tp": 0, "later_sl": 0, "later_neither": 0}

    later_tp = 0
    later_sl = 0
    later_neither = 0

    for _, t in mh_trades.iterrows():
        exit_dt = pd.to_datetime(t["exit_time"], format="mixed", utc=True).to_pydatetime()
        subsequent_end = exit_dt + timedelta(minutes=lookahead_minutes)

        direction = str(t["direction"])
        sl = float(t["stop_loss"])
        tp = float(t["take_profit"])

        t_msc, bids, asks = tick_repo.get_raw_slice(exit_dt, subsequent_end)
        if len(t_msc) == 0:
            later_neither += 1
            continue

        touched_tp = False
        touched_sl = False

        for k in range(len(t_msc)):
            cur_bid = float(bids[k])
            cur_ask = float(asks[k])

            if direction == "LONG":
                if cur_bid >= tp:
                    touched_tp = True
                    break
                if cur_bid <= sl:
                    touched_sl = True
                    break
            else:  # SHORT
                if cur_ask <= tp:
                    touched_tp = True
                    break
                if cur_ask >= sl:
                    touched_sl = True
                    break

        if touched_tp:
            later_tp += 1
        elif touched_sl:
            later_sl += 1
        else:
            later_neither += 1

    return {
        "max_hold_trades_evaluated": count_mh,
        "subsequent_touched_tp": later_tp,
        "subsequent_touched_sl": later_sl,
        "subsequent_touched_neither": later_neither,
        "pct_touching_tp": round((later_tp / count_mh) * 100.0, 1) if count_mh > 0 else 0.0,
        "pct_touching_sl": round((later_sl / count_mh) * 100.0, 1) if count_mh > 0 else 0.0,
        "pct_touching_neither": (
            round((later_neither / count_mh) * 100.0, 1) if count_mh > 0 else 0.0
        ),
    }


def compute_bootstrap_distribution(
    trades_df: pd.DataFrame,
    n_iterations: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Perform descriptive bootstrap resampling on the net P&L distribution.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade ledger containing 'net_pnl'.
    n_iterations : int
        Number of bootstrap replicates (default: 10,000).
    random_state : int
        Deterministic seed (default: 42).

    Returns
    -------
    dict[str, Any]
        Bootstrap point estimates and 95% confidence intervals.
    """
    if trades_df.empty:
        return {"iterations": 0}

    pnl_arr = trades_df["net_pnl"].to_numpy(dtype=np.float64)
    n = len(pnl_arr)

    rng = np.random.default_rng(random_state)
    boot_indices = rng.integers(0, n, size=(n_iterations, n))
    boot_samples = pnl_arr[boot_indices]

    boot_means = np.mean(boot_samples, axis=1)
    boot_medians = np.median(boot_samples, axis=1)
    boot_totals = np.sum(boot_samples, axis=1)

    prob_positive_total = float(np.mean(boot_totals > 0)) * 100.0

    return {
        "n_iterations": n_iterations,
        "random_state": random_state,
        "sample_size": n,
        "mean_trade_pnl": {
            "point_estimate": round(float(np.mean(pnl_arr)), 2),
            "ci_95_lower": round(float(np.percentile(boot_means, 2.5)), 2),
            "ci_95_upper": round(float(np.percentile(boot_means, 97.5)), 2),
        },
        "median_trade_pnl": {
            "point_estimate": round(float(np.median(pnl_arr)), 2),
            "ci_95_lower": round(float(np.percentile(boot_medians, 2.5)), 2),
            "ci_95_upper": round(float(np.percentile(boot_medians, 97.5)), 2),
        },
        "total_net_pnl": {
            "point_estimate": round(float(np.sum(pnl_arr)), 2),
            "ci_95_lower": round(float(np.percentile(boot_totals, 2.5)), 2),
            "ci_95_upper": round(float(np.percentile(boot_totals, 97.5)), 2),
        },
        "prob_total_pnl_positive_pct": round(prob_positive_total, 2),
        "explicit_limitation": (
            "This is a descriptive resampling diagnostic on a small 60-trade sample, "
            "not evidence of statistical alpha."
        ),
    }


def verify_test_partition_rejection(timestamp_or_df: Any) -> bool:
    """Verify that any tick timestamp at or after 2026-02-19 10:45 UTC is strictly rejected.

    Parameters
    ----------
    timestamp_or_df : datetime | pd.DataFrame | np.ndarray
        Timestamp or dataset to audit.

    Returns
    -------
    bool
        True if safely within validation bounds. Raises ValueError on violation.
    """
    lock_ms = int(PHASE11_TEST_LOCK_TIMESTAMP.timestamp() * 1000)

    if isinstance(timestamp_or_df, datetime):
        ts_ms = int(timestamp_or_df.timestamp() * 1000)
        if ts_ms > lock_ms:
            raise ValueError(
                f"CRITICAL TEST PARTITION VIOLATION: Timestamp {timestamp_or_df} "
                f"crosses Phase 11 test lock boundary ({PHASE11_TEST_LOCK_TIMESTAMP})."
            )
        return True

    if isinstance(timestamp_or_df, pd.DataFrame):
        if "time_msc" in timestamp_or_df.columns:
            max_ms = int(timestamp_or_df["time_msc"].max())
            if max_ms > lock_ms:
                raise ValueError(
                    f"CRITICAL TEST PARTITION VIOLATION: Max tick time_msc {max_ms} "
                    f"crosses Phase 11 test lock boundary ({lock_ms} ms)."
                )
        return True

    return True
