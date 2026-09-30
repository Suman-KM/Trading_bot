"""Trade-level reconciliation and difference attribution between candle and tick engines.

Compares simulated trade ledgers from CandleBacktestEngine and TickBacktestEngine,
matching trades by signal timestamp and direction, and attributing variances to:
- Entry price differences (bar open vs first tick Ask/Bid)
- Exit price differences
- Exit event differences (SL vs TP vs MAX_HOLD priority)
- Max-hold timing differences (bar close vs last tick)
- Spread cost variability (static bar spread vs dynamic tick spread)
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def reconcile_candle_vs_tick(
    candle_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    point_value: float = 1e-05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match trades between candle and tick backtests and categorize differences.

    Parameters
    ----------
    candle_df : pd.DataFrame
        Trade ledger from CandleBacktestEngine.
    tick_df : pd.DataFrame
        Trade ledger from TickBacktestEngine.
    point_value : float
        Unit per point (1e-5 for EURUSD).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        (Reconciliation DataFrame, Summary Metrics Dictionary)
    """
    if candle_df.empty and tick_df.empty:
        return pd.DataFrame(), {
            "total_candle_trades": 0,
            "total_tick_trades": 0,
            "matched_trades": 0,
            "classification_counts": {},
        }

    # Normalize timestamps
    c_df = candle_df.copy()
    t_df = tick_df.copy()

    c_df["signal_time_str"] = pd.to_datetime(c_df["signal_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    t_df["signal_time_str"] = pd.to_datetime(t_df["signal_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Outer merge on signal_time and direction
    merged = pd.merge(
        c_df,
        t_df,
        on=["signal_time_str", "direction"],
        how="outer",
        suffixes=("_candle", "_tick"),
    )

    reconciled_rows = []
    classifications = {
        "IDENTICAL": 0,
        "ENTRY_PRICE_DIFF": 0,
        "EXIT_PRICE_DIFF": 0,
        "EXIT_EVENT_DIFF": 0,
        "MAX_HOLD_TIMING_DIFF": 0,
        "SPREAD_COST_DIFF": 0,
        "UNMATCHED_TRADE": 0,
    }

    for _, row in merged.iterrows():
        sig_time = row["signal_time_str"]
        direction = row["direction"]

        has_candle = pd.notna(row.get("trade_id_candle"))
        has_tick = pd.notna(row.get("trade_id_tick"))

        if not has_candle or not has_tick:
            classifications["UNMATCHED_TRADE"] += 1
            reconciled_rows.append(
                {
                    "signal_time": sig_time,
                    "direction": direction,
                    "candle_trade_id": row.get("trade_id_candle"),
                    "tick_trade_id": row.get("trade_id_tick"),
                    "classification": "UNMATCHED_TRADE",
                    "reason": "Trade present in only one engine",
                    "candle_exit_reason": row.get("exit_reason_candle"),
                    "tick_exit_reason": row.get("exit_reason_tick"),
                    "candle_pnl": row.get("net_pnl_candle"),
                    "tick_pnl": row.get("net_pnl_tick"),
                    "pnl_diff": None,
                    "candle_entry_price": row.get("entry_price_candle"),
                    "tick_entry_price": row.get("entry_price_tick"),
                    "entry_price_diff_pts": None,
                    "candle_exit_price": row.get("exit_price_candle"),
                    "tick_exit_price": row.get("exit_price_tick"),
                    "exit_price_diff_pts": None,
                }
            )
            continue

        c_entry = float(row["entry_price_candle"])
        t_entry = float(row["entry_price_tick"])
        c_exit = float(row["exit_price_candle"])
        t_exit = float(row["exit_price_tick"])
        c_reason = str(row["exit_reason_candle"])
        t_reason = str(row["exit_reason_tick"])
        c_pnl = float(row["net_pnl_candle"])
        t_pnl = float(row["net_pnl_tick"])
        c_hold = int(row["holding_bars_candle"])
        t_hold = int(row["holding_bars_tick"])

        entry_diff_pts = round((t_entry - c_entry) / point_value, 2)
        exit_diff_pts = round((t_exit - c_exit) / point_value, 2)
        pnl_diff = round(t_pnl - c_pnl, 2)

        # Classification hierarchy
        if c_reason != t_reason:
            cls_name = "EXIT_EVENT_DIFF"
            reason_desc = (
                f"Exit event changed: Candle={c_reason} vs Tick={t_reason} "
                f"(chronological tick path resolved actual touch order)"
            )
        elif c_reason == "MAX_HOLD" and (
            abs(entry_diff_pts) > 2.0 or abs(exit_diff_pts) > 2.0 or c_hold != t_hold
        ):
            cls_name = "MAX_HOLD_TIMING_DIFF"
            reason_desc = f"Max-hold variation: Candle exit={c_exit:.5f} vs Tick exit={t_exit:.5f}"
        elif abs(entry_diff_pts) > 2.0 and abs(exit_diff_pts) <= 1.0:
            cls_name = "ENTRY_PRICE_DIFF"
            reason_desc = (
                f"Entry fill differed by {entry_diff_pts} pts (bar open vs first tick quote)"
            )
        elif abs(exit_diff_pts) > 2.0:
            cls_name = "EXIT_PRICE_DIFF"
            reason_desc = f"Exit fill differed by {exit_diff_pts} pts"
        elif abs(pnl_diff) > 0.05:
            cls_name = "SPREAD_COST_DIFF"
            reason_desc = (
                f"PnL difference ${pnl_diff:+.2f} attributable to floating tick spread variation"
            )
        else:
            cls_name = "IDENTICAL"
            reason_desc = "Execution and outcome effectively identical"

        classifications[cls_name] += 1
        reconciled_rows.append(
            {
                "signal_time": sig_time,
                "direction": direction,
                "candle_trade_id": int(row["trade_id_candle"]),
                "tick_trade_id": int(row["trade_id_tick"]),
                "classification": cls_name,
                "reason": reason_desc,
                "candle_exit_reason": c_reason,
                "tick_exit_reason": t_reason,
                "candle_pnl": round(c_pnl, 2),
                "tick_pnl": round(t_pnl, 2),
                "pnl_diff": pnl_diff,
                "candle_entry_price": round(c_entry, 5),
                "tick_entry_price": round(t_entry, 5),
                "entry_price_diff_pts": entry_diff_pts,
                "candle_exit_price": round(c_exit, 5),
                "tick_exit_price": round(t_exit, 5),
                "exit_price_diff_pts": exit_diff_pts,
            }
        )

    rec_df = pd.DataFrame(reconciled_rows)

    matched = rec_df[rec_df["classification"] != "UNMATCHED_TRADE"]
    pnl_diffs = matched["pnl_diff"].dropna()

    candle_pnl_sum = float(candle_df["net_pnl"].sum()) if not candle_df.empty else 0.0
    tick_pnl_sum = float(tick_df["net_pnl"].sum()) if not tick_df.empty else 0.0
    mean_diff = float(pnl_diffs.mean()) if not pnl_diffs.empty else 0.0
    median_diff = float(pnl_diffs.median()) if not pnl_diffs.empty else 0.0
    min_diff = float(pnl_diffs.min()) if not pnl_diffs.empty else 0.0
    max_diff = float(pnl_diffs.max()) if not pnl_diffs.empty else 0.0

    summary: dict[str, Any] = {
        "total_candle_trades": len(candle_df),
        "total_tick_trades": len(tick_df),
        "matched_trades": len(matched),
        "unmatched_trades": classifications["UNMATCHED_TRADE"],
        "classifications": classifications,
        "candle_total_pnl": round(candle_pnl_sum, 2),
        "tick_total_pnl": round(tick_pnl_sum, 2),
        "net_pnl_difference": round(tick_pnl_sum - candle_pnl_sum, 2),
        "mean_pnl_diff_per_trade": round(mean_diff, 2),
        "median_pnl_diff_per_trade": round(median_diff, 2),
        "max_adverse_diff": round(min_diff, 2),
        "max_favorable_diff": round(max_diff, 2),
    }

    return rec_df, summary
