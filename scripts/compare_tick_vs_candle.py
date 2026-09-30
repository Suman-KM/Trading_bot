#!/usr/bin/env python3
"""Phase 20: Standalone Candle vs Tick Execution Comparison & Reconciliation Viewer.

Parses and formats trade-level reconciliation and execution metrics comparing:
- Phase 12 Candle-Level Backtest
- Phase 20 Tick-Realistic Execution (Exact Bid/Ask, Intrabar Path, Floating Spread)
- Phase 20 Tick-Realistic Execution with Slippage Stress
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

COMPARISON_JSON = Path("reports/phase20_execution_comparison.json")
RECONCILIATION_CSV = Path("reports/phase20_trade_reconciliation.csv")
TICK_SUMMARY_CSV = Path("reports/phase20_tick_trade_summary.csv")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 20: Compare Tick vs Candle Execution Results"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON summary instead of formatted report",
    )
    parser.add_argument(
        "--top-divergences",
        type=int,
        default=5,
        help="Number of largest favorable and adverse divergences to show (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    """Load comparison data and format comparative report."""
    args = parse_args()

    if not COMPARISON_JSON.exists() or not RECONCILIATION_CSV.exists():
        print(
            f"[ERROR] Required comparison files not found:\n"
            f"  {COMPARISON_JSON}: {COMPARISON_JSON.exists()}\n"
            f"  {RECONCILIATION_CSV}: {RECONCILIATION_CSV.exists()}\n"
            f"Please run `python3 scripts/run_tick_backtest.py` first.",
            file=sys.stderr,
        )
        return 1

    with open(COMPARISON_JSON, encoding="utf-8") as f:
        comp = json.load(f)

    if args.json:
        print(json.dumps(comp, indent=2))
        return 0

    rec_df = pd.read_csv(RECONCILIATION_CSV)

    candle = comp["conf_50_sensitivity"]["candle_engine"]
    tick = comp["conf_50_sensitivity"]["tick_engine_baseline"]
    tick_slip = comp["conf_50_sensitivity"]["tick_engine_slippage_stress"]
    rec_sum = comp["reconciliation_summary"]
    tick_data = comp["tick_data_summary"]

    sep = "=" * 80
    subsep = "-" * 80

    print(sep)
    print("PHASE 20: TICK-REALISTIC vs CANDLE EXECUTION COMPARISON REPORT")
    print(sep)
    print(f"Timestamp: {comp.get('timestamp_utc')}")
    print(f"Phase 11 Test Set Locked & Respected: {tick_data.get('phase11_test_lock_respected')}")
    print(f"Total Validation Ticks Analyzed: {tick_data.get('tick_count'):,}")
    print(f"Earliest Tick: {tick_data.get('earliest_timestamp')}")
    print(f"Latest Tick:   {tick_data.get('latest_timestamp')}")
    print(f"Timestamp Monotonic: {tick_data.get('is_monotonic')}")
    print(f"Negative Spreads:    {tick_data.get('negative_spreads')}")
    print()

    print("1. BASELINE EXECUTION VERIFICATION (Threshold = 0.60)")
    print(subsep)
    print(f"Candle Baseline Trades: {comp['baseline_reproduction']['candle_trades']}")
    print(f"Tick Baseline Trades:   {comp['baseline_reproduction']['tick_trades']}")
    print(f"Status: Exactly Reproduced ({comp['baseline_reproduction']['confirmed_reproduced']})")
    print()

    print("2. PERFORMANCE COMPARISON TABLE (Sensitivity Threshold = 0.50)")
    print(subsep)
    header = (
        f"{'Metric':<25} | {'Candle Engine':<15} | {'Tick Realistic':<15} | {'Tick + Slippage':<15}"
    )
    print(header)
    print("-" * len(header))
    c_tr, t_tr, ts_tr = candle["trades"], tick["trades"], tick_slip["trades"]
    c_wr, t_wr, ts_wr = candle["win_rate"], tick["win_rate"], tick_slip["win_rate"]
    c_pnl, t_pnl, ts_pnl = candle["net_pnl"], tick["net_pnl"], tick_slip["net_pnl"]
    c_pf, t_pf = candle["profit_factor"], tick["profit_factor"]
    c_dd, t_dd = candle["max_drawdown_pct"], tick["max_drawdown_pct"]
    c_sp, t_sp = candle["spread_cost"], tick["spread_cost"]
    c_sl, t_sl = candle["exit_reasons"]["STOP_LOSS"], tick["exit_reasons"]["STOP_LOSS"]
    c_tp, t_tp = candle["exit_reasons"]["TAKE_PROFIT"], tick["exit_reasons"]["TAKE_PROFIT"]
    c_mh, t_mh = candle["exit_reasons"]["MAX_HOLD"], tick["exit_reasons"]["MAX_HOLD"]

    print(f"{'Total Trades':<25} | {c_tr:<15} | {t_tr:<15} | {ts_tr:<15}")
    print(f"{'Win Rate (%)':<25} | {c_wr:<15.2f} | {t_wr:<15.2f} | {ts_wr:<15.2f}")
    print(f"{'Net P&L ($)':<25} | {c_pnl:<15.2f} | {t_pnl:<15.2f} | {ts_pnl:<15.2f}")
    print(f"{'Profit Factor':<25} | {c_pf:<15.4f} | {t_pf:<15.4f} | {'N/A':<15}")
    print(f"{'Max Drawdown (%)':<25} | {c_dd:<15.2f} | {t_dd:<15.2f} | {'N/A':<15}")
    print(f"{'Total Spread Cost ($)':<25} | {c_sp:<15.2f} | {t_sp:<15.2f} | {'N/A':<15}")
    print(f"{'Stop Loss Exits':<25} | {c_sl:<15} | {t_sl:<15} | {'N/A':<15}")
    print(f"{'Take Profit Exits':<25} | {c_tp:<15} | {t_tp:<15} | {'N/A':<15}")
    print(f"{'Max Hold Exits':<25} | {c_mh:<15} | {t_mh:<15} | {'N/A':<15}")
    print()

    print("3. SPREAD REALISM AUDIT")
    print(subsep)
    sp = tick_data["spread_metrics"]
    diff_sp = tick["spread_cost"] - candle["spread_cost"]
    pct_sp = ((tick["spread_cost"] / candle["spread_cost"]) - 1.0) * 100.0
    print(f"Mean Spread:     {sp['mean_points']:.2f} pts ({sp['mean_points'] / 10:.2f} pips)")
    print(f"Median Spread:   {sp['median_points']:.2f} pts ({sp['median_points'] / 10:.2f} pips)")
    print(f"95th Pct Spread: {sp['p95_points']:.2f} pts ({sp['p95_points'] / 10:.2f} pips)")
    print(f"99th Pct Spread: {sp['p99_points']:.2f} pts ({sp['p99_points'] / 10:.2f} pips)")
    print(f"Max Spread:      {sp['max_points']:.2f} pts ({sp['max_points'] / 10:.2f} pips)")
    print("Static Candle Spread:   4.00 points (0.40 pips)")
    print(f"Spread Cost Expansion:  +${diff_sp:.2f} (+{pct_sp:.1f}%)")
    print()

    print("4. TRADE-BY-TRADE RECONCILIATION BREAKDOWN")
    print(subsep)
    print(f"Total Trades Evaluated: {rec_sum['total_candle_trades']}")
    print(f"Matched Trades:         {rec_sum['matched_trades']}")
    print(f"Unmatched Trades:       {rec_sum['unmatched_trades']}")
    print()
    print("Variance Categorization:")
    for cat, count in rec_sum["classifications"].items():
        pct = (count / rec_sum["matched_trades"]) * 100 if rec_sum["matched_trades"] > 0 else 0
        print(f"  - {cat:<22}: {count:>3} trades ({pct:>5.1f}%)")
    print()

    print("P&L Divergence Summary:")
    print(f"  Net P&L Difference (Tick - Candle):  ${rec_sum['net_pnl_difference']:+,.2f}")
    print(f"  Mean P&L Diff per Trade:             ${rec_sum['mean_pnl_diff_per_trade']:+.2f}")
    print(f"  Median P&L Diff per Trade:           ${rec_sum['median_pnl_diff_per_trade']:+.2f}")
    print(f"  Max Favorable Divergence:            ${rec_sum['max_favorable_diff']:+.2f}")
    print(f"  Max Adverse Divergence:              ${rec_sum['max_adverse_diff']:+.2f}")
    print()

    if "pnl_diff" in rec_df.columns:
        valid_diffs = rec_df.dropna(subset=["pnl_diff"]).sort_values("pnl_diff", ascending=False)

        top_n = min(args.top_divergences, len(valid_diffs))
        print(f"5. TOP {top_n} FAVORABLE DIVERGENCES (Tick > Candle)")
        print(subsep)
        fav = valid_diffs.head(top_n)
        for _, r in fav.iterrows():
            c_desc = f"{r['candle_exit_reason']:<11} (${r['candle_pnl']:>7.2f})"
            t_desc = f"{r['tick_exit_reason']:<11} (${r['tick_pnl']:>7.2f})"
            print(
                f"  [{r['signal_time']}] {r['direction']:<5} | C: {c_desc} -> T: {t_desc} "
                f"| Diff: ${r['pnl_diff']:>+7.2f} | {r['classification']}"
            )
        print()

        print(f"6. TOP {top_n} ADVERSE DIVERGENCES (Candle > Tick)")
        print(subsep)
        adv = valid_diffs.tail(top_n).sort_values("pnl_diff")
        for _, r in adv.iterrows():
            c_desc = f"{r['candle_exit_reason']:<11} (${r['candle_pnl']:>7.2f})"
            t_desc = f"{r['tick_exit_reason']:<11} (${r['tick_pnl']:>7.2f})"
            print(
                f"  [{r['signal_time']}] {r['direction']:<5} | C: {c_desc} -> T: {t_desc} "
                f"| Diff: ${r['pnl_diff']:>+7.2f} | {r['classification']}"
            )
        print()

    print("7. RESEARCH INTERPRETATION & ROOT CAUSE SUMMARY")
    print(subsep)
    print(
        "1. Conservative Collision Bias in Candle Simulation:\n"
        "   When a 15-minute bar touches both SL and TP price levels, the candle engine\n"
        "   strictly assumes the Stop Loss was hit first (conservative assumption).\n"
        "   Chronological tick playback reveals that in 7 cases, Take Profit was reached\n"
        "   BEFORE Stop Loss, eliminating false loss attributions.\n"
        "\n"
        "2. Max Hold Intrabar Excursions:\n"
        "   In 6 cases, trades categorized as MAX_HOLD under candle simulation actually\n"
        "   reached Take Profit during intrabar tick trajectory before the holding limit.\n"
        "\n"
        "3. Realistic Spread Expansion:\n"
        "   Historical floating spread averaged 7.38 points (vs static 4.0 points in M15),\n"
        "   increasing spread friction by +32.5% ($118.64 -> $157.25). Despite higher friction,\n"
        "   true intrabar TP executions dominated net performance.\n"
        "\n"
        "4. Critical Caution:\n"
        "   This divergence occurs across a small 60-trade research sample at non-baseline\n"
        "   threshold (0.50). The baseline (0.60) remains at 0 trades. These findings do NOT\n"
        "   demonstrate general commercial viability, but rather quantify candle simulation\n"
        "   inaccuracies."
    )
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
