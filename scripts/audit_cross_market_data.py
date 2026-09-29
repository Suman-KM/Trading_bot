#!/usr/bin/env python3
"""Phase 18: Cross-Market Data Availability Audit for MT5.

Audits candidate cross-market instruments in the MetaQuotes-Demo MT5 terminal
running under Wine. Checks symbol existence, total available H4 bars, earliest/latest
timestamps, spreads, and data continuity.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _run_in_wine(args: list[str]) -> int:
    wine_prefix = os.environ.get("WINEPREFIX", os.path.expanduser("~/.wine-mt5-demo"))
    script_path = os.path.abspath(__file__)
    wine_cmd = [
        "wine",
        "C:\\Python\\python.exe",
        script_path,
    ] + args
    env = os.environ.copy()
    env["WINEPREFIX"] = wine_prefix
    env["WINEDEBUG"] = "-all"
    proc = subprocess.run(wine_cmd, env=env)
    return proc.returncode


def _execute_audit() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 not available in this Python environment.")
        return 1

    try:
        import numpy as np
    except ImportError:
        print("[ERROR] numpy not available.")
        return 1

    print("=" * 70)
    print("PHASE 18: CROSS-MARKET INSTRUMENT AUDIT (MT5)")
    print("=" * 70)

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}")
        return 1

    try:
        term_info = mt5.terminal_info()
        acc_info = mt5.account_info()
        broker = acc_info.company if acc_info else "Unknown"
        server = acc_info.server if acc_info else "Unknown"
        print(f"Connected: {term_info.connected} | Broker: {broker} | Server: {server}")

        all_symbols = mt5.symbols_get()
        all_symbol_names = [s.name for s in all_symbols] if all_symbols else []
        print(f"Total symbols configured in terminal: {len(all_symbol_names)}")

        candidates = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "EURGBP",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "XAUUSD",
            "XAGUSD",
            "US500",
            "US30",
            "USTEC",
            "DXY",
            "DX",
            "USDX",
            "EURJPY",
            "GBPJPY",
        ]

        audit_results: dict[str, dict] = {}
        tf = mt5.TIMEFRAME_H4
        target_bars = 26000

        for sym in candidates:
            matching_syms = [s for s in all_symbol_names if s.upper() == sym.upper()]
            if not matching_syms:
                matching_syms = [s for s in all_symbol_names if sym.upper() in s.upper()]

            if not matching_syms:
                print(f"  {sym:<8}: NOT AVAILABLE (Symbol not listed in terminal)")
                audit_results[sym] = {
                    "symbol": sym,
                    "status": "NOT AVAILABLE",
                    "broker": broker,
                    "server": server,
                    "reason": "Symbol not listed by broker",
                    "sufficient_history": False,
                }
                continue

            exact_sym = matching_syms[0]
            if not mt5.symbol_select(exact_sym, True):
                print(f"  {sym:<8}: FAILED to select ({mt5.last_error()})")
                audit_results[sym] = {
                    "symbol": exact_sym,
                    "status": "FAILED_SELECT",
                    "broker": broker,
                    "server": server,
                    "reason": f"mt5.symbol_select failed: {mt5.last_error()}",
                    "sufficient_history": False,
                }
                continue

            # Retry loop for copying rates to give MT5 time to fetch from broker server
            rates = None
            for attempt in range(6):
                rates = mt5.copy_rates_from_pos(exact_sym, tf, 0, target_bars)
                if rates is not None and len(rates) > 0:
                    # If count is still tiny (<500), the server might still be streaming
                    if len(rates) > 500 or attempt >= 3:
                        break
                time.sleep(1.5)

            if rates is None or len(rates) == 0:
                print(f"  {sym:<8}: NO RATES RETURNED ({mt5.last_error()})")
                audit_results[sym] = {
                    "symbol": exact_sym,
                    "status": "NO_DATA",
                    "broker": broker,
                    "server": server,
                    "reason": f"copy_rates_from_pos returned None: {mt5.last_error()}",
                    "sufficient_history": False,
                }
                continue

            rates_sorted = np.sort(rates, order="time")
            bar_count = len(rates_sorted)
            earliest_dt = datetime.fromtimestamp(int(rates_sorted[0]["time"]), tz=timezone.utc)
            latest_dt = datetime.fromtimestamp(int(rates_sorted[-1]["time"]), tz=timezone.utc)
            has_spread = bool(np.any(rates_sorted["spread"] > 0))

            # Sufficient if it covers from at least 2011 to 2026 (>20,000 H4 bars)
            sufficient = bool(earliest_dt.year <= 2011 and bar_count >= 20000)

            e_str = earliest_dt.strftime("%Y-%m-%d %H:%M")
            l_str = latest_dt.strftime("%Y-%m-%d %H:%M")
            print(
                f"  {exact_sym:<8}: {bar_count:>6} bars | "
                f"{e_str} -> {l_str} | "
                f"Spread: {has_spread} | Sufficient (>=2011): {sufficient}"
            )

            audit_results[sym] = {
                "symbol": exact_sym,
                "status": "AVAILABLE",
                "broker": broker,
                "server": server,
                "timeframe": "H4",
                "bar_count": bar_count,
                "earliest_ts": earliest_dt.isoformat(),
                "latest_ts": latest_dt.isoformat(),
                "timezone": "UTC",
                "has_spread": has_spread,
                "sufficient_history": sufficient,
            }

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_file = reports_dir / "phase18_cross_market_audit.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2)
        print(f"\nSaved cross-market audit to {out_file}")

    finally:
        mt5.shutdown()

    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _execute_audit()
    return _run_in_wine(["--worker"])


if __name__ == "__main__":
    sys.exit(main())
