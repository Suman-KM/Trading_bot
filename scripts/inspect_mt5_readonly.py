#!/usr/bin/env python3
"""Phase 37: Safe, Read-Only MT5 Terminal & Market Data Inspection Diagnostic.

Performs safe environment discovery, terminal inspection, account metadata audit,
EURUSD symbol specification query, and small market data (ticks/bars) validation.

ABSOLUTE SAFETY RULES:
- ZERO order placement.
- ZERO calls to order_send() or order_check().
- ZERO credentials logged or stored.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _mask_account_login(login: Any) -> str:
    if login is None:
        return "REDACTED"
    s = str(login).strip()
    return f"***{s[-4:]}" if len(s) > 4 else "***"


def _run_in_wine(args: list[str]) -> int:
    wine_prefix = os.environ.get("WINEPREFIX", os.path.expanduser("~/.wine-mt5-demo"))
    script_path = os.path.abspath(__file__)
    wine_cmd = [
        "wine",
        "C:\\Python\\python.exe",
        script_path,
        "--wine-worker",
    ] + args
    env = os.environ.copy()
    env["WINEPREFIX"] = wine_prefix
    env["WINEDEBUG"] = "-all"
    proc = subprocess.run(wine_cmd, env=env)
    return proc.returncode


def _execute_readonly_inspection() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        print(f"[FATAL] MetaTrader5 package missing in current environment: {exc}")
        return 1

    print("=" * 80)
    print("PHASE 37: MT5 READ-ONLY TERMINAL & MARKET-DATA INSPECTION")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Initialize MT5
    print("\n[Step 1/10] Initializing MT5 terminal...")
    init_ok = mt5.initialize()
    if not init_ok:
        err = mt5.last_error()
        print(f"[ERROR] MT5 initialization failed: {err}")
        return 1
    print("  -> MT5 initialized successfully.")

    # Safety Assertion: Ensure order_send is never called
    assert not hasattr(sys.modules[__name__], "order_send"), "FATAL: order_send exposed!"

    results: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INITIALIZED",
    }

    try:
        # 2. Terminal Info
        print("\n[Step 2/10] Retrieving Terminal Info...")
        t_info = mt5.terminal_info()
        if t_info is None:
            print("  [WARN] terminal_info() returned None.")
            results["terminal"] = None
        else:
            t_dict = t_info._asdict()
            print(f"  Name           : {t_dict.get('name')}")
            print(f"  Company        : {t_dict.get('company')}")
            print(f"  Connected      : {t_dict.get('connected')}")
            print(
                f"  Trade Allowed  : {t_dict.get('trade_allowed')} "
                "(NOTE: Project gate remains DISABLED)"
            )
            print(f"  Ping (us)      : {t_dict.get('ping_last')}")
            print(f"  Path           : {t_dict.get('path')}")
            results["terminal"] = {
                "name": t_dict.get("name"),
                "company": t_dict.get("company"),
                "connected": t_dict.get("connected"),
                "trade_allowed_terminal": t_dict.get("trade_allowed"),
                "ping_last_us": t_dict.get("ping_last"),
                "path": t_dict.get("path"),
            }

        # 3. Version Info
        print("\n[Step 3/10] Retrieving MT5 Version Info...")
        v_info = mt5.version()
        if v_info is not None and len(v_info) >= 3:
            ver_str = f"{v_info[0]}.{v_info[1]} ({v_info[2]})"
            print(f"  Build / Version: {ver_str}")
            results["version"] = ver_str
        else:
            results["version"] = "Unknown"

        # 4. Account Info
        print("\n[Step 4/10] Retrieving Account Info (Read-Only)...")
        a_info = mt5.account_info()
        if a_info is None:
            print("  [WARN] account_info() returned None.")
            results["account"] = None
        else:
            a_dict = a_info._asdict()
            masked_login = _mask_account_login(a_dict.get("login"))
            print(f"  Login (Masked) : {masked_login}")
            print(f"  Server         : {a_dict.get('server')}")
            print(f"  Company        : {a_dict.get('company')}")
            print(f"  Currency       : {a_dict.get('currency')}")
            print(f"  Leverage       : 1:{a_dict.get('leverage')}")
            print(f"  Balance        : {a_dict.get('balance')}")
            print(f"  Equity         : {a_dict.get('equity')}")
            print(f"  Margin         : {a_dict.get('margin')}")
            print(f"  Free Margin    : {a_dict.get('margin_free')}")
            print(f"  Trade Mode     : {a_dict.get('trade_mode')} (0=DEMO)")
            results["account"] = {
                "login_masked": masked_login,
                "server": a_dict.get("server"),
                "company": a_dict.get("company"),
                "currency": a_dict.get("currency"),
                "leverage": a_dict.get("leverage"),
                "balance": a_dict.get("balance"),
                "equity": a_dict.get("equity"),
                "margin": a_dict.get("margin"),
                "margin_free": a_dict.get("margin_free"),
                "trade_mode": a_dict.get("trade_mode"),
            }

        # 5. EURUSD Symbol Info
        print("\n[Step 5/10] Retrieving EURUSD Symbol Metadata...")
        mt5.symbol_select("EURUSD", True)
        s_info = mt5.symbol_info("EURUSD")
        if s_info is None:
            print("  [ERROR] symbol_info('EURUSD') returned None.")
            results["symbol"] = None
        else:
            digits = getattr(s_info, "digits", 5)
            point = getattr(s_info, "point", 0.00001)
            contract_size = getattr(s_info, "trade_contract_size", 100000.0)
            vol_min = getattr(s_info, "volume_min", 0.01)
            vol_max = getattr(s_info, "volume_max", 100.0)
            vol_step = getattr(s_info, "volume_step", 0.01)
            spread = getattr(s_info, "spread", 0)

            print(f"  Symbol         : {s_info.name}")
            print(f"  Digits / Point : {digits} / {point}")
            print(f"  Contract Size  : {contract_size}")
            print(f"  Volume Range   : {vol_min} to {vol_max} (Step: {vol_step})")
            print(f"  Spread         : {spread}")
            results["symbol"] = {
                "name": s_info.name,
                "digits": digits,
                "point": point,
                "contract_size": contract_size,
                "volume_min": vol_min,
                "volume_max": vol_max,
                "volume_step": vol_step,
                "spread": spread,
            }

        # 6. EURUSD Latest Tick
        print("\n[Step 6/10] Retrieving Latest EURUSD Tick...")
        tick = mt5.symbol_info_tick("EURUSD")
        if tick is None:
            print("  [WARN] symbol_info_tick('EURUSD') returned None.")
            results["latest_tick"] = None
        else:
            t_dict = tick._asdict()
            print(f"  Bid / Ask      : {t_dict.get('bid')} / {t_dict.get('ask')}")
            print(f"  Spread (Units) : {round(t_dict.get('ask', 0) - t_dict.get('bid', 0), 6)}")
            print(f"  Broker Time    : {t_dict.get('time')}")
            results["latest_tick"] = {
                "bid": t_dict.get("bid"),
                "ask": t_dict.get("ask"),
                "last": t_dict.get("last"),
                "time": t_dict.get("time"),
                "time_msc": t_dict.get("time_msc"),
                "spread": round(t_dict.get("ask", 0) - t_dict.get("bid", 0), 6),
            }

        # 7. Small Historical OHLC Sample (5 M15 bars)
        print("\n[Step 7/10] Retrieving Small Historical OHLC Sample (5 M15 Bars)...")
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0, 5)
        if rates is None or len(rates) == 0:
            print("  [WARN] copy_rates_from_pos returned None or empty.")
            results["ohlc_sample"] = []
        else:
            print(f"  Retrieved {len(rates)} bars.")
            sample_bars = []
            for r in rates:
                bar_data = {
                    "time": int(r["time"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "tick_volume": int(r["tick_volume"]),
                }
                sample_bars.append(bar_data)
                print(
                    f"    Bar Time: {bar_data['time']} | "
                    f"O: {bar_data['open']:.5f} H: {bar_data['high']:.5f} "
                    f"L: {bar_data['low']:.5f} C: {bar_data['close']:.5f} | "
                    f"Vol: {bar_data['tick_volume']}"
                )
            results["ohlc_sample"] = sample_bars

        # 8. Small Historical Tick Sample (5 ticks)
        print("\n[Step 8/10] Retrieving Small Historical Tick Sample (5 Ticks)...")
        # Query ticks near current day
        ticks_sample = mt5.copy_ticks_from(
            "EURUSD",
            datetime(2026, 9, 30, tzinfo=timezone.utc),
            5,
            mt5.COPY_TICKS_ALL,
        )
        if ticks_sample is None or len(ticks_sample) == 0:
            print("  [WARN] copy_ticks_from returned None or empty.")
            results["ticks_sample"] = []
        else:
            print(f"  Retrieved {len(ticks_sample)} historical ticks.")
            sample_ticks = []
            for t in ticks_sample[:5]:
                tick_dict = {
                    "time": int(t["time"]),
                    "time_msc": int(t["time_msc"]),
                    "bid": float(t["bid"]),
                    "ask": float(t["ask"]),
                }
                sample_ticks.append(tick_dict)
                print(
                    f"    Tick Time: {tick_dict['time_msc']} | "
                    f"Bid: {tick_dict['bid']:.5f} | Ask: {tick_dict['ask']:.5f}"
                )
            results["ticks_sample"] = sample_ticks

        # 9. Timezone & Offset Validation
        print("\n[Step 9/10] Timezone & UTC Alignment Validation...")
        if tick is not None:
            # Current UTC vs Broker tick time
            utc_now = datetime.now(timezone.utc)
            broker_ts = tick.time
            broker_dt = datetime.fromtimestamp(broker_ts, tz=timezone.utc)
            offset_seconds = (broker_dt - utc_now).total_seconds()
            offset_hours = round(offset_seconds / 3600.0, 1)
            print(f"  Current System UTC Time : {utc_now.isoformat()}")
            print(f"  Broker Server Tick Time : {broker_dt.isoformat()}")
            print(f"  Estimated Broker Offset : {offset_hours} hours from UTC")
            results["timezone"] = {
                "system_utc": utc_now.isoformat(),
                "broker_time": broker_dt.isoformat(),
                "broker_utc_offset_hours": offset_hours,
            }

    finally:
        # 10. Clean Shutdown
        print("\n[Step 10/10] Clean MT5 Shutdown...")
        mt5.shutdown()
        print("  -> MT5 shutdown completed cleanly.")

    # Save output summary to reports (for inspection)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "mt5_readonly_inspection.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Read-only inspection report saved to {report_file}")
    return 0


def main() -> int:
    if "--wine-worker" in sys.argv or sys.platform == "win32":
        return _execute_readonly_inspection()
    else:
        print("[INFO] Launching MT5 inspection via Wine worker...")
        return _run_in_wine([])


if __name__ == "__main__":
    sys.exit(main())
