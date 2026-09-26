#!/usr/bin/env python3
"""EURUSD M15 Raw Historical Data Export Utility.

Connects to the MetaTrader 5 demo environment (MetaQuotes Ltd. / MetaQuotes-Demo),
retrieves the complete available EURUSD M15 historical bar dataset in deterministic
chunks, verifies raw integrity, and exports directly to Apache Parquet.

Preserves exact raw MT5 fields:
- time (Unix epoch seconds, int64)
- open (float64)
- high (float64)
- low (float64)
- close (float64)
- tick_volume (uint64)
- spread (int32)
- real_volume (uint64)
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def validate_bar_records(
    records: list[dict[str, Any]] | list[Any],
) -> dict[str, Any]:
    """Perform read-only integrity and financial sanity checks on raw bar records.

    Accepts a list of dicts (or objects supporting key access) with keys:
    time, open, high, low, close, tick_volume, spread, real_volume.
    """
    total_rows = len(records)
    if total_rows == 0:
        return {
            "valid": False,
            "total_rows": 0,
            "error": "No records to validate",
        }

    duplicate_timestamps = 0
    strictly_chronological = True
    min_interval: int | None = None
    max_interval: int | None = None

    ohlc_violations = 0
    zero_or_negative_prices = 0
    nan_count = 0
    inf_count = 0

    tick_vol_min = float("inf")
    tick_vol_max = float("-inf")
    tick_vol_zeros = 0
    tick_vol_negatives = 0

    spread_min = float("inf")
    spread_max = float("-inf")
    spread_negatives = 0

    real_vol_zeros = 0
    real_vol_min = float("inf")
    real_vol_max = float("-inf")

    seen_timestamps: set[int] = set()
    prev_time: int | None = None

    for i, row in enumerate(records):
        t = int(row["time"])
        o = float(row["open"])
        h = float(row["high"])
        low_val = float(row["low"])
        c = float(row["close"])
        tv = int(row["tick_volume"])
        sp = int(row["spread"])
        rv = int(row["real_volume"])

        # Check NaN / Inf
        for val in (o, h, low_val, c):
            if math.isnan(val):
                nan_count += 1
            if math.isinf(val):
                inf_count += 1

        # Check positive prices
        if o <= 0 or h <= 0 or low_val <= 0 or c <= 0:
            zero_or_negative_prices += 1

        # Check OHLC validity
        is_ohlc_valid = (
            h >= o and h >= c and h >= low_val and low_val <= o and low_val <= c and low_val <= h
        )
        if not is_ohlc_valid:
            ohlc_violations += 1

        # Check duplicates
        if t in seen_timestamps:
            duplicate_timestamps += 1
        seen_timestamps.add(t)

        # Check chronology
        if prev_time is not None:
            step = t - prev_time
            if step <= 0:
                strictly_chronological = False
            if min_interval is None or step < min_interval:
                min_interval = step
            if max_interval is None or step > max_interval:
                max_interval = step
        prev_time = t

        # Tick volume stats
        if tv < tick_vol_min:
            tick_vol_min = tv
        if tv > tick_vol_max:
            tick_vol_max = tv
        if tv == 0:
            tick_vol_zeros += 1
        if tv < 0:
            tick_vol_negatives += 1

        # Spread stats
        if sp < spread_min:
            spread_min = sp
        if sp > spread_max:
            spread_max = sp
        if sp < 0:
            spread_negatives += 1

        # Real volume stats
        if rv < real_vol_min:
            real_vol_min = rv
        if rv > real_vol_max:
            real_vol_max = rv
        if rv == 0:
            real_vol_zeros += 1

    earliest_ts = int(records[0]["time"])
    latest_ts = int(records[-1]["time"])

    is_valid = (
        duplicate_timestamps == 0
        and strictly_chronological
        and ohlc_violations == 0
        and zero_or_negative_prices == 0
        and nan_count == 0
        and inf_count == 0
        and tick_vol_negatives == 0
        and spread_negatives == 0
    )

    earliest_utc_str = datetime.fromtimestamp(earliest_ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    latest_utc_str = datetime.fromtimestamp(latest_ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    return {
        "valid": is_valid,
        "total_rows": total_rows,
        "earliest_timestamp": earliest_ts,
        "latest_timestamp": latest_ts,
        "earliest_utc": earliest_utc_str,
        "latest_utc": latest_utc_str,
        "duplicate_timestamps": duplicate_timestamps,
        "strictly_chronological": strictly_chronological,
        "min_interval_seconds": min_interval,
        "max_interval_seconds": max_interval,
        "ohlc_violations": ohlc_violations,
        "zero_or_negative_prices": zero_or_negative_prices,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "tick_volume_min": tick_vol_min,
        "tick_volume_max": tick_vol_max,
        "tick_volume_zeros": tick_vol_zeros,
        "tick_volume_negatives": tick_vol_negatives,
        "spread_min": spread_min,
        "spread_max": spread_max,
        "spread_negatives": spread_negatives,
        "real_volume_min": real_vol_min,
        "real_volume_max": real_vol_max,
        "real_volume_zeros": real_vol_zeros,
    }


def _run_in_wine(args: list[str]) -> int:
    """Delegate script execution to Wine Python when running on Linux."""
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


def _execute_mt5_export(
    symbol: str,
    timeframe_str: str,
    output_path: Path,
    chunk_size: int = 25000,
    max_bars: int = 100000,
) -> int:
    """Execute raw MT5 history retrieval and Parquet export inside Wine Python."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 package is not available in the current Python environment.")
        return 1

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"[ERROR] Required packages (numpy, pyarrow) missing: {exc}")
        return 1

    start_time = time.time()
    print("=" * 60)
    print("EURUSD M15 RAW HISTORICAL DATA EXPORT")
    print("=" * 60)
    print(f"Target symbol:      {symbol}")
    print(f"Target timeframe:   {timeframe_str}")
    print(f"Destination file:   {output_path.resolve()}")
    print(f"Chunk size:         {chunk_size}")
    print(f"Max bars ceiling:   {max_bars}")

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}")
        return 1

    try:
        # Check connection and terminal info
        term_info = mt5.terminal_info()
        acc_info = mt5.account_info()
        print(f"Connected:          {term_info.connected}")
        print(f"Broker:             {acc_info.company if acc_info else 'Unknown'}")
        print(f"Server:             {acc_info.server if acc_info else 'Unknown'}")
        print(f"Demo Account:       {acc_info.login if acc_info else 'Unknown'}")
        print(f"Terminal Max Bars:  {term_info.maxbars}")

        # Ensure symbol is selected
        if not mt5.symbol_select(symbol, True):
            print(f"[FATAL] Failed to select symbol {symbol}: {mt5.last_error()}")
            return 1

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            print(f"[FATAL] Symbol info returned None for {symbol}")
            return 1

        print(f"Symbol visible:     {sym_info.visible}")
        print(f"Symbol trade mode:  {sym_info.trade_mode}")

        tf = getattr(mt5, f"TIMEFRAME_{timeframe_str.upper()}", None)
        if tf is None:
            print(f"[FATAL] Invalid timeframe string: {timeframe_str}")
            return 1

        # Deterministic chunked retrieval
        print("\nRetrieving historical bars in deterministic chunks...")
        chunks: list[np.ndarray] = []
        pos = 0

        while pos < max_bars:
            req_count = min(chunk_size, max_bars - pos)
            rates = mt5.copy_rates_from_pos(symbol, tf, pos, req_count)
            if rates is None or len(rates) == 0:
                print(f"  Pos {pos}: Reached end of available data ({mt5.last_error()})")
                break

            chunks.append(rates)
            msg = f"  Pos {pos:6d} -> retrieved {len(rates):5d} bars"
            print(f"{msg} (earliest in chunk: {rates[0]['time']})")

            if len(rates) < req_count:
                # MT5 returned fewer bars than requested; terminal history exhausted
                break

            pos += len(rates)

        if not chunks:
            print("[FATAL] No bars retrieved from MT5.")
            return 1

        # Concatenate chunks
        combined = np.concatenate(chunks)
        print(f"\nTotal raw bars collected from chunks: {len(combined):,}")

        # Sort strictly ascending by timestamp
        combined = np.sort(combined, order="time")

        # Detect and remove exact boundary duplicates
        unique_mask = np.empty(len(combined), dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = combined["time"][1:] != combined["time"][:-1]
        deduped = combined[unique_mask]

        dropped = len(combined) - len(deduped)
        if dropped > 0:
            print(f"[INFO] Removed {dropped} exact boundary duplicate bar(s).")
        else:
            print("[INFO] Zero duplicate timestamps across chunks.")

        print(f"Final validated raw bar count: {len(deduped):,}")

        # Run full validation
        val_result = validate_bar_records(deduped)
        if not val_result["valid"]:
            print(f"[FATAL] Raw data validation failed: {val_result}")
            return 1

        # Write Parquet with explicit schema
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pa_schema = pa.schema(
            [
                ("time", pa.int64()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("tick_volume", pa.uint64()),
                ("spread", pa.int32()),
                ("real_volume", pa.uint64()),
            ]
        )

        table = pa.Table.from_arrays(
            [
                pa.array(deduped["time"], type=pa.int64()),
                pa.array(deduped["open"], type=pa.float64()),
                pa.array(deduped["high"], type=pa.float64()),
                pa.array(deduped["low"], type=pa.float64()),
                pa.array(deduped["close"], type=pa.float64()),
                pa.array(deduped["tick_volume"], type=pa.uint64()),
                pa.array(deduped["spread"], type=pa.int32()),
                pa.array(deduped["real_volume"], type=pa.uint64()),
            ],
            schema=pa_schema,
        )

        pq.write_table(table, str(output_path), compression="snappy")
        duration = time.time() - start_time
        file_size = output_path.stat().st_size

        print("\n" + "=" * 60)
        print("EXPORT & VALIDATION REPORT")
        print("=" * 60)
        print("Status:                 SUCCESS")
        print(f"File Path:              {output_path.resolve()}")
        print(f"File Size:              {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")
        print("Format:                 Apache Parquet (Snappy)")
        print(f"Total Rows:             {val_result['total_rows']:,}")
        print(
            f"Earliest Timestamp:     {val_result['earliest_timestamp']} "
            f"({val_result['earliest_utc']})"
        )
        print(
            f"Latest Timestamp:       {val_result['latest_timestamp']} "
            f"({val_result['latest_utc']})"
        )
        print(f"Duplicates:             {val_result['duplicate_timestamps']}")
        print(f"Chronological:          {val_result['strictly_chronological']}")
        print(f"Min Interval:           {val_result['min_interval_seconds']} seconds")
        print(f"Max Interval:           {val_result['max_interval_seconds']} seconds")
        print(f"OHLC Violations:        {val_result['ohlc_violations']}")
        print(f"Zero/Negative Prices:   {val_result['zero_or_negative_prices']}")
        print(f"NaN Count:              {val_result['nan_count']}")
        print(f"Inf Count:              {val_result['inf_count']}")
        print(
            f"Tick Volume Range:      [{val_result['tick_volume_min']}, "
            f"{val_result['tick_volume_max']}]"
        )
        print(f"Spread Range:           [{val_result['spread_min']}, {val_result['spread_max']}]")
        print(
            f"Real Volume Range:      [{val_result['real_volume_min']}, "
            f"{val_result['real_volume_max']}]"
        )
        print(f"Export Duration:        {duration:.2f} seconds")
        print(
            f"Completeness:           MT5-LIMITED "
            f"(Terminal maxbars buffer = {term_info.maxbars:,})"
        )
        print("=" * 60)
        return 0

    finally:
        mt5.shutdown()


def main() -> int:
    """CLI entrypoint with automatic Wine Python delegation on Linux."""
    parser = argparse.ArgumentParser(description="Export EURUSD M15 raw data from MT5.")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol name (default: EURUSD)")
    parser.add_argument("--timeframe", default="M15", help="Timeframe (default: M15)")
    parser.add_argument(
        "--output",
        default="data/raw/eurusd_m15/eurusd_m15_raw.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Chunk size for rates retrieval (default: 25000)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=100000,
        help="Max bars ceiling to query (default: 100000)",
    )
    args = parser.parse_args()

    # If MetaTrader5 is not importable directly (e.g. running on Linux host), delegate
    try:
        import MetaTrader5  # noqa: F401
    except ImportError:
        return _run_in_wine(sys.argv[1:])

    output_path = Path(args.output)
    return _execute_mt5_export(
        symbol=args.symbol,
        timeframe_str=args.timeframe,
        output_path=output_path,
        chunk_size=args.chunk_size,
        max_bars=args.max_bars,
    )


if __name__ == "__main__":
    sys.exit(main())
