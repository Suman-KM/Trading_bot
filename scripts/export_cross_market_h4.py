#!/usr/bin/env python3
"""Phase 18: Cross-Market H4 Historical Data Exporter.

Retrieves complete available H4 history for verified cross-market instruments:
GBPUSD, USDJPY, EURGBP, USDCHF, AUDUSD, USDCAD, NZDUSD, XAUUSD
from the MetaQuotes-Demo MT5 terminal via Wine, validates data integrity,
and saves Parquet files to data/raw/cross_market_h4/ and data/processed/cross_market_h4/.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

RAW_CROSS_DIR = Path("data/raw/cross_market_h4")
PROCESSED_CROSS_DIR = Path("data/processed/cross_market_h4")
REPORTS_DIR = Path("reports")

VERIFIED_SYMBOLS = [
    "GBPUSD",
    "USDJPY",
    "EURGBP",
    "USDCHF",
    "AUDUSD",
    "USDCAD",
    "NZDUSD",
    "XAUUSD",
]


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


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _execute_export_worker() -> int:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("[ERROR] MetaTrader5 not available in this Python environment.")
        return 1

    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"[ERROR] Required packages (numpy, pyarrow) missing: {exc}")
        return 1

    print("=" * 70)
    print("PHASE 18: CROSS-MARKET H4 HISTORICAL DATA EXPORT (WINE WORKER)")
    print("=" * 70)

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}")
        return 1

    try:
        RAW_CROSS_DIR.mkdir(parents=True, exist_ok=True)
        tf = mt5.TIMEFRAME_H4
        target_bars = 26000

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

        for sym in VERIFIED_SYMBOLS:
            print(f"Exporting {sym} H4 from MT5...")
            if not mt5.symbol_select(sym, True):
                print(f"[ERROR] Failed to select {sym}: {mt5.last_error()}")
                return 1

            rates = None
            for attempt in range(5):
                rates = mt5.copy_rates_from_pos(sym, tf, 0, target_bars)
                if rates is not None and len(rates) >= 20000:
                    break
                cnt = len(rates) if rates is not None else 0
                print(f"  [{sym}] attempt {attempt + 1}: {cnt} bars, retrying...")
                time.sleep(1.5)

            if rates is None or len(rates) == 0:
                print(f"[FATAL] Failed to retrieve rates for {sym}: {mt5.last_error()}")
                return 1

            rates_sorted = np.sort(rates, order="time")
            raw_file = RAW_CROSS_DIR / f"{sym.lower()}_h4_raw.parquet"

            table = pa.Table.from_arrays(
                [
                    pa.array(rates_sorted["time"], type=pa.int64()),
                    pa.array(rates_sorted["open"], type=pa.float64()),
                    pa.array(rates_sorted["high"], type=pa.float64()),
                    pa.array(rates_sorted["low"], type=pa.float64()),
                    pa.array(rates_sorted["close"], type=pa.float64()),
                    pa.array(rates_sorted["tick_volume"], type=pa.uint64()),
                    pa.array(rates_sorted["spread"], type=pa.int32()),
                    pa.array(rates_sorted["real_volume"], type=pa.uint64()),
                ],
                schema=pa_schema,
            )
            pq.write_table(table, str(raw_file))
            print(f"  Saved {len(rates_sorted):,} bars -> {raw_file}")

    finally:
        mt5.shutdown()

    return 0


def _process_exported_data() -> int:
    """Run in Linux Python: process raw files, add UTC timestamps, deduplicate, compute hashes."""
    import pandas as pd

    print("\nProcessing raw Parquet files into processed format...")
    PROCESSED_CROSS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}

    for sym in VERIFIED_SYMBOLS:
        raw_file = RAW_CROSS_DIR / f"{sym.lower()}_h4_raw.parquet"
        if not raw_file.exists():
            print(f"[ERROR] Missing raw file for {sym}: {raw_file}")
            return 1

        df = pd.read_parquet(raw_file)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = (
            df[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "tick_volume",
                    "spread",
                    "real_volume",
                ]
            ]
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        proc_file = PROCESSED_CROSS_DIR / f"{sym.lower()}_h4_processed.parquet"
        df.to_parquet(proc_file, index=False)

        sha = compute_file_hash(proc_file)
        earliest_ts = df["timestamp"].iloc[0].isoformat()
        latest_ts = df["timestamp"].iloc[-1].isoformat()
        bar_count = len(df)

        print(
            f"  {sym:<8}: {bar_count:>6,} bars ({earliest_ts[:16]} -> {latest_ts[:16]}) | "
            f"SHA: {sha[:16]}..."
        )

        metadata[sym] = {
            "symbol": sym,
            "bar_count": bar_count,
            "earliest_timestamp": earliest_ts,
            "latest_timestamp": latest_ts,
            "raw_file": str(raw_file),
            "processed_file": str(proc_file),
            "sha256": sha,
        }

    meta_file = REPORTS_DIR / "cross_market_export_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[SUCCESS] Cross-market export complete. Metadata saved to {meta_file}")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _execute_export_worker()

    # Step 1: Export raw files via Wine MT5
    rc = _run_in_wine(["--worker"])
    if rc != 0:
        print(f"[FATAL] Wine export worker failed with return code {rc}")
        return rc

    # Step 2: Process in Linux Python
    return _process_exported_data()


if __name__ == "__main__":
    sys.exit(main())
