#!/usr/bin/env python3
"""Phase 17: EURUSD H4 Expanded Historical Data Chunk Exporter.

Connects to the MetaTrader 5 demo environment (MetaQuotes Ltd. / MetaQuotes-Demo)
via Wine, retrieves the complete available EURUSD H4 historical dataset (~25,000 bars
spanning 2010 to 2026) in overlapping deterministic chunks, verifies chunk integrity,
and exports chunk Parquet files to data/raw/eurusd_h4_expanded/chunks/.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNKS_DIR = Path("data/raw/eurusd_h4_expanded/chunks")
REPORTS_DIR = Path("reports")


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


def compute_chunk_hash(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _execute_h4_chunk_export() -> int:
    """Execute raw MT5 H4 history retrieval in overlapping chunks inside Wine Python."""
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

    print("=" * 70)
    print("PHASE 17: EURUSD H4 EXPANDED HISTORICAL CHUNK EXPORT")
    print("=" * 70)

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}")
        return 1

    try:
        symbol = "EURUSD"
        if not mt5.symbol_select(symbol, True):
            print(f"[FATAL] Failed to select symbol {symbol}: {mt5.last_error()}")
            return 1

        term_info = mt5.terminal_info()
        acc_info = mt5.account_info()
        print(f"Connected:         {term_info.connected}")
        print(f"Broker:            {acc_info.company if acc_info else 'Unknown'}")
        print(f"Server:            {acc_info.server if acc_info else 'Unknown'}")
        print(f"Terminal Max Bars: {term_info.maxbars}")

        tf = mt5.TIMEFRAME_H4
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Define 3 overlapping chunk specifications
        # Chunk 1: pos 0, count 10,000 (Recent: ~2020 to 2026)
        # Chunk 2: pos 9,900, count 10,000 (Middle: ~2014 to 2020, 100 bars overlap with Chunk 1)
        # Chunk 3: pos 19,800, count 6,000 (Oldest: ~2010 to 2014, 100 bars overlap with Chunk 2)
        chunk_specs = [
            {"chunk_id": 1, "pos": 0, "count": 10000},
            {"chunk_id": 2, "pos": 9900, "count": 10000},
            {"chunk_id": 3, "pos": 19800, "count": 6000},
        ]

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

        metadata_records: list[dict[str, Any]] = []

        for spec in chunk_specs:
            cid = spec["chunk_id"]
            pos = spec["pos"]
            cnt = spec["count"]

            print(f"\nRetrieving Chunk {cid} (pos={pos}, req_count={cnt})...")
            rates = mt5.copy_rates_from_pos(symbol, tf, pos, cnt)
            if rates is None or len(rates) == 0:
                print(f"[FATAL] Failed to retrieve Chunk {cid}: {mt5.last_error()}")
                return 1

            # Sort ascending by time
            rates_sorted = np.sort(rates, order="time")
            actual_count = len(rates_sorted)
            earliest_ts = int(rates_sorted[0]["time"])
            latest_ts = int(rates_sorted[-1]["time"])

            # Check duplicates within chunk
            times = rates_sorted["time"]
            dup_count = int(np.sum(times[1:] == times[:-1]))

            # OHLC validity
            high_ok = bool(
                np.all(
                    (rates_sorted["high"] >= rates_sorted["open"])
                    & (rates_sorted["high"] >= rates_sorted["close"])
                )
            )
            low_ok = bool(
                np.all(
                    (rates_sorted["low"] <= rates_sorted["open"])
                    & (rates_sorted["low"] <= rates_sorted["close"])
                )
            )
            prices_pos = bool(
                np.all(
                    (rates_sorted["open"] > 0)
                    & (rates_sorted["high"] > 0)
                    & (rates_sorted["low"] > 0)
                    & (rates_sorted["close"] > 0)
                )
            )
            spread_ok = bool(np.all(rates_sorted["spread"] >= 0))
            tick_vol_ok = bool(np.all(rates_sorted["tick_volume"] >= 0))

            chunk_filename = f"chunk_{cid}.parquet"
            chunk_path = CHUNKS_DIR / chunk_filename

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
            pq.write_table(table, str(chunk_path))

            sha256 = compute_chunk_hash(chunk_path)
            earliest_str = datetime.fromtimestamp(earliest_ts, tz=timezone.utc).isoformat()
            latest_str = datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat()

            print(
                f"  Saved {chunk_filename}: {actual_count:,} bars ({earliest_str} -> {latest_str})"
            )
            print(
                f"  Integrity: OHLC={high_ok and low_ok and prices_pos}, "
                f"SpreadOk={spread_ok}, VolOk={tick_vol_ok}, Dups={dup_count}"
            )
            print(f"  SHA256: {sha256}")

            metadata_records.append(
                {
                    "chunk_id": cid,
                    "file_name": chunk_filename,
                    "pos": pos,
                    "requested_count": cnt,
                    "actual_count": actual_count,
                    "earliest_ts": earliest_ts,
                    "latest_ts": latest_ts,
                    "earliest_utc": earliest_str,
                    "latest_utc": latest_str,
                    "duplicate_count": dup_count,
                    "ohlc_valid": bool(high_ok and low_ok and prices_pos),
                    "spread_valid": spread_ok,
                    "volume_valid": tick_vol_ok,
                    "sha256": sha256,
                }
            )

        meta_path = REPORTS_DIR / "h4_chunk_export_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata_records, f, indent=2)
        print(f"\nMetadata successfully saved to {meta_path}")

    finally:
        mt5.shutdown()

    return 0


def main() -> int:
    try:
        import MetaTrader5  # noqa: F401

        is_wine = False
    except ImportError:
        is_wine = True

    if is_wine and sys.platform.startswith("linux"):
        print("[INFO] Delegating H4 chunk export to Wine Python...")
        return _run_in_wine(sys.argv[1:])
    else:
        return _execute_h4_chunk_export()


if __name__ == "__main__":
    sys.exit(main())
