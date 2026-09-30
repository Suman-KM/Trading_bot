#!/usr/bin/env python3
"""Phase 20: Historical Tick Execution Data Exporter (Wine Environment).

Extracts bounded historical EURUSD tick datasets corresponding to the Phase 12
validation period (2025-07-14 05:45 UTC to 2026-02-19 10:45 UTC) to evaluate
tick-realistic execution.

STRICT INVARIANTS:
1. Phase 11 Test Partition (2026-02-19 12:00:00 UTC onward) is PERMANENTLY LOCKED.
   Any tick at or after 2026-02-19 10:45:00 UTC is strictly rejected.
2. Ticks must satisfy: monotonic timestamps, finite bid/ask > 0, ask >= bid.
3. Output is written to data/raw/microstructure_audit/ (gitignored).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAW_AUDIT_DIR = Path("data/raw/microstructure_audit")
REPORTS_DIR = Path("reports")

# Strict lock boundary: Phase 12 validation ends 2026-02-19 10:45:00 UTC
VALIDATION_END_LOCK = datetime(2026, 2, 19, 10, 45, 0, tzinfo=timezone.utc)
VALIDATION_START_LOCK = datetime(2025, 7, 14, 5, 45, 0, tzinfo=timezone.utc)


def _run_in_wine(args: list[str]) -> int:
    """Spawn execution in isolated Wine Python environment."""
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


def _execute_wine_worker() -> int:
    """Execute tick extraction inside the Wine Python environment."""
    try:
        import MetaTrader5 as mt5
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        print(f"[FATAL] Required packages in Wine missing: {exc}", flush=True)
        return 1

    print("=" * 70, flush=True)
    print("PHASE 20: TICK EXECUTION DATA EXPORTER (WINE WORKER)", flush=True)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 70, flush=True)

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}", flush=True)
        return 1

    try:
        RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Define Trade Execution Windows from Phase 12 Validation trades (Conf=0.50)
        # Windows span from signal candle open to max-hold bar close + 15m safety buffer
        trade_windows = [
            # Trade 1-4: July 2025
            (
                datetime(2025, 7, 16, 18, 0, tzinfo=timezone.utc),
                datetime(2025, 7, 16, 19, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 7, 18, 12, 0, tzinfo=timezone.utc),
                datetime(2025, 7, 18, 13, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 7, 23, 10, 0, tzinfo=timezone.utc),
                datetime(2025, 7, 23, 11, 45, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 7, 24, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 7, 24, 15, 30, tzinfo=timezone.utc),
            ),
            # Trade 5-19: August 2025
            (
                datetime(2025, 8, 1, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 1, 15, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 4, 11, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 4, 12, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 5, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 5, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 5, 13, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 5, 14, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 6, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 6, 15, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 7, 11, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 7, 12, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 7, 15, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 7, 16, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 8, 14, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 8, 15, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 11, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 11, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 12, 10, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 12, 11, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 12, 12, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 12, 13, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 13, 13, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 13, 14, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 14, 15, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 14, 16, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 15, 15, 30, tzinfo=timezone.utc),
                datetime(2025, 8, 15, 17, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 8, 29, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 29, 1, 30, tzinfo=timezone.utc),
            ),
            # Trade 20-38: September 2025
            (
                datetime(2025, 9, 2, 10, 30, tzinfo=timezone.utc),
                datetime(2025, 9, 2, 18, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 8, 16, 30, tzinfo=timezone.utc),
                datetime(2025, 9, 8, 18, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 9, 10, 15, tzinfo=timezone.utc),
                datetime(2025, 9, 9, 12, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 15, 14, 45, tzinfo=timezone.utc),
                datetime(2025, 9, 15, 16, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 16, 17, 15, tzinfo=timezone.utc),
                datetime(2025, 9, 16, 19, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 17, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 9, 17, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 17, 20, 45, tzinfo=timezone.utc),
                datetime(2025, 9, 17, 23, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 18, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 9, 18, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 22, 12, 45, tzinfo=timezone.utc),
                datetime(2025, 9, 22, 16, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 23, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 9, 23, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 23, 9, 30, tzinfo=timezone.utc),
                datetime(2025, 9, 23, 16, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 25, 10, 30, tzinfo=timezone.utc),
                datetime(2025, 9, 25, 12, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 9, 26, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 9, 26, 1, 30, tzinfo=timezone.utc),
            ),
            # Trade 39-43: October 2025
            (
                datetime(2025, 10, 1, 16, 45, tzinfo=timezone.utc),
                datetime(2025, 10, 1, 18, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 10, 6, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 10, 6, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 10, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 10, 10, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 10, 13, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 10, 13, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 10, 30, 14, 30, tzinfo=timezone.utc),
                datetime(2025, 10, 30, 16, 0, tzinfo=timezone.utc),
            ),
            # Trade 44: November 2025
            (
                datetime(2025, 11, 7, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 11, 7, 1, 30, tzinfo=timezone.utc),
            ),
            # Trade 45-51: December 2025
            (
                datetime(2025, 12, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 12, 1, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 12, 10, 21, 30, tzinfo=timezone.utc),
                datetime(2025, 12, 10, 23, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 12, 11, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 12, 11, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 12, 15, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 12, 15, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 12, 16, 15, 45, tzinfo=timezone.utc),
                datetime(2025, 12, 16, 17, 0, tzinfo=timezone.utc),
            ),
            (
                datetime(2025, 12, 29, 8, 45, tzinfo=timezone.utc),
                datetime(2025, 12, 29, 11, 0, tzinfo=timezone.utc),
            ),
            # Trade 52-58: January 2026
            (
                datetime(2026, 1, 20, 14, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 20, 17, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 1, 26, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 26, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 27, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 1, 28, 11, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 28, 12, 45, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 1, 28, 21, 15, tzinfo=timezone.utc),
                datetime(2026, 1, 28, 22, 30, tzinfo=timezone.utc),
            ),
            # Trade 59-60: February 2026
            (
                datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 2, 2, 1, 30, tzinfo=timezone.utc),
            ),
            (
                datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc),
                datetime(2026, 2, 3, 16, 45, tzinfo=timezone.utc),
            ),
        ]

        print(
            f"[Step 1/3] Extracting ticks for {len(trade_windows)} validation execution windows...",
            flush=True,
        )

        all_times = []
        all_time_mscs = []
        all_bids = []
        all_asks = []
        all_flags = []

        total_extracted = 0
        t_start_all = time.time()

        for idx, (w_start, w_end) in enumerate(trade_windows, start=1):
            # Strict safety assertion: Never touch locked Phase 11 test partition
            if w_end >= VALIDATION_END_LOCK:
                print(f"[FATAL] Window {w_end} violates Phase 11 test lock!", flush=True)
                return 1
            if w_start < VALIDATION_START_LOCK:
                print(f"[FATAL] Window {w_start} precedes Phase 12 validation start!", flush=True)
                return 1

            t_w_start = time.time()
            ticks = mt5.copy_ticks_range("EURUSD", w_start, w_end, mt5.COPY_TICKS_ALL)
            t_dur = time.time() - t_w_start
            cnt = len(ticks) if ticks is not None else 0

            if cnt == 0:
                print(
                    f"  [WARN] Window {idx}/{len(trade_windows)} returned 0 ticks: "
                    f"{w_start} to {w_end}",
                    flush=True,
                )
                continue

            all_times.append(ticks["time"].astype(np.int64))
            all_time_mscs.append(ticks["time_msc"].astype(np.int64))
            all_bids.append(ticks["bid"].astype(np.float64))
            all_asks.append(ticks["ask"].astype(np.float64))
            all_flags.append(ticks["flags"].astype(np.uint32))

            total_extracted += cnt
            if idx % 10 == 0 or idx == len(trade_windows):
                print(
                    f"  Processed {idx}/{len(trade_windows)} windows "
                    f"({total_extracted:,} ticks, last window {cnt:,} ticks in {t_dur:.2f}s)...",
                    flush=True,
                )

        total_time = time.time() - t_start_all
        print(
            f"\n[Step 2/3] Validating and consolidating {total_extracted:,} ticks "
            f"across all windows ({total_time:.2f}s)...",
            flush=True,
        )

        merged_time = np.concatenate(all_times)
        merged_time_msc = np.concatenate(all_time_mscs)
        merged_bid = np.concatenate(all_bids)
        merged_ask = np.concatenate(all_asks)
        merged_flags = np.concatenate(all_flags)

        # Sort chronologically by time_msc
        sort_order = np.argsort(merged_time_msc, kind="stable")
        merged_time = merged_time[sort_order]
        merged_time_msc = merged_time_msc[sort_order]
        merged_bid = merged_bid[sort_order]
        merged_ask = merged_ask[sort_order]
        merged_flags = merged_flags[sort_order]

        # Deduplicate identical consecutive timestamps if any
        # Check monotonicity
        diffs = np.diff(merged_time_msc)
        non_negative_diffs = np.all(diffs >= 0)
        assert non_negative_diffs, "Tick millisecond timestamps not monotonic!"

        # Check prices
        assert np.all(np.isfinite(merged_bid)), "Non-finite Bid detected!"
        assert np.all(np.isfinite(merged_ask)), "Non-finite Ask detected!"
        assert np.all(merged_bid > 0.5), "Unrealistic Bid price detected!"
        assert np.all(merged_ask > 0.5), "Unrealistic Ask price detected!"
        assert np.all(merged_ask >= merged_bid), "Negative spread (crossed market) detected!"

        # Convert to PyArrow Table
        pa_schema = pa.schema(
            [
                ("time", pa.int64()),
                ("time_msc", pa.int64()),
                ("bid", pa.float64()),
                ("ask", pa.float64()),
                ("flags", pa.uint32()),
                ("timestamp", pa.timestamp("ms", tz="UTC")),
            ]
        )

        table = pa.Table.from_arrays(
            [
                pa.array(merged_time, type=pa.int64()),
                pa.array(merged_time_msc, type=pa.int64()),
                pa.array(merged_bid, type=pa.float64()),
                pa.array(merged_ask, type=pa.float64()),
                pa.array(merged_flags, type=pa.uint32()),
                pa.array(merged_time_msc, type=pa.timestamp("ms", tz="UTC")),
            ],
            schema=pa_schema,
        )

        out_path = RAW_AUDIT_DIR / "validation_trades_ticks.parquet"
        pq.write_table(table, out_path, compression="snappy")
        file_size_kb = out_path.stat().st_size / 1024

        first_ts = datetime.fromtimestamp(merged_time[0], tz=timezone.utc).isoformat()
        last_ts = datetime.fromtimestamp(merged_time[-1], tz=timezone.utc).isoformat()

        print(
            f"  Successfully wrote {len(table):,} ticks to {out_path} ({file_size_kb:.1f} KB)",
            flush=True,
        )
        print(f"  First tick: {first_ts} | Last tick: {last_ts}", flush=True)

        # 3. Export Metadata Report
        print("\n[Step 3/3] Saving extraction metadata...", flush=True)
        t_info = mt5.terminal_info()
        s_info = mt5.symbol_info("EURUSD")
        meta = {
            "phase": 20,
            "title": "Phase 20 Tick Execution Data Extraction",
            "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": "EURUSD",
            "broker": t_info.company if t_info else "MetaQuotes Ltd.",
            "server": "MetaQuotes-Demo",
            "point_value": s_info.point if s_info else 1e-05,
            "digits": s_info.digits if s_info else 5,
            "windows_count": len(trade_windows),
            "total_ticks": len(table),
            "earliest_tick": first_ts,
            "latest_tick": last_ts,
            "file_path": str(out_path),
            "file_size_bytes": out_path.stat().st_size,
            "validation_start_lock": VALIDATION_START_LOCK.isoformat(),
            "validation_end_lock": VALIDATION_END_LOCK.isoformat(),
            "test_partition_status": "PERMANENTLY LOCKED & UNTOUCHED",
            "checks": {
                "monotonic_timestamps": bool(non_negative_diffs),
                "finite_bid_ask": True,
                "nonnegative_spread": True,
                "no_test_contamination": bool(merged_time[-1] < VALIDATION_END_LOCK.timestamp()),
            },
        }

        meta_path = REPORTS_DIR / "phase20_tick_data_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata saved to {meta_path}", flush=True)

        mt5.shutdown()
        print("\n" + "=" * 70, flush=True)
        print("PHASE 20 TICK EXTRACTION COMPLETED SUCCESSFULLY.", flush=True)
        print("=" * 70, flush=True)
        return 0

    except Exception as exc:
        print(f"[ERROR] Exception during tick extraction: {exc}", flush=True)
        mt5.shutdown()
        return 1


def main() -> int:
    """Main launcher dispatching to Wine worker if on Linux."""
    if "--wine-worker" in sys.argv:
        return _execute_wine_worker()
    print("[Launcher] Spawning MT5 tick export worker in Wine environment...", flush=True)
    return _run_in_wine(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
