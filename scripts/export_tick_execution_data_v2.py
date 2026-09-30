#!/usr/bin/env python3
"""Phase 21.1: Corrected Historical Tick Data Exporter (v2).

Repairs tick coverage by performing bounded extractions from MT5 for all missing
Phase 12 validation trade execution windows, producing:
data/raw/microstructure_audit/validation_trades_ticks_v2.parquet

STRICT INVARIANTS:
1. Phase 11 Test Partition (2026-02-19 12:00:00 UTC onward) is PERMANENTLY LOCKED.
   Any tick at or after 2026-02-19 10:45:00 UTC is strictly rejected.
2. The original Phase 20 file (validation_trades_ticks.parquet) is preserved untouched.
3. Bounded retrieval only (signal candle to max-hold bar close + 15m buffer).
4. Hard timeout of 2 minutes on extraction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAW_AUDIT_DIR = Path("data/raw/microstructure_audit")
REPORTS_DIR = Path("reports")

VALIDATION_END_LOCK = datetime(2026, 2, 19, 10, 45, 0, tzinfo=timezone.utc)
VALIDATION_START_LOCK = datetime(2025, 7, 14, 5, 45, 0, tzinfo=timezone.utc)


def _run_in_wine(args: list[str]) -> int:
    """Spawn execution in isolated Wine Python environment with timeout."""
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
    try:
        proc = subprocess.run(wine_cmd, env=env, timeout=120)  # 2 minute hard timeout
        return proc.returncode
    except subprocess.TimeoutExpired:
        print("[FATAL] MT5 tick extraction timed out after 120s!", flush=True)
        return 1


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
    print("PHASE 21.1: TICK EXECUTION DATA EXPORTER V2 (WINE WORKER)", flush=True)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 70, flush=True)

    if not mt5.initialize():
        print(f"[FATAL] MT5 initialization failed: {mt5.last_error()}", flush=True)
        return 1

    try:
        RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load original Phase 20 tick file via PyArrow
        v1_path = RAW_AUDIT_DIR / "validation_trades_ticks.parquet"
        if not v1_path.exists():
            print(f"[FATAL] Original v1 file not found: {v1_path}", flush=True)
            mt5.shutdown()
            return 1

        v1_table = pq.read_table(v1_path)
        print(f"[Step 1/4] Loaded original Phase 20 dataset: {len(v1_table):,} ticks", flush=True)

        # 2. Load missing extraction windows from json plan
        windows_path = RAW_AUDIT_DIR / "extraction_windows.json"
        if not windows_path.exists():
            print(f"[FATAL] Windows specification file not found: {windows_path}", flush=True)
            mt5.shutdown()
            return 1

        with open(windows_path, "r", encoding="utf-8") as f:
            windows_data = json.load(f)

        print(
            f"[Step 2/4] Loaded {len(windows_data)} missing execution windows to extract",
            flush=True,
        )

        all_times = [v1_table.column("time").to_numpy().astype(np.int64)]
        all_time_mscs = [v1_table.column("time_msc").to_numpy().astype(np.int64)]
        all_bids = [v1_table.column("bid").to_numpy().astype(np.float64)]
        all_asks = [v1_table.column("ask").to_numpy().astype(np.float64)]
        if "flags" in v1_table.column_names:
            all_flags = [v1_table.column("flags").to_numpy().astype(np.uint32)]
        else:
            all_flags = [np.zeros(len(v1_table), dtype=np.uint32)]

        # 3. For each missing window, extract from MT5
        new_extracted_ticks = 0
        extracted_windows_count = 0

        print(
            "\n[Step 3/4] Performing bounded extractions for missing trade windows...", flush=True
        )
        for item in windows_data:
            t_id = int(item["trade_id"])
            w_start = datetime.fromisoformat(item["window_start"])
            w_end = datetime.fromisoformat(item["window_end"])

            # Strict safety assertion
            if w_end >= VALIDATION_END_LOCK:
                print(f"[FATAL] Window for trade {t_id} violates Phase 11 test lock!", flush=True)
                mt5.shutdown()
                return 1

            ticks = mt5.copy_ticks_range("EURUSD", w_start, w_end, mt5.COPY_TICKS_ALL)
            cnt = len(ticks) if ticks is not None else 0
            if cnt > 0:
                all_times.append(ticks["time"].astype(np.int64))
                all_time_mscs.append(ticks["time_msc"].astype(np.int64))
                all_bids.append(ticks["bid"].astype(np.float64))
                all_asks.append(ticks["ask"].astype(np.float64))
                all_flags.append(ticks["flags"].astype(np.uint32))
                new_extracted_ticks += cnt
                extracted_windows_count += 1
                q_str = f"{ticks[0][1]:.5f}/{ticks[0][2]:.5f}"
                print(
                    f"  Trade {t_id:2d} ({w_start.strftime('%Y-%m-%d %H:%M')}): "
                    f"retrieved {cnt:,} ticks (Quote: {q_str})",
                    flush=True,
                )
            else:
                print(f"  Trade {t_id:2d}: [WARN] MT5 returned 0 ticks for {w_start}", flush=True)

        print(
            f"\nRetrieved {new_extracted_ticks:,} new ticks across "
            f"{extracted_windows_count} missing windows.",
            flush=True,
        )

        # 4. Consolidate, sort, and deduplicate v2 dataset
        print("\n[Step 4/4] Consolidating and validating v2 dataset...", flush=True)
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

        # Simple deduplication using numpy: adjacent ticks with identical time_msc, bid, ask
        # Create mask of duplicates
        diff_time = np.diff(merged_time_msc)
        diff_bid = np.diff(merged_bid)
        diff_ask = np.diff(merged_ask)
        is_dup = (diff_time == 0) & (diff_bid == 0.0) & (diff_ask == 0.0)
        keep_mask = np.ones(len(merged_time_msc), dtype=bool)
        keep_mask[1:] = ~is_dup

        merged_time = merged_time[keep_mask]
        merged_time_msc = merged_time_msc[keep_mask]
        merged_bid = merged_bid[keep_mask]
        merged_ask = merged_ask[keep_mask]
        merged_flags = merged_flags[keep_mask]

        # Validate invariants
        diffs = np.diff(merged_time_msc)
        assert np.all(diffs >= 0), "Monotonicity violated in v2!"
        assert np.all(np.isfinite(merged_bid)), "Non-finite bid in v2!"
        assert np.all(np.isfinite(merged_ask)), "Non-finite ask in v2!"
        assert np.all(merged_bid > 0.5), "Unrealistic bid in v2!"
        assert np.all(merged_ask > 0.5), "Unrealistic ask in v2!"
        assert np.all(merged_ask >= merged_bid), "Negative spread in v2!"

        lock_ms = int(VALIDATION_END_LOCK.timestamp() * 1000)
        assert merged_time_msc[-1] < lock_ms, (
            f"Test lock violation: {merged_time_msc[-1]} >= {lock_ms}"
        )

        # Write to validation_trades_ticks_v2.parquet
        out_v2_path = RAW_AUDIT_DIR / "validation_trades_ticks_v2.parquet"
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
        table_v2 = pa.Table.from_arrays(
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

        pq.write_table(table_v2, out_v2_path, compression="snappy")
        file_size_kb = out_v2_path.stat().st_size / 1024

        first_ts = datetime.fromtimestamp(merged_time_msc[0] / 1000.0, tz=timezone.utc).isoformat()
        last_ts = datetime.fromtimestamp(merged_time_msc[-1] / 1000.0, tz=timezone.utc).isoformat()

        print(
            f"Successfully created v2 dataset: {len(table_v2):,} ticks ({file_size_kb:.1f} KB)",
            flush=True,
        )
        print(f"First tick: {first_ts} | Last tick: {last_ts}", flush=True)

        mt5.shutdown()
        print("\n" + "=" * 70, flush=True)
        print("PHASE 21.1 TICK EXTRACTION V2 COMPLETED SUCCESSFULLY.", flush=True)
        print("=" * 70, flush=True)
        return 0

    except Exception as exc:
        print(f"[ERROR] Exception during tick extraction v2: {exc}", flush=True)
        mt5.shutdown()
        return 1


def main() -> int:
    """Main launcher generating extraction plan and dispatching to Wine worker."""
    if "--wine-worker" in sys.argv:
        return _execute_wine_worker()

    import pandas as pd

    print("[Launcher] Preparing extraction plan for missing windows...", flush=True)
    RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    v1_path = RAW_AUDIT_DIR / "validation_trades_ticks.parquet"
    ledger_path = REPORTS_DIR / "phase21_trade_ledger.csv"

    v1_df = pd.read_parquet(v1_path)
    v1_times_ms = v1_df["time_msc"].to_numpy()
    ledger_df = pd.read_csv(ledger_path)

    missing_windows = []
    for _, row in ledger_df.iterrows():
        t_id = int(row["trade_id"])
        sig_dt = pd.to_datetime(row["signal_time"], format="mixed", utc=True).to_pydatetime()
        w_start = sig_dt
        w_end = sig_dt + timedelta(minutes=90)
        start_ms = int(w_start.timestamp() * 1000)
        end_ms = int(w_end.timestamp() * 1000)

        idx_0 = int(v1_times_ms.searchsorted(start_ms))
        idx_1 = int(v1_times_ms.searchsorted(end_ms))
        cnt = idx_1 - idx_0
        if cnt < 50:
            missing_windows.append(
                {
                    "trade_id": t_id,
                    "window_start": w_start.isoformat(),
                    "window_end": w_end.isoformat(),
                }
            )

    plan_path = RAW_AUDIT_DIR / "extraction_windows.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(missing_windows, f, indent=2)

    print(
        f"[Launcher] Identified {len(missing_windows)} missing windows. Plan saved to {plan_path}.",
        flush=True,
    )
    print("[Launcher] Spawning MT5 tick export worker v2 in Wine environment...", flush=True)
    return _run_in_wine(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
