"""Expanded historical dataset ingestion, chunk validation, and merging engine for H4 swing
research.

Validates chunk integrity, verifies bit-for-bit consistency across overlapping chunk boundaries,
merges chunks chronologically, and preserves strict holdout test governance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHUNKS_DIR = Path("data/raw/eurusd_h4_expanded/chunks")
RAW_EXPANDED_DIR = Path("data/raw/eurusd_h4_expanded")
PROCESSED_EXPANDED_DIR = Path("data/processed/eurusd_h4_expanded")
REPORTS_DIR = Path("reports")

LOCKED_TEST_START_TS = pd.Timestamp("2026-02-19 12:00:00+00:00")
RESEARCH_END_TS = pd.Timestamp("2026-02-19 10:45:00+00:00")


def validate_chunk_overlaps(
    chunks: list[pd.DataFrame],
) -> list[dict[str, Any]]:
    """Verify bit-for-bit equality across overlapping boundary bars between adjacent chunks.

    Parameters
    ----------
    chunks : list[pd.DataFrame]
        List of chunk DataFrames ordered chronologically (oldest to newest).

    Returns
    -------
    list[dict[str, Any]]
        List of overlap validation results.
    """
    overlap_results = []
    check_cols = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]

    for i in range(len(chunks) - 1):
        c_older = chunks[i]
        c_newer = chunks[i + 1]

        t_older = set(c_older["time"])
        t_newer = set(c_newer["time"])
        overlap_times = sorted(t_older.intersection(t_newer))

        if not overlap_times:
            raise ValueError(f"No overlap found between chunk {i + 1} and chunk {i + 2}.")

        sub_older = (
            c_older[c_older["time"].isin(overlap_times)].sort_values("time").reset_index(drop=True)
        )
        sub_newer = (
            c_newer[c_newer["time"].isin(overlap_times)].sort_values("time").reset_index(drop=True)
        )

        field_matches = {}
        for col in check_cols:
            is_match = bool(np.array_equal(sub_older[col].to_numpy(), sub_newer[col].to_numpy()))
            field_matches[col] = is_match
            if not is_match:
                diff_count = int((sub_older[col] != sub_newer[col]).sum())
                raise ValueError(
                    f"Discrepancy in field '{col}' between chunk {i + 1} and {i + 2}: "
                    f"{diff_count} differing rows."
                )

        t_earliest = pd.to_datetime(overlap_times[0], unit="s", utc=True).isoformat()
        t_latest = pd.to_datetime(overlap_times[-1], unit="s", utc=True).isoformat()

        overlap_results.append(
            {
                "pair": f"chunk_{i + 1}_vs_chunk_{i + 2}",
                "overlap_bars_count": len(overlap_times),
                "earliest_overlap_ts": t_earliest,
                "latest_overlap_ts": t_latest,
                "field_matches": field_matches,
                "all_fields_identical": all(field_matches.values()),
            }
        )

    return overlap_results


def merge_and_validate_h4_chunks(
    chunks: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Merge H4 chunks, remove exact duplicate boundary bars, and perform integrity checks.

    Returns
    -------
    pd.DataFrame
        Validated, strictly monotonic H4 DataFrame spanning 2010 to 2026.
    """
    if chunks is None:
        c1 = pd.read_parquet(CHUNKS_DIR / "chunk_1.parquet")
        c2 = pd.read_parquet(CHUNKS_DIR / "chunk_2.parquet")
        c3 = pd.read_parquet(CHUNKS_DIR / "chunk_3.parquet")
        # Ordered oldest to newest: chunk_3 (2010-2014), chunk_2 (2013-2020), chunk_1 (2020-2026)
        chunks = [c3, c2, c1]

    # 1. Validate bit-for-bit overlap equality
    validate_chunk_overlaps(chunks)

    # 2. Merge and deduplicate
    combined = pd.concat(chunks, ignore_index=True)
    deduped = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    deduped["timestamp"] = pd.to_datetime(deduped["time"], unit="s", utc=True)

    # 3. Integrity validations
    ts = deduped["timestamp"]
    if not ts.is_monotonic_increasing:
        raise ValueError("Merged timestamps must be strictly monotonic increasing.")

    if ts.duplicated().any():
        raise ValueError("Merged timestamps contain duplicates.")

    high_ok = (deduped["high"] >= deduped["open"]) & (deduped["high"] >= deduped["close"])
    low_ok = (deduped["low"] <= deduped["open"]) & (deduped["low"] <= deduped["close"])
    if not (high_ok.all() and low_ok.all()):
        raise ValueError("Merged H4 bars contain OHLC geometric violations.")

    if (deduped[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Merged H4 bars contain non-positive prices.")

    if deduped.isna().any().any():
        raise ValueError("Merged H4 bars contain unexpected NaN values.")

    # 4. Save merged Parquet files to research paths (never overwriting canonical M15 files)
    RAW_EXPANDED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_EXPANDED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_EXPANDED_DIR / "eurusd_h4_raw.parquet"
    proc_path = PROCESSED_EXPANDED_DIR / "eurusd_h4_processed.parquet"

    deduped.to_parquet(raw_path, index=False)
    deduped.to_parquet(proc_path, index=False)

    return deduped


def get_expanded_research_data(
    df_h4: pd.DataFrame | None = None,
    research_end_ts: pd.Timestamp = RESEARCH_END_TS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract expanded pre-test research partition and held-out test partition.

    Guarantees that test partition (from 2026-02-19 12:00:00 UTC onward) is strictly separated.
    """
    if df_h4 is None:
        proc_path = PROCESSED_EXPANDED_DIR / "eurusd_h4_processed.parquet"
        if not proc_path.exists():
            df_h4 = merge_and_validate_h4_chunks()
        else:
            df_h4 = pd.read_parquet(proc_path)
            df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"], utc=True)

    ts = df_h4["timestamp"]
    res_mask = ts <= research_end_ts
    test_mask = ts >= LOCKED_TEST_START_TS

    df_research = df_h4[res_mask].copy().reset_index(drop=True)
    df_test = df_h4[test_mask].copy().reset_index(drop=True)

    return df_research, df_test
