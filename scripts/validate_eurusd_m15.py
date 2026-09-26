#!/usr/bin/env python3
"""EURUSD M15 Market Data Validation and Ingestion Pipeline CLI.

Usage:
    uv run python scripts/validate_eurusd_m15.py [--raw-path PATH] [--output-dir PATH] [--export-csv]

If raw market data has not yet been imported from Member 1, the script reports:
'RAW EURUSD M15 DATA NOT YET AVAILABLE LOCALLY — PIPELINE READY FOR DATA IMPORT.'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure root repository is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.data.ingestion import (
    generate_dataset_metadata,
    load_raw_dataset,
    process_and_validate,
    save_dataset_metadata,
    save_processed_dataset,
    save_quality_report,
)


def find_raw_file(raw_dir: Path, preferred_path: Path | None = None) -> Path | None:
    """Locate the raw EURUSD M15 data file."""
    if preferred_path and preferred_path.is_file():
        return preferred_path

    # Check standard filenames in raw_dir
    candidates = [
        raw_dir / "eurusd_m15_raw.parquet",
        raw_dir / "eurusd_m15_raw.csv",
        raw_dir / "EURUSD_M15.parquet",
        raw_dir / "EURUSD_M15.csv",
        raw_dir / "eurusd_m15.parquet",
        raw_dir / "eurusd_m15.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # Search for any non-hidden .parquet or .csv in raw_dir
    if raw_dir.is_dir():
        for p in raw_dir.glob("*.parquet"):
            if p.is_file() and not p.name.startswith("."):
                return p
        for c in raw_dir.glob("*.csv"):
            if c.is_file() and not c.name.startswith("."):
                return c

    return None


def run_pipeline(
    raw_path: Path | None,
    output_dir: Path,
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    export_csv: bool = False,
) -> int:
    """Execute the data validation pipeline."""
    raw_dir = REPO_ROOT / "data" / "raw" / "eurusd_m15"
    target_raw_file = find_raw_file(raw_dir, raw_path)

    if target_raw_file is None:
        print("=" * 78)
        print("RAW EURUSD M15 DATA NOT YET AVAILABLE LOCALLY — PIPELINE READY FOR DATA IMPORT.")
        print("=" * 78)
        print("\nExpected MT5 Data Specification from Member 1:")
        print("  - Target directory: data/raw/eurusd_m15/")
        print("  - Recommended name: eurusd_m15_raw.parquet (or eurusd_m15_raw.csv)")
        print(
            "  - Required columns: time, open, high, low, close, tick_volume, spread, real_volume"
        )
        print("  - Timezone:         Unix epoch seconds (UTC)")
        print("  - Expected candles: ~75,000 M15 bars (2023-09-18 to 2026-09-25)")
        print("\nOnce Member 1 transfers the MT5 export file into data/raw/eurusd_m15/, re-run:")
        print("  uv run python scripts/validate_eurusd_m15.py\n")
        return 0

    print(f"Loading raw dataset from: {target_raw_file}")
    df_raw = load_raw_dataset(target_raw_file)
    print(f"Raw dataset loaded: {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

    print("Running validation and UTC timestamp enrichment...")
    df_processed, report = process_and_validate(
        df_raw,
        symbol=symbol,
        timeframe=timeframe,
        source="MT5 (MetaQuotes Ltd. / MetaQuotes-Demo)",
    )

    # Persist processed artifacts
    processed_parquet = save_processed_dataset(
        df_processed,
        output_dir=output_dir,
        symbol=symbol,
        timeframe=timeframe,
        export_csv=export_csv,
    )
    quality_report_path = save_quality_report(report, output_dir=output_dir)
    metadata = generate_dataset_metadata(
        df_processed,
        report,
        broker="MetaQuotes Ltd.",
        server="MetaQuotes-Demo",
        pipeline_version="1.0.0",
    )
    metadata_path = save_dataset_metadata(metadata, output_dir=output_dir)

    # Print summary report
    print("\n" + "=" * 78)
    print(f"DATA VALIDATION REPORT: {symbol} {timeframe}")
    print("=" * 78)
    print(f"Validation Status:          {report.validation_status}")
    print(f"Total Rows:                 {report.row_count:,}")
    print(f"Earliest Candle (UTC):      {report.earliest_timestamp_utc}")
    print(f"Latest Candle (UTC):        {report.latest_timestamp_utc}")
    print("-" * 78)
    print("Quality Checks:")
    print(f"  - Duplicate Timestamps:   {report.timestamps.duplicate_count}")
    print(f"  - Duplicate Rows:         {report.duplicate_rows_count}")
    print(
        f"  - Chronological Order:    {'PASS' if report.timestamps.is_strictly_monotonic else 'FAIL'}"
    )
    print(f"  - OHLC Violations:        {report.ohlc.total_violations}")
    print(
        f"  - Invalid Prices (<=0/NaN): {report.ohlc.non_positive_prices_count + report.ohlc.nan_prices_count}"
    )
    print(f"  - Missing Values:         {report.missing_values_count}")
    print(f"  - Negative Spread:        {report.spread.negative_spread_count}")
    print(f"  - Zero Tick Volume:       {report.volume.zero_tick_volume_count}")
    print(f"  - Real Volume Zero Rate:  {report.volume.real_volume_zero_percentage:.1f}%")
    print("-" * 78)
    print("Gap Analysis:")
    print(f"  - Total Gaps (>15m):      {report.gaps.total_gaps}")
    print(f"  - Expected Weekends:      {report.gaps.expected_weekend_count}")
    print(f"  - Expected Holidays:      {report.gaps.expected_holiday_count}")
    print(f"  - Unexpected Gaps:        {report.gaps.unexpected_gap_count}")
    print(f"  - Unclassified Gaps:      {report.gaps.unclassified_gap_count}")
    print("-" * 78)
    print("Artifacts Generated:")
    print(f"  - Processed Dataset:      {processed_parquet}")
    print(f"  - Quality Report:         {quality_report_path}")
    print(f"  - Dataset Metadata:       {metadata_path}")
    print("=" * 78)

    if report.validation_status == "PASS":
        print("\nDATASET READY FOR PHASE 4 (Exploratory Data Analysis).\n")
        return 0
    else:
        print("\nVALIDATION FAILED WITH CRITICAL ERRORS:")
        for err in report.critical_errors:
            print(f"  [ERROR] {err}")
        print()
        return 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="EURUSD M15 Data Validation Pipeline")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to raw market data file (.parquet or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "eurusd_m15",
        help="Directory where processed artifacts will be written",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export a processed CSV copy alongside Parquet",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Trading instrument symbol (default: EURUSD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="M15",
        help="Candle timeframe (default: M15)",
    )

    args = parser.parse_args()
    code = run_pipeline(
        raw_path=args.raw_path,
        output_dir=args.output_dir,
        symbol=args.symbol,
        timeframe=args.timeframe,
        export_csv=args.export_csv,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
