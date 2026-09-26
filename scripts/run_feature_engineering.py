#!/usr/bin/env python3
"""EURUSD M15 Feature Engineering Pipeline Runner.

Loads the validated M15 dataset, computes 80 point-in-time quantitative features,
saves the output feature dataset (Parquet) and generates comprehensive feature metadata (JSON).

Usage:
    uv run python scripts/run_feature_engineering.py [--input-path PATH]
                                                     [--output-path PATH]
                                                     [--metadata-path PATH]
                                                     [--drop-warmup]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure root repository is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from ai.features import (  # noqa: E402
    MAX_LOOKBACK_BARS,
    build_feature_pipeline,
    generate_feature_metadata,
)


def find_input_file(preferred_path: Path | None = None) -> Path:
    """Locate the validated processed or raw EURUSD M15 dataset."""
    if preferred_path and preferred_path.is_file():
        return preferred_path

    candidates = [
        REPO_ROOT / "data" / "processed" / "eurusd_m15" / "eurusd_m15_processed.parquet",
        REPO_ROOT / "data" / "raw" / "eurusd_m15" / "eurusd_m15_raw.parquet",
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(f"EURUSD M15 dataset not found in candidate locations: {candidates}")


def main() -> int:
    """Execute the feature engineering pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Run Phase 5 EURUSD M15 Feature Engineering Pipeline."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to input Parquet market data file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "data" / "features" / "eurusd_m15" / "eurusd_m15_features.parquet",
        help="Path to save output feature Parquet file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "reports" / "feature_metadata.json",
        help="Path to save JSON feature metadata.",
    )
    parser.add_argument(
        "--drop-warmup",
        action="store_true",
        help=f"Drop initial {MAX_LOOKBACK_BARS} warm-up bars where rolling features are NaN.",
    )

    args = parser.parse_args()

    input_file = find_input_file(args.input_path)
    print(f"Loading input dataset from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df):,} candles.")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    print("Computing quantitative features across 7 groups...")
    df_features, feature_defs = build_feature_pipeline(
        df, drop_warmup=args.drop_warmup, include_raw_columns=True
    )

    print(f"Generating feature metadata for {len(feature_defs)} features...")
    metadata = generate_feature_metadata(df_features, feature_defs)

    # Persist feature dataset
    print(f"Saving feature dataset to: {output_path}")
    df_features.to_parquet(output_path, compression="snappy", index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Persist metadata JSON
    print(f"Saving feature metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Console summary
    print("=" * 65)
    print("EURUSD M15 FEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"Input Candles:         {len(df):,}")
    print(f"Output Dimensions:     {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")
    print(f"Total Features:        {len(feature_defs)}")
    print(f"Max Lookback Window:   {MAX_LOOKBACK_BARS} bars (20.0 hours)")
    print(f"Warm-up Rows Dropped:  {args.drop_warmup}")
    print(f"Output File Size:      {file_size_mb:.2f} MB")
    print("Feature Group Counts:")
    for group, count in sorted(metadata["group_counts"].items()):
        print(f"  - {group:<12}: {count:>2} features")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
