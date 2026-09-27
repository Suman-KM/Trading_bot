#!/usr/bin/env python3
"""EURUSD M15 Label Engineering Pipeline Runner.

Loads the validated M15 dataset, generates multi-horizon forward returns and
ternary directional classification labels, saves the output dataset (Parquet),
and generates comprehensive label metadata (JSON).

Usage:
    uv run python scripts/run_label_engineering.py [--input-path PATH]
                                                   [--output-path PATH]
                                                   [--metadata-path PATH]
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

from ai.labels import (  # noqa: E402
    DEFAULT_HORIZONS,
    build_label_pipeline,
    generate_label_metadata,
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
    """Execute the label engineering pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Run Phase 6 EURUSD M15 Label Engineering Pipeline."
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
        default=REPO_ROOT / "data" / "labels" / "eurusd_m15" / "eurusd_m15_labels.parquet",
        help="Path to save output target label Parquet file.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=REPO_ROOT / "reports" / "label_metadata.json",
        help="Path to save JSON label metadata.",
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

    print(f"Generating targets across horizons {DEFAULT_HORIZONS}...")
    df_labels, label_defs = build_label_pipeline(
        df, horizons=DEFAULT_HORIZONS, include_timestamp=True
    )

    spread_series = df["spread"] if "spread" in df.columns else None
    print(f"Generating label metadata for {len(label_defs)} targets...")
    metadata = generate_label_metadata(df_labels, label_defs, spread_series=spread_series)

    # Persist label dataset
    print(f"Saving label dataset to: {output_path}")
    df_labels.to_parquet(output_path, compression="snappy", index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Persist metadata JSON
    print(f"Saving label metadata to: {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Console summary
    print("=" * 68)
    print("EURUSD M15 LABEL ENGINEERING SUMMARY")
    print("=" * 68)
    print(f"Input Candles:         {len(df):,}")
    print(f"Output Dimensions:     {df_labels.shape[0]:,} rows x {df_labels.shape[1]} columns")
    print(f"Total Targets:         {len(label_defs)}")
    print(f"Output File Size:      {file_size_mb:.2f} MB")
    print("\nDirectional Class Distributions (Fixed Thresholds):")
    for ld in metadata["labels"]:
        if ld["label_type"] == "multiclass_direction":
            h = ld["horizon_bars"]
            mins = ld["horizon_minutes"]
            pcts = ld["class_percentages"]
            print(
                f"  - Horizon {h:>2} bars ({mins:>3}m) [thresh={ld['threshold'] * 10000:.1f} bps]: "
                f"SHORT: {pcts['SHORT (-1)']:>5.2f}% | "
                f"NEUTRAL: {pcts['NEUTRAL (0)']:>5.2f}% | "
                f"LONG: {pcts['LONG (+1)']:>5.2f}% "
                f"(NaNs: {ld['nan_count']})"
            )

    print("\nDirectional Class Distributions (Vol-Scaled Thresholds):")
    for ld in metadata["labels"]:
        if ld["label_type"] == "volatility_scaled_direction":
            h = ld["horizon_bars"]
            mins = ld["horizon_minutes"]
            pcts = ld["class_percentages"]
            print(
                f"  - Horizon {h:>2} bars ({mins:>3}m) [thresh=0.5*sqrt(h)*ATR]: "
                f"SHORT: {pcts['SHORT (-1)']:>5.2f}% | "
                f"NEUTRAL: {pcts['NEUTRAL (0)']:>5.2f}% | "
                f"LONG: {pcts['LONG (+1)']:>5.2f}% "
                f"(NaNs: {ld['nan_count']})"
            )

    print("\nForward Return Dispersion:")
    for ld in metadata["labels"]:
        if ld["label_type"] == "continuous_return":
            h = ld["horizon_bars"]
            std_bps = ld["std"] * 10000
            p05 = ld["quantiles"]["p05"] * 10000
            p95 = ld["quantiles"]["p95"] * 10000
            print(
                f"  - Horizon {h:>2} bars: std={std_bps:>5.1f} bps | "
                f"min={ld['min'] * 100:>6.2f}% | max={ld['max'] * 100:>6.2f}% | "
                f"p05={p05:>6.1f} bps | p95={p95:>6.1f} bps"
            )
    print("=" * 68)

    return 0


if __name__ == "__main__":
    sys.exit(main())
