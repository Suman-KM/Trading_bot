#!/usr/bin/env python3
"""EURUSD M15 Exploratory Data Analysis (EDA) Runner.

Loads the validated Parquet dataset, executes the full descriptive EDA pipeline,
generates research figures, and persists comprehensive summary metrics.

Usage:
    uv run python scripts/run_eda.py [--data-path PATH]
                                    [--figures-dir DIR]
                                    [--summary-path PATH]
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

from ai.data.eda import run_full_eda  # noqa: E402


def find_data_file(preferred_path: Path | None = None) -> Path:
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
    """Execute the EDA pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Run Phase 4 EURUSD M15 Exploratory Data Analysis."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to Parquet market data file.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "figures",
        help="Directory to save generated research plots.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=REPO_ROOT / "reports" / "eda_summary.json",
        help="Path to save JSON summary metrics.",
    )

    args = parser.parse_args()

    data_file = find_data_file(args.data_path)
    print(f"Loading dataset from: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df):,} candles across {len(df.columns)} columns.")

    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Executing EDA suite and generating plots in: {figures_dir}")
    eda_results = run_full_eda(df, figures_dir=figures_dir)

    # Persist JSON summary
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=2, default=str)
    print(f"Persisted EDA summary metrics to: {summary_path}")

    # Output executive console summary
    p = eda_results["price"]
    v = eda_results["volume_spread"]
    g = eda_results["gaps"]

    print("=" * 60)
    print("EURUSD M15 EDA SUMMARY")
    print("=" * 60)
    print(f"Candles analyzed:     {p['count']:,}")
    print(f"Close price range:    {p['close']['min']:.5f} -> {p['close']['max']:.5f}")
    print(f"Close price mean:     {p['close']['mean']:.5f} (std: {p['close']['std']:.5f})")
    print(f"Log Return Mean:      {p['log_returns']['mean']:.7f}")
    print(f"Log Return Std:       {p['log_returns']['std']:.7f}")
    print(f"Log Return Skewness:  {p['log_returns']['skewness']:.4f}")
    print(f"Log Return Kurtosis:  {p['log_returns']['kurtosis']:.4f} (excess kurtosis)")
    print(f"Annualized Vol:       {p['volatility']['annualized_vol'] * 100:.2f}%")
    print(f"Average 14-bar ATR:   {p['volatility']['atr_14_mean']:.5f}")
    print(f"Average Spread:       {v['spread_points']['mean']:.1f} points")
    print(f"Median Spread:        {v['spread_points']['median']:.1f} points")
    print(f"Average Tick Volume:  {v['tick_volume']['mean']:.1f}")
    print(f"Real Volume Zero Pct: {v['real_volume']['zero_pct']:.1f}%")
    print(f"Total Gaps (>900s):   {g['total_gaps']}")
    print(f"  - Weekends:         {g['expected_weekend_count']}")
    print(f"  - Holidays:         {g['expected_holiday_count']}")
    print(f"  - Unexpected:       {g['unexpected_gap_count']}")
    print(f"  - Unclassified:     {g['unclassified_gap_count']}")
    print(f"Figures generated:    {len(eda_results['figures'])}")
    for fig_path in eda_results["figures"]:
        print(f"  * {fig_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
