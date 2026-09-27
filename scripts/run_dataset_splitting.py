"""Executable pipeline for EURUSD M15 dataset assembly and chronological splitting.

Performs point-in-time feature-target assembly, calculates chronological 70/15/15 partitions,
enforces forward-purge and embargo safety gaps, verifies leakage guarantees, and generates
the machine-readable splitting metadata contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.dataset.assembly import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_LABELS_PATH,
    assemble_dataset,
)
from ai.dataset.splits import SplitConfig, split_dataset
from ai.dataset.validation import validate_dataset_splits
from ai.labels.returns import DEFAULT_HORIZONS

REPORTS_DIR = Path("reports")
METADATA_OUTPUT_PATH = REPORTS_DIR / "dataset_split_metadata.json"


def run_dataset_splitting_pipeline() -> dict[str, Any]:
    """Execute the dataset assembly, temporal splitting, and metadata generation pipeline."""
    print("=" * 80)
    print("EURUSD M15 DATASET ASSEMBLY & TEMPORAL SPLITTING PIPELINE (PHASE 7)")
    print("=" * 80)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "dataset_name": "EURUSD_M15_RESEARCH_DATASET",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "source_features_path": str(DEFAULT_FEATURES_PATH),
        "source_labels_path": str(DEFAULT_LABELS_PATH),
        "total_raw_candles": 100000,
        "feature_warmup_bars": 80,
        "post_warmup_candles": 99920,
        "baseline_split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "horizons_metadata": {},
    }

    # Primary baseline evaluation for each candidate horizon
    for h in DEFAULT_HORIZONS:
        target_name = f"direction_{h}"
        print(f"\n[Processing Horizon H={h} (Target: {target_name})]")

        # 1. Assemble dataset
        assembled = assemble_dataset(target_column=target_name)
        usable_count = int(assembled.usable_mask.sum())
        tail_nans = int((~assembled.usable_mask).sum())
        print(f"  Assembled post-warmup rows: {len(assembled.df):,}")
        print(f"  Complete usable rows:       {usable_count:,}")
        print(f"  Tail label NaNs preserved:  {tail_nans}")

        # 2. Baseline split: Purge only (purge=H, embargo=0)
        config_purge = SplitConfig(purge_bars=h, embargo_bars=0)
        splits_purge = split_dataset(assembled, config_purge)
        report_purge = validate_dataset_splits(splits_purge, raise_on_error=True)
        pct_t = splits_purge.train.pct_of_usable
        pct_v = splits_purge.val.pct_of_usable
        pct_te = splits_purge.test.pct_of_usable
        print("  Split [Purge Only (gap=H)]:")
        print(
            f"    Train: {len(splits_purge.train):,} rows ({pct_t:.2f}%) "
            f"[{splits_purge.train.start_timestamp} -> {splits_purge.train.end_timestamp}]"
        )
        print(
            f"    Val:   {len(splits_purge.val):,} rows ({pct_v:.2f}%) "
            f"[{splits_purge.val.start_timestamp} -> {splits_purge.val.end_timestamp}]"
        )
        print(
            f"    Test:  {len(splits_purge.test):,} rows ({pct_te:.2f}%) "
            f"[{splits_purge.test.start_timestamp} -> {splits_purge.test.end_timestamp}]"
        )
        print(
            f"    Purged: Train={splits_purge.purged_train_bars}, "
            f"Val={splits_purge.purged_val_bars}"
        )
        print(f"    Leakage validation: {'PASSED' if report_purge.is_valid else 'FAILED'}")

        # 3. Enhanced split: Purge + Embargo (purge=H, embargo=H)
        config_embargo = SplitConfig(purge_bars=h, embargo_bars=h)
        splits_embargo = split_dataset(assembled, config_embargo)
        report_embargo = validate_dataset_splits(splits_embargo, raise_on_error=True)
        print(f"  Split [Purge + Embargo (gap={h * 2} bars / {h * 30} min)]:")
        print(
            f"    Train: {len(splits_embargo.train):,} rows | "
            f"Val: {len(splits_embargo.val):,} rows | "
            f"Test: {len(splits_embargo.test):,} rows"
        )
        print(f"    Leakage validation: {'PASSED' if report_embargo.is_valid else 'FAILED'}")

        # Store horizon metadata
        metadata["horizons_metadata"][f"H_{h}"] = {
            "horizon_bars": h,
            "horizon_minutes": h * 15,
            "primary_classification_target": target_name,
            "primary_continuous_target": f"future_return_{h}",
            "assembled_rows": len(assembled.df),
            "usable_rows": usable_count,
            "tail_label_nans": tail_nans,
            "feature_count": len(assembled.feature_names),
            "baseline_purge_only": splits_purge.to_summary_dict(),
            "purge_plus_embargo": splits_embargo.to_summary_dict(),
        }

    # Write metadata JSON
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved splitting metadata specification to: {METADATA_OUTPUT_PATH}")
    print("=" * 80)
    print("DATASET ASSEMBLY & SPLITTING PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)
    return metadata


if __name__ == "__main__":
    run_dataset_splitting_pipeline()
