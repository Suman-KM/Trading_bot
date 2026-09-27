"""Executable script for Phase 10 Controlled Model Improvement Study.

Runs controlled experiments on EURUSD M15 directional forecasting (H=4, target: direction_4):
1. Experiment A: Logistic Regression (balanced) vs Logistic Regression (unweighted)
2. Experiment B: Random Forest (balanced) vs Random Forest (unweighted)
3. Experiment C: Extra Trees Classifier (balanced & unweighted)
4. Experiment D: Confidence / Decision Filtering Analysis
5. Temporal Robustness across Validation Blocks A, B, and C

CRITICAL SAFETY:
- The TEST partition is strictly locked and NEVER accessed, evaluated, or predicted on.
- Validation is the sole out-of-sample evaluation partition.
- Scaler is fitted strictly on the training partition.
- No broad hyperparameter search, grid search, or external ML frameworks.
- Neutral, descriptive reporting; no claims of profitability or trading viability.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.experiments import run_model_improvement_experiments


def main() -> None:
    """Execute controlled model improvement study."""
    parser = argparse.ArgumentParser(
        description="Phase 10: Controlled Model Improvement Study for EURUSD M15"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Output directory for reports and metadata (default: reports)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 10 — CONTROLLED MODEL IMPROVEMENT STUDY")
    print("Instrument: EURUSD | Timeframe: M15 | Horizon: H=4 | Target: direction_4")
    print("=" * 80)

    # 1. Assemble Dataset
    print("\n[1] Assembling dataset for Horizon H=4 (target: direction_4)...")
    assembled = assemble_dataset(target_column="direction_4")

    # 2. Chronological Splitting (70% Train, 15% Val, 15% Test with H=4 purge)
    print("\n[2] Splitting into chronological partitions (70% Train / 15% Val / 15% Test)...")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    splits = split_dataset(assembled, config)

    # 3. Print Dataset Partition Summary
    n_train = len(splits.train)
    n_val = len(splits.val)
    n_test = len(splits.test)
    n_features = len(splits.feature_names)

    print("\n" + "-" * 80)
    print("DATASET PARTITION SUMMARY:")
    print(f"  Input Features:   {n_features} derived point-in-time features")
    print(f"  Target:           {splits.target_name} (SHORT: -1.0, NEUTRAL: 0.0, LONG: 1.0)")
    print(
        f"  Training Split:   {n_train:,} rows "
        f"[{splits.train.start_timestamp} -> {splits.train.end_timestamp}]"
    )
    print(
        f"  Validation Split: {n_val:,} rows "
        f"[{splits.val.start_timestamp} -> {splits.val.end_timestamp}]"
    )
    print(f"  Test Split:       {n_test:,} rows [STRICTLY PROTECTED / NEVER ACCESSED]")
    print("-" * 80)

    # 4. Run Model Improvement Study
    print("\n[3] Running controlled model improvement experiments strictly on Validation...")
    print("    (Training scaler strictly on Train, evaluating on Validation only)")
    results = run_model_improvement_experiments(
        splits=splits,
        random_seed=args.random_seed,
        save_artifacts=True,
        reports_dir=args.reports_dir,
    )

    df_comp = results["comparison_df"]
    df_temp = results["temporal_df"]
    df_conf = results["confidence_df"]

    # 5. Display Overall Model Comparison Table
    print("\n" + "=" * 90)
    print("OVERALL MODEL PERFORMANCE COMPARISON (VALIDATION SET):")
    print("=" * 90)
    fmt_hdr = (
        f"{'Model':32s} | {'Weight':8s} | {'Accuracy':8s} | {'Bal Acc':8s} | "
        f"{'Macro F1':8s} | {'Rec Short':9s} | {'Rec Long':8s} | {'ROC-AUC':8s}"
    )
    print(fmt_hdr)
    print("-" * 90)

    for _, row in df_comp.iterrows():
        auc_str = f"{row['macro_roc_auc']:.4f}" if pd.notna(row["macro_roc_auc"]) else "N/A"
        print(
            f"{row['model_name']:32s} | {row['class_weight']:8s} | "
            f"{row['accuracy']:8.4f} | {row['balanced_accuracy']:8.4f} | "
            f"{row['macro_f1']:8.4f} | {row['recall_short']:9.4f} | "
            f"{row['recall_long']:8.4f} | {auc_str:8s}"
        )

    # 6. Display Temporal Stability Across Validation Blocks A, B, C
    print("\n" + "=" * 90)
    print("TEMPORAL ROBUSTNESS SUMMARY (BALANCED ACCURACY & MACRO F1 BY BLOCK):")
    print("=" * 90)
    print(
        f"{'Model':32s} | {'Block':12s} | {'Bal Acc':8s} | "
        f"{'Macro F1':8s} | {'Rec S':8s} | {'Rec L':8s}"
    )
    print("-" * 90)
    for _, row in df_temp.iterrows():
        print(
            f"{row['model']:32s} | {row['block']:12s} | "
            f"{row['balanced_accuracy']:8.4f} | {row['macro_f1']:8.4f} | "
            f"{row['recall_short']:8.4f} | {row['recall_long']:8.4f}"
        )

    # 7. Display Decision / Confidence Filtering Summary
    print("\n" + "=" * 90)
    print("CONFIDENCE FILTERING SUMMARY (EXPERIMENT D - RANDOM FOREST BALANCED):")
    print("=" * 90)
    rf_conf = df_conf[df_conf["model"] == "Random Forest (balanced)"]
    print(
        f"{'Thresh':6s} | {'Covered':7s} | {'Coverage%':9s} | {'Accuracy':8s} | "
        f"{'Bal Acc':8s} | {'Macro F1':8s} | {'Pred S':6s} | {'Pred N':6s} | {'Pred L':6s}"
    )
    print("-" * 90)
    for _, row in rf_conf.iterrows():
        acc_s = f"{row['accuracy']:.4f}" if pd.notna(row["accuracy"]) else "N/A"
        bal_s = f"{row['balanced_accuracy']:.4f}" if pd.notna(row["balanced_accuracy"]) else "N/A"
        f1_s = f"{row['macro_f1']:.4f}" if pd.notna(row["macro_f1"]) else "N/A"
        ps = int(row["pred_count_short"])
        pn = int(row["pred_count_neutral"])
        pl = int(row["pred_count_long"])
        print(
            f"{row['threshold']:6.2f} | {int(row['covered_count']):7d} | "
            f"{row['coverage_pct']:8.2f}% | {acc_s:8s} | {bal_s:8s} | {f1_s:8s} | "
            f"{ps:6d} | {pn:6d} | {pl:6d}"
        )

    # 8. Report Artifact Locations
    print("\n" + "=" * 90)
    print("PHASE 10 ARTIFACTS GENERATED:")
    print("=" * 90)
    print(f"  Comparison Table:    {args.reports_dir}/model_improvement_comparison.csv")
    print(f"  Temporal Stability:  {args.reports_dir}/model_improvement_temporal.csv")
    print(f"  Confidence Analysis: {args.reports_dir}/model_improvement_confidence.csv")
    print(f"  Study Metadata:      {args.reports_dir}/model_experiments_metadata.json")
    print("\n[SAFETY NOTE] The TEST partition was NOT accessed, inspected, or evaluated.")
    print("All evaluations were performed strictly on the VALIDATION partition.")
    print("No trading viability or profitability is claimed.")


if __name__ == "__main__":
    main()
