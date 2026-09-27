"""Executable CLI script for Phase 11 Final Out-of-Sample Test Evaluation.

Executes final holdout evaluation of locked Phase 8 and Phase 10 models on the
EURUSD M15 test partition (14,988 observations).

CRITICAL GOVERNANCE:
- Evaluates models on the untouched test partition for the first time.
- Models are trained strictly on the training partition (69,937 rows).
- Scaler is fitted strictly on the training partition.
- No model tuning, retraining, feature selection, or threshold tuning.
- Results are reported descriptively with zero claims of profitability or trading viability.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.test_evaluation import run_final_test_evaluation


def main() -> None:
    """Execute final out-of-sample test evaluation workflow."""
    parser = argparse.ArgumentParser(
        description="Phase 11: Final Out-of-Sample Test Evaluation for EURUSD M15"
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

    print("=" * 95)
    print("PHASE 11 — FINAL OUT-OF-SAMPLE TEST EVALUATION (EURUSD M15)")
    print("Target: direction_4 (Horizon H=4, Fixed Threshold = 0.00050 / 5.0 pips)")
    print("=" * 95)

    # 1. Assemble and Split Dataset
    print("\n[1] Assembling dataset and partitioning into chronological splits...")
    assembled = assemble_dataset(target_column="direction_4")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=4)
    splits = split_dataset(assembled, config)

    n_train = len(splits.train)
    n_val = len(splits.val)
    n_test = len(splits.test)

    print("\n" + "-" * 95)
    print("DATASET PARTITION SUMMARY:")
    print(
        f"  Training Split:   {n_train:,} rows "
        f"[{splits.train.start_timestamp} -> {splits.train.end_timestamp}]"
    )
    print(
        f"  Validation Split: {n_val:,} rows "
        f"[{splits.val.start_timestamp} -> {splits.val.end_timestamp}]"
    )
    print(
        f"  Test Split:       {n_test:,} rows "
        f"[{splits.test.start_timestamp} -> {splits.test.end_timestamp}]"
    )
    print(f"  Features:         {len(splits.feature_names)} point-in-time causal features")
    print(f"  Target:           {splits.target_name} (Fixed threshold: 0.00050 / 5.0 pips)")
    print("-" * 95)

    # 2. Test Set Class Distribution Breakdown
    tr_counts = splits.train_y.value_counts().sort_index()
    val_counts = splits.val_y.value_counts().sort_index()
    test_counts = splits.test_y.value_counts().sort_index()

    tr_pcts = splits.train_y.value_counts(normalize=True).sort_index() * 100.0
    val_pcts = splits.val_y.value_counts(normalize=True).sort_index() * 100.0
    test_pcts = splits.test_y.value_counts(normalize=True).sort_index() * 100.0

    print("\nCLASS DISTRIBUTION COMPARISON ACROSS PARTITIONS:")
    print(
        f"{'Class':16s} | {'Train Count':12s} {'Train %':8s} | "
        f"{'Val Count':10s} {'Val %':8s} | {'Test Count':11s} {'Test %':8s}"
    )
    print("-" * 95)
    class_names = {-1.0: "SHORT (-1)", 0.0: "NEUTRAL (0)", 1.0: "LONG (+1)"}
    for c in [-1.0, 0.0, 1.0]:
        print(
            f"{class_names[c]:16s} | {tr_counts.get(c, 0):12,d} {tr_pcts.get(c, 0.0):7.2f}% | "
            f"{val_counts.get(c, 0):10,d} {val_pcts.get(c, 0.0):7.2f}% | "
            f"{test_counts.get(c, 0):11,d} {test_pcts.get(c, 0.0):7.2f}%"
        )
    print("-" * 95)

    # 3. Execute Phase 11 Final Holdout Evaluation
    print("\n[2] Executing Phase 11 final out-of-sample holdout evaluation...")
    print(
        "    (Training models strictly on Train; unlocking Test partition "
        "exclusively for evaluation)"
    )
    results = run_final_test_evaluation(
        splits=splits,
        random_seed=args.random_seed,
        unlock_test_set=True,
        save_artifacts=True,
        reports_dir=args.reports_dir,
    )

    df_comp = results["comparison_df"]
    df_temp = results["temporal_df"]
    df_conf = results["confidence_df"]

    # 4. Display Full Test Performance Summary Table
    print("\n" + "=" * 95)
    print("FINAL OUT-OF-SAMPLE TEST PERFORMANCE (ALL 7 CANDIDATE ARCHITECTURES):")
    print("=" * 95)
    print(
        f"{'Model':32s} | {'Weight':8s} | {'Accuracy':8s} | {'Bal Acc':8s} | "
        f"{'Macro F1':8s} | {'Rec Short':9s} | {'Rec Long':8s} | {'ROC-AUC':8s} | {'Brier':7s}"
    )
    print("-" * 95)
    for _, row in df_comp.iterrows():
        auc_val = row["test_macro_roc_auc"]
        auc_str = f"{auc_val:.4f}" if pd.notna(auc_val) else "N/A"
        print(
            f"{row['model_name']:32s} | {row['class_weight']:8s} | "
            f"{row['test_accuracy']:8.4f} | {row['test_balanced_accuracy']:8.4f} | "
            f"{row['test_macro_f1']:8.4f} | {row['test_recall_short']:9.4f} | "
            f"{row['test_recall_long']:8.4f} | {auc_str:8s} | {row['test_brier_multiclass']:7.4f}"
        )

    # 5. Display Validation vs Test Generalization Gap Table
    print("\n" + "=" * 95)
    print("VALIDATION VS TEST GENERALIZATION GAP ANALYSIS (DESCRIPTIVE HOLDOUT DELTAS):")
    print("=" * 95)
    print(
        f"{'Model':32s} | {'Val Acc':7s} -> {'Test Acc':8s} {'(Gap)':8s} | "
        f"{'Val Bal':7s} -> {'Test Bal':8s} {'(Gap)':8s} | "
        f"{'Val F1':6s} -> {'Test F1':7s} {'(Gap)':7s}"
    )
    print("-" * 95)
    for _, row in df_comp.iterrows():
        acc_str = (
            f"{row['val_accuracy']:7.4f} -> {row['test_accuracy']:8.4f} "
            f"({row['gap_accuracy']:+7.4f})"
        )
        bal_str = (
            f"{row['val_balanced_accuracy']:7.4f} -> {row['test_balanced_accuracy']:8.4f} "
            f"({row['gap_balanced_accuracy']:+7.4f})"
        )
        f1_str = (
            f"{row['val_macro_f1']:6.4f} -> {row['test_macro_f1']:7.4f} "
            f"({row['gap_macro_f1']:+6.4f})"
        )
        print(f"{row['model_name']:32s} | {acc_str} | {bal_str} | {f1_str}")

    # 6. Display Directional Recall & ROC-AUC Generalization Gaps
    print("\n" + "=" * 95)
    print("DIRECTIONAL RECALL & ROC-AUC GENERALIZATION GAPS:")
    print("=" * 95)
    print(
        f"{'Model':32s} | {'Val RecS':8s} -> {'Test RecS':9s} {'(Gap)':8s} | "
        f"{'Val RecL':8s} -> {'Test RecL':9s} {'(Gap)':8s} | "
        f"{'Val AUC':7s} -> {'Test AUC':8s} {'(Gap)':8s}"
    )
    print("-" * 95)
    for _, row in df_comp.iterrows():
        auc_v = f"{row['val_macro_roc_auc']:.4f}" if pd.notna(row["val_macro_roc_auc"]) else "N/A"
        auc_t = f"{row['test_macro_roc_auc']:.4f}" if pd.notna(row["test_macro_roc_auc"]) else "N/A"
        gap_a = f"{row['gap_macro_roc_auc']:+7.4f}" if pd.notna(row["gap_macro_roc_auc"]) else "N/A"
        recs_str = (
            f"{row['val_recall_short']:8.4f} -> {row['test_recall_short']:9.4f} "
            f"({row['gap_recall_short']:+7.4f})"
        )
        recl_str = (
            f"{row['val_recall_long']:8.4f} -> {row['test_recall_long']:9.4f} "
            f"({row['gap_recall_long']:+7.4f})"
        )
        auc_str = f"{auc_v:7s} -> {auc_t:8s} ({gap_a:7s})"
        print(f"{row['model_name']:32s} | {recs_str} | {recl_str} | {auc_str}")

    # 7. Display Temporal Test Breakdown for Preselected Candidate
    print("\n" + "=" * 95)
    print("TEMPORAL TEST BLOCKS (PRESELECTED CANDIDATE: RANDOM FOREST BALANCED):")
    print("=" * 95)
    rf_temp = df_temp[df_temp["model"] == "Random Forest (balanced)"]
    print(
        f"{'Block':13s} | {'Start Time':22s} -> {'End Time':22s} | "
        f"{'Bal Acc':8s} | {'Macro F1':8s} | {'Rec S':8s} | {'Rec L':8s} | {'ROC-AUC':8s}"
    )
    print("-" * 95)
    for _, row in rf_temp.iterrows():
        auc_s = f"{row['roc_auc_ovr']:.4f}" if pd.notna(row["roc_auc_ovr"]) else "N/A"
        print(
            f"{row['block']:13s} | {row['start_time'][:19]:19s} -> {row['end_time'][:19]:19s} | "
            f"{row['balanced_accuracy']:8.4f} | {row['macro_f1']:8.4f} | "
            f"{row['recall_short']:8.4f} | {row['recall_long']:8.4f} | {auc_s:8s}"
        )

    # 8. Display Confidence Analysis for Preselected Candidate
    print("\n" + "=" * 95)
    print("POST-HOC DESCRIPTIVE CONFIDENCE ANALYSIS (RANDOM FOREST BALANCED ON TEST):")
    print("=" * 95)
    rf_conf = df_conf[df_conf["model"] == "Random Forest (balanced)"]
    print(
        f"{'Thresh':6s} | {'Covered':7s} | {'Coverage%':9s} | {'Accuracy':8s} | "
        f"{'Bal Acc':8s} | {'Macro F1':8s} | {'Pred S':6s} | {'Pred N':6s} | {'Pred L':6s}"
    )
    print("-" * 95)
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

    # 9. Calibration & Brier Score Diagnostics
    print("\n" + "=" * 95)
    print("PROBABILITY CALIBRATION & BRIER SCORE DIAGNOSTICS (TEST SET):")
    print("=" * 95)
    print(
        f"{'Model':32s} | {'Multiclass Brier':17s} | {'Brier Short':12s} | "
        f"{'Brier Neutral':14s} | {'Brier Long':12s}"
    )
    print("-" * 95)
    for _, row in df_comp.iterrows():
        print(
            f"{row['model_name']:32s} | {row['test_brier_multiclass']:17.4f} | "
            f"{row['test_brier_short']:12.4f} | {row['test_brier_neutral']:14.4f} | "
            f"{row['test_brier_long']:12.4f}"
        )

    # 10. Summary & Generated Artifacts
    print("\n" + "=" * 95)
    print("PHASE 11 ARTIFACTS GENERATED:")
    print("=" * 95)
    print(f"  Test Comparison & Gaps: {args.reports_dir}/final_test_comparison.csv")
    print(f"  Temporal Test Blocks:   {args.reports_dir}/final_test_temporal.csv")
    print(f"  Test Confidence Grid:   {args.reports_dir}/final_test_confidence.csv")
    print(f"  Study Metadata:         {args.reports_dir}/final_test_metadata.json")
    print("\n[GOVERNANCE CONFIRMATION] The Test partition was evaluated strictly as a holdout.")
    print("No test information was used for model tuning, selection, or optimization.")
    print("No trading viability or profitability is claimed.")


if __name__ == "__main__":
    main()
