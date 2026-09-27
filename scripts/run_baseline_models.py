"""Executable script to run Phase 8 Baseline Machine Learning Models.

Loads Phase 7 chronological splits for EURUSD M15 (H=4, target: direction_4), fits
Majority, Logistic Regression, and Random Forest baselines strictly on Train, evaluates
performance on Validation, outputs comparison tables and confusion matrices, and enforces
strict Test set protection.
"""

from __future__ import annotations

from pathlib import Path

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.models.pipeline import run_baseline_training_pipeline


def main() -> None:
    """Execute baseline models training, validation, and reporting."""
    print("=" * 80)
    print("PHASE 8 — BASELINE MACHINE LEARNING MODELS (EURUSD M15)")
    print("=" * 80)

    # 1. Load dataset via Phase 7 pipeline
    horizon = 4
    target_name = f"direction_{horizon}"
    print(f"\n[1] Assembling dataset for Horizon H={horizon} (Target: {target_name})...")
    assembled = assemble_dataset(target_column=target_name)

    # 2. Chronological split (70% Train, 15% Val, 15% Test with H=4 purge)
    print("\n[2] Splitting into chronological partitions (70% Train / 15% Val / 15% Test)...")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=horizon)
    splits = split_dataset(assembled, config)

    # 3. Print dataset sizes and class distributions
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
    print(f"  Test Split:       {n_test:,} rows (UNTOUCHED / RESERVED FOR FUTURE PHASES)")
    print("-" * 80)

    tr_dist = splits.train_y.value_counts(normalize=True).sort_index()
    val_dist = splits.val_y.value_counts(normalize=True).sort_index()

    print("\nCLASS DISTRIBUTIONS:")
    print("  Class          Train %        Val %")
    class_labels = {-1.0: "SHORT (-1)", 0.0: "NEUTRAL (0)", 1.0: "LONG (+1)"}
    for c in [-1.0, 0.0, 1.0]:
        t_pct = tr_dist.get(c, 0.0) * 100.0
        v_pct = val_dist.get(c, 0.0) * 100.0
        print(f"  {class_labels[c]:14s} {t_pct:6.2f}%       {v_pct:6.2f}%")

    # 4. Train & Evaluate Baselines
    print("\n" + "=" * 80)
    print("[3] Training and evaluating baseline models on VALIDATION set...")
    print("    (Note: Test set is strictly protected and never accessed)")
    print("=" * 80)

    results = run_baseline_training_pipeline(
        splits=splits,
        random_seed=42,
        generate_figures=True,
        save_metadata=True,
        figures_dir=Path("reports/figures"),
        metadata_path=Path("reports/baseline_model_metadata.json"),
    )

    # 5. Print Comparison Table
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON (VALIDATION SET):")
    print("=" * 80)
    print(
        f"{'Model':26s} | {'Accuracy':8s} | {'Bal. Acc':8s} | "
        f"{'Macro P':8s} | {'Macro R':8s} | {'Macro F1':8s} | {'ROC-AUC':8s}"
    )
    print("-" * 80)

    for key in ["majority", "logistic_regression", "random_forest"]:
        res = results[key]
        auc_str = f"{res.roc_auc_ovr:.4f}" if res.roc_auc_ovr is not None else "N/A"
        print(
            f"{res.model_name:26s} | {res.accuracy:8.4f} | {res.balanced_accuracy:8.4f} | "
            f"{res.macro_precision:8.4f} | {res.macro_recall:8.4f} | "
            f"{res.macro_f1:8.4f} | {auc_str:8s}"
        )

    # 6. Detailed Per-Class Breakdown
    print("\n" + "-" * 80)
    print("PER-CLASS VALIDATION F1-SCORES:")
    print("-" * 80)
    print(f"{'Model':26s} | {'SHORT (-1)':12s} | {'NEUTRAL (0)':12s} | {'LONG (+1)':12s}")
    print("-" * 80)
    for key in ["majority", "logistic_regression", "random_forest"]:
        res = results[key]
        f1_short = res.per_class_f1.get("SHORT (-1)", 0.0)
        f1_neut = res.per_class_f1.get("NEUTRAL (0)", 0.0)
        f1_long = res.per_class_f1.get("LONG (+1)", 0.0)
        print(f"{res.model_name:26s} | {f1_short:12.4f} | {f1_neut:12.4f} | {f1_long:12.4f}")

    # 7. Differences relative to majority baseline
    maj_acc = results["majority"].accuracy
    lr_acc = results["logistic_regression"].accuracy
    rf_acc = results["random_forest"].accuracy
    lr_diff = (lr_acc - maj_acc) * 100.0
    rf_diff = (rf_acc - maj_acc) * 100.0

    print("\n" + "-" * 80)
    print("SUMMARY RELATIVE TO NAIVE MAJORITY BASELINE:")
    print(f"  Majority Baseline Accuracy:   {maj_acc * 100.0:.2f}%")
    print(
        f"  Logistic Regression Accuracy: {lr_acc * 100.0:.2f}% "
        f"(difference: {lr_diff:+.2f} percentage points)"
    )
    print(
        f"  Random Forest Accuracy:       {rf_acc * 100.0:.2f}% "
        f"(difference: {rf_diff:+.2f} percentage points)"
    )
    print("-" * 80)
    print("\n[Artifacts Generated]")
    print("  - Confusion matrices: reports/figures/confusion_matrix_{model}.png")
    print("  - Metadata contract:  reports/baseline_model_metadata.json")
    print("  - Test set status:    STRICTLY UNTOUCHED (0 test evaluations performed)")
    print("=" * 80)
    print("PHASE 8 BASELINE EXECUTION COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    main()
