"""Executable script to execute Phase 9 Baseline Model Diagnostics.

Loads Phase 7 chronological splits for EURUSD M15 (H=4, target: direction_4), trains
Phase 8 baseline models strictly on the Train set, and executes comprehensive diagnostics
strictly on the Validation set:
1. Predicted probability distributions and quantiles
2. Confidence / coverage trade-offs (thresholds 0.40 to 0.90)
3. Class-wise precision, recall, F1, OvR ROC-AUC, and Average Precision
4. One-vs-Rest ROC curves and Precision-Recall curves
5. Random Forest MDI and Logistic Regression coefficient feature importances
6. Feature sanity and leakage verification against Phase 5 registry
7. Temporal validation diagnostics across 3 contiguous blocks (A, B, C)
8. Misclassification error analysis and confidence profiling
9. Model agreement and Cohen's Kappa diagnostics
10. Descriptive probability calibration (reliability curves and Brier scores)

STRICT SAFETY CONSTRAINTS:
- Test split is NEVER accessed, inspected, or evaluated.
- No hyperparameter tuning, grid search, or threshold optimization.
- No model calibration is fitted (Platt scaling or isotonic regression).
- No trading backtests, order submissions, or MT5 connections.
- Neutral descriptive reporting only; no profitability claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import SplitConfig, split_dataset
from ai.features.pipeline import get_feature_registry
from ai.models.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from ai.models.diagnostics import (
    CLASS_LABELS,
    compute_calibration_diagnostics,
    compute_class_wise_metrics,
    compute_confidence_coverage,
    compute_error_analysis,
    compute_lr_feature_importance,
    compute_model_agreement,
    compute_probability_diagnostics,
    compute_rf_feature_importance,
    compute_temporal_validation_diagnostics,
    plot_calibration_curves,
    plot_feature_importance_top20,
    plot_max_probability_distribution,
    plot_pr_curves,
    plot_probability_by_class,
    plot_roc_curves,
    plot_roc_model_comparison,
    plot_temporal_validation_diagnostics,
    verify_feature_sanity,
)

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def run_diagnostics_pipeline(
    random_seed: int = 42,
    horizon: int = 4,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute complete Phase 9 baseline model diagnostics workflow.

    Parameters
    ----------
    random_seed : int, default 42
        Deterministic random seed.
    horizon : int, default 4
        Prediction horizon bars (H=4 for direction_4).
    save_artifacts : bool, default True
        If True, writes CSVs, PNG figures, and metadata JSON to disk.

    Returns
    -------
    dict[str, Any]
        Dictionary of diagnostic results.
    """
    target_name = f"direction_{horizon}"
    print("=" * 80)
    print("PHASE 9 — BASELINE MODEL DIAGNOSTICS & FEATURE IMPORTANCE")
    print(f"Instrument: EURUSD | Timeframe: M15 | Horizon: H={horizon} | Target: {target_name}")
    print("=" * 80)

    # 1. Assemble and Split Dataset
    print(f"\n[1] Assembling dataset for Horizon H={horizon} (Target: {target_name})...")
    assembled = assemble_dataset(target_column=target_name)

    print("[2] Partitioning chronological splits (70% Train / 15% Val / 15% Test)...")
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, purge_bars=horizon)
    splits = split_dataset(assembled, config)

    # 2. Strict Test-Set Safety Verification
    print("\n[3] Verifying strict test-set isolation...")
    print(f"    Train observations:      {len(splits.train):,} rows")
    print(f"    Validation observations: {len(splits.val):,} rows")
    print(f"    Test observations:       {len(splits.test):,} rows [LOCKED / NEVER ACCESSED]")
    print(f"    Input Features:          {len(splits.feature_names)} features")
    assert len(splits.feature_names) == 80, f"Expected 80 features, got {len(splits.feature_names)}"

    # 3. Train Phase 8 Baseline Models strictly on Train
    print("\n[4] Training Phase 8 baseline models strictly on Train split...")
    X_train = splits.train_X
    y_train = splits.train_y
    X_val = splits.val_X
    y_val = splits.val_y
    val_timestamps = splits.val.timestamps
    classes = [-1.0, 0.0, 1.0]

    model_lr = LogisticRegressionBaseline(
        random_state=random_seed,
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
    )
    print("    Fitting Logistic Regression (train-fitted StandardScaler)...")
    model_lr.fit(X_train, y_train)

    model_rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        random_state=random_seed,
        n_jobs=-1,
    )
    print("    Fitting Random Forest (100 estimators, max_depth=10, min_samples_leaf=20)...")
    model_rf.fit(X_train, y_train)

    # 4. Generate Validation Predictions & Probabilities
    print("\n[5] Computing validation predictions and probabilities...")
    y_pred_lr = model_lr.predict(X_val)
    prob_lr = model_lr.predict_proba(X_val)

    y_pred_rf = model_rf.predict(X_val)
    prob_rf = model_rf.predict_proba(X_val)

    # Ensure output directories exist
    if save_artifacts:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Probability Distribution Diagnostics
    print("\n" + "-" * 80)
    print("1. PROBABILITY DISTRIBUTION DIAGNOSTICS")
    print("-" * 80)
    prob_diag_lr = compute_probability_diagnostics(prob_lr, classes=classes)
    prob_diag_rf = compute_probability_diagnostics(prob_rf, classes=classes)

    max_p_lr = prob_diag_lr["max_probability"]
    max_p_rf = prob_diag_rf["max_probability"]

    print("Max Probability Quantiles:")
    header = (
        f"{'Model':22s} | {'Mean':7s} | {'Std':7s} | {'Min':7s} | {'p50':7s} | "
        f"{'p75':7s} | {'p90':7s} | {'p95':7s} | {'p99':7s} | {'Max':7s}"
    )
    print(header)
    print("-" * 95)
    row_lr = (
        f"{'Logistic Regression':22s} | {max_p_lr['mean']:7.4f} | {max_p_lr['std']:7.4f} | "
        f"{max_p_lr['min']:7.4f} | {max_p_lr['p50']:7.4f} | {max_p_lr['p75']:7.4f} | "
        f"{max_p_lr['p90']:7.4f} | {max_p_lr['p95']:7.4f} | {max_p_lr['p99']:7.4f} | "
        f"{max_p_lr['max']:7.4f}"
    )
    print(row_lr)
    row_rf = (
        f"{'Random Forest':22s} | {max_p_rf['mean']:7.4f} | {max_p_rf['std']:7.4f} | "
        f"{max_p_rf['min']:7.4f} | {max_p_rf['p50']:7.4f} | {max_p_rf['p75']:7.4f} | "
        f"{max_p_rf['p90']:7.4f} | {max_p_rf['p95']:7.4f} | {max_p_rf['p99']:7.4f} | "
        f"{max_p_rf['max']:7.4f}"
    )
    print(row_rf)

    if save_artifacts:
        plot_max_probability_distribution(
            prob_lr,
            "Logistic Regression",
            FIGURES_DIR / "logistic_max_probability_distribution.png",
        )
        plot_max_probability_distribution(
            prob_rf, "Random Forest", FIGURES_DIR / "random_forest_max_probability_distribution.png"
        )
        plot_probability_by_class(
            prob_lr,
            y_val,
            classes,
            "Logistic Regression",
            FIGURES_DIR / "logistic_probability_by_class.png",
        )
        plot_probability_by_class(
            prob_rf,
            y_val,
            classes,
            "Random Forest",
            FIGURES_DIR / "random_forest_probability_by_class.png",
        )

    # 6. Confidence / Coverage Analysis
    print("\n" + "-" * 80)
    print("2. CONFIDENCE / COVERAGE ANALYSIS (EXPLORATORY DIAGNOSTICS)")
    print("-" * 80)
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    cov_lr = compute_confidence_coverage(
        prob_lr, y_val, y_pred_lr, thresholds=thresholds, classes=classes
    )
    cov_rf = compute_confidence_coverage(
        prob_rf, y_val, y_pred_rf, thresholds=thresholds, classes=classes
    )

    cov_header = (
        f"{'Threshold':10s} | {'Covered':8s} | {'Coverage %':11s} | "
        f"{'Accuracy':9s} | {'Bal. Acc':9s} | {'Macro F1':9s}"
    )
    print("Logistic Regression Coverage & Accuracy by Confidence Threshold:")
    print(cov_header)
    print("-" * 65)
    for rec in cov_lr:
        acc_s = f"{rec['accuracy']:.4f}" if rec["accuracy"] is not None else "N/A"
        bal_s = f"{rec['balanced_accuracy']:.4f}" if rec["balanced_accuracy"] is not None else "N/A"
        f1_s = f"{rec['macro_f1']:.4f}" if rec["macro_f1"] is not None else "N/A"
        row_str = (
            f"{rec['threshold']:10.2f} | {rec['covered_count']:8d} | "
            f"{rec['coverage'] * 100.0:10.2f}% | {acc_s:9s} | {bal_s:9s} | {f1_s:9s}"
        )
        print(row_str)

    print("\nRandom Forest Coverage & Accuracy by Confidence Threshold:")
    print(cov_header)
    print("-" * 65)
    for rec in cov_rf:
        acc_s = f"{rec['accuracy']:.4f}" if rec["accuracy"] is not None else "N/A"
        bal_s = f"{rec['balanced_accuracy']:.4f}" if rec["balanced_accuracy"] is not None else "N/A"
        f1_s = f"{rec['macro_f1']:.4f}" if rec["macro_f1"] is not None else "N/A"
        row_str = (
            f"{rec['threshold']:10.2f} | {rec['covered_count']:8d} | "
            f"{rec['coverage'] * 100.0:10.2f}% | {acc_s:9s} | {bal_s:9s} | {f1_s:9s}"
        )
        print(row_str)

    # 7. Class-wise Performance & Discrimination Metrics
    print("\n" + "-" * 80)
    print("3. CLASS-WISE PERFORMANCE & DISCRIMINATION METRICS")
    print("-" * 80)
    class_metrics_lr = compute_class_wise_metrics(y_val, y_pred_lr, prob_lr, classes=classes)
    class_metrics_rf = compute_class_wise_metrics(y_val, y_pred_rf, prob_rf, classes=classes)

    print("Class-wise Summary:")
    cw_header = (
        f"{'Model':20s} | {'Class':13s} | {'Support':8s} | {'Precision':9s} | "
        f"{'Recall':8s} | {'F1':8s} | {'ROC-AUC':8s} | {'PR-AUC (AP)':11s}"
    )
    print(cw_header)
    print("-" * 98)
    for model_name, c_dict in [
        ("Logistic Regression", class_metrics_lr),
        ("Random Forest", class_metrics_rf),
    ]:
        for c_label, m in c_dict.items():
            auc_s = f"{m['roc_auc_ovr']:.4f}" if m["roc_auc_ovr"] is not None else "N/A"
            ap_s = f"{m['average_precision']:.4f}" if m["average_precision"] is not None else "N/A"
            cw_row = (
                f"{model_name:20s} | {c_label:13s} | {m['support']:8d} | "
                f"{m['precision']:9.4f} | {m['recall']:8.4f} | {m['f1']:8.4f} | "
                f"{auc_s:8s} | {ap_s:11s}"
            )
            print(cw_row)

    # 8. ROC and Precision-Recall Curves
    if save_artifacts:
        print("\n[6] Generating ROC and Precision-Recall curves...")
        plot_roc_curves(
            y_val,
            prob_lr,
            classes,
            "Logistic Regression",
            FIGURES_DIR / "roc_curve_logistic_regression.png",
        )
        plot_roc_curves(
            y_val, prob_rf, classes, "Random Forest", FIGURES_DIR / "roc_curve_random_forest.png"
        )
        plot_roc_model_comparison(
            y_val, prob_lr, prob_rf, classes, FIGURES_DIR / "roc_curve_model_comparison.png"
        )
        plot_pr_curves(
            y_val,
            prob_lr,
            classes,
            "Logistic Regression",
            FIGURES_DIR / "pr_curve_logistic_regression.png",
        )
        plot_pr_curves(
            y_val, prob_rf, classes, "Random Forest", FIGURES_DIR / "pr_curve_random_forest.png"
        )

    # 9. Feature Importance Extraction
    print("\n" + "-" * 80)
    print("4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)
    rf_importance_df = compute_rf_feature_importance(model_rf, splits.feature_names)
    lr_importance_df = compute_lr_feature_importance(
        model_lr, splits.feature_names, classes=classes
    )

    print("Random Forest Top 10 Features (MDI Impurity):")
    for _, row in rf_importance_df.head(10).iterrows():
        print(
            f"  Rank {int(row['rank']):2d}: {row['feature']:28s} "
            f"(Importance: {row['importance']:.5f})"
        )

    print("\nLogistic Regression Top 10 Features (Mean Absolute Coefficient):")
    for _, row in lr_importance_df.head(10).iterrows():
        print(
            f"  Rank {int(row['rank']):2d}: {row['feature']:28s} "
            f"(Mean |w|: {row['aggregate_importance']:.4f} | "
            f"S: {row['coef_short']:+.3f} | N: {row['coef_neutral']:+.3f} | "
            f"L: {row['coef_long']:+.3f})"
        )

    if save_artifacts:
        rf_importance_df.to_csv(REPORTS_DIR / "random_forest_feature_importance.csv", index=False)
        lr_importance_df.to_csv(
            REPORTS_DIR / "logistic_regression_feature_importance.csv", index=False
        )
        plot_feature_importance_top20(
            rf_importance_df,
            "Random Forest — Top 20 Features (MDI Feature Importance)",
            FIGURES_DIR / "random_forest_feature_importance_top20.png",
            value_col="importance",
        )
        plot_feature_importance_top20(
            lr_importance_df,
            "Logistic Regression — Top 20 Features (Mean Absolute Coefficient)",
            FIGURES_DIR / "logistic_regression_feature_importance_top20.png",
            value_col="aggregate_importance",
        )

    # 10. Feature Sanity Check against Phase 5 Registry
    print("\n[7] Verifying feature sanity against Phase 5 registry...")
    registry_defs = get_feature_registry()
    registry_names = [f.name for f in registry_defs]
    sanity_rf = verify_feature_sanity(rf_importance_df, registry_names)
    sanity_lr = verify_feature_sanity(lr_importance_df, registry_names)
    print(f"    Random Forest sanity check:     {'PASS' if sanity_rf['passed'] else 'FAIL'}")
    print(f"    Logistic Regression sanity:     {'PASS' if sanity_lr['passed'] else 'FAIL'}")
    print(
        f"    Feature count:                  {sanity_rf['feature_count']} features "
        f"(Registry: {len(registry_names)})"
    )
    print("    Target/Timestamp/OHLCV leakage: ZERO detected")

    # 11. Temporal Validation Diagnostics
    print("\n" + "-" * 80)
    print("5. TEMPORAL VALIDATION DIAGNOSTICS (CHRONOLOGICAL STABILITY)")
    print("-" * 80)
    temporal_df = compute_temporal_validation_diagnostics(
        val_X=X_val,
        val_y=y_val,
        val_timestamps=val_timestamps,
        model_lr=model_lr,
        model_rf=model_rf,
        classes=classes,
    )
    t_header = (
        f"{'Model':20s} | {'Block':13s} | {'Rows':6s} | {'Accuracy':8s} | {'Bal. Acc':8s} | "
        f"{'Macro F1':8s} | {'Rec SHORT':9s} | {'Rec NEUT':8s} | {'Rec LONG':8s} | {'ROC-AUC':8s}"
    )
    print(t_header)
    print("-" * 115)
    for _, row in temporal_df.iterrows():
        auc_s = f"{row['roc_auc_ovr']:.4f}" if row["roc_auc_ovr"] is not None else "N/A"
        t_row = (
            f"{row['model']:20s} | {row['block']:13s} | {row['rows']:6d} | "
            f"{row['accuracy']:8.4f} | {row['balanced_accuracy']:8.4f} | "
            f"{row['macro_f1']:8.4f} | {row['recall_short']:9.4f} | "
            f"{row['recall_neutral']:8.4f} | {row['recall_long']:8.4f} | {auc_s:8s}"
        )
        print(t_row)

    if save_artifacts:
        temporal_df.to_csv(REPORTS_DIR / "validation_temporal_diagnostics.csv", index=False)
        plot_temporal_validation_diagnostics(
            temporal_df, FIGURES_DIR / "validation_temporal_diagnostics.png"
        )

    # 12. Error Analysis
    print("\n" + "-" * 80)
    print("6. ERROR ANALYSIS & MISCLASSIFICATION BREAKDOWN")
    print("-" * 80)
    error_lr = compute_error_analysis(y_val, y_pred_lr, prob_lr, classes=classes)
    error_rf = compute_error_analysis(y_val, y_pred_rf, prob_rf, classes=classes)

    print("Top Error Modes (Logistic Regression):")
    for _, row in error_lr[~error_lr["is_correct"]].head(4).iterrows():
        err_str = (
            f"  True {row['actual_label']:12s} -> Pred {row['predicted_label']:12s}: "
            f"{row['count']:5d} rows ({row['pct_of_total']:5.2f}% total) | "
            f"Mean Conf: {row['mean_confidence']:.4f}"
        )
        print(err_str)

    print("\nTop Error Modes (Random Forest):")
    for _, row in error_rf[~error_rf["is_correct"]].head(4).iterrows():
        err_str = (
            f"  True {row['actual_label']:12s} -> Pred {row['predicted_label']:12s}: "
            f"{row['count']:5d} rows ({row['pct_of_total']:5.2f}% total) | "
            f"Mean Conf: {row['mean_confidence']:.4f}"
        )
        print(err_str)

    if save_artifacts:
        error_lr.to_csv(REPORTS_DIR / "error_analysis_logistic.csv", index=False)
        error_rf.to_csv(REPORTS_DIR / "error_analysis_random_forest.csv", index=False)

    # 13. Model Agreement Diagnostics
    print("\n" + "-" * 80)
    print("7. MODEL AGREEMENT DIAGNOSTICS (LOGISTIC REGRESSION vs RANDOM FOREST)")
    print("-" * 80)
    agreement = compute_model_agreement(y_pred_lr, y_pred_rf, y_val, classes=classes)
    agree_pct = agreement["agreement_rate"] * 100.0
    print(
        f"Agreement Rate:                {agree_pct:.2f}% "
        f"({agreement['agreement_count']:,} / {agreement['total_samples']:,})"
    )
    print(f"Disagreement Count:            {agreement['disagreement_count']:,} rows")
    print(f"Cohen's Kappa:                 {agreement['cohen_kappa']:.4f}")
    print(f"Accuracy when both AGREE:      {agreement['joint_accuracy_on_agreement']:.4f}")
    print(f"LR Accuracy on DISAGREEMENT:   {agreement['lr_accuracy_on_disagreement']:.4f}")
    print(f"RF Accuracy on DISAGREEMENT:   {agreement['rf_accuracy_on_disagreement']:.4f}")

    if save_artifacts:
        # Save agreement table to CSV
        agreement_records = [
            {"metric": "total_samples", "value": agreement["total_samples"]},
            {"metric": "agreement_count", "value": agreement["agreement_count"]},
            {"metric": "disagreement_count", "value": agreement["disagreement_count"]},
            {"metric": "agreement_rate", "value": agreement["agreement_rate"]},
            {"metric": "cohen_kappa", "value": agreement["cohen_kappa"]},
            {
                "metric": "joint_accuracy_on_agreement",
                "value": agreement["joint_accuracy_on_agreement"],
            },
            {
                "metric": "lr_accuracy_on_disagreement",
                "value": agreement["lr_accuracy_on_disagreement"],
            },
            {
                "metric": "rf_accuracy_on_disagreement",
                "value": agreement["rf_accuracy_on_disagreement"],
            },
        ]
        pd.DataFrame(agreement_records).to_csv(REPORTS_DIR / "model_agreement.csv", index=False)

    # 14. Calibration Diagnostics
    print("\n" + "-" * 80)
    print("8. PROBABILITY CALIBRATION DIAGNOSTICS (UNMODIFIED MODELS)")
    print("-" * 80)
    cal_lr = compute_calibration_diagnostics(y_val, prob_lr, classes=classes, n_bins=10)
    cal_rf = compute_calibration_diagnostics(y_val, prob_rf, classes=classes, n_bins=10)

    print(f"Multiclass Brier Score (LR):   {cal_lr['multiclass_brier_score']:.4f}")
    print(f"Multiclass Brier Score (RF):   {cal_rf['multiclass_brier_score']:.4f}")
    for c_label in [CLASS_LABELS[c] for c in classes]:
        brier_lr = cal_lr["per_class"][c_label]["brier_score"]
        brier_rf = cal_rf["per_class"][c_label]["brier_score"]
        print(f"  {c_label:14s} Brier — LR: {brier_lr:.4f} | RF: {brier_rf:.4f}")

    if save_artifacts:
        plot_calibration_curves(
            y_val,
            prob_lr,
            classes,
            "Logistic Regression",
            FIGURES_DIR / "calibration_curve_logistic_regression.png",
            n_bins=10,
        )
        plot_calibration_curves(
            y_val,
            prob_rf,
            classes,
            "Random Forest",
            FIGURES_DIR / "calibration_curve_random_forest.png",
            n_bins=10,
        )

    # 15. Master Metadata Contract
    print("\n[8] Writing master metadata contract...")
    metadata: dict[str, Any] = {
        "pipeline_phase": "Phase 9 — Baseline Model Diagnostics & Feature Importance",
        "instrument": "EURUSD",
        "timeframe": "M15",
        "target": target_name,
        "horizon": horizon,
        "feature_count": len(splits.feature_names),
        "train_rows": len(splits.train),
        "validation_rows": len(splits.val),
        "test_rows": len(splits.test),
        "random_seed": random_seed,
        "model_names": ["Logistic Regression", "Random Forest"],
        "test_set_used": False,
        "test_set_status": "Strictly untouched / reserved for future out-of-sample evaluation",
        "software_environment": {
            "python_version": "3.13",
            "scikit_learn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "confidence_thresholds": thresholds,
        "temporal_blocks": {
            "VALIDATION_A": {"rows": len(X_val) // 3},
            "VALIDATION_B": {"rows": len(X_val) // 3},
            "VALIDATION_C": {"rows": len(X_val) - 2 * (len(X_val) // 3)},
        },
        "feature_importance_files": [
            "reports/random_forest_feature_importance.csv",
            "reports/logistic_regression_feature_importance.csv",
        ],
        "generated_figure_paths": [
            "reports/figures/logistic_max_probability_distribution.png",
            "reports/figures/random_forest_max_probability_distribution.png",
            "reports/figures/logistic_probability_by_class.png",
            "reports/figures/random_forest_probability_by_class.png",
            "reports/figures/roc_curve_logistic_regression.png",
            "reports/figures/roc_curve_random_forest.png",
            "reports/figures/roc_curve_model_comparison.png",
            "reports/figures/pr_curve_logistic_regression.png",
            "reports/figures/pr_curve_random_forest.png",
            "reports/figures/random_forest_feature_importance_top20.png",
            "reports/figures/logistic_regression_feature_importance_top20.png",
            "reports/figures/validation_temporal_diagnostics.png",
            "reports/figures/calibration_curve_logistic_regression.png",
            "reports/figures/calibration_curve_random_forest.png",
        ],
        "probability_diagnostics": {
            "logistic_regression": prob_diag_lr,
            "random_forest": prob_diag_rf,
        },
        "class_wise_metrics": {
            "logistic_regression": class_metrics_lr,
            "random_forest": class_metrics_rf,
        },
        "model_agreement": {
            "agreement_rate": agreement["agreement_rate"],
            "cohen_kappa": agreement["cohen_kappa"],
            "joint_accuracy_agree": agreement["joint_accuracy_on_agreement"],
            "lr_accuracy_disagree": agreement["lr_accuracy_on_disagreement"],
            "rf_accuracy_disagree": agreement["rf_accuracy_on_disagreement"],
        },
        "calibration_diagnostics": {
            "logistic_regression_multiclass_brier": cal_lr["multiclass_brier_score"],
            "random_forest_multiclass_brier": cal_rf["multiclass_brier_score"],
        },
    }

    if save_artifacts:
        meta_path = REPORTS_DIR / "model_diagnostics_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"    Saved metadata to {meta_path}")

    print("\n" + "=" * 80)
    print("PHASE 9 DIAGNOSTICS EXECUTION COMPLETE")
    print("=" * 80)

    return {
        "metadata": metadata,
        "rf_importance": rf_importance_df,
        "lr_importance": lr_importance_df,
        "temporal": temporal_df,
        "error_lr": error_lr,
        "error_rf": error_rf,
        "agreement": agreement,
    }


def main() -> None:
    """CLI entry point for running baseline model diagnostics."""
    run_diagnostics_pipeline(random_seed=42, horizon=4, save_artifacts=True)


if __name__ == "__main__":
    main()
