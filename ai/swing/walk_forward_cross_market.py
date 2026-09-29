"""Phase 18: Walk-forward cross-market evaluation and feature ablation engine.

Executes controlled feature group ablation across 9 expanding-window chronological folds
spanning 2010 to late 2024 (pre-holdout research partition), followed by single
one-shot out-of-sample evaluation on the sealed Fresh Research Holdout (Nov 2024 to Feb 2026).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.walk_forward_expanded import WalkForwardFold


def generate_pre_holdout_folds(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    n_folds: int = 9,
    purge_gap_bars: int = 8,
) -> list[WalkForwardFold]:
    """Generate 9 chronological expanding folds on pre-holdout research data."""
    valid_mask = target_series.notna() & features_df.notna().all(axis=1)
    valid_indices = np.where(valid_mask)[0]

    n_valid = len(valid_indices)
    # Reserve initial ~20% of valid samples for fold 1 initial train (~2,300 samples)
    min_train_samples = int(n_valid * 0.22)
    remaining_samples = n_valid - min_train_samples
    val_size = remaining_samples // n_folds

    folds: list[WalkForwardFold] = []

    for fold_idx in range(1, n_folds + 1):
        val_start_pos = min_train_samples + (fold_idx - 1) * val_size
        val_end_pos = min_train_samples + fold_idx * val_size if fold_idx < n_folds else n_valid

        val_target_indices = valid_indices[val_start_pos:val_end_pos]
        first_val_idx = val_target_indices[0]
        last_val_idx = val_target_indices[-1]

        # Purge gap: train ends at least purge_gap_bars before first_val_idx
        train_end_idx = first_val_idx - purge_gap_bars - 1
        train_target_indices = valid_indices[valid_indices <= train_end_idx]

        train_start_ts = df["timestamp"].iloc[train_target_indices[0]]
        train_end_ts = df["timestamp"].iloc[train_target_indices[-1]]
        val_start_ts = df["timestamp"].iloc[first_val_idx]
        val_end_ts = df["timestamp"].iloc[last_val_idx]

        tr_y = target_series.iloc[train_target_indices]
        va_y = target_series.iloc[val_target_indices]
        tr_dist = {"long": int((tr_y == 1.0).sum()), "short": int((tr_y == -1.0).sum())}
        va_dist = {"long": int((va_y == 1.0).sum()), "short": int((va_y == -1.0).sum())}

        folds.append(
            WalkForwardFold(
                fold_idx=fold_idx,
                train_start_ts=train_start_ts,
                train_end_ts=train_end_ts,
                val_start_ts=val_start_ts,
                val_end_ts=val_end_ts,
                train_indices=train_target_indices,
                val_indices=val_target_indices,
                train_sample_count=len(train_target_indices),
                val_sample_count=len(val_target_indices),
                purge_gap_bars=purge_gap_bars,
                train_class_dist=tr_dist,
                val_class_dist=va_dist,
            )
        )

    return folds


def evaluate_feature_ablation_experiment(
    h4_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    folds: list[WalkForwardFold],
    rf_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a feature set across all walk-forward folds using the frozen Random Forest."""
    if rf_config is None:
        rf_config = FROZEN_RF_CONFIG.copy()

    fold_metrics: list[dict[str, Any]] = []
    all_oof_preds: list[float] = []
    all_oof_probs: list[float] = []
    all_oof_indices: list[int] = []

    for f in folds:
        train_idx = f.train_indices
        val_idx = f.val_indices

        X_train = features_df.iloc[train_idx].to_numpy()
        y_train = target_series.iloc[train_idx].to_numpy()
        X_val = features_df.iloc[val_idx].to_numpy()
        y_val = target_series.iloc[val_idx].to_numpy()

        rf = RandomForestClassifier(**rf_config)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_val)
        probs_all = rf.predict_proba(X_val)
        idx_long = list(rf.classes_).index(1.0) if 1.0 in rf.classes_ else 1
        probs = probs_all[:, idx_long]

        bal_acc = float(balanced_accuracy_score(y_val, preds))
        acc = float(accuracy_score(y_val, preds))
        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        prec = float(precision_score(y_val, preds, pos_label=1.0, zero_division=0))
        rec = float(recall_score(y_val, preds, pos_label=1.0, zero_division=0))
        try:
            y_val_bin = (y_val == 1.0).astype(int)
            auc = float(roc_auc_score(y_val_bin, probs))
        except Exception:
            auc = float("nan")

        fold_metrics.append(
            {
                "fold_idx": f.fold_idx,
                "train_start": f.train_start_ts.isoformat(),
                "train_end": f.train_end_ts.isoformat(),
                "val_start": f.val_start_ts.isoformat(),
                "val_end": f.val_end_ts.isoformat(),
                "train_samples": f.train_sample_count,
                "val_samples": f.val_sample_count,
                "balanced_accuracy": bal_acc,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "roc_auc": auc,
                "precision": prec,
                "recall": rec,
            }
        )

        all_oof_preds.extend(preds.tolist())
        all_oof_probs.extend(probs.tolist())
        all_oof_indices.extend(val_idx.tolist())

    bal_accs = [m["balanced_accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]

    summary = {
        "mean_balanced_accuracy": float(np.mean(bal_accs)),
        "median_balanced_accuracy": float(np.median(bal_accs)),
        "std_balanced_accuracy": float(np.std(bal_accs)),
        "min_balanced_accuracy": float(np.min(bal_accs)),
        "max_balanced_accuracy": float(np.max(bal_accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "folds_above_50_pct": int(sum(b > 0.50 for b in bal_accs)),
        "folds_above_55_pct": int(sum(b > 0.55 for b in bal_accs)),
        "folds_below_50_pct": int(sum(b < 0.50 for b in bal_accs)),
        "total_folds": len(folds),
        "total_oof_predictions": len(all_oof_preds),
    }

    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "oof_predictions": {
            "indices": all_oof_indices,
            "predictions": all_oof_preds,
            "probabilities": all_oof_probs,
        },
    }


def evaluate_single_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    rf_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single out-of-sample evaluation on the Fresh Research Holdout."""
    if rf_config is None:
        rf_config = FROZEN_RF_CONFIG.copy()

    # Align valid samples
    train_mask = y_train.notna() & X_train.notna().all(axis=1)
    holdout_mask = y_holdout.notna() & X_holdout.notna().all(axis=1)

    X_tr = X_train[train_mask].to_numpy()
    y_tr = y_train[train_mask].to_numpy()

    X_ho = X_holdout[holdout_mask].to_numpy()
    y_ho = y_holdout[holdout_mask].to_numpy()

    # Random Forest evaluation
    rf = RandomForestClassifier(**rf_config)
    rf.fit(X_tr, y_tr)
    preds_rf = rf.predict(X_ho)
    probs_rf_all = rf.predict_proba(X_ho)
    idx_long_rf = list(rf.classes_).index(1.0) if 1.0 in rf.classes_ else 1
    probs_rf = probs_rf_all[:, idx_long_rf]

    # Logistic Regression evaluation
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_ho_scaled = scaler.transform(X_ho)

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_tr_scaled, y_tr)
    preds_lr = lr.predict(X_ho_scaled)
    probs_lr_all = lr.predict_proba(X_ho_scaled)
    idx_long_lr = list(lr.classes_).index(1.0) if 1.0 in lr.classes_ else 1
    probs_lr = probs_lr_all[:, idx_long_lr]

    y_ho_bin = (y_ho == 1.0).astype(int)
    cm_rf = confusion_matrix(y_ho, preds_rf, labels=[-1.0, 1.0]).tolist()

    return {
        "holdout_samples": len(y_ho),
        "random_forest": {
            "balanced_accuracy": float(balanced_accuracy_score(y_ho, preds_rf)),
            "accuracy": float(accuracy_score(y_ho, preds_rf)),
            "macro_f1": float(f1_score(y_ho, preds_rf, average="macro", zero_division=0)),
            "precision": float(precision_score(y_ho, preds_rf, pos_label=1.0, zero_division=0)),
            "recall": float(recall_score(y_ho, preds_rf, pos_label=1.0, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_ho_bin, probs_rf)),
            "confusion_matrix": cm_rf,
            "feature_importances": dict(zip(X_train.columns, rf.feature_importances_.tolist())),
        },
        "logistic_regression": {
            "balanced_accuracy": float(balanced_accuracy_score(y_ho, preds_lr)),
            "accuracy": float(accuracy_score(y_ho, preds_lr)),
            "macro_f1": float(f1_score(y_ho, preds_lr, average="macro", zero_division=0)),
            "precision": float(precision_score(y_ho, preds_lr, pos_label=1.0, zero_division=0)),
            "recall": float(recall_score(y_ho, preds_lr, pos_label=1.0, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_ho_bin, probs_lr)),
        },
    }
