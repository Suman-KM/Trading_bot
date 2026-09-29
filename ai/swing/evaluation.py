"""Independent confirmation evaluation engine for H4 swing candidates and baselines.

Implements frozen model configurations (Random Forest candidate, Extra Trees,
Logistic Regression, Majority class, Naive persistence) across chronological
walk-forward folds with full metric tracking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

from ai.swing.walk_forward import WalkForwardFold

# Frozen Candidate Configuration (Phase 15 specification — strictly immutable)
FROZEN_RF_CONFIG: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# Frozen Extra Trees Configuration
FROZEN_ET_CONFIG: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# Frozen Logistic Regression Configuration
FROZEN_LR_CONFIG: dict[str, Any] = {
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 1000,
}


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
    classes: list[float] | None = None,
) -> dict[str, Any]:
    """Compute complete binary classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[-1.0, 1.0])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    prec_short = float(precision_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    prec_long = float(precision_score(y_true, y_pred, pos_label=1.0, zero_division=0))
    rec_short = float(recall_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    rec_long = float(recall_score(y_true, y_pred, pos_label=1.0, zero_division=0))
    f1_short = float(f1_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    f1_long = float(f1_score(y_true, y_pred, pos_label=1.0, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    auc = float("nan")
    if probs is not None and classes is not None and len(classes) == 2:
        try:
            idx_long = classes.index(1.0)
            p_long = probs[:, idx_long]
            y_bin = np.where(y_true == 1.0, 1, 0)
            auc = float(roc_auc_score(y_bin, p_long))
        except Exception:
            auc = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "precision_short": prec_short,
        "precision_long": prec_long,
        "recall_short": rec_short,
        "recall_long": rec_long,
        "f1_short": f1_short,
        "f1_long": f1_long,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": auc,
    }


def evaluate_walk_forward_candidate(
    h4_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    folds: list[WalkForwardFold],
) -> dict[str, Any]:
    """Execute complete walk-forward evaluation across all folds and models.

    Parameters
    ----------
    h4_df : pd.DataFrame
        H4 OHLCV DataFrame.
    features_df : pd.DataFrame
        Computed swing features.
    target_series : pd.Series
        Target labels (e.g. direction_vol_8).
    folds : list[WalkForwardFold]
        List of chronological WalkForwardFold objects.

    Returns
    -------
    dict[str, Any]
        Dictionary of fold-by-fold results, model comparisons, and robustness summary.
    """
    model_fold_results: dict[str, list[dict[str, Any]]] = {
        "Majority": [],
        "Persistence": [],
        "LogisticRegression": [],
        "RandomForest": [],
        "ExtraTrees": [],
    }

    # Store full out-of-sample predictions for candidate
    rf_oof_indices: list[int] = []
    rf_oof_preds: list[float] = []
    rf_oof_probs: list[list[float]] = []

    for fold in folds:
        tr_idx = fold.train_indices
        va_idx = fold.val_indices

        X_tr = features_df.iloc[tr_idx].reset_index(drop=True)
        y_tr = target_series.iloc[tr_idx].reset_index(drop=True).to_numpy()
        X_va = features_df.iloc[va_idx].reset_index(drop=True)
        y_va = target_series.iloc[va_idx].reset_index(drop=True).to_numpy()

        # -------------------------------------------------------------
        # Baseline A: Majority Class
        # -------------------------------------------------------------
        vals, counts = np.unique(y_tr, return_counts=True)
        maj_cls = vals[np.argmax(counts)]
        pred_maj = np.full(len(y_va), maj_cls)
        m_maj = compute_binary_metrics(y_va, pred_maj)
        m_maj["fold_idx"] = fold.fold_idx
        model_fold_results["Majority"].append(m_maj)

        # -------------------------------------------------------------
        # Baseline B: Naive Persistence (previous 8-bar return sign)
        # -------------------------------------------------------------
        # Using feature 'return_8' as realized past 8-bar return direction
        if "return_8" in features_df.columns:
            past_ret = features_df.iloc[va_idx]["return_8"].to_numpy()
            pred_pers = np.where(past_ret > 0.0, 1.0, -1.0)
            m_pers = compute_binary_metrics(y_va, pred_pers)
        else:
            m_pers = m_maj.copy()
        m_pers["fold_idx"] = fold.fold_idx
        model_fold_results["Persistence"].append(m_pers)

        # -------------------------------------------------------------
        # Baseline C: Logistic Regression (StandardScaler on train)
        # -------------------------------------------------------------
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        lr = LogisticRegression(**FROZEN_LR_CONFIG)
        lr.fit(X_tr_sc, y_tr)
        pred_lr = lr.predict(X_va_sc)
        prob_lr = lr.predict_proba(X_va_sc)
        m_lr = compute_binary_metrics(y_va, pred_lr, prob_lr, list(lr.classes_))
        m_lr["fold_idx"] = fold.fold_idx
        model_fold_results["LogisticRegression"].append(m_lr)

        # -------------------------------------------------------------
        # Candidate D: Frozen Random Forest
        # -------------------------------------------------------------
        rf = RandomForestClassifier(**FROZEN_RF_CONFIG)
        rf.fit(X_tr, y_tr)
        pred_rf = rf.predict(X_va)
        prob_rf = rf.predict_proba(X_va)
        m_rf = compute_binary_metrics(y_va, pred_rf, prob_rf, list(rf.classes_))
        m_rf["fold_idx"] = fold.fold_idx
        model_fold_results["RandomForest"].append(m_rf)

        rf_oof_indices.extend(va_idx)
        rf_oof_preds.extend(pred_rf)
        rf_oof_probs.extend(prob_rf.tolist())

        # -------------------------------------------------------------
        # Baseline E: Frozen Extra Trees
        # -------------------------------------------------------------
        et = ExtraTreesClassifier(**FROZEN_ET_CONFIG)
        et.fit(X_tr, y_tr)
        pred_et = et.predict(X_va)
        prob_et = et.predict_proba(X_va)
        m_et = compute_binary_metrics(y_va, pred_et, prob_et, list(et.classes_))
        m_et["fold_idx"] = fold.fold_idx
        model_fold_results["ExtraTrees"].append(m_et)

    # -----------------------------------------------------------------
    # Compute Across-Fold Robustness Summary
    # -----------------------------------------------------------------
    summary: dict[str, Any] = {}
    for model_name, m_list in model_fold_results.items():
        bal_accs = [m["balanced_accuracy"] for m in m_list]
        accs = [m["accuracy"] for m in m_list]
        f1s = [m["macro_f1"] for m in m_list]
        aucs = [m["roc_auc"] for m in m_list if not np.isnan(m["roc_auc"])]

        # Fold-to-fold degradation: max drop between adjacent chronological folds
        fold_drops = [bal_accs[i] - bal_accs[i + 1] for i in range(len(bal_accs) - 1)]
        max_degradation = float(max(fold_drops)) if fold_drops else 0.0

        summary[model_name] = {
            "mean_balanced_accuracy": float(np.mean(bal_accs)),
            "median_balanced_accuracy": float(np.median(bal_accs)),
            "std_balanced_accuracy": float(np.std(bal_accs)),
            "min_balanced_accuracy": float(np.min(bal_accs)),
            "max_balanced_accuracy": float(np.max(bal_accs)),
            "max_fold_to_fold_degradation": max_degradation,
            "mean_accuracy": float(np.mean(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "folds_above_50_pct": int(sum(b > 0.50 for b in bal_accs)),
            "folds_above_55_pct": int(sum(b > 0.55 for b in bal_accs)),
            "total_folds": len(bal_accs),
        }

    return {
        "fold_results": model_fold_results,
        "summary": summary,
        "oof_candidate_predictions": {
            "indices": rf_oof_indices,
            "predictions": rf_oof_preds,
            "probabilities": rf_oof_probs,
        },
    }
