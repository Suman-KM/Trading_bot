"""Expanded 16-year chronological walk-forward evaluation generator for H4 swing research.

Executes expanding-window rolling-origin evaluation across 10 chronological folds
spanning 2013 to 2026, enforcing an exact 8-bar (32-hour) de Prado purge gap at every boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ai.swing.evaluation import (
    FROZEN_ET_CONFIG,
    FROZEN_LR_CONFIG,
    FROZEN_RF_CONFIG,
    compute_binary_metrics,
)
from ai.swing.walk_forward import (
    DEFAULT_RESEARCH_CUTOFF_TS,
    LOCKED_TEST_START_TS,
    WalkForwardFold,
)


def generate_expanded_walk_forward_folds(
    h4_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    initial_train_bars: int = 5000,
    n_folds: int = 10,
    purge_bars: int = 8,
    research_end_ts: pd.Timestamp = DEFAULT_RESEARCH_CUTOFF_TS,
) -> list[WalkForwardFold]:
    """Generate 10 chronological expanding-window walk-forward folds across expanded history.

    Parameters
    ----------
    h4_df : pd.DataFrame
        H4 OHLCV DataFrame (2010 to 2026).
    features_df : pd.DataFrame
        Computed swing features.
    target_series : pd.Series
        Target labels (direction_vol_8).
    initial_train_bars : int, default 5000
        Initial training warmup window (~3.2 years).
    n_folds : int, default 10
        Number of forward chronological validation folds.
    purge_bars : int, default 8
        Horizon in bars to purge between train and validation.
    research_end_ts : pd.Timestamp, default DEFAULT_RESEARCH_CUTOFF_TS
        Upper cutoff. Strictly excludes locked test partition.

    Returns
    -------
    list[WalkForwardFold]
        List of WalkForwardFold objects.
    """
    ts = pd.to_datetime(h4_df["timestamp"], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("h4_df timestamps must be strictly monotonic increasing.")

    # Strictly filter to pre-test research partition
    research_mask = ts <= research_end_ts
    if (ts[research_mask] > LOCKED_TEST_START_TS).any():
        raise PermissionError("Attempted to access locked test partition data.")

    h4_res = h4_df[research_mask].copy().reset_index(drop=True)
    y_res = target_series.loc[research_mask].reset_index(drop=True)
    X_res = features_df.loc[research_mask].reset_index(drop=True)
    ts_res = ts[research_mask].reset_index(drop=True)

    n_raw = len(h4_res)
    remaining_bars = n_raw - initial_train_bars
    fold_size = remaining_bars // n_folds

    folds: list[WalkForwardFold] = []

    for f in range(n_folds):
        val_start = initial_train_bars + f * fold_size
        val_end = val_start + fold_size if f < n_folds - 1 else n_raw

        train_end = val_start - purge_bars
        val_eval_end = val_end - purge_bars

        tr_raw_indices = np.arange(0, train_end)
        va_raw_indices = np.arange(val_start, val_eval_end)

        tr_valid_mask = y_res.iloc[tr_raw_indices].notna() & ~X_res.iloc[tr_raw_indices].isna().any(
            axis=1
        )
        va_valid_mask = y_res.iloc[va_raw_indices].notna() & ~X_res.iloc[va_raw_indices].isna().any(
            axis=1
        )

        tr_indices = tr_raw_indices[tr_valid_mask]
        va_indices = va_raw_indices[va_valid_mask]

        t_tr_start = ts_res.iloc[tr_indices[0]]
        t_tr_end = ts_res.iloc[tr_indices[-1]]
        t_va_start = ts_res.iloc[va_indices[0]]
        t_va_end = ts_res.iloc[va_indices[-1]]

        assert t_tr_end < t_va_start, f"Fold {f + 1}: Temporal overlap detected!"

        tr_y_fold = y_res.iloc[tr_indices]
        va_y_fold = y_res.iloc[va_indices]

        tr_dist = {
            "long": int((tr_y_fold == 1.0).sum()),
            "short": int((tr_y_fold == -1.0).sum()),
        }
        va_dist = {
            "long": int((va_y_fold == 1.0).sum()),
            "short": int((va_y_fold == -1.0).sum()),
        }

        fold = WalkForwardFold(
            fold_idx=f + 1,
            train_indices=tr_indices,
            val_indices=va_indices,
            train_start_ts=t_tr_start,
            train_end_ts=t_tr_end,
            val_start_ts=t_va_start,
            val_end_ts=t_va_end,
            purge_gap_bars=purge_bars,
            train_sample_count=len(tr_indices),
            val_sample_count=len(va_indices),
            train_class_dist=tr_dist,
            val_class_dist=va_dist,
        )
        folds.append(fold)

    return folds


def evaluate_expanded_walk_forward(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    folds: list[WalkForwardFold],
) -> dict[str, Any]:
    """Execute complete walk-forward evaluation across all 10 expanded folds.

    Parameters
    ----------
    features_df : pd.DataFrame
        Computed swing features for expanded H4 dataset.
    target_series : pd.Series
        Target labels (direction_vol_8).
    folds : list[WalkForwardFold]
        List of WalkForwardFold objects.

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

        # Majority
        vals, counts = np.unique(y_tr, return_counts=True)
        maj_cls = vals[np.argmax(counts)]
        pred_maj = np.full(len(y_va), maj_cls)
        m_maj = compute_binary_metrics(y_va, pred_maj)
        m_maj["fold_idx"] = fold.fold_idx
        model_fold_results["Majority"].append(m_maj)

        # Persistence
        if "return_8" in features_df.columns:
            past_ret = features_df.iloc[va_idx]["return_8"].to_numpy()
            pred_pers = np.where(past_ret > 0.0, 1.0, -1.0)
            m_pers = compute_binary_metrics(y_va, pred_pers)
        else:
            m_pers = m_maj.copy()
        m_pers["fold_idx"] = fold.fold_idx
        model_fold_results["Persistence"].append(m_pers)

        # Logistic Regression
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

        # Frozen Random Forest Candidate
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

        # Extra Trees
        et = ExtraTreesClassifier(**FROZEN_ET_CONFIG)
        et.fit(X_tr, y_tr)
        pred_et = et.predict(X_va)
        prob_et = et.predict_proba(X_va)
        m_et = compute_binary_metrics(y_va, pred_et, prob_et, list(et.classes_))
        m_et["fold_idx"] = fold.fold_idx
        model_fold_results["ExtraTrees"].append(m_et)

    # Compute summary
    summary: dict[str, Any] = {}
    for model_name, m_list in model_fold_results.items():
        bal_accs = [m["balanced_accuracy"] for m in m_list]
        accs = [m["accuracy"] for m in m_list]
        f1s = [m["macro_f1"] for m in m_list]
        aucs = [m["roc_auc"] for m in m_list if not np.isnan(m["roc_auc"])]

        summary[model_name] = {
            "mean_balanced_accuracy": float(np.mean(bal_accs)),
            "median_balanced_accuracy": float(np.median(bal_accs)),
            "std_balanced_accuracy": float(np.std(bal_accs)),
            "min_balanced_accuracy": float(np.min(bal_accs)),
            "max_balanced_accuracy": float(np.max(bal_accs)),
            "mean_accuracy": float(np.mean(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "mean_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "folds_above_50_pct": int(sum(b > 0.50 for b in bal_accs)),
            "folds_above_55_pct": int(sum(b > 0.55 for b in bal_accs)),
            "folds_below_50_pct": int(sum(b < 0.50 for b in bal_accs)),
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
