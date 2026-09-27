"""Comprehensive baseline model diagnostics, probability analysis, and feature importance.

Phase 9 Diagnostic Module:
- Validation-only probability distributions and quantile analysis
- Confidence / coverage threshold trade-offs (0.40 to 0.90)
- Class-wise performance (precision, recall, F1, OvR ROC-AUC, Average Precision)
- OvR ROC curves and Precision-Recall curves
- Random Forest MDI feature importance ranking
- Logistic Regression mean absolute coefficient importance ranking
- Feature sanity checks against Phase 5 registry (no target, timestamp, or raw OHLCV leakage)
- Temporal validation stability analysis across contiguous chronological blocks (A, B, C)
- Misclassification error analysis and confidence profiling
- Model agreement and disagreement dynamics between LR and RF
- Descriptive probability calibration analysis (Brier scores and reliability curves)

CRITICAL SAFETY:
- The TEST partition is NEVER accessed, loaded, or evaluated.
- Models are NOT modified, hyperparameter-tuned, or calibrated.
- Diagnostics are purely descriptive and do not make profitability claims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Use headless backend for automated execution
matplotlib.use("Agg")

CLASS_LABELS: dict[float, str] = {
    -1.0: "SHORT (-1)",
    0.0: "NEUTRAL (0)",
    1.0: "LONG (+1)",
}

DEFAULT_CLASSES: list[float] = [-1.0, 0.0, 1.0]
DEFAULT_CONFIDENCE_THRESHOLDS: list[float] = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# =====================================================================
# 1. Probability Distribution Diagnostics
# =====================================================================


def compute_probability_diagnostics(
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Compute comprehensive probability distribution statistics and quantiles.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Class labels corresponding to probability matrix columns.

    Returns
    -------
    dict[str, Any]
        Dictionary of probability distribution metrics and quantiles.
    """
    prob_arr = np.asarray(probabilities, dtype=float)
    if prob_arr.ndim != 2 or prob_arr.shape[1] != len(classes):
        raise ValueError(
            f"Expected probability array of shape (N, {len(classes)}), got {prob_arr.shape}"
        )

    # Verify probability bounds and row summation
    if (prob_arr < 0.0).any() or (prob_arr > 1.0).any():
        raise ValueError("Probabilities must lie strictly in [0.0, 1.0].")

    row_sums = np.sum(prob_arr, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("Probability rows must sum to 1.0.")

    max_probs = np.max(prob_arr, axis=1)

    max_prob_stats = {
        "mean": float(np.mean(max_probs)),
        "std": float(np.std(max_probs)),
        "min": float(np.min(max_probs)),
        "p25": float(np.percentile(max_probs, 25)),
        "p50": float(np.percentile(max_probs, 50)),
        "p75": float(np.percentile(max_probs, 75)),
        "p90": float(np.percentile(max_probs, 90)),
        "p95": float(np.percentile(max_probs, 95)),
        "p99": float(np.percentile(max_probs, 99)),
        "max": float(np.max(max_probs)),
    }

    per_class_stats: dict[str, dict[str, float]] = {}
    for idx, c in enumerate(classes):
        label = CLASS_LABELS.get(float(c), f"Class {c}")
        col_probs = prob_arr[:, idx]
        per_class_stats[label] = {
            "mean": float(np.mean(col_probs)),
            "std": float(np.std(col_probs)),
            "min": float(np.min(col_probs)),
            "p25": float(np.percentile(col_probs, 25)),
            "p50": float(np.percentile(col_probs, 50)),
            "p75": float(np.percentile(col_probs, 75)),
            "p90": float(np.percentile(col_probs, 90)),
            "p95": float(np.percentile(col_probs, 95)),
            "p99": float(np.percentile(col_probs, 99)),
            "max": float(np.max(col_probs)),
        }

    return {
        "sample_count": len(prob_arr),
        "max_probability": max_prob_stats,
        "per_class_probability": per_class_stats,
    }


# =====================================================================
# 2. Confidence / Coverage Analysis
# =====================================================================


def compute_confidence_coverage(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> list[dict[str, Any]]:
    """Evaluate classification performance conditional on minimum prediction confidence.

    Exploratory diagnostics assessing precision/accuracy trade-offs as prediction
    confidence thresholds increase. This analysis is purely descriptive and does not
    establish an operational trading filter.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).
    y_true : pd.Series | np.ndarray
        True target labels.
    y_pred : pd.Series | np.ndarray
        Predicted class labels.
    thresholds : list[float] | None
        Confidence thresholds in [0, 1]. Defaults to [0.40, 0.50, 0.60, 0.70, 0.80, 0.90].
    classes : list[float] | np.ndarray
        Known class labels.

    Returns
    -------
    list[dict[str, Any]]
        List of coverage and performance metrics for each threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_CONFIDENCE_THRESHOLDS

    prob_arr = np.asarray(probabilities, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    total_samples = len(y_true_arr)
    max_probs = np.max(prob_arr, axis=1)

    records: list[dict[str, Any]] = []

    for thresh in thresholds:
        mask = max_probs >= thresh
        covered_count = int(np.sum(mask))
        coverage = float(covered_count / total_samples) if total_samples > 0 else 0.0

        if covered_count > 0:
            cov_true = y_true_arr[mask]
            cov_pred = y_pred_arr[mask]

            acc = float(accuracy_score(cov_true, cov_pred))
            # Balanced accuracy requires >= 2 distinct classes in subset to avoid warnings
            unique_classes_in_subset = np.unique(cov_true)
            if len(unique_classes_in_subset) > 1:
                bal_acc: float | None = float(balanced_accuracy_score(cov_true, cov_pred))
            else:
                bal_acc = None

            macro_f1 = float(f1_score(cov_true, cov_pred, average="macro", zero_division=0.0))

            pred_class_counts: dict[str, int] = {}
            for c in classes:
                label = CLASS_LABELS.get(float(c), f"Class {c}")
                pred_class_counts[label] = int(np.sum(cov_pred == c))
        else:
            acc = None
            bal_acc = None
            macro_f1 = None
            pred_class_counts = {CLASS_LABELS.get(float(c), f"Class {c}"): 0 for c in classes}

        records.append(
            {
                "threshold": float(thresh),
                "covered_count": covered_count,
                "total_count": total_samples,
                "coverage": coverage,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "predicted_class_counts": pred_class_counts,
            }
        )

    return records


# =====================================================================
# 3. Class-wise Performance & Discrimination Metrics
# =====================================================================


def compute_class_wise_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> dict[str, dict[str, Any]]:
    """Compute detailed per-class precision, recall, F1, OvR ROC-AUC, and Average Precision.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True class labels.
    y_pred : pd.Series | np.ndarray
        Predicted class labels.
    probabilities : np.ndarray
        Predicted probability array of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Known classes aligned with probability matrix columns.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-class diagnostic metrics mapped by readable class name.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)

    class_list = [float(c) for c in classes]
    results: dict[str, dict[str, Any]] = {}

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)
        y_pred_bin = (y_pred_arr == c).astype(int)
        prob_c = prob_arr[:, idx]

        support = int(np.sum(y_true_bin))
        pred_count = int(np.sum(y_pred_bin))

        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0.0))
        rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0.0))
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0.0))

        # OvR ROC-AUC
        if 0 < support < len(y_true_bin):
            roc_auc = float(roc_auc_score(y_true_bin, prob_c))
            avg_prec = float(average_precision_score(y_true_bin, prob_c))
        else:
            roc_auc = None
            avg_prec = None

        results[label] = {
            "class_value": c,
            "support": support,
            "predicted_count": pred_count,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc_ovr": roc_auc,
            "average_precision": avg_prec,
        }

    return results


# =====================================================================
# 4. Feature Importance Extraction
# =====================================================================


def compute_rf_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract and rank Random Forest Mean Decrease in Impurity (MDI) feature importances.

    Parameters
    ----------
    model : Any
        Fitted RandomForestBaseline instance or scikit-learn RandomForestClassifier.
    feature_names : list[str]
        List of 80 feature names matching model input columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['rank', 'feature', 'importance'] sorted descending.
    """
    if hasattr(model, "model_") and model.model_ is not None:
        raw_model = model.model_
    else:
        raw_model = model

    if not hasattr(raw_model, "feature_importances_"):
        raise ValueError("Provided model does not have feature_importances_ attribute.")

    importances = raw_model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: model has {len(importances)} features, "
            f"names list has {len(feature_names)}."
        )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def compute_lr_feature_importance(
    model: Any,
    feature_names: list[str],
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Extract and rank Logistic Regression coefficient magnitudes.

    For multiclass Logistic Regression, coefficients have shape (n_classes, n_features).
    The aggregate importance measure is the mean absolute coefficient across classes:
        aggregate_importance_j = (1 / K) * sum_{k=1}^K |w_{k, j}|

    Individual class coefficients (coef_short, coef_neutral, coef_long) are also provided.

    Parameters
    ----------
    model : Any
        Fitted LogisticRegressionBaseline instance or scikit-learn LogisticRegression.
    feature_names : list[str]
        List of 80 feature names matching model input columns.
    classes : list[float] | np.ndarray
        Class labels corresponding to model coefficient rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['rank', 'feature', 'aggregate_importance',
        'coef_short', 'coef_neutral', 'coef_long'] sorted descending by aggregate importance.
    """
    if hasattr(model, "model_") and model.model_ is not None:
        raw_model = model.model_
    else:
        raw_model = model

    if not hasattr(raw_model, "coef_"):
        raise ValueError("Provided model does not have coef_ attribute.")

    coef = raw_model.coef_  # Shape: (n_classes, n_features)
    if coef.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: coef has {coef.shape[1]} features, "
            f"names list has {len(feature_names)}."
        )

    agg_importance = np.mean(np.abs(coef), axis=0)

    # Map class rows to specific column names
    class_list = [float(c) for c in classes]
    coef_dict: dict[str, np.ndarray] = {}
    for idx, c in enumerate(class_list):
        if c == -1.0:
            coef_dict["coef_short"] = coef[idx]
        elif c == 0.0:
            coef_dict["coef_neutral"] = coef[idx]
        elif c == 1.0:
            coef_dict["coef_long"] = coef[idx]
        else:
            coef_dict[f"coef_class_{c}"] = coef[idx]

    data = {
        "feature": feature_names,
        "aggregate_importance": agg_importance,
        **coef_dict,
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="aggregate_importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# =====================================================================
# 5. Feature Sanity Check
# =====================================================================


def verify_feature_sanity(
    importance_df: pd.DataFrame,
    feature_registry_names: list[str],
) -> dict[str, Any]:
    """Verify that feature importances adhere strictly to Phase 5 definitions and constraints.

    Checks:
    1. Exactly 80 features.
    2. All features match Phase 5 feature registry.
    3. No target columns (direction_*, forward_return_*).
    4. No timestamp columns (open_time, timestamp, date, time).
    5. No raw OHLCV base columns (open, high, low, close, tick_volume, spread, real_volume).

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame containing a 'feature' column.
    feature_registry_names : list[str]
        Canonical 80 feature names from Phase 5 pipeline registry.

    Returns
    -------
    dict[str, Any]
        Sanity verification results dictionary.

    Raises
    ------
    ValueError
        If any feature sanity or leakage check fails.
    """
    if "feature" not in importance_df.columns:
        raise ValueError("importance_df must contain a 'feature' column.")

    features = list(importance_df["feature"])
    feature_set = set(features)
    registry_set = set(feature_registry_names)

    # 1. Feature count check
    if len(features) != 80:
        raise ValueError(
            f"Feature sanity violation: Expected exactly 80 features, found {len(features)}."
        )

    # 2. Registry match
    missing_from_registry = feature_set - registry_set
    if missing_from_registry:
        raise ValueError(
            f"Feature sanity violation: Features not in Phase 5 registry: {missing_from_registry}"
        )

    # 3. No target columns (direction_*, future_*, forward_*, target, label)
    prohibited_target_prefixes = ("direction_", "direction_vol_", "future_", "forward_")
    prohibited_target_exact = ("target", "label")
    leaked_targets = [
        f
        for f in features
        if f.lower().startswith(prohibited_target_prefixes) or f.lower() in prohibited_target_exact
    ]
    if leaked_targets:
        raise ValueError(f"Leakage violation: Target columns found in features: {leaked_targets}")

    # 4. No timestamp columns (strictly check raw timestamp names, allowing cyclical hour_sin etc)
    prohibited_timestamp_exact = ["open_time", "timestamp", "time", "date", "datetime"]
    leaked_timestamps = [f for f in features if f.lower() in prohibited_timestamp_exact]
    if leaked_timestamps:
        raise ValueError(
            f"Leakage violation: Timestamp columns found in features: {leaked_timestamps}"
        )

    # 5. No raw base columns
    raw_base_columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    leaked_base = [f for f in features if f.lower() in raw_base_columns]
    if leaked_base:
        raise ValueError(
            f"Sanity violation: Raw OHLCV base columns found in features: {leaked_base}"
        )

    return {
        "passed": True,
        "feature_count": len(features),
        "registry_count": len(feature_registry_names),
        "target_leakage_detected": False,
        "timestamp_leakage_detected": False,
        "raw_base_columns_detected": False,
    }


# =====================================================================
# 6. Temporal Validation Diagnostics
# =====================================================================


def compute_temporal_validation_diagnostics(
    val_X: pd.DataFrame,
    val_y: pd.Series,
    val_timestamps: pd.Series,
    model_lr: Any,
    model_rf: Any,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Evaluate model performance stability across three contiguous chronological validation blocks.

    Partitions the validation period into:
    - Block A: First third of observations (chronologically earliest)
    - Block B: Middle third of observations
    - Block C: Final third of observations (chronologically latest)

    CRITICAL:
    - Models are NOT retrained on validation blocks.
    - Diagnostics assess temporal stability of fixed Phase 8 models.
    - This is NOT model selection or hyperparameter tuning.

    Parameters
    ----------
    val_X : pd.DataFrame
        Validation feature matrix.
    val_y : pd.Series
        Validation true targets.
    val_timestamps : pd.Series
        Validation timestamps.
    model_lr : Any
        Fitted Logistic Regression baseline model.
    model_rf : Any
        Fitted Random Forest baseline model.
    classes : list[float] | np.ndarray
        Class labels.

    Returns
    -------
    pd.DataFrame
        Temporal validation metrics across blocks A, B, and C for both models.
    """
    n_val = len(val_X)
    block_size = n_val // 3
    class_list = [float(c) for c in classes]

    blocks = [
        ("VALIDATION_A", 0, block_size),
        ("VALIDATION_B", block_size, 2 * block_size),
        ("VALIDATION_C", 2 * block_size, n_val),
    ]

    models = [
        ("Logistic Regression", model_lr),
        ("Random Forest", model_rf),
    ]

    records: list[dict[str, Any]] = []

    for model_name, model in models:
        # Precompute predictions and probabilities across validation
        y_pred_all = model.predict(val_X)
        prob_all = model.predict_proba(val_X)

        for block_name, start_idx, end_idx in blocks:
            b_X = val_X.iloc[start_idx:end_idx]
            b_y = np.asarray(val_y.iloc[start_idx:end_idx], dtype=float)
            b_timestamps = val_timestamps.iloc[start_idx:end_idx]

            b_pred = y_pred_all[start_idx:end_idx]
            b_prob = prob_all[start_idx:end_idx]

            start_time_str = str(b_timestamps.iloc[0])
            end_time_str = str(b_timestamps.iloc[-1])
            b_rows = len(b_X)

            acc = float(accuracy_score(b_y, b_pred))
            bal_acc = float(balanced_accuracy_score(b_y, b_pred))
            macro_f1 = float(f1_score(b_y, b_pred, average="macro", zero_division=0.0))

            # Per-class recalls
            rec_short = float(recall_score(b_y == -1.0, b_pred == -1.0, zero_division=0.0))
            rec_neutral = float(recall_score(b_y == 0.0, b_pred == 0.0, zero_division=0.0))
            rec_long = float(recall_score(b_y == 1.0, b_pred == 1.0, zero_division=0.0))

            # Multiclass OvR ROC-AUC if all classes present in block
            unique_classes = np.unique(b_y)
            if len(unique_classes) == len(class_list):
                try:
                    roc_auc = float(roc_auc_score(b_y, b_prob, multi_class="ovr", average="macro"))
                except ValueError:
                    roc_auc = None
            else:
                roc_auc = None

            records.append(
                {
                    "model": model_name,
                    "block": block_name,
                    "start_time": start_time_str,
                    "end_time": end_time_str,
                    "rows": b_rows,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "macro_f1": macro_f1,
                    "recall_short": rec_short,
                    "recall_neutral": rec_neutral,
                    "recall_long": rec_long,
                    "roc_auc_ovr": roc_auc,
                }
            )

    return pd.DataFrame(records)


# =====================================================================
# 7. Error Analysis
# =====================================================================


def compute_error_analysis(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> pd.DataFrame:
    """Analyze validation misclassifications and prediction confidence profiles.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target labels.
    y_pred : pd.Series | np.ndarray
        Predicted class labels.
    probabilities : np.ndarray
        Predicted probability array of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Known class labels.

    Returns
    -------
    pd.DataFrame
        Detailed breakdown of (actual, predicted) pairs with frequency and confidence stats.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)

    total_samples = len(y_true_arr)
    total_errors = int(np.sum(y_true_arr != y_pred_arr))
    max_probs = np.max(prob_arr, axis=1)

    records: list[dict[str, Any]] = []

    for true_c in classes:
        for pred_c in classes:
            mask = (y_true_arr == true_c) & (y_pred_arr == pred_c)
            count = int(np.sum(mask))
            is_correct = bool(true_c == pred_c)

            pct_total = float(count / total_samples * 100.0) if total_samples > 0 else 0.0
            if not is_correct:
                pct_errors = float(count / total_errors * 100.0) if total_errors > 0 else 0.0
            else:
                pct_errors = 0.0

            if count > 0:
                conf_subset = max_probs[mask]
                mean_conf = float(np.mean(conf_subset))
                median_conf = float(np.median(conf_subset))
                min_conf = float(np.min(conf_subset))
                max_conf = float(np.max(conf_subset))
            else:
                mean_conf = 0.0
                median_conf = 0.0
                min_conf = 0.0
                max_conf = 0.0

            records.append(
                {
                    "actual_class": float(true_c),
                    "predicted_class": float(pred_c),
                    "actual_label": CLASS_LABELS.get(float(true_c), f"{true_c}"),
                    "predicted_label": CLASS_LABELS.get(float(pred_c), f"{pred_c}"),
                    "is_correct": is_correct,
                    "count": count,
                    "pct_of_total": pct_total,
                    "pct_of_errors": pct_errors,
                    "mean_confidence": mean_conf,
                    "median_confidence": median_conf,
                    "min_confidence": min_conf,
                    "max_confidence": max_conf,
                }
            )

    df = pd.DataFrame(records)
    # Sort with errors first (largest error count descending), then correct
    df = df.sort_values(by=["is_correct", "count"], ascending=[True, False]).reset_index(drop=True)
    return df


# =====================================================================
# 8. Model Agreement Diagnostics
# =====================================================================


def compute_model_agreement(
    y_pred_lr: pd.Series | np.ndarray,
    y_pred_rf: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
) -> dict[str, Any]:
    """Compute agreement and disagreement dynamics between Logistic Regression and Random Forest.

    Parameters
    ----------
    y_pred_lr : pd.Series | np.ndarray
        Predictions from Logistic Regression.
    y_pred_rf : pd.Series | np.ndarray
        Predictions from Random Forest.
    y_true : pd.Series | np.ndarray
        True validation targets.
    classes : list[float] | np.ndarray
        Known class labels.

    Returns
    -------
    dict[str, Any]
        Agreement statistics, Cohen's Kappa, contingency table, and conditional accuracies.
    """
    lr_arr = np.asarray(y_pred_lr, dtype=float)
    rf_arr = np.asarray(y_pred_rf, dtype=float)
    y_true_arr = np.asarray(y_true, dtype=float)

    total_samples = len(y_true_arr)
    agree_mask = lr_arr == rf_arr
    disagree_mask = ~agree_mask

    agree_count = int(np.sum(agree_mask))
    disagree_count = int(np.sum(disagree_mask))
    agree_rate = float(agree_count / total_samples) if total_samples > 0 else 0.0

    kappa = float(cohen_kappa_score(lr_arr, rf_arr))

    # Agreement by predicted class
    class_agreement: dict[str, dict[str, Any]] = {}
    for c in classes:
        label = CLASS_LABELS.get(float(c), f"Class {c}")
        lr_pred_c = lr_arr == c
        rf_pred_c = rf_arr == c
        both_c = int(np.sum(lr_pred_c & rf_pred_c))
        lr_only_c = int(np.sum(lr_pred_c & ~rf_pred_c))
        rf_only_c = int(np.sum(~lr_pred_c & rf_pred_c))

        class_agreement[label] = {
            "both_predicted_count": both_c,
            "lr_predicted_count": int(np.sum(lr_pred_c)),
            "rf_predicted_count": int(np.sum(rf_pred_c)),
            "lr_only_count": lr_only_c,
            "rf_only_count": rf_only_c,
        }

    # Contingency matrix (Rows: LR, Columns: RF)
    matrix: list[list[int]] = []
    for lr_c in classes:
        row: list[int] = []
        for rf_c in classes:
            cnt = int(np.sum((lr_arr == lr_c) & (rf_arr == rf_c)))
            row.append(cnt)
        matrix.append(row)

    # Accuracy when models agree vs disagree
    if agree_count > 0:
        joint_accuracy_agree = float(accuracy_score(y_true_arr[agree_mask], lr_arr[agree_mask]))
    else:
        joint_accuracy_agree = 0.0

    if disagree_count > 0:
        lr_acc = accuracy_score(y_true_arr[disagree_mask], lr_arr[disagree_mask])
        rf_acc = accuracy_score(y_true_arr[disagree_mask], rf_arr[disagree_mask])
        lr_accuracy_disagree = float(lr_acc)
        rf_accuracy_disagree = float(rf_acc)
    else:
        lr_accuracy_disagree = 0.0
        rf_accuracy_disagree = 0.0

    return {
        "total_samples": total_samples,
        "agreement_count": agree_count,
        "disagreement_count": disagree_count,
        "agreement_rate": agree_rate,
        "cohen_kappa": kappa,
        "contingency_matrix_lr_rows_rf_cols": matrix,
        "classes": [float(c) for c in classes],
        "class_labels": [CLASS_LABELS.get(float(c), f"{c}") for c in classes],
        "class_agreement": class_agreement,
        "joint_accuracy_on_agreement": joint_accuracy_agree,
        "lr_accuracy_on_disagreement": lr_accuracy_disagree,
        "rf_accuracy_on_disagreement": rf_accuracy_disagree,
    }


# =====================================================================
# 9. Probability Calibration Diagnostics (No Calibration Fitted)
# =====================================================================


def compute_calibration_diagnostics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray = (-1.0, 0.0, 1.0),
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute descriptive probability calibration diagnostics (Brier score, reliability curves).

    IMPORTANT:
    - This is purely descriptive diagnostics.
    - NO calibration models (Platt scaling, isotonic regression) are fitted.
    - The baseline models remain unmodified.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True validation targets.
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Class labels corresponding to columns.
    n_bins : int
        Number of bins for reliability curve computation.

    Returns
    -------
    dict[str, Any]
        Per-class Brier score, multiclass Brier score, and reliability curve coordinates.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    class_list = [float(c) for c in classes]

    per_class_results: dict[str, Any] = {}
    y_true_one_hot = np.zeros_like(prob_arr)

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)
        y_true_one_hot[:, idx] = y_true_bin
        prob_c = prob_arr[:, idx]

        brier = float(brier_score_loss(y_true_bin, prob_c))
        prob_true, prob_pred = calibration_curve(
            y_true_bin, prob_c, n_bins=n_bins, strategy="uniform"
        )

        per_class_results[label] = {
            "brier_score": brier,
            "prob_true": [float(p) for p in prob_true],
            "prob_pred": [float(p) for p in prob_pred],
        }

    # Multiclass Brier score: mean squared error between one-hot truth and probability vector
    multiclass_brier = float(np.mean(np.sum((y_true_one_hot - prob_arr) ** 2, axis=1)))

    return {
        "multiclass_brier_score": multiclass_brier,
        "n_bins": n_bins,
        "per_class": per_class_results,
    }


# =====================================================================
# 10. Visualization Plotting Functions
# =====================================================================


def plot_max_probability_distribution(
    probabilities: np.ndarray,
    model_name: str,
    save_path: Path | str,
) -> None:
    """Plot histogram and KDE summary of maximum predicted probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).
    model_name : str
        Descriptive model identifier.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    max_probs = np.max(probabilities, axis=1)
    mean_val = float(np.mean(max_probs))
    median_val = float(np.median(max_probs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        max_probs,
        bins=50,
        density=True,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.4f}",
    )
    ax.axvline(
        median_val,
        color="green",
        linestyle=":",
        linewidth=1.8,
        label=f"Median: {median_val:.4f}",
    )

    ax.set_title(
        f"{model_name} — Maximum Predicted Probability Distribution\n(Validation Set, H=4)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Maximum Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Density", fontsize=10)
    ax.set_xlim(0.30, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_probability_by_class(
    probabilities: np.ndarray,
    y_true: pd.Series | np.ndarray,
    classes: list[float] | np.ndarray,
    model_name: str,
    save_path: Path | str,
) -> None:
    """Plot boxplot of predicted probabilities grouped by true target class.

    Parameters
    ----------
    probabilities : np.ndarray
        Predicted class probabilities of shape (n_samples, n_classes).
    y_true : pd.Series | np.ndarray
        True target labels.
    classes : list[float] | np.ndarray
        Known class labels.
    model_name : str
        Descriptive model identifier.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    class_list = [float(c) for c in classes]

    fig, axes = plt.subplots(1, len(class_list), figsize=(14, 4.5), sharey=True)
    colors = ["#d62728", "#7f7f7f", "#2ca02c"]

    for idx, c in enumerate(class_list):
        ax = axes[idx]
        target_label = CLASS_LABELS.get(c, f"Class {c}")
        # Predicted probability for class c conditioned on true class
        data_by_true = [prob_arr[y_true_arr == true_c, idx] for true_c in class_list]
        labels = [CLASS_LABELS.get(tc, f"{tc}").split()[0] for tc in class_list]

        bp = ax.boxplot(
            data_by_true,
            tick_labels=labels,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.6)

        ax.set_title(f"Predicted P({target_label})", fontsize=11, fontweight="bold")
        ax.set_xlabel("True Class", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Predicted Probability", fontsize=10)

    fig.suptitle(
        f"{model_name} — Predicted Probabilities by True Class (Validation Set)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray,
    model_name: str,
    save_path: Path | str,
) -> None:
    """Plot One-vs-Rest ROC curves for each target class.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target labels.
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Known class labels.
    model_name : str
        Descriptive model identifier.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    class_list = [float(c) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"SHORT (-1)": "#d62728", "NEUTRAL (0)": "#7f7f7f", "LONG (+1)": "#2ca02c"}

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)
        prob_c = prob_arr[:, idx]

        fpr, tpr, _ = roc_curve(y_true_bin, prob_c)
        auc_val = roc_auc_score(y_true_bin, prob_c)

        ax.plot(
            fpr,
            tpr,
            label=f"{label} (AUC = {auc_val:.4f})",
            color=colors.get(label, "#1f77b4"),
            linewidth=2.0,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random Guess (AUC = 0.50)")
    ax.set_title(
        f"{model_name} — One-vs-Rest ROC Curves\n(Validation Set, H=4)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_model_comparison(
    y_true: pd.Series | np.ndarray,
    prob_lr: np.ndarray,
    prob_rf: np.ndarray,
    classes: list[float] | np.ndarray,
    save_path: Path | str,
) -> None:
    """Plot comparative One-vs-Rest ROC curves between Logistic Regression and Random Forest.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target labels.
    prob_lr : np.ndarray
        Predicted probabilities from Logistic Regression.
    prob_rf : np.ndarray
        Predicted probabilities from Random Forest.
    classes : list[float] | np.ndarray
        Known class labels.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    class_list = [float(c) for c in classes]

    fig, ax = plt.subplots(figsize=(8, 7))
    class_styles = {
        "SHORT (-1)": ("#d62728", "-"),
        "NEUTRAL (0)": ("#7f7f7f", "-"),
        "LONG (+1)": ("#2ca02c", "-"),
    }

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)

        # LR
        fpr_lr, tpr_lr, _ = roc_curve(y_true_bin, prob_lr[:, idx])
        auc_lr = roc_auc_score(y_true_bin, prob_lr[:, idx])
        color, _ = class_styles.get(label, ("#1f77b4", "-"))

        ax.plot(
            fpr_lr,
            tpr_lr,
            label=f"LR: {label} (AUC = {auc_lr:.4f})",
            color=color,
            linestyle="--",
            linewidth=1.8,
        )

        # RF
        fpr_rf, tpr_rf, _ = roc_curve(y_true_bin, prob_rf[:, idx])
        auc_rf = roc_auc_score(y_true_bin, prob_rf[:, idx])
        ax.plot(
            fpr_rf,
            tpr_rf,
            label=f"RF: {label} (AUC = {auc_rf:.4f})",
            color=color,
            linestyle="-",
            linewidth=2.0,
        )

    ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Random Guess (0.50)")
    ax.set_title(
        "Baseline Model Comparison — One-vs-Rest ROC Curves\n(Validation Set, H=4)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray,
    model_name: str,
    save_path: Path | str,
) -> None:
    """Plot One-vs-Rest Precision-Recall curves with Average Precision and no-skill baselines.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target labels.
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Known class labels.
    model_name : str
        Descriptive model identifier.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    class_list = [float(c) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"SHORT (-1)": "#d62728", "NEUTRAL (0)": "#7f7f7f", "LONG (+1)": "#2ca02c"}

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)
        prob_c = prob_arr[:, idx]

        precision, recall, _ = precision_recall_curve(y_true_bin, prob_c)
        ap_val = average_precision_score(y_true_bin, prob_c)
        prevalence = float(np.mean(y_true_bin))

        color = colors.get(label, "#1f77b4")
        ax.plot(
            recall,
            precision,
            label=f"{label} (AP = {ap_val:.4f})",
            color=color,
            linewidth=2.0,
        )
        ax.axhline(
            prevalence,
            color=color,
            linestyle=":",
            alpha=0.6,
            linewidth=1.2,
            label=f"{label} baseline ({prevalence:.1%})",
        )

    ax.set_title(
        f"{model_name} — Precision-Recall Curves\n(Validation Set, H=4)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance_top20(
    importance_df: pd.DataFrame,
    title: str,
    save_path: Path | str,
    value_col: str = "importance",
) -> None:
    """Plot horizontal bar chart of the top 20 ranked features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature ranking.
    title : str
        Chart title.
    save_path : Path | str
        Destination path for saved PNG figure.
    value_col : str
        Column name for feature importance values (e.g. 'importance' or 'aggregate_importance').
    """
    top20 = importance_df.head(20).copy()
    # Invert order for horizontal bar chart (top rank at top)
    top20 = top20.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        top20["feature"],
        top20[value_col],
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Metric", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add numeric labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max(top20[value_col]) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_temporal_validation_diagnostics(
    temporal_df: pd.DataFrame,
    save_path: Path | str,
) -> None:
    """Plot performance metric variation across chronological validation blocks A, B, and C.

    Parameters
    ----------
    temporal_df : pd.DataFrame
        Temporal validation metrics DataFrame.
    save_path : Path | str
        Destination path for saved PNG figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    models = ["Logistic Regression", "Random Forest"]
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_labels = ["Accuracy", "Balanced Acc", "Macro F1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        sub_df = temporal_df[temporal_df["model"] == model_name].copy()
        blocks = list(sub_df["block"])
        x = np.arange(len(blocks))
        width = 0.25

        for m_idx, (m_col, m_name) in enumerate(zip(metrics, metric_labels)):
            vals = sub_df[m_col].values
            ax.bar(
                x + (m_idx - 1) * width,
                vals,
                width=width,
                label=m_name,
                color=colors[m_idx],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_title(f"{model_name}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Block A (Early)", "Block B (Mid)", "Block C (Late)"], fontsize=9)
        ax.set_ylim(0.0, 0.8)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        if idx == 0:
            ax.set_ylabel("Validation Score", fontsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.suptitle(
        "Temporal Validation Diagnostics Across Contiguous Chronological Blocks",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_calibration_curves(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: list[float] | np.ndarray,
    model_name: str,
    save_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Plot reliability curves (calibration curves) for each target class.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target labels.
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    classes : list[float] | np.ndarray
        Known class labels.
    model_name : str
        Descriptive model identifier.
    save_path : Path | str
        Destination path for saved PNG figure.
    n_bins : int
        Number of bins for probability binning.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    class_list = [float(c) for c in classes]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"SHORT (-1)": "#d62728", "NEUTRAL (0)": "#7f7f7f", "LONG (+1)": "#2ca02c"}

    for idx, c in enumerate(class_list):
        label = CLASS_LABELS.get(c, f"Class {c}")
        y_true_bin = (y_true_arr == c).astype(int)
        prob_c = prob_arr[:, idx]

        prob_true, prob_pred = calibration_curve(
            y_true_bin, prob_c, n_bins=n_bins, strategy="uniform"
        )
        brier = brier_score_loss(y_true_bin, prob_c)

        color = colors.get(label, "#1f77b4")
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            label=f"{label} (Brier = {brier:.4f})",
            color=color,
            linewidth=1.8,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect Calibration")
    ax.set_title(
        f"{model_name} — Reliability Curves (Probability Calibration Diagnostics)\n"
        f"(Validation Set, {n_bins} Bins, Uncalibrated)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
