"""Phase 13: Intraday Signal Improvement Research.

Executes all research tasks analyzing EURUSD M15 signal distribution, directional
separation, confidence calibration, feature information, temporal stability, and
controlled model comparison.

Strict Holdout Rule:
- Uses ONLY Training (69,937 rows) and Validation (14,983 rows) partitions.
- Phase 11 Test partition (14,988 rows) remains PERMANENTLY LOCKED and untouched.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.models.baselines import LogisticRegressionBaseline, RandomForestBaseline

REPORTS_DIR = Path("reports")


def categorize_feature(fname: str) -> str:
    """Classify feature into functional category based on naming convention."""
    fname_lower = fname.lower()
    categories = {
        "volatility": [
            "atr",
            "natr",
            "bb_",
            "bollinger",
            "keltner",
            "std",
            "volatility",
            "range",
            "hist_vol",
        ],
        "momentum": ["rsi", "roc", "macd", "stoch", "cci", "momentum", "williams", "cmo"],
        "trend": ["adx", "ema", "sma", "trend", "linear_reg", "dmi"],
        "time_of_day": ["hour", "day_of_week", "sin_", "cos_", "session"],
        "activity": ["volume", "spread", "tick_volume", "count"],
    }
    for cat, kw_list in categories.items():
        if any(kw in fname_lower for kw in kw_list):
            return cat
    return "other"


def compute_distribution_metrics(probs: np.ndarray) -> dict[str, float]:
    """Compute standard summary statistics for a probability array."""
    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "min": float(np.min(probs)),
        "p25": float(np.percentile(probs, 25)),
        "median": float(np.median(probs)),
        "p75": float(np.percentile(probs, 75)),
        "p90": float(np.percentile(probs, 90)),
        "p95": float(np.percentile(probs, 95)),
        "max": float(np.max(probs)),
    }


def compute_confidence_buckets(
    p_dir: np.ndarray,
    val_preds: np.ndarray,
    y_val_arr: np.ndarray,
    buckets: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Compute directional signal quality metrics across specified confidence intervals."""
    records = []
    n_total = len(y_val_arr)

    for low, high in buckets:
        mask = (p_dir >= low) & (p_dir < high) if high <= 1.0 else (p_dir >= low)
        cnt = int(np.sum(mask))
        pct_val = (cnt / n_total) * 100.0 if n_total > 0 else 0.0

        if cnt > 0:
            sub_preds = val_preds[mask]
            sub_true = y_val_arr[mask]

            n_pred_long = int(np.sum(sub_preds == 1.0))
            n_pred_short = int(np.sum(sub_preds == -1.0))
            n_pred_neutral = int(np.sum(sub_preds == 0.0))

            true_long_freq = float(np.mean(sub_true == 1.0) * 100.0)
            true_short_freq = float(np.mean(sub_true == -1.0) * 100.0)
            true_neutral_freq = float(np.mean(sub_true == 0.0) * 100.0)

            dir_pred_mask = sub_preds != 0.0
            if np.sum(dir_pred_mask) > 0:
                acc_val = accuracy_score(sub_true[dir_pred_mask], sub_preds[dir_pred_mask])
                dir_acc = float(acc_val * 100.0)
                n_dir_pred = int(np.sum(dir_pred_mask))
            else:
                dir_acc = None
                n_dir_pred = 0
        else:
            n_pred_long = 0
            n_pred_short = 0
            n_pred_neutral = 0
            true_long_freq = 0.0
            true_short_freq = 0.0
            true_neutral_freq = 0.0
            dir_acc = None
            n_dir_pred = 0

        label_str = f"{low:.2f}–{high:.2f}" if high <= 1.0 else f"{low:.2f}+"
        records.append(
            {
                "bucket": label_str,
                "count": cnt,
                "pct_validation": pct_val,
                "pred_long": n_pred_long,
                "pred_short": n_pred_short,
                "pred_neutral": n_pred_neutral,
                "dir_predictions_count": n_dir_pred,
                "dir_prediction_accuracy": dir_acc,
                "true_long_pct": true_long_freq,
                "true_short_pct": true_short_freq,
                "true_neutral_pct": true_neutral_freq,
            }
        )
    return records


def run_phase13_research() -> dict[str, Any]:
    """Execute complete Phase 13 intraday signal research study."""
    t_start_total = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 13: INTRADAY SIGNAL IMPROVEMENT RESEARCH")
    print(f"Timestamp (UTC): {now_utc}")
    print("=" * 80)

    # 1. Dataset Loading & Partition Verification
    print("\n[Step 1/8] Loading dataset and verifying partition isolation...")
    t0 = time.time()
    ds = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits = split_dataset(ds)

    assert len(splits.train) == 69937, f"Train size unexpected: {len(splits.train)}"
    assert len(splits.val) == 14983, f"Validation size unexpected: {len(splits.val)}"
    assert len(splits.test) == 14988, f"ALERT: Test partition altered: {len(splits.test)}"

    print(
        f"  Training partition:   {len(splits.train):,} rows "
        f"({splits.train.start_timestamp} -> {splits.train.end_timestamp})"
    )
    print(
        f"  Validation partition: {len(splits.val):,} rows "
        f"({splits.val.start_timestamp} -> {splits.val.end_timestamp})"
    )
    print(f"  Test partition:       {len(splits.test):,} rows [STRICTLY LOCKED / UNTOUCHED]")
    print(f"  Partition setup verified in {time.time() - t0:.2f}s")

    X_train, y_train = splits.train.X, splits.train.y
    X_val, y_val = splits.val.X, splits.val.y
    val_timestamps = splits.val.timestamps

    # 2. Fit Random Forest Baseline Once
    print("\n[Step 2/8] Fitting Random Forest baseline on Training data (69,937 rows)...")
    t0 = time.time()
    rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  RF fitted in {time.time() - t0:.2f}s across {len(X_train):,} training observations.")

    # 3. Vectorized Batch Inference on Validation Set
    print("\n[Step 3/8] Executing vectorized batch inference across Validation set...")
    t0 = time.time()
    val_probs = rf.predict_proba(X_val)
    val_preds = rf.predict(X_val)
    print(f"  Validation batch inference completed in {time.time() - t0:.2f}s")

    classes = list(rf.classes_)
    idx_short = classes.index(-1.0)
    idx_neutral = classes.index(0.0)
    idx_long = classes.index(1.0)

    p_short = val_probs[:, idx_short]
    p_neutral = val_probs[:, idx_neutral]
    p_long = val_probs[:, idx_long]
    p_dir = np.maximum(p_long, p_short)
    p_max = np.max(val_probs, axis=1)
    y_val_arr = np.asarray(y_val, dtype=float)

    results: dict[str, Any] = {}

    # Task 1: Audit Current Signal Distribution
    task1 = {
        "validation_rows": len(y_val_arr),
        "long_prob_dist": compute_distribution_metrics(p_long),
        "short_prob_dist": compute_distribution_metrics(p_short),
        "neutral_prob_dist": compute_distribution_metrics(p_neutral),
        "directional_confidence": {
            **compute_distribution_metrics(p_dir),
            "p99": float(np.percentile(p_dir, 99)),
        },
        "pct_dir_ge_40": float(np.mean(p_dir >= 0.40) * 100),
        "pct_dir_ge_45": float(np.mean(p_dir >= 0.45) * 100),
        "pct_dir_ge_50": float(np.mean(p_dir >= 0.50) * 100),
        "pct_dir_ge_55": float(np.mean(p_dir >= 0.55) * 100),
        "pct_dir_ge_60": float(np.mean(p_dir >= 0.60) * 100),
        "count_dir_ge_40": int(np.sum(p_dir >= 0.40)),
        "count_dir_ge_45": int(np.sum(p_dir >= 0.45)),
        "count_dir_ge_50": int(np.sum(p_dir >= 0.50)),
        "count_dir_ge_55": int(np.sum(p_dir >= 0.55)),
        "count_dir_ge_60": int(np.sum(p_dir >= 0.60)),
    }
    results["task1"] = task1

    # Task 2: Separate Directional Signal from Neutral Class
    acc_3c = accuracy_score(y_val_arr, val_preds)
    bal_acc_3c = balanced_accuracy_score(y_val_arr, val_preds)
    prec_3c = precision_score(y_val_arr, val_preds, average="macro", zero_division=0)
    rec_3c = recall_score(y_val_arr, val_preds, average="macro", zero_division=0)
    f1_3c = f1_score(y_val_arr, val_preds, average="macro", zero_division=0)
    cm_3c = confusion_matrix(y_val_arr, val_preds, labels=[-1.0, 0.0, 1.0])

    dir_mask = y_val_arr != 0.0
    y_dir_true = y_val_arr[dir_mask]
    n_dir = len(y_dir_true)
    n_long = int(np.sum(y_dir_true == 1.0))
    n_short = int(np.sum(y_dir_true == -1.0))

    pred_binary = np.where(p_long[dir_mask] >= p_short[dir_mask], 1.0, -1.0)
    acc_bin = accuracy_score(y_dir_true, pred_binary)
    bal_acc_bin = balanced_accuracy_score(y_dir_true, pred_binary)
    prec_bin = precision_score(y_dir_true, pred_binary, average="macro", zero_division=0)
    rec_bin = recall_score(y_dir_true, pred_binary, average="macro", zero_division=0)
    f1_bin = f1_score(y_dir_true, pred_binary, average="macro", zero_division=0)
    cm_bin = confusion_matrix(y_dir_true, pred_binary, labels=[-1.0, 1.0])

    pred_3c_on_dir = val_preds[dir_mask]
    neutral_on_dir_cnt = int(np.sum(pred_3c_on_dir == 0.0))
    neutral_on_dir_pct = (neutral_on_dir_cnt / n_dir) * 100.0 if n_dir > 0 else 0.0

    task2 = {
        "three_class_overall": {
            "accuracy": float(acc_3c),
            "balanced_accuracy": float(bal_acc_3c),
            "macro_precision": float(prec_3c),
            "macro_recall": float(rec_3c),
            "macro_f1": float(f1_3c),
            "confusion_matrix": cm_3c.tolist(),
        },
        "binary_directional_diagnostic": {
            "directional_sample_count": n_dir,
            "pct_of_validation": float(n_dir / len(y_val_arr) * 100),
            "true_long_count": n_long,
            "true_short_count": n_short,
            "directional_accuracy": float(acc_bin),
            "balanced_accuracy": float(bal_acc_bin),
            "macro_precision": float(prec_bin),
            "macro_recall": float(rec_bin),
            "macro_f1": float(f1_bin),
            "confusion_matrix": cm_bin.tolist(),
            "neutral_suppression_count": neutral_on_dir_cnt,
            "neutral_suppression_pct": float(neutral_on_dir_pct),
        },
    }
    results["task2"] = task2

    # Task 3: Signal Quality by Confidence
    buckets = [
        (0.40, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 1.01),
    ]
    bucket_stats = compute_confidence_buckets(p_dir, val_preds, y_val_arr, buckets)
    results["task3_directional_buckets"] = bucket_stats

    # Task 4: Class Probability Calibration
    y_val_onehot = np.zeros_like(val_probs)
    for col_idx, cls_val in enumerate(classes):
        y_val_onehot[:, col_idx] = (y_val_arr == cls_val).astype(float)

    brier_short = brier_score_loss(y_val_onehot[:, idx_short], p_short)
    brier_neutral = brier_score_loss(y_val_onehot[:, idx_neutral], p_neutral)
    brier_long = brier_score_loss(y_val_onehot[:, idx_long], p_long)
    brier_overall = float(np.mean([brier_short, brier_neutral, brier_long]))

    calib_bins = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60]
    calib_records = []
    for i in range(len(calib_bins) - 1):
        b_low, b_high = calib_bins[i], calib_bins[i + 1]
        l_mask = (p_long >= b_low) & (p_long < b_high)
        l_cnt = int(np.sum(l_mask))
        l_obs = float(np.mean(y_val_arr[l_mask] == 1.0) * 100.0) if l_cnt > 0 else 0.0
        l_conf = float(np.mean(p_long[l_mask]) * 100.0) if l_cnt > 0 else 0.0

        s_mask = (p_short >= b_low) & (p_short < b_high)
        s_cnt = int(np.sum(s_mask))
        s_obs = float(np.mean(y_val_arr[s_mask] == -1.0) * 100.0) if s_cnt > 0 else 0.0
        s_conf = float(np.mean(p_short[s_mask]) * 100.0) if s_cnt > 0 else 0.0

        calib_records.append(
            {
                "prob_range": f"{b_low:.2f}–{b_high:.2f}",
                "long_count": l_cnt,
                "long_mean_conf_pct": l_conf,
                "long_observed_acc_pct": l_obs,
                "short_count": s_cnt,
                "short_mean_conf_pct": s_conf,
                "short_observed_acc_pct": s_obs,
            }
        )

    results["task4_brier"] = {
        "brier_short": float(brier_short),
        "brier_neutral": float(brier_neutral),
        "brier_long": float(brier_long),
        "brier_overall": brier_overall,
    }
    results["task4_calibration"] = calib_records

    # Task 5: Signal Frequency
    dir_argmax = val_preds != 0.0
    conf_dir_argmax = p_max[dir_argmax]

    max_dconf = float(np.max(conf_dir_argmax)) if len(conf_dir_argmax) > 0 else 0.0
    mean_dconf = float(np.mean(conf_dir_argmax)) if len(conf_dir_argmax) > 0 else 0.0
    task5 = {
        "max_directional_confidence": max_dconf,
        "mean_directional_confidence": mean_dconf,
        "directional_predictions_above_50": int(np.sum((p_max >= 0.50) & dir_argmax)),
        "directional_predictions_above_55": int(np.sum((p_max >= 0.55) & dir_argmax)),
        "directional_predictions_above_60": int(np.sum((p_max >= 0.60) & dir_argmax)),
        "directional_predictions_above_65": int(np.sum((p_max >= 0.65) & dir_argmax)),
        "directional_predictions_above_70": int(np.sum((p_max >= 0.70) & dir_argmax)),
        "neutral_predictions_count": int(np.sum(val_preds == 0.0)),
        "neutral_predictions_pct": float(np.mean(val_preds == 0.0) * 100.0),
        "max_neutral_confidence": float(np.max(p_neutral)),
    }
    results["task5"] = task5

    # Task 6: Feature Information Diagnostic
    importances = rf.model_.feature_importances_
    feat_names = list(X_val.columns)
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

    top30_cats = [categorize_feature(f[0]) for f in feat_imp[:30]]
    cat_counts = pd.Series(top30_cats).value_counts().to_dict()

    dir_rows_mask = p_dir >= 0.50
    diff_records = []
    for name, imp in feat_imp[:15]:
        val_series = X_val[name]
        mean_all = float(val_series.mean())
        std_all = float(val_series.std()) if val_series.std() > 1e-12 else 1.0
        if np.sum(dir_rows_mask) > 0:
            mean_dir = float(val_series[dir_rows_mask].mean())
        else:
            mean_dir = mean_all
        z_shift = (mean_dir - mean_all) / std_all
        diff_records.append(
            {
                "feature": name,
                "importance": float(imp),
                "category": categorize_feature(name),
                "mean_all": mean_all,
                "mean_dir_signals": mean_dir,
                "z_shift": float(z_shift),
            }
        )

    results["task6_top_features"] = [
        {"name": f[0], "importance": float(f[1])} for f in feat_imp[:25]
    ]
    results["task6_category_breakdown"] = cat_counts
    results["task6_feature_shift"] = diff_records

    # Task 7: Temporal Stability
    n_val = len(y_val_arr)
    block_size = n_val // 3
    blocks = [
        ("Block 1 (Early)", 0, block_size),
        ("Block 2 (Mid)", block_size, 2 * block_size),
        ("Block 3 (Late)", 2 * block_size, n_val),
    ]

    block_records = []
    for label, start_idx, end_idx in blocks:
        b_y_true = y_val_arr[start_idx:end_idx]
        b_preds = val_preds[start_idx:end_idx]
        b_p_dir = p_dir[start_idx:end_idx]
        b_ts = val_timestamps.iloc[start_idx:end_idx]

        b_long_cnt = int(np.sum(b_y_true == 1.0))
        b_short_cnt = int(np.sum(b_y_true == -1.0))
        b_neut_cnt = int(np.sum(b_y_true == 0.0))

        b_dir_preds_cnt = int(np.sum(b_preds != 0.0))
        b_dir_ge50_cnt = int(np.sum(b_p_dir >= 0.50))

        b_acc = float(accuracy_score(b_y_true, b_preds))
        b_bal_acc = float(balanced_accuracy_score(b_y_true, b_preds))
        b_f1 = float(f1_score(b_y_true, b_preds, average="macro", zero_division=0))

        block_records.append(
            {
                "block": label,
                "start_time": str(b_ts.iloc[0]),
                "end_time": str(b_ts.iloc[-1]),
                "rows": len(b_y_true),
                "true_long_pct": float(b_long_cnt / len(b_y_true) * 100),
                "true_short_pct": float(b_short_cnt / len(b_y_true) * 100),
                "true_neut_pct": float(b_neut_cnt / len(b_y_true) * 100),
                "mean_dir_conf": float(np.mean(b_p_dir)),
                "max_dir_conf": float(np.max(b_p_dir)),
                "dir_preds_count": b_dir_preds_cnt,
                "dir_ge50_count": b_dir_ge50_cnt,
                "accuracy": b_acc,
                "balanced_accuracy": b_bal_acc,
                "macro_f1": b_f1,
            }
        )
    results["task7_blocks"] = block_records

    # Task 8: Controlled Model Comparison (Logistic Regression)
    t0 = time.time()
    lr = LogisticRegressionBaseline(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    t_fit_lr = time.time() - t0

    lr_probs = lr.predict_proba(X_val)
    lr_preds = lr.predict(X_val)
    lr_p_short = lr_probs[:, classes.index(-1.0)]
    lr_p_long = lr_probs[:, classes.index(1.0)]
    lr_p_dir = np.maximum(lr_p_long, lr_p_short)

    lr_acc = float(accuracy_score(y_val_arr, lr_preds))
    lr_bal_acc = float(balanced_accuracy_score(y_val_arr, lr_preds))
    lr_f1 = float(f1_score(y_val_arr, lr_preds, average="macro", zero_division=0))

    lr_pred_binary = np.where(lr_p_long[dir_mask] >= lr_p_short[dir_mask], 1.0, -1.0)
    lr_bin_acc = float(accuracy_score(y_dir_true, lr_pred_binary))
    lr_bin_f1 = float(f1_score(y_dir_true, lr_pred_binary, average="macro", zero_division=0))

    task8 = {
        "model": "LogisticRegression_balanced",
        "fit_time_seconds": float(t_fit_lr),
        "mean_directional_conf": float(np.mean(lr_p_dir)),
        "max_directional_conf": float(np.max(lr_p_dir)),
        "count_dir_ge_50": int(np.sum(lr_p_dir >= 0.50)),
        "count_dir_ge_60": int(np.sum(lr_p_dir >= 0.60)),
        "count_dir_ge_70": int(np.sum(lr_p_dir >= 0.70)),
        "three_class_accuracy": lr_acc,
        "three_class_balanced_acc": lr_bal_acc,
        "three_class_macro_f1": lr_f1,
        "binary_directional_acc": lr_bin_acc,
        "binary_directional_f1": lr_bin_f1,
    }
    results["task8_lr"] = task8

    # Save to reports directory
    with open(REPORTS_DIR / "phase13_research_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start_total
    print(f"\n[Completed] Phase 13 research execution finished in {total_time:.2f}s.")
    return results


if __name__ == "__main__":
    run_phase13_research()
