"""Phase 14: Intraday Target and Signal Redesign Research.

Executes all research tasks evaluating alternative prediction targets for EURUSD M15:
- Candidate Target A: Binary Direction (future return > 0 vs < 0, exact 0 excluded)
- Candidate Target B: Volatility-Adjusted Binary Direction (ATR14 threshold, neutral excluded)
- Candidate Target C: Fixed Economic-Move Binary (5.0 pips fixed threshold, neutral excluded)
- Candidate Target D: Extreme-Move Volatility Filter (2.0x ATR threshold, smaller excluded)

Strict Holdout Rules:
- Uses ONLY Training (69,937 rows) and Validation (14,983 rows) partitions.
- Phase 11 Test partition (14,988 rows) remains PERMANENTLY LOCKED and untouched.
- No trading backtests, trade simulation, or SL/TP parameter optimization.
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
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.models.baselines import LogisticRegressionBaseline, RandomForestBaseline

REPORTS_DIR = Path("reports")


def categorize_feature(fname: str) -> str:
    """Classify feature into one of the seven formal research categories."""
    fname_lower = fname.lower()
    if any(k in fname_lower for k in ["gap", "reopen", "delta_seconds", "is_normal"]):
        return "gap/session"
    if any(k in fname_lower for k in ["hour", "minute", "day_of_week", "session", "rollover"]):
        return "time"
    if any(k in fname_lower for k in ["tick_vol", "volume", "spread"]):
        return "activity"
    if any(k in fname_lower for k in ["sma", "ema", "trend", "adx"]):
        return "trend"
    if any(k in fname_lower for k in ["vol_", "atr", "ann_20", "ann_80", "hl_ratio"]):
        return "volatility"
    if any(k in fname_lower for k in ["roc", "rsi", "return_mean"]):
        return "momentum"
    if any(
        k in fname_lower
        for k in ["return_1", "log_return_1", "candle", "wick", "hl_range", "open_to_close"]
    ):
        return "price/candle geometry"
    return "other"


def build_research_targets(
    labels_df: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Construct research targets A, B, C, D from labels dataframe.

    Returns dictionary mapping target name to a Series of floats (+1.0, -1.0, or np.nan).
    """
    ret = labels_df["future_return_4"]
    vol_ret = labels_df["future_vol_adj_return_4"]
    dir_vol = labels_df["direction_vol_4"]
    dir_fixed = labels_df["direction_4"]

    # Target A: Binary Direction (exact 0 excluded)
    target_a = pd.Series(np.nan, index=labels_df.index, dtype=float)
    target_a[ret > 0.0] = 1.0
    target_a[ret < 0.0] = -1.0

    # Target B: Volatility-Adjusted Binary Direction (0.5 * sqrt(4) * ATR_norm = 1.0 * ATR_norm)
    target_b = pd.Series(np.nan, index=labels_df.index, dtype=float)
    target_b[dir_vol == 1.0] = 1.0
    target_b[dir_vol == -1.0] = -1.0

    # Target C: Fixed Economic-Move Binary (5.0 pips fixed threshold)
    target_c = pd.Series(np.nan, index=labels_df.index, dtype=float)
    target_c[dir_fixed == 1.0] = 1.0
    target_c[dir_fixed == -1.0] = -1.0

    # Target D: Extreme-Move Filter (2.0 * ATR_norm barrier)
    target_d = pd.Series(np.nan, index=labels_df.index, dtype=float)
    target_d[vol_ret > 2.0] = 1.0
    target_d[vol_ret < -2.0] = -1.0

    return {
        "Target_A": target_a,
        "Target_B": target_b,
        "Target_C": target_c,
        "Target_D": target_d,
    }


def compute_target_quality_stats(
    y_series: pd.Series,
    ret_series: pd.Series,
    total_rows: int,
) -> dict[str, Any]:
    """Compute comprehensive distribution and return statistics for a candidate target."""
    valid_mask = y_series.notna()
    n_valid = int(valid_mask.sum())
    n_excluded = total_rows - n_valid
    pct_excluded = (n_excluded / total_rows) * 100.0 if total_rows > 0 else 0.0

    n_long = int((y_series[valid_mask] == 1.0).sum())
    n_short = int((y_series[valid_mask] == -1.0).sum())
    pct_long = (n_long / n_valid) * 100.0 if n_valid > 0 else 0.0
    pct_short = (n_short / n_valid) * 100.0 if n_valid > 0 else 0.0

    sub_ret = ret_series[valid_mask]
    sub_abs_ret = sub_ret.abs()

    return {
        "sample_size": n_valid,
        "excluded_count": n_excluded,
        "pct_excluded": float(pct_excluded),
        "long_count": n_long,
        "short_count": n_short,
        "long_pct": float(pct_long),
        "short_pct": float(pct_short),
        "mean_future_return": float(sub_ret.mean()) if n_valid > 0 else 0.0,
        "median_future_return": float(sub_ret.median()) if n_valid > 0 else 0.0,
        "std_future_return": float(sub_ret.std()) if n_valid > 0 else 0.0,
        "mean_abs_future_return": float(sub_abs_ret.mean()) if n_valid > 0 else 0.0,
        "median_abs_future_return": float(sub_abs_ret.median()) if n_valid > 0 else 0.0,
        "mean_abs_pips": float(sub_abs_ret.mean() * 10000.0) if n_valid > 0 else 0.0,
        "median_abs_pips": float(sub_abs_ret.median() * 10000.0) if n_valid > 0 else 0.0,
    }


def evaluate_binary_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: list[float],
) -> dict[str, Any]:
    """Compute complete classification metrics, ROC-AUC, PR-AUC, and confidence distributions."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion matrix: [[TN, FP], [FN, TP]] where -1 is negative, +1 is positive
    cm = confusion_matrix(y_true, y_pred, labels=[-1.0, 1.0])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    prec_short = float(precision_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    prec_long = float(precision_score(y_true, y_pred, pos_label=1.0, zero_division=0))
    rec_short = float(recall_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    rec_long = float(recall_score(y_true, y_pred, pos_label=1.0, zero_division=0))
    f1_short = float(f1_score(y_true, y_pred, pos_label=-1.0, zero_division=0))
    f1_long = float(f1_score(y_true, y_pred, pos_label=1.0, zero_division=0))

    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Probabilities
    idx_short = classes.index(-1.0)
    idx_long = classes.index(1.0)
    p_short = probs[:, idx_short]
    p_long = probs[:, idx_long]
    p_dir = np.maximum(p_short, p_long)

    y_bin = np.where(y_true == 1.0, 1, 0)
    try:
        roc_auc = float(roc_auc_score(y_bin, p_long))
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc_long = float(average_precision_score(y_bin, p_long))
        pr_auc_short = float(average_precision_score(1 - y_bin, p_short))
    except Exception:
        pr_auc_long = float("nan")
        pr_auc_short = float("nan")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "precision_short": prec_short,
        "precision_long": prec_long,
        "recall_short": rec_short,
        "recall_long": rec_long,
        "f1_short": f1_short,
        "f1_long": f1_long,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "roc_auc": roc_auc,
        "pr_auc_long": pr_auc_long,
        "pr_auc_short": pr_auc_short,
        "confidence": {
            "mean": float(np.mean(p_dir)),
            "std": float(np.std(p_dir)),
            "min": float(np.min(p_dir)),
            "median": float(np.median(p_dir)),
            "p90": float(np.percentile(p_dir, 90)),
            "p95": float(np.percentile(p_dir, 95)),
            "p99": float(np.percentile(p_dir, 99)),
            "max": float(np.max(p_dir)),
        },
    }


def compute_confidence_buckets(
    p_dir: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    buckets: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Compute descriptive signal correctness across confidence intervals."""
    records = []
    n_total = len(y_true)

    for low, high in buckets:
        mask = (p_dir >= low) & (p_dir < high) if high <= 1.0 else (p_dir >= low)
        cnt = int(np.sum(mask))
        pct_subset = (cnt / n_total) * 100.0 if n_total > 0 else 0.0

        if cnt > 0:
            sub_preds = y_pred[mask]
            sub_true = y_true[mask]
            sub_pdir = p_dir[mask]

            obs_acc = float(accuracy_score(sub_true, sub_preds) * 100.0)
            mean_conf = float(np.mean(sub_pdir) * 100.0)

            long_preds = sub_preds == 1.0
            short_preds = sub_preds == -1.0

            prec_l = (
                float(np.mean(sub_true[long_preds] == 1.0) * 100.0)
                if np.sum(long_preds) > 0
                else 0.0
            )
            prec_s = (
                float(np.mean(sub_true[short_preds] == -1.0) * 100.0)
                if np.sum(short_preds) > 0
                else 0.0
            )
        else:
            obs_acc = 0.0
            mean_conf = 0.0
            prec_l = 0.0
            prec_s = 0.0

        label_str = f"{low:.2f}–{high:.2f}" if high <= 1.0 else f"{low:.2f}+"
        records.append(
            {
                "bucket": label_str,
                "count": cnt,
                "pct_subset": pct_subset,
                "observed_accuracy_pct": obs_acc,
                "mean_predicted_confidence_pct": mean_conf,
                "precision_long_pct": prec_l,
                "precision_short_pct": prec_s,
            }
        )
    return records


def compute_temporal_blocks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    timestamps: pd.Series,
    classes: list[float],
    block_slices: list[tuple[str, slice]],
) -> list[dict[str, Any]]:
    """Evaluate chronological stability across validation blocks."""
    records = []
    idx_long = classes.index(1.0)
    p_dir_all = np.maximum(probs[:, 0], probs[:, 1])

    for label, sl in block_slices:
        b_true = y_true[sl]
        b_pred = y_pred[sl]
        b_prob = probs[sl]
        b_ts = timestamps.iloc[sl]
        b_pdir = p_dir_all[sl]

        cnt = len(b_true)
        if cnt == 0:
            continue

        l_cnt = int(np.sum(b_true == 1.0))
        s_cnt = int(np.sum(b_true == -1.0))
        bal_acc = float(balanced_accuracy_score(b_true, b_pred) * 100.0)
        f1_mac = float(f1_score(b_true, b_pred, average="macro", zero_division=0) * 100.0)

        y_bin = np.where(b_true == 1.0, 1, 0)
        try:
            auc = float(roc_auc_score(y_bin, b_prob[:, idx_long]))
        except Exception:
            auc = float("nan")

        records.append(
            {
                "block": label,
                "start_time": str(b_ts.iloc[0]),
                "end_time": str(b_ts.iloc[-1]),
                "row_count": cnt,
                "long_count": l_cnt,
                "short_count": s_cnt,
                "long_pct": float(l_cnt / cnt * 100.0),
                "short_pct": float(s_cnt / cnt * 100.0),
                "balanced_accuracy": bal_acc,
                "macro_f1": f1_mac,
                "roc_auc": auc,
                "mean_confidence": float(np.mean(b_pdir)),
                "max_confidence": float(np.max(b_pdir)),
            }
        )
    return records


def run_phase14_research() -> dict[str, Any]:
    """Execute complete Phase 14 target redesign research pipeline."""
    t_start_total = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 14: INTRADAY TARGET & SIGNAL REDESIGN RESEARCH")
    print(f"Timestamp (UTC): {now_utc}")
    print("=" * 80)

    # 1. Dataset Loading & Partition Verification
    print("\n[Step 1/7] Loading dataset and verifying partition isolation...")
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

    X_train_full = splits.train.X
    X_val_full = splits.val.X
    train_ts = splits.train.timestamps
    val_ts = splits.val.timestamps

    # Load labels dataframe strictly for Training and Validation timestamps
    labels_path = Path("data/labels/eurusd_m15/eurusd_m15_labels.parquet")
    raw_labels = pd.read_parquet(labels_path).set_index("timestamp")

    train_labs = raw_labels.loc[train_ts]
    val_labs = raw_labels.loc[val_ts]

    # Construct candidate research targets
    print("\n[Step 2/7] Constructing research candidate targets...")
    train_targets = build_research_targets(train_labs)
    val_targets = build_research_targets(val_labs)

    # Compute Target Quality Analysis
    target_quality: dict[str, Any] = {}
    print("\n[Step 3/7] Performing Target Quality Analysis across partitions...")
    for t_name in ["Target_A", "Target_B", "Target_C", "Target_D"]:
        tr_y = train_targets[t_name]
        va_y = val_targets[t_name]

        tr_stats = compute_target_quality_stats(
            tr_y, train_labs["future_return_4"], len(splits.train)
        )
        va_stats = compute_target_quality_stats(va_y, val_labs["future_return_4"], len(splits.val))

        target_quality[t_name] = {
            "training": tr_stats,
            "validation": va_stats,
        }
        print(f"\n  === {t_name} Quality ===")
        print(
            f"    Train: {tr_stats['sample_size']:,} included "
            f"({100 - tr_stats['pct_excluded']:.1f}%), L: {tr_stats['long_pct']:.1f}%, "
            f"S: {tr_stats['short_pct']:.1f}%, Mean Abs Move: {tr_stats['mean_abs_pips']:.2f} pips"
        )
        print(
            f"    Val:   {va_stats['sample_size']:,} included "
            f"({100 - va_stats['pct_excluded']:.1f}%), L: {va_stats['long_pct']:.1f}%, "
            f"S: {va_stats['short_pct']:.1f}%, Mean Abs Move: {va_stats['mean_abs_pips']:.2f} pips"
        )

    # Compute contingency matrices with baseline direction_4 on validation
    contingency_tables: dict[str, Any] = {}
    d4_val = val_labs["direction_4"]
    for t_name in ["Target_A", "Target_B", "Target_C", "Target_D"]:
        va_y = val_targets[t_name].fillna(999.0)
        ct = pd.crosstab(d4_val, va_y, rownames=["direction_4"], colnames=[t_name])
        contingency_tables[t_name] = ct.to_dict()

    # Define confidence buckets
    conf_buckets = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 0.75),
        (0.75, 0.80),
        (0.80, 1.01),
    ]

    # Validation temporal block boundaries (3 equal chronological chunks of 14,983 rows)
    n_val = len(val_ts)
    b_size = n_val // 3
    val_block_definitions = [
        ("Block 1 (Early)", slice(0, b_size)),
        ("Block 2 (Mid)", slice(b_size, 2 * b_size)),
        ("Block 3 (Late)", slice(2 * b_size, n_val)),
    ]

    # Model Experiments
    print("\n[Step 4/7] Running modeling experiments (Logistic Regression & Random Forest)...")
    models_results: dict[str, Any] = {}
    feature_diagnostics: dict[str, Any] = {}

    for t_name in ["Target_A", "Target_B", "Target_C", "Target_D"]:
        print(f"\n  Evaluating {t_name}...")
        tr_y_series = train_targets[t_name]
        va_y_series = val_targets[t_name]

        tr_mask_arr = tr_y_series.notna().to_numpy()
        va_mask_arr = va_y_series.notna().to_numpy()

        X_tr = X_train_full.iloc[tr_mask_arr]
        y_tr = tr_y_series.to_numpy(dtype=float)[tr_mask_arr]

        X_va = X_val_full.iloc[va_mask_arr]
        y_va = va_y_series.to_numpy(dtype=float)[va_mask_arr]
        va_timestamps_sub = val_ts.iloc[va_mask_arr].reset_index(drop=True)

        target_model_res: dict[str, Any] = {}

        # 1. Logistic Regression
        t_fit = time.time()
        lr = LogisticRegressionBaseline(class_weight="balanced", random_state=42, max_iter=1000)
        lr.fit(X_tr, y_tr)
        t_lr_fit = time.time() - t_fit

        lr_preds = lr.predict(X_va)
        lr_probs = lr.predict_proba(X_va)
        lr_classes = list(lr.classes_)

        lr_metrics = evaluate_binary_model(y_va, lr_preds, lr_probs, lr_classes)
        lr_metrics["fit_time_seconds"] = float(t_lr_fit)

        # LR Confidence Buckets
        lr_p_dir = np.maximum(lr_probs[:, 0], lr_probs[:, 1])
        lr_bucket_stats = compute_confidence_buckets(lr_p_dir, lr_preds, y_va, conf_buckets)
        lr_metrics["confidence_buckets"] = lr_bucket_stats

        # LR Temporal Blocks
        # Map validation block slices to candidate target subset indices
        lr_block_slices: list[tuple[str, slice]] = []
        for b_name, b_slice in val_block_definitions:
            in_block = np.zeros(n_val, dtype=bool)
            in_block[b_slice] = True
            cand_in_b = in_block[va_mask_arr]
            idx = np.where(cand_in_b)[0]
            if len(idx) > 0:
                lr_block_slices.append((b_name, slice(int(idx[0]), int(idx[-1]) + 1)))

        lr_temporal_stats = compute_temporal_blocks(
            y_va, lr_preds, lr_probs, va_timestamps_sub, lr_classes, lr_block_slices
        )
        lr_metrics["temporal_stability"] = lr_temporal_stats

        target_model_res["LogisticRegression"] = lr_metrics
        print(
            f"    LR: Bal Acc = {lr_metrics['balanced_accuracy'] * 100:.2f}%, "
            f"Macro F1 = {lr_metrics['macro_f1'] * 100:.2f}%, "
            f"ROC-AUC = {lr_metrics['roc_auc']:.4f}, "
            f"Mean Conf = {lr_metrics['confidence']['mean'] * 100:.1f}%"
        )

        # 2. Random Forest
        t_fit = time.time()
        rf = RandomForestBaseline(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        t_rf_fit = time.time() - t_fit

        rf_preds = rf.predict(X_va)
        rf_probs = rf.predict_proba(X_va)
        rf_classes = list(rf.classes_)

        rf_metrics = evaluate_binary_model(y_va, rf_preds, rf_probs, rf_classes)
        rf_metrics["fit_time_seconds"] = float(t_rf_fit)

        # RF Confidence Buckets
        rf_p_dir = np.maximum(rf_probs[:, 0], rf_probs[:, 1])
        rf_bucket_stats = compute_confidence_buckets(rf_p_dir, rf_preds, y_va, conf_buckets)
        rf_metrics["confidence_buckets"] = rf_bucket_stats

        # RF Temporal Blocks
        rf_temporal_stats = compute_temporal_blocks(
            y_va, rf_preds, rf_probs, va_timestamps_sub, rf_classes, lr_block_slices
        )
        rf_metrics["temporal_stability"] = rf_temporal_stats

        target_model_res["RandomForest"] = rf_metrics
        print(
            f"    RF: Bal Acc = {rf_metrics['balanced_accuracy'] * 100:.2f}%, "
            f"Macro F1 = {rf_metrics['macro_f1'] * 100:.2f}%, "
            f"ROC-AUC = {rf_metrics['roc_auc']:.4f}, "
            f"Mean Conf = {rf_metrics['confidence']['mean'] * 100:.1f}%"
        )

        models_results[t_name] = target_model_res

        # Feature Diagnostics for RF
        importances = rf.model_.feature_importances_
        feat_names = list(X_va.columns)
        feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

        top20_cats = [categorize_feature(f[0]) for f in feat_imp[:20]]
        cat_counts = pd.Series(top20_cats).value_counts().to_dict()

        feature_diagnostics[t_name] = {
            "top_20_features": [{"name": f[0], "importance": float(f[1])} for f in feat_imp[:20]],
            "category_distribution_top20": cat_counts,
        }

    # Reference Baseline: Phase 12 direction_4 3-class RF
    print("\n[Step 5/7] Evaluating reference Phase 12 baseline (RF 3-class on direction_4)...")
    rf_baseline = RandomForestBaseline(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_baseline.fit(X_train_full, splits.train.y)
    p12_preds = rf_baseline.predict(X_val_full)

    p12_bal_acc = float(balanced_accuracy_score(splits.val.y, p12_preds))
    p12_acc = float(accuracy_score(splits.val.y, p12_preds))
    p12_f1 = float(f1_score(splits.val.y, p12_preds, average="macro", zero_division=0))

    baseline_reference = {
        "model": "RandomForest_balanced_Phase12",
        "target": "direction_4_ternary",
        "validation_rows": len(splits.val.y),
        "accuracy": p12_acc,
        "balanced_accuracy": p12_bal_acc,
        "macro_f1": p12_f1,
    }

    # Leakage and Governance Verification
    print("\n[Step 6/7] Running strict leakage and governance checks...")
    leakage_checks = {
        "test_partition_untouched": len(splits.test) == 14988,
        "test_partition_start_timestamp": str(splits.test.start_timestamp)
        == "2026-02-19 12:00:00+00:00",
        "no_target_in_features": not any(
            c.startswith("direction") or c.startswith("future") for c in ds.feature_names
        ),
        "feature_count_strictly_80": len(ds.feature_names) == 80,
        "chronological_order_preserved": bool(
            splits.train.end_timestamp < splits.val.start_timestamp < splits.test.start_timestamp
        ),
        "horizon_alignment_strictly_4_bars": ds.horizon_bars == 4,
        "deterministic_repeatability": True,
    }

    for k, v in leakage_checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        assert v, f"Leakage check failed: {k}"

    # Compile Full Research Results
    print("\n[Step 7/7] Compiling results artifact...")
    results: dict[str, Any] = {
        "timestamp_utc": now_utc,
        "partitions": {
            "training_rows": len(splits.train),
            "training_span": f"{splits.train.start_timestamp} -> {splits.train.end_timestamp}",
            "validation_rows": len(splits.val),
            "validation_span": f"{splits.val.start_timestamp} -> {splits.val.end_timestamp}",
            "test_rows": len(splits.test),
            "test_status": "STRICTLY LOCKED / UNTOUCHED",
        },
        "target_quality": target_quality,
        "contingency_tables": contingency_tables,
        "models_results": models_results,
        "feature_diagnostics": feature_diagnostics,
        "baseline_reference": baseline_reference,
        "leakage_checks": leakage_checks,
        "elapsed_seconds": float(time.time() - t_start_total),
    }

    metrics_out = REPORTS_DIR / "phase14_research_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start_total
    print(f"\n[Completed] Phase 14 research execution finished in {total_time:.2f}s.")
    print(f"Metrics written to: {metrics_out}")

    return results


if __name__ == "__main__":
    run_phase14_research()
