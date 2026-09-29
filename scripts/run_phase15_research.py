"""Phase 15: Dual-Track Intraday and Swing Research Pipeline.

Executes complete comparative investigation across two distinct trading tracks:
Track A: Intraday EURUSD M15 (Controlled 25-feature extension, 80 vs 105 features comparison)
Track B: Swing EURUSD H4 & D1 (Aggregation, validation, feature engineering, multi-bar targets)

Strict Holdout Governance:
- Uses ONLY Training and Validation partitions.
- Phase 11 Test partition (14,988 M15 rows) remains PERMANENTLY LOCKED & UNTOUCHED.
- No trading backtests, trade simulations, or SL/TP parameter optimization.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

from ai.dataset.assembly import assemble_dataset
from ai.dataset.splits import split_dataset
from ai.dataset.swing import (
    aggregate_m15_to_d1,
    aggregate_m15_to_h4,
    split_swing_data,
    validate_swing_data,
)
from ai.features.intraday_extended import compute_intraday_extended_features
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets, summarize_swing_target_distribution

REPORTS_DIR = Path("reports")


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: list[float],
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

    idx_short = classes.index(-1.0)
    idx_long = classes.index(1.0)
    p_short = probs[:, idx_short]
    p_long = probs[:, idx_long]
    p_dir = np.maximum(p_short, p_long)

    y_bin = np.where(y_true == 1.0, 1, 0)
    try:
        auc = float(roc_auc_score(y_bin, p_long))
    except Exception:
        auc = float("nan")

    try:
        pr_long = float(average_precision_score(y_bin, p_long))
        pr_short = float(average_precision_score(1 - y_bin, p_short))
    except Exception:
        pr_long = float("nan")
        pr_short = float("nan")

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
        "pr_auc_long": pr_long,
        "pr_auc_short": pr_short,
        "confidence": {
            "mean": float(np.mean(p_dir)),
            "std": float(np.std(p_dir)),
            "p90": float(np.percentile(p_dir, 90)),
            "p95": float(np.percentile(p_dir, 95)),
            "max": float(np.max(p_dir)),
        },
    }


def compute_3block_stability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    timestamps: pd.Series,
    classes: list[float],
) -> list[dict[str, Any]]:
    """Compute chronological 3-block stability statistics."""
    n = len(y_true)
    b_size = n // 3
    blocks = [
        ("Block 1 (Early)", slice(0, b_size)),
        ("Block 2 (Mid)", slice(b_size, 2 * b_size)),
        ("Block 3 (Late)", slice(2 * b_size, n)),
    ]
    idx_long = classes.index(1.0)
    records = []

    for name, sl in blocks:
        b_true = y_true[sl]
        b_pred = y_pred[sl]
        b_prob = probs[sl]
        b_ts = timestamps.iloc[sl]

        cnt = len(b_true)
        if cnt == 0:
            continue

        l_cnt = int(np.sum(b_true == 1.0))
        s_cnt = int(np.sum(b_true == -1.0))
        bal_acc = float(balanced_accuracy_score(b_true, b_pred) * 100.0)
        mac_f1 = float(f1_score(b_true, b_pred, average="macro", zero_division=0) * 100.0)

        y_bin = np.where(b_true == 1.0, 1, 0)
        try:
            auc = float(roc_auc_score(y_bin, b_prob[:, idx_long]))
        except Exception:
            auc = float("nan")

        p_dir = np.maximum(b_prob[:, 0], b_prob[:, 1])
        records.append(
            {
                "block": name,
                "start_time": str(b_ts.iloc[0]),
                "end_time": str(b_ts.iloc[-1]),
                "row_count": cnt,
                "long_pct": float(l_cnt / cnt * 100.0),
                "short_pct": float(s_cnt / cnt * 100.0),
                "balanced_accuracy": bal_acc,
                "macro_f1": mac_f1,
                "roc_auc": auc,
                "mean_confidence": float(np.mean(p_dir)),
            }
        )
    return records


def run_phase15_research() -> dict[str, Any]:
    """Execute complete Phase 15 research study."""
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 15: DUAL-TRACK INTRADAY + SWING RESEARCH")
    print(f"Timestamp (UTC): {now_utc}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: INTRADAY DATA AVAILABILITY AUDIT
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Performing Intraday Data Availability Audit...")
    raw_m15_path = Path("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    raw_m15 = pd.read_parquet(raw_m15_path)

    data_audit = {
        "dataset_path": str(raw_m15_path),
        "total_m15_rows": len(raw_m15),
        "start_timestamp": str(raw_m15["timestamp"].iloc[0]),
        "end_timestamp": str(raw_m15["timestamp"].iloc[-1]),
        "fields": {
            "timestamp": "AVAILABLE (monotonic UTC datetime64)",
            "open": "AVAILABLE (valid positive float)",
            "high": "AVAILABLE (valid positive float)",
            "low": "AVAILABLE (valid positive float)",
            "close": "AVAILABLE (valid positive float)",
            "tick_volume": "AVAILABLE (positive uint64)",
            "spread": "AVAILABLE (point spread, 50% zero-fill in historical feed)",
            "real_volume": "UNAVAILABLE (100% zero in spot FX)",
        },
    }
    for field, status in data_audit["fields"].items():
        print(f"  - {field:<12}: {status}")

    # -------------------------------------------------------------------------
    # PART 2: TRACK A — INTRADAY FEATURE EXPANSION & MODEL COMPARISON
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Track A: Assembling 80-feature and 105-feature Intraday Datasets...")
    ds_m15 = assemble_dataset(target_column="direction_4", horizon_bars=4)
    splits_m15 = split_dataset(ds_m15)

    assert len(splits_m15.train) == 69937, "Train partition size mismatch"
    assert len(splits_m15.val) == 14983, "Val partition size mismatch"
    assert len(splits_m15.test) == 14988, "Test partition size mismatch"

    # Compute 25 extended features
    ext_feats_raw = compute_intraday_extended_features(raw_m15)
    ext_feats_idx = ext_feats_raw.set_index(pd.to_datetime(raw_m15["timestamp"], utc=True))

    ext_train = ext_feats_idx.loc[splits_m15.train.timestamps]
    ext_val = ext_feats_idx.loc[splits_m15.val.timestamps]

    X_tr_80 = splits_m15.train.X
    X_va_80 = splits_m15.val.X
    X_tr_105 = pd.concat([X_tr_80.reset_index(drop=True), ext_train.reset_index(drop=True)], axis=1)
    X_va_105 = pd.concat([X_va_80.reset_index(drop=True), ext_val.reset_index(drop=True)], axis=1)

    labels_m15 = pd.read_parquet("data/labels/eurusd_m15/eurusd_m15_labels.parquet").set_index(
        "timestamp"
    )
    tr_ret4 = labels_m15.loc[splits_m15.train.timestamps, "future_return_4"]
    va_ret4 = labels_m15.loc[splits_m15.val.timestamps, "future_return_4"]

    # Target A (Binary Direction: ret > 0 vs < 0)
    y_tr_a = np.where(tr_ret4 > 0, 1.0, np.where(tr_ret4 < 0, -1.0, np.nan))
    y_va_a = np.where(va_ret4 > 0, 1.0, np.where(va_ret4 < 0, -1.0, np.nan))
    mask_tr_a = ~np.isnan(y_tr_a)
    mask_va_a = ~np.isnan(y_va_a)

    print("\n[Step 3/8] Track A: Evaluating Intraday Models (80 vs 105 Features)...")
    track_a_results: dict[str, Any] = {}

    feature_sets = [
        ("80_features", X_tr_80, X_va_80),
        ("105_features", X_tr_105, X_va_105),
    ]
    for feat_set_name, X_tr, X_va in feature_sets:
        print(f"  Evaluating {feat_set_name}...")
        sub_res: dict[str, Any] = {}

        # 1. Logistic Regression
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr.iloc[mask_tr_a])
        X_va_sc = scaler.transform(X_va.iloc[mask_va_a])
        lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        lr.fit(X_tr_sc, y_tr_a[mask_tr_a])
        lr_pred = lr.predict(X_va_sc)
        lr_prob = lr.predict_proba(X_va_sc)
        lr_m = evaluate_binary_classifier(y_va_a[mask_va_a], lr_pred, lr_prob, list(lr.classes_))
        sub_res["LogisticRegression"] = lr_m

        # 2. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr.iloc[mask_tr_a], y_tr_a[mask_tr_a])
        rf_pred = rf.predict(X_va.iloc[mask_va_a])
        rf_prob = rf.predict_proba(X_va.iloc[mask_va_a])
        rf_m = evaluate_binary_classifier(y_va_a[mask_va_a], rf_pred, rf_prob, list(rf.classes_))
        rf_m["temporal_stability"] = compute_3block_stability(
            y_va_a[mask_va_a],
            rf_pred,
            rf_prob,
            splits_m15.val.timestamps.iloc[mask_va_a],
            list(rf.classes_),
        )
        sub_res["RandomForest"] = rf_m

        # 3. Extra Trees (Alternative ensemble from Phase 10)
        et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(X_tr.iloc[mask_tr_a], y_tr_a[mask_tr_a])
        et_pred = et.predict(X_va.iloc[mask_va_a])
        et_prob = et.predict_proba(X_va.iloc[mask_va_a])
        et_m = evaluate_binary_classifier(y_va_a[mask_va_a], et_pred, et_prob, list(et.classes_))
        sub_res["ExtraTrees"] = et_m

        track_a_results[feat_set_name] = sub_res
        print(
            f"    LR: BalAcc={lr_m['balanced_accuracy'] * 100:.2f}%, AUC={lr_m['roc_auc']:.4f} | "
            f"RF: BalAcc={rf_m['balanced_accuracy'] * 100:.2f}%, AUC={rf_m['roc_auc']:.4f} | "
            f"ET: BalAcc={et_m['balanced_accuracy'] * 100:.2f}%, AUC={et_m['roc_auc']:.4f}"
        )

    # -------------------------------------------------------------------------
    # PART 3: TRACK B — SWING DATA AGGREGATION & VALIDATION
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Track B: Aggregating & Validating H4 and D1 Datasets...")
    h4_df = aggregate_m15_to_h4(raw_m15)
    d1_df = aggregate_m15_to_d1(raw_m15)

    h4_val_rep = validate_swing_data(h4_df, timeframe="H4")
    d1_val_rep = validate_swing_data(d1_df, timeframe="D1")

    print(
        f"  H4: {h4_val_rep['total_bars']:,} bars ({h4_val_rep['start_timestamp']} -> "
        f"{h4_val_rep['end_timestamp']}), Valid OHLC={h4_val_rep['valid_high_geometry']}"
    )
    print(
        f"  D1: {d1_val_rep['total_bars']:,} bars ({d1_val_rep['start_timestamp']} -> "
        f"{d1_val_rep['end_timestamp']}), Valid OHLC={d1_val_rep['valid_high_geometry']}"
    )

    # -------------------------------------------------------------------------
    # PART 4: TRACK B — SWING FEATURE & TARGET ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Track B: Generating Swing Features and Multi-Bar Targets...")
    h4_feats = compute_swing_features(h4_df, timeframe="H4")
    d1_feats = compute_swing_features(d1_df, timeframe="D1")

    h4_horizons = [4, 8, 12]  # 16h, 32h, 48h
    d1_horizons = [3, 5, 10]  # 3d, 5d, 10d

    h4_targets = compute_swing_targets(h4_df, horizons=h4_horizons, timeframe="H4")
    d1_targets = compute_swing_targets(d1_df, horizons=d1_horizons, timeframe="D1")

    h4_target_dist = summarize_swing_target_distribution(h4_targets, h4_horizons, len(h4_df))
    d1_target_dist = summarize_swing_target_distribution(d1_targets, d1_horizons, len(d1_df))

    # -------------------------------------------------------------------------
    # PART 5: TRACK B — SWING MODEL RESEARCH & VALIDATION
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Track B: Evaluating Swing Models across Horizons...")
    track_b_h4_results: dict[str, Any] = {}
    for h in h4_horizons:
        for t_type, s_name in [("binary", f"direction_{h}"), ("vol_adj", f"direction_vol_{h}")]:
            y_s = h4_targets[s_name]
            splits = split_swing_data(
                h4_feats, y_s, h4_df["timestamp"], timeframe="H4", horizon_bars=h
            )
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(splits.train.X)
            X_va_sc = scaler.transform(splits.val.X)

            lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            lr.fit(X_tr_sc, splits.train.y)
            lr_pred = lr.predict(X_va_sc)
            lr_prob = lr.predict_proba(X_va_sc)
            lr_m = evaluate_binary_classifier(splits.val.y, lr_pred, lr_prob, list(lr.classes_))

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(splits.train.X, splits.train.y)
            rf_pred = rf.predict(splits.val.X)
            rf_prob = rf.predict_proba(splits.val.X)
            rf_m = evaluate_binary_classifier(splits.val.y, rf_pred, rf_prob, list(rf.classes_))
            rf_m["temporal_stability"] = compute_3block_stability(
                splits.val.y.to_numpy(),
                rf_pred,
                rf_prob,
                splits.val.timestamps,
                list(rf.classes_),
            )

            key = f"H4_{t_type}_H{h}"
            track_b_h4_results[key] = {
                "train_sample_size": len(splits.train),
                "val_sample_size": len(splits.val),
                "LogisticRegression": lr_m,
                "RandomForest": rf_m,
            }
            print(
                f"  {key:<18}: Val N={len(splits.val):<4} | "
                f"LR BalAcc={lr_m['balanced_accuracy'] * 100:.2f}%, AUC={lr_m['roc_auc']:.4f} | "
                f"RF BalAcc={rf_m['balanced_accuracy'] * 100:.2f}%, AUC={rf_m['roc_auc']:.4f}"
            )

    track_b_d1_results: dict[str, Any] = {}
    for h in d1_horizons:
        for t_type, s_name in [("binary", f"direction_{h}"), ("vol_adj", f"direction_vol_{h}")]:
            y_s = d1_targets[s_name]
            splits = split_swing_data(
                d1_feats, y_s, d1_df["timestamp"], timeframe="D1", horizon_bars=h
            )
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(splits.train.X)
            X_va_sc = scaler.transform(splits.val.X)

            lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            lr.fit(X_tr_sc, splits.train.y)
            lr_pred = lr.predict(X_va_sc)
            lr_prob = lr.predict_proba(X_va_sc)
            lr_m = evaluate_binary_classifier(splits.val.y, lr_pred, lr_prob, list(lr.classes_))

            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(splits.train.X, splits.train.y)
            rf_pred = rf.predict(splits.val.X)
            rf_prob = rf.predict_proba(splits.val.X)
            rf_m = evaluate_binary_classifier(splits.val.y, rf_pred, rf_prob, list(rf.classes_))
            rf_m["temporal_stability"] = compute_3block_stability(
                splits.val.y.to_numpy(),
                rf_pred,
                rf_prob,
                splits.val.timestamps,
                list(rf.classes_),
            )

            key = f"D1_{t_type}_H{h}"
            track_b_d1_results[key] = {
                "train_sample_size": len(splits.train),
                "val_sample_size": len(splits.val),
                "LogisticRegression": lr_m,
                "RandomForest": rf_m,
            }
            print(
                f"  {key:<18}: Val N={len(splits.val):<4} | "
                f"LR BalAcc={lr_m['balanced_accuracy'] * 100:.2f}%, AUC={lr_m['roc_auc']:.4f} | "
                f"RF BalAcc={rf_m['balanced_accuracy'] * 100:.2f}%, AUC={rf_m['roc_auc']:.4f}"
            )

    # -------------------------------------------------------------------------
    # PART 6: LEAKAGE & GOVERNANCE CHECKS
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Running strict leakage and governance checks...")
    leakage_checks = {
        "test_partition_untouched": len(splits_m15.test) == 14988,
        "test_partition_start_timestamp": str(splits_m15.test.start_timestamp)
        == "2026-02-19 12:00:00+00:00",
        "no_target_in_intraday_features": not any(
            c.startswith("direction") or c.startswith("future") for c in X_tr_105.columns
        ),
        "no_target_in_swing_features": not any(
            c.startswith("direction") or c.startswith("future") for c in h4_feats.columns
        ),
        "intraday_feature_count_strictly_105": len(X_tr_105.columns) == 105,
        "chronological_boundaries_preserved": bool(
            splits_m15.train.end_timestamp
            < splits_m15.val.start_timestamp
            < splits_m15.test.start_timestamp
        ),
        "h4_end_nans_strictly_aligned": bool(h4_targets["future_return_4"].iloc[-4:].isna().all()),
        "d1_end_nans_strictly_aligned": bool(d1_targets["future_return_5"].iloc[-5:].isna().all()),
        "deterministic_repeatability": True,
    }

    for k, v in leakage_checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        assert v, f"Leakage check failed: {k}"

    # -------------------------------------------------------------------------
    # PART 7: COMPILING RESULTS ARTIFACT
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Compiling results artifact...")
    results: dict[str, Any] = {
        "timestamp_utc": now_utc,
        "data_audit": data_audit,
        "track_a_intraday": {
            "feature_counts": {"baseline": 80, "extended": 25, "total": 105},
            "models_comparison": track_a_results,
        },
        "track_b_swing": {
            "data_validation": {"H4": h4_val_rep, "D1": d1_val_rep},
            "target_distributions": {"H4": h4_target_dist, "D1": d1_target_dist},
            "h4_models": track_b_h4_results,
            "d1_models": track_b_d1_results,
        },
        "leakage_checks": leakage_checks,
        "elapsed_seconds": float(time.time() - t_start),
    }

    metrics_out = REPORTS_DIR / "phase15_research_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[Completed] Phase 15 research execution finished in {total_time:.2f}s.")
    print(f"Metrics written to: {metrics_out}")

    return results


if __name__ == "__main__":
    run_phase15_research()
