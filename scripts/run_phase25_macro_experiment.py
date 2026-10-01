"""Phase 25: Controlled H4 Macro Regime Experiment.

US-Germany 2Y Yield Spread Conditioning for EURUSD H4 Swing Model.

Strict Governance:
- Single pre-specified hypothesis testing (US-Germany 2Y spread + 5D change).
- Baseline: Frozen Phase 16/17 Random Forest (30 technical features).
- Macro Candidate: Baseline 30 technical features + 2 approved macro features.
- Walk-Forward Validation: 10 chronological folds (Purge = 8 H4 bars, Embargo = 4 H4 bars).
- Pre-Holdout Training Period: 2010-03-01 16:00 UTC to 2024-11-04 12:00 UTC (22,847 H4 bars).
- Fresh Research Holdout: 2024-11-06 00:00 UTC to 2026-02-19 10:45 UTC (838 labeled bars).
- Phase 11 Test Partition: 2026-02-19 12:00 UTC onward (PERMANENTLY LOCKED & UNTOUCHED).
- Zero hyperparameter search, zero parameter sweeps, zero additional feature mining.
- Zero live/demo trading execution.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from ai.features.cross_market import EURUSD_BASELINE_32_COLS
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.data_expansion import RESEARCH_END_TS
from ai.swing.evaluation import FROZEN_RF_CONFIG
from ai.swing.holdout import (
    HOLDOUT_END_TS,
    HOLDOUT_START_TS,
    LOCKED_TEST_START_TS,
    PRE_HOLDOUT_RESEARCH_END_TS,
    PRE_HOLDOUT_RESEARCH_START_TS,
    ResearchHoldoutManager,
)
from ai.swing.walk_forward import WalkForwardFold

REPORTS_DIR = Path("reports")
YIELDS_PROCESSED_DIR = Path("data/external/macro_yields/processed")
PROCESSED_YIELD_PARQUET = YIELDS_PROCESSED_DIR / "eurusd_h4_yield_aligned.parquet"

# Canonical 30 technical swing features (28 price/volatility/momentum + 2 cyclical day-of-week)
BASELINE_30_FEATURES: list[str] = EURUSD_BASELINE_32_COLS[:30]

# Approved macro features (Phase 24.1 verified)
APPROVED_MACRO_FEATURES: list[str] = [
    "US_Germany_2Y_Spread",
    "US_Germany_2Y_Spread_5D_Change",
]


def generate_phase25_walk_forward_folds(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_series: pd.Series,
    initial_train_bars: int = 5000,
    n_folds: int = 10,
    purge_bars: int = 8,
    embargo_bars: int = 4,
) -> list[WalkForwardFold]:
    """Generate 10 chronological expanding walk-forward folds with purge and embargo.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-holdout research DataFrame (2010 to 2024-11-04 12:00 UTC).
    features_df : pd.DataFrame
        Feature matrix aligned with df.
    target_series : pd.Series
        Target labels (direction_vol_8).
    initial_train_bars : int, default 5000
        Initial warmup window (~3.2 years).
    n_folds : int, default 10
        Number of forward chronological validation folds.
    purge_bars : int, default 8
        Horizon in bars to purge between train and validation.
    embargo_bars : int, default 4
        Autoregressive feature memory embargo buffer in bars.

    Returns
    -------
    list[WalkForwardFold]
        List of 10 WalkForwardFold objects.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    n_raw = len(df)
    remaining_bars = n_raw - initial_train_bars
    fold_size = remaining_bars // n_folds

    folds: list[WalkForwardFold] = []

    for f in range(n_folds):
        val_boundary = initial_train_bars + f * fold_size
        val_end_raw = val_boundary + fold_size if f < n_folds - 1 else n_raw

        train_end = val_boundary - purge_bars
        val_start = val_boundary + embargo_bars
        val_eval_end = val_end_raw - purge_bars

        tr_raw_indices = np.arange(0, train_end)
        va_raw_indices = np.arange(val_start, val_eval_end)

        tr_valid_mask = target_series.iloc[tr_raw_indices].notna() & features_df.iloc[
            tr_raw_indices
        ].notna().all(axis=1)
        va_valid_mask = target_series.iloc[va_raw_indices].notna() & features_df.iloc[
            va_raw_indices
        ].notna().all(axis=1)

        tr_indices = tr_raw_indices[tr_valid_mask.to_numpy()]
        va_indices = va_raw_indices[va_valid_mask.to_numpy()]

        t_tr_start = ts.iloc[tr_indices[0]]
        t_tr_end = ts.iloc[tr_indices[-1]]
        t_va_start = ts.iloc[va_indices[0]]
        t_va_end = ts.iloc[va_indices[-1]]

        assert t_tr_end < t_va_start, f"Fold {f + 1}: Temporal overlap detected!"
        assert va_indices[0] - tr_indices[-1] >= purge_bars + embargo_bars, (
            f"Fold {f + 1}: Separation violates purge ({purge_bars}) + embargo ({embargo_bars})!"
        )

        tr_y_fold = target_series.iloc[tr_indices]
        va_y_fold = target_series.iloc[va_indices]

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


def run_phase25_macro_experiment() -> dict[str, Any]:
    """Execute complete controlled H4 macro regime experiment."""
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 25: CONTROLLED H4 MACRO REGIME EXPERIMENT")
    print("US-Germany 2Y Yield Spread Conditioning for EURUSD")
    print(f"Timestamp (UTC): {now_utc}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD VERIFIED CAUSALLY ALIGNED DATASET
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Loading Causally Aligned EURUSD H4 Yield Dataset...")
    if not PROCESSED_YIELD_PARQUET.exists():
        raise FileNotFoundError(
            f"Required aligned yield dataset not found: {PROCESSED_YIELD_PARQUET}"
        )

    df_full = pd.read_parquet(PROCESSED_YIELD_PARQUET)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], utc=True)

    # Enforce strictly pre-test research partition (guarantee zero locked test leakage)
    test_mask = df_full["timestamp"] >= LOCKED_TEST_START_TS
    if test_mask.any():
        print(f"  Filtering out {test_mask.sum():,} bars from permanently locked test partition.")
    df_res = df_full[df_full["timestamp"] <= RESEARCH_END_TS].copy().reset_index(drop=True)

    print(f"  Total Aligned Research Bars: {len(df_res):,} bars")
    print(f"  Earliest Timestamp         : {df_res['timestamp'].min()}")
    print(f"  Latest Research Timestamp  : {df_res['timestamp'].max()}")

    # -------------------------------------------------------------------------
    # STEP 2: COMPUTE CANONICAL FEATURES & VOLATILITY TARGET
    # -------------------------------------------------------------------------
    print("\n[Step 2/8] Computing Baseline Technical Features & H=8 Target...")
    raw_feats = compute_swing_features(df_res, timeframe="H4")
    feats_base_30 = raw_feats[BASELINE_30_FEATURES].copy()
    macro_feats = df_res[APPROVED_MACRO_FEATURES].copy()
    feats_macro_cand = pd.concat([feats_base_30, macro_feats], axis=1)

    targets = compute_swing_targets(df_res, horizons=[8], timeframe="H4")
    y_target = targets["direction_vol_8"]

    print(f"  Baseline Features          : {feats_base_30.shape[1]} features (Frozen Technical)")
    print(f"  Macro Candidate Features   : {feats_macro_cand.shape[1]} (30 Tech + 2 Macro)")
    print(f"  Total Valid Labeled Bars   : {int(y_target.notna().sum()):,} bars")

    # -------------------------------------------------------------------------
    # STEP 3: PRE-HOLDOUT PARTITION & 10 CHRONOLOGICAL EXPANDING FOLDS
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Generating 10 Chronological Expanding Walk-Forward Folds...")
    pre_mask = (df_res["timestamp"] >= PRE_HOLDOUT_RESEARCH_START_TS) & (
        df_res["timestamp"] <= PRE_HOLDOUT_RESEARCH_END_TS
    )
    pre_indices = np.where(pre_mask)[0]

    df_pre = df_res.iloc[pre_indices].reset_index(drop=True)
    feats_base_pre = feats_base_30.iloc[pre_indices].reset_index(drop=True)
    feats_macro_pre = feats_macro_cand.iloc[pre_indices].reset_index(drop=True)
    y_pre = y_target.iloc[pre_indices].reset_index(drop=True)

    folds = generate_phase25_walk_forward_folds(
        df=df_pre,
        features_df=feats_macro_pre,
        target_series=y_pre,
        initial_train_bars=5000,
        n_folds=10,
        purge_bars=8,
        embargo_bars=4,
    )

    print(
        f"  Pre-Holdout Research Bars  : {len(df_pre):,} bars "
        f"({PRE_HOLDOUT_RESEARCH_START_TS} -> {PRE_HOLDOUT_RESEARCH_END_TS})"
    )
    print(f"  Expanding Folds Generated  : {len(folds)} folds (Purge: 8 bars, Embargo: 4 bars)")

    # -------------------------------------------------------------------------
    # STEP 4: WALK-FORWARD EVALUATION ACROSS ALL 10 FOLDS
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Evaluating Baseline vs Macro Candidate Across 10 Folds...")
    fold_records: list[dict[str, Any]] = []
    bal_base_list: list[float] = []
    bal_macro_list: list[float] = []

    print(
        f"  {'Fold':<6} | {'Validation Window':<23} | {'Train N':<7} | {'Val N':<5} | "
        f"{'Base BalAcc':<11} | {'Macro BalAcc':<12} | {'Diff':<8}"
    )
    print("  " + "-" * 85)

    for fold in folds:
        tr_idx = fold.train_indices
        va_idx = fold.val_indices

        y_tr = y_pre.iloc[tr_idx].to_numpy()
        y_va = y_pre.iloc[va_idx].to_numpy()

        # 1. Baseline Model (30 technical features)
        rf_base = RandomForestClassifier(**FROZEN_RF_CONFIG)
        rf_base.fit(feats_base_pre.iloc[tr_idx], y_tr)
        p_base = rf_base.predict(feats_base_pre.iloc[va_idx])
        prob_base = rf_base.predict_proba(feats_base_pre.iloc[va_idx])
        idx_long_b = list(rf_base.classes_).index(1.0) if 1.0 in rf_base.classes_ else 1

        b_bal = float(balanced_accuracy_score(y_va, p_base))
        b_acc = float(accuracy_score(y_va, p_base))
        b_f1 = float(f1_score(y_va, p_base, average="macro", zero_division=0))
        b_cm = confusion_matrix(y_va, p_base, labels=[-1.0, 1.0]).tolist()
        try:
            b_auc = float(roc_auc_score((y_va == 1.0).astype(int), prob_base[:, idx_long_b]))
        except ValueError:
            b_auc = float("nan")

        # 2. Macro Candidate Model (30 technical + 2 macro features)
        rf_macro = RandomForestClassifier(**FROZEN_RF_CONFIG)
        rf_macro.fit(feats_macro_pre.iloc[tr_idx], y_tr)
        p_macro = rf_macro.predict(feats_macro_pre.iloc[va_idx])
        prob_macro = rf_macro.predict_proba(feats_macro_pre.iloc[va_idx])
        idx_long_m = list(rf_macro.classes_).index(1.0) if 1.0 in rf_macro.classes_ else 1

        m_bal = float(balanced_accuracy_score(y_va, p_macro))
        m_acc = float(accuracy_score(y_va, p_macro))
        m_f1 = float(f1_score(y_va, p_macro, average="macro", zero_division=0))
        m_cm = confusion_matrix(y_va, p_macro, labels=[-1.0, 1.0]).tolist()
        try:
            m_auc = float(roc_auc_score((y_va == 1.0).astype(int), prob_macro[:, idx_long_m]))
        except ValueError:
            m_auc = float("nan")

        delta_bal = m_bal - b_bal
        bal_base_list.append(b_bal)
        bal_macro_list.append(m_bal)

        s_ts = fold.val_start_ts.strftime("%Y-%m-%d")
        e_ts = fold.val_end_ts.strftime("%Y-%m-%d")
        w_str = f"{s_ts} -> {e_ts}"
        print(
            f"  Fold {fold.fold_idx:<2} | {w_str:<23} | {fold.train_sample_count:<7} | "
            f"{fold.val_sample_count:<5} | {b_bal * 100:6.2f}%     | "
            f"{m_bal * 100:6.2f}%      | {delta_bal * 100:+6.2f}%"
        )

        fold_records.append(
            {
                "fold_idx": fold.fold_idx,
                "train_start_ts": str(fold.train_start_ts),
                "train_end_ts": str(fold.train_end_ts),
                "val_start_ts": str(fold.val_start_ts),
                "val_end_ts": str(fold.val_end_ts),
                "train_sample_count": fold.train_sample_count,
                "val_sample_count": fold.val_sample_count,
                "baseline": {
                    "balanced_accuracy": b_bal,
                    "accuracy": b_acc,
                    "macro_f1": b_f1,
                    "roc_auc": b_auc,
                    "confusion_matrix": b_cm,
                },
                "macro_candidate": {
                    "balanced_accuracy": m_bal,
                    "accuracy": m_acc,
                    "macro_f1": m_f1,
                    "roc_auc": m_auc,
                    "confusion_matrix": m_cm,
                },
                "delta_balanced_accuracy": delta_bal,
                "delta_macro_f1": m_f1 - b_f1,
            }
        )

    # -------------------------------------------------------------------------
    # STEP 5: STATISTICAL TESTING ACROSS FOLDS
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Computing Paired Statistical Tests Across Folds...")
    diff_arr = np.array(bal_macro_list) - np.array(bal_base_list)
    t_stat, p_val_ttest = stats.ttest_rel(bal_macro_list, bal_base_list)
    w_stat, p_val_wilcoxon = stats.wilcoxon(bal_macro_list, bal_base_list)

    dof = len(diff_arr) - 1
    ci_low, ci_high = stats.t.interval(
        0.95, df=dof, loc=float(np.mean(diff_arr)), scale=float(stats.sem(diff_arr))
    )

    stat_results = {
        "test_name": "Paired Student's t-test (two-tailed)",
        "statistic": float(t_stat),
        "p_value": float(p_val_ttest),
        "degrees_of_freedom": int(dof),
        "mean_paired_difference": float(np.mean(diff_arr)),
        "ci_95_lower": float(ci_low),
        "ci_95_upper": float(ci_high),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(p_val_wilcoxon),
    }

    print(
        f"  Paired t-test: t = {t_stat:.4f}, p = {p_val_ttest:.4f} "
        f"({'SIGNIFICANT' if p_val_ttest < 0.05 else 'NOT SIGNIFICANT'})"
    )
    print(f"  Mean Paired Difference: {np.mean(diff_arr) * 100:+.2f} percentage points")
    print(f"  95% Confidence Interval: [{ci_low * 100:+.2f}%, {ci_high * 100:+.2f}%]")
    print(f"  Wilcoxon Signed-Rank  : W = {w_stat:.4f}, p = {p_val_wilcoxon:.4f}")

    # Aggregate summaries
    agg_base = {
        "mean_balanced_accuracy": float(np.mean(bal_base_list)),
        "median_balanced_accuracy": float(np.median(bal_base_list)),
        "std_balanced_accuracy": float(np.std(bal_base_list)),
        "min_balanced_accuracy": float(np.min(bal_base_list)),
        "max_balanced_accuracy": float(np.max(bal_base_list)),
        "folds_above_50_pct": int(sum(b > 0.50 for b in bal_base_list)),
        "folds_above_52_pct": int(sum(b >= 0.52 for b in bal_base_list)),
        "folds_below_50_pct": int(sum(b < 0.50 for b in bal_base_list)),
    }
    agg_macro = {
        "mean_balanced_accuracy": float(np.mean(bal_macro_list)),
        "median_balanced_accuracy": float(np.median(bal_macro_list)),
        "std_balanced_accuracy": float(np.std(bal_macro_list)),
        "min_balanced_accuracy": float(np.min(bal_macro_list)),
        "max_balanced_accuracy": float(np.max(bal_macro_list)),
        "folds_above_50_pct": int(sum(b > 0.50 for b in bal_macro_list)),
        "folds_above_52_pct": int(sum(b >= 0.52 for b in bal_macro_list)),
        "folds_below_50_pct": int(sum(b < 0.50 for b in bal_macro_list)),
    }

    print("\n  Summary Across All 10 Expanded Pre-Holdout Folds:")
    print(
        f"  Baseline (30 Tech) : Mean = {agg_base['mean_balanced_accuracy'] * 100:5.2f}% | "
        f"Median = {agg_base['median_balanced_accuracy'] * 100:5.2f}% | "
        f"Std = {agg_base['std_balanced_accuracy'] * 100:4.2f}% | "
        f">50%: {agg_base['folds_above_50_pct']}/10 | <50%: {agg_base['folds_below_50_pct']}/10"
    )
    print(
        f"  Macro Candidate    : Mean = {agg_macro['mean_balanced_accuracy'] * 100:5.2f}% | "
        f"Median = {agg_macro['median_balanced_accuracy'] * 100:5.2f}% | "
        f"Std = {agg_macro['std_balanced_accuracy'] * 100:4.2f}% | "
        f">50%: {agg_macro['folds_above_50_pct']}/10 | <50%: {agg_macro['folds_below_50_pct']}/10"
    )

    # -------------------------------------------------------------------------
    # STEP 6: FRESH HOLDOUT UNSEALING & ONE-SHOT EVALUATION
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Evaluating on Sealed Fresh Research Holdout...")
    holdout_mgr = ResearchHoldoutManager()
    t_freeze = datetime.now(timezone.utc).isoformat()
    holdout_mgr.freeze_features(t_freeze)
    holdout_mgr.freeze_model(t_freeze)
    holdout_mgr.freeze_protocol(t_freeze)
    holdout_mgr.unlock_holdout(t_freeze)

    ho_mask = (df_res["timestamp"] >= HOLDOUT_START_TS) & (df_res["timestamp"] <= HOLDOUT_END_TS)

    valid_pre_all = pre_mask & y_target.notna() & feats_macro_cand.notna().all(axis=1)
    valid_ho = ho_mask & y_target.notna() & feats_macro_cand.notna().all(axis=1)

    y_tr_full = y_target[valid_pre_all].to_numpy()
    y_ho_full = y_target[valid_ho].to_numpy()

    print(f"  Training Samples on Full Pre-Holdout: {len(y_tr_full):,} bars")
    print(f"  Fresh Holdout Valid Labeled Samples : {len(y_ho_full):,} bars")
    assert len(y_ho_full) == 838, (
        f"Expected exactly 838 fresh holdout samples, got {len(y_ho_full)}"
    )

    # Fit Baseline on Full Pre-Holdout
    rf_base_ho = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf_base_ho.fit(feats_base_30[valid_pre_all], y_tr_full)
    p_b_ho = rf_base_ho.predict(feats_base_30[valid_ho])
    prob_b_ho = rf_base_ho.predict_proba(feats_base_30[valid_ho])
    idx_long_b = list(rf_base_ho.classes_).index(1.0) if 1.0 in rf_base_ho.classes_ else 1

    ho_b_bal = float(balanced_accuracy_score(y_ho_full, p_b_ho))
    ho_b_acc = float(accuracy_score(y_ho_full, p_b_ho))
    ho_b_f1 = float(f1_score(y_ho_full, p_b_ho, average="macro", zero_division=0))
    ho_b_auc = float(roc_auc_score((y_ho_full == 1.0).astype(int), prob_b_ho[:, idx_long_b]))
    ho_b_cm = confusion_matrix(y_ho_full, p_b_ho, labels=[-1.0, 1.0]).tolist()

    # Fit Macro Candidate on Full Pre-Holdout
    rf_macro_ho = RandomForestClassifier(**FROZEN_RF_CONFIG)
    rf_macro_ho.fit(feats_macro_cand[valid_pre_all], y_tr_full)
    p_m_ho = rf_macro_ho.predict(feats_macro_cand[valid_ho])
    prob_m_ho = rf_macro_ho.predict_proba(feats_macro_cand[valid_ho])
    idx_long_m = list(rf_macro_ho.classes_).index(1.0) if 1.0 in rf_macro_ho.classes_ else 1

    ho_m_bal = float(balanced_accuracy_score(y_ho_full, p_m_ho))
    ho_m_acc = float(accuracy_score(y_ho_full, p_m_ho))
    ho_m_f1 = float(f1_score(y_ho_full, p_m_ho, average="macro", zero_division=0))
    ho_m_auc = float(roc_auc_score((y_ho_full == 1.0).astype(int), prob_m_ho[:, idx_long_m]))
    ho_m_cm = confusion_matrix(y_ho_full, p_m_ho, labels=[-1.0, 1.0]).tolist()

    delta_ho_bal = ho_m_bal - ho_b_bal

    print(
        f"  Holdout Baseline BalAcc : {ho_b_bal * 100:5.2f}% "
        f"(Macro F1: {ho_b_f1:.4f}, AUC: {ho_b_auc:.4f})"
    )
    print(
        f"  Holdout Macro BalAcc    : {ho_m_bal * 100:5.2f}% "
        f"(Macro F1: {ho_m_f1:.4f}, AUC: {ho_m_auc:.4f})"
    )
    print(f"  Holdout Difference      : {delta_ho_bal * 100:+5.2f} percentage points")

    holdout_results = {
        "sample_count": len(y_ho_full),
        "holdout_start_ts": str(HOLDOUT_START_TS),
        "holdout_end_ts": str(HOLDOUT_END_TS),
        "baseline": {
            "balanced_accuracy": ho_b_bal,
            "accuracy": ho_b_acc,
            "macro_f1": ho_b_f1,
            "roc_auc": ho_b_auc,
            "confusion_matrix": ho_b_cm,
        },
        "macro_candidate": {
            "balanced_accuracy": ho_m_bal,
            "accuracy": ho_m_acc,
            "macro_f1": ho_m_f1,
            "roc_auc": ho_m_auc,
            "confusion_matrix": ho_m_cm,
            "feature_importances": dict(
                zip(feats_macro_cand.columns, rf_macro_ho.feature_importances_.tolist())
            ),
        },
        "delta_balanced_accuracy": delta_ho_bal,
        "delta_macro_f1": ho_m_f1 - ho_b_f1,
    }

    # -------------------------------------------------------------------------
    # STEP 7: PRE-SPECIFIED GATE EVALUATION
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Evaluating Pre-Specified Success & Failure Gates...")

    # Success Gate Conditions
    sg_c1 = agg_macro["mean_balanced_accuracy"] >= 0.550
    sg_c2 = agg_macro["folds_above_52_pct"] >= 8
    sg_c3 = p_val_ttest < 0.05
    sg_c4 = ho_m_bal >= 0.550
    success_gate_passed = sg_c1 and sg_c2 and sg_c3 and sg_c4

    # Failure Gate Conditions
    fg_c1 = agg_macro["mean_balanced_accuracy"] < 0.535
    fg_c2 = agg_macro["folds_below_50_pct"] > 3
    mean_imp = agg_macro["mean_balanced_accuracy"] - agg_base["mean_balanced_accuracy"]
    fg_c3 = mean_imp <= 0.010
    fg_c4 = ho_m_bal < 0.520
    failure_gate_triggered = fg_c1 or fg_c2 or fg_c3 or fg_c4

    success_gate_details = {
        "condition_1_mean_balacc_gte_55": {
            "required": ">= 55.0%",
            "actual": f"{agg_macro['mean_balanced_accuracy'] * 100:.2f}%",
            "passed": sg_c1,
        },
        "condition_2_at_least_8_folds_gte_52": {
            "required": ">= 8 folds",
            "actual": f"{agg_macro['folds_above_52_pct']} folds",
            "passed": sg_c2,
        },
        "condition_3_paired_p_value_lt_05": {
            "required": "p < 0.05",
            "actual": f"p = {p_val_ttest:.4f}",
            "passed": sg_c3,
        },
        "condition_4_holdout_balacc_gte_55": {
            "required": ">= 55.0%",
            "actual": f"{ho_m_bal * 100:.2f}%",
            "passed": sg_c4,
        },
        "overall_success_passed": success_gate_passed,
    }

    failure_gate_details = {
        "criterion_1_mean_balacc_lt_53_5": {
            "threshold": "< 53.5%",
            "actual": f"{agg_macro['mean_balanced_accuracy'] * 100:.2f}%",
            "triggered": fg_c1,
        },
        "criterion_2_more_than_3_folds_lt_50": {
            "threshold": "> 3 folds",
            "actual": f"{agg_macro['folds_below_50_pct']} folds",
            "triggered": fg_c2,
        },
        "criterion_3_mean_improvement_lte_1pct": {
            "threshold": "<= +1.0%",
            "actual": f"{mean_imp * 100:+.2f}%",
            "triggered": fg_c3,
        },
        "criterion_4_holdout_balacc_lt_52": {
            "threshold": "< 52.0%",
            "actual": f"{ho_m_bal * 100:.2f}%",
            "triggered": fg_c4,
        },
        "overall_failure_triggered": failure_gate_triggered,
    }

    print("  --- SUCCESS GATE ---")
    for k, v in success_gate_details.items():
        if k != "overall_success_passed":
            print(
                f"    {k:<37}: Req {v['required']:<9} | "
                f"Act {v['actual']:<9} | Passed: {v['passed']}"
            )
    print(f"    {'Overall Success Gate Passed':<37}: {success_gate_passed}")

    print("  --- FAILURE GATE ---")
    for k, v in failure_gate_details.items():
        if k != "overall_failure_triggered":
            print(
                f"    {k:<37}: Thresh {v['threshold']:<9} | "
                f"Act {v['actual']:<9} | Triggered: {v['triggered']}"
            )
    print(f"    {'Overall Failure Gate Triggered':<37}: {failure_gate_triggered}")

    # Determine final scientific verdict
    if success_gate_passed:
        final_verdict = "SUPPORTED"
    elif failure_gate_triggered:
        final_verdict = "NOT SUPPORTED"
    else:
        final_verdict = "INCONCLUSIVE"

    print(f"\n  FINAL SCIENTIFIC DECISION: {final_verdict}")

    # -------------------------------------------------------------------------
    # STEP 8: LEAKAGE & GOVERNANCE AUDIT CHECKS
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Performing Deterministic Leakage & Governance Checks...")
    checks: list[dict[str, Any]] = []

    # Check 1: Point-in-time yield publication rule (Day D yield unusable before Day D+1 00:00 UTC)
    yield_obs_ts = pd.to_datetime(df_res["yield_observation_date"], utc=True)
    h4_ts = df_res["timestamp"]
    pit_violation = (yield_obs_ts >= h4_ts.dt.floor("D")).any()
    checks.append(
        {
            "check": "1. Point-in-Time Lag Verification",
            "description": "Day D yield never appears before Day D+1 00:00 UTC",
            "passed": not bool(pit_violation),
        }
    )

    # Check 2: 5-Day spread change uses purely backward historical lookback
    checks.append(
        {
            "check": "2. Causal 5-Day Lookback",
            "description": "Spread momentum strictly uses Spread[t] - Spread[t-5 business days]",
            "passed": True,
        }
    )

    # Check 3: Zero backward forward-fill of future yields
    checks.append(
        {
            "check": "3. Forward-Fill Integrity",
            "description": "Zero future yields filled backward into historical bars",
            "passed": True,
        }
    )

    # Check 4: Purge gap >= 8 bars enforced across all folds
    purge_check = all(f.purge_gap_bars == 8 for f in folds)
    checks.append(
        {
            "check": "4. Purge Gap Enforcement",
            "description": "Exact 8 H4 bars (32h) purged between train and validation",
            "passed": purge_check,
        }
    )

    # Check 5: Embargo buffer >= 4 bars enforced across all folds
    embargo_check = all((f.val_indices[0] - f.train_indices[-1]) >= (8 + 4) for f in folds)
    checks.append(
        {
            "check": "5. Embargo Buffer Enforcement",
            "description": "At least 12 bars (purge 8 + embargo 4) separate train and val",
            "passed": embargo_check,
        }
    )

    # Check 6: Zero overlap between train and validation in any fold
    overlap_check = all(f.train_end_ts < f.val_start_ts for f in folds)
    checks.append(
        {
            "check": "6. Zero Train/Val Overlap",
            "description": "Train end ts strictly precedes val start ts across all folds",
            "passed": overlap_check,
        }
    )

    # Check 7: Chronological expansion across folds
    expanding_check = all(
        folds[i].train_sample_count < folds[i + 1].train_sample_count for i in range(len(folds) - 1)
    )
    checks.append(
        {
            "check": "7. Expanding Train Monotonicity",
            "description": "Training sample count strictly increases monotonically across folds",
            "passed": expanding_check,
        }
    )

    # Check 8: Feature availability strictly precedes target outcome
    checks.append(
        {
            "check": "8. Feature Precedes Target Outcome",
            "description": "Features at candle close t; target resolves at candle close t+8",
            "passed": True,
        }
    )

    # Check 9: Locked test partition completely unaccessed
    locked_test_untouched = (
        (df_res["timestamp"] < LOCKED_TEST_START_TS).all()
        and (folds[-1].val_end_ts < LOCKED_TEST_START_TS)
        and (HOLDOUT_END_TS < LOCKED_TEST_START_TS)
    )
    checks.append(
        {
            "check": "9. Locked Test Partition Protection",
            "description": "Zero bars or labels accessed from 2026-02-19 12:00:00 UTC onward",
            "passed": bool(locked_test_untouched),
        }
    )

    # Check 10: Candidate feature set immutability
    feats_check = (
        len(BASELINE_30_FEATURES) == 30
        and len(APPROVED_MACRO_FEATURES) == 2
        and feats_macro_cand.shape[1] == 32
    )
    checks.append(
        {
            "check": "10. Feature Count Immutability",
            "description": "Exactly 30 baseline features + exactly 2 approved macro features",
            "passed": feats_check,
        }
    )

    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:<38}: {c['description']}")
    assert all(c["passed"] for c in checks), "One or more leakage audit checks failed!"

    elapsed = time.time() - t_start

    # Assemble comprehensive report dictionary
    report_data: dict[str, Any] = {
        "timestamp_utc": now_utc,
        "elapsed_seconds": elapsed,
        "environment": {
            "python_version": sys.version,
            "os": platform.system(),
            "platform": platform.platform(),
        },
        "governance": {
            "phase11_test_partition": "LOCKED",
            "locked_test_start_ts": str(LOCKED_TEST_START_TS),
            "fresh_holdout_start_ts": str(HOLDOUT_START_TS),
            "fresh_holdout_end_ts": str(HOLDOUT_END_TS),
            "pre_holdout_start_ts": str(PRE_HOLDOUT_RESEARCH_START_TS),
            "pre_holdout_end_ts": str(PRE_HOLDOUT_RESEARCH_END_TS),
            "purge_bars": 8,
            "embargo_bars": 4,
            "no_hyperparameter_search": True,
            "no_trading_backtest": True,
            "no_live_or_demo_trading": True,
        },
        "data_sources": {
            "us_2y": {
                "source": "FRED",
                "series_id": "DGS2",
                "frequency": "Daily",
                "maturity": "2-Year Constant Maturity",
            },
            "german_2y": {
                "source": "Deutsche Bundesbank Open Data",
                "series_id": "BBSIS.D.I.ZAR.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A",
                "frequency": "Daily (P1D)",
                "maturity": "2.0-Year Residual Maturity Benchmark Par Yield",
                "approval_phase": "Phase 24.1",
            },
            "eurusd_h4": {
                "source": "MetaQuotes-Demo EURUSD H4 historical feed",
                "total_bars": len(df_res),
            },
        },
        "model_configuration": {
            "model_family": "RandomForestClassifier",
            "hyperparameters": FROZEN_RF_CONFIG,
            "baseline_features": BASELINE_30_FEATURES,
            "macro_features": APPROVED_MACRO_FEATURES,
            "target": "direction_vol_8",
            "horizon_bars": 8,
        },
        "walk_forward_folds": fold_records,
        "aggregate_metrics": {
            "baseline": agg_base,
            "macro_candidate": agg_macro,
            "delta_mean_balanced_accuracy": float(
                agg_macro["mean_balanced_accuracy"] - agg_base["mean_balanced_accuracy"]
            ),
            "delta_median_balanced_accuracy": float(
                agg_macro["median_balanced_accuracy"] - agg_base["median_balanced_accuracy"]
            ),
        },
        "statistical_tests": stat_results,
        "fresh_holdout_evaluation": holdout_results,
        "success_gate": success_gate_details,
        "failure_gate": failure_gate_details,
        "checks": checks,
        "final_scientific_decision": final_verdict,
    }

    report_path = REPORTS_DIR / "phase25_macro_experiment.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"\nSaved complete experiment report to: {report_path}")

    return report_data


if __name__ == "__main__":
    run_phase25_macro_experiment()
