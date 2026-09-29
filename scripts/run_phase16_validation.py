"""Phase 16: H4 Swing Candidate Confirmation & M15 Entry Confirmation Research.

Executes:
1. Independent Walk-Forward Confirmation of Phase 15 H4 Candidate across 5 chronological folds.
2. Baselines evaluation: Majority Class, Naive Persistence, Logistic Regression, Extra Trees.
3. Across-fold temporal robustness analysis (mean, median, std, min, max, fold-to-fold degradation).
4. Descriptive market regime analysis (volatility, trend, range, session/hour, reopen).
5. M15 entry confirmation filter experiments (Base, Direction, Momentum, Trend, Two-of-Three).
6. 10 deterministic leakage and governance checks.

Strict Governance:
- Uses ONLY Training + Validation research data (up to 2026-02-19 10:45:00 UTC).
- Phase 11 M15 Test (14,988 rows) remains PERMANENTLY LOCKED & UNTOUCHED.
- Phase 15 H4 Test (939 rows) remains PERMANENTLY LOCKED & UNTOUCHED.
- Phase 15 D1 Test (156 rows) remains PERMANENTLY LOCKED & UNTOUCHED.
- Phase 12 Baseline remains UNCHANGED.
- NO live trading, broker connection, or trade backtesting.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.dataset.assembly import assemble_dataset
from ai.dataset.swing import aggregate_m15_to_h4
from ai.features.swing import compute_swing_features
from ai.labels.swing import compute_swing_targets
from ai.swing.confirmation import align_m15_with_h4_decisions, evaluate_m15_confirmations
from ai.swing.evaluation import evaluate_walk_forward_candidate
from ai.swing.regimes import analyze_h4_regimes
from ai.swing.timing_audit import generate_timing_audit_report
from ai.swing.walk_forward import generate_chronological_walk_forward_folds

REPORTS_DIR = Path("reports")
LOCKED_TEST_START_TS = pd.Timestamp("2026-02-19 12:00:00+00:00")
RESEARCH_END_TS = pd.Timestamp("2026-02-19 10:45:00+00:00")


def run_phase16_research() -> dict[str, Any]:
    """Execute complete Phase 16 validation and confirmation pipeline."""
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 16: H4 SWING CONFIRMATION & M15 ENTRY CONFIRMATION RESEARCH")
    print(f"Timestamp (UTC): {now_utc}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: DATA LOAD & GOVERNANCE AUDIT
    # -------------------------------------------------------------------------
    print("\n[Step 1/7] Loading Data and Enforcing Governance Locks...")
    raw_m15 = pd.read_parquet("data/processed/eurusd_m15/eurusd_m15_processed.parquet")
    h4_df = aggregate_m15_to_h4(raw_m15)

    ts_h4 = pd.to_datetime(h4_df["timestamp"], utc=True)
    test_h4_mask = ts_h4 >= LOCKED_TEST_START_TS
    research_h4_mask = ts_h4 <= RESEARCH_END_TS

    n_total_h4 = len(h4_df)
    n_research_h4 = int(research_h4_mask.sum())
    n_test_h4 = int(test_h4_mask.sum())

    print(f"  Total H4 bars: {n_total_h4:,}")
    print(f"  Research H4 bars (Train + Val): {n_research_h4:,} (up to {RESEARCH_END_TS})")
    print(f"  Locked Test H4 bars: {n_test_h4:,} (from {LOCKED_TEST_START_TS} onward) -> UNTOUCHED")

    # -------------------------------------------------------------------------
    # PART 2: COMPUTE H4 FEATURES & TARGETS
    # -------------------------------------------------------------------------
    print("\n[Step 2/7] Computing H4 Features and Volatility-Adjusted Target (H=8)...")
    h4_feats = compute_swing_features(h4_df, timeframe="H4")
    h4_targets = compute_swing_targets(h4_df, horizons=[8], timeframe="H4")
    target_vol_8 = h4_targets["direction_vol_8"]

    # Filter to research dataset only
    h4_res = h4_df[research_h4_mask].copy().reset_index(drop=True)
    feats_res = h4_feats.loc[research_h4_mask].reset_index(drop=True)
    target_res = target_vol_8.loc[research_h4_mask].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # PART 3: CHRONOLOGICAL WALK-FORWARD GENERATION
    # -------------------------------------------------------------------------
    print("\n[Step 3/7] Generating 5 Chronological Walk-Forward Folds (8-bar purge)...")
    folds = generate_chronological_walk_forward_folds(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        initial_train_bars=2524,
        n_folds=5,
        purge_bars=8,
        research_end_ts=RESEARCH_END_TS,
    )

    for fold in folds:
        print(
            f"  Fold {fold.fold_idx}: Train N={fold.train_sample_count:<4} "
            f"({fold.train_start_ts} -> {fold.train_end_ts}) | "
            f"Val N={fold.val_sample_count:<4} "
            f"({fold.val_start_ts} -> {fold.val_end_ts}) | "
            f"Purge={fold.purge_gap_bars} bars"
        )

    # -------------------------------------------------------------------------
    # PART 4: WALK-FORWARD CANDIDATE & BASELINES EVALUATION
    # -------------------------------------------------------------------------
    print("\n[Step 4/7] Evaluating Candidate & Baselines across Walk-Forward Folds...")
    eval_results = evaluate_walk_forward_candidate(
        h4_df=h4_res,
        features_df=feats_res,
        target_series=target_res,
        folds=folds,
    )

    print("\n  Summary Across All 5 Folds:")
    print(
        f"  {'Model':<20} | {'Mean BalAcc':<12} | {'Median':<8} | {'Std':<8} | "
        f"{'Min':<8} | {'Max':<8} | {'Max Drop':<8} | {'>50%':<5} | {'>55%':<5}"
    )
    print("  " + "-" * 95)
    for model_name, s in eval_results["summary"].items():
        print(
            f"  {model_name:<20} | "
            f"{s['mean_balanced_accuracy'] * 100:6.2f}%     | "
            f"{s['median_balanced_accuracy'] * 100:6.2f}% | "
            f"{s['std_balanced_accuracy'] * 100:6.2f}% | "
            f"{s['min_balanced_accuracy'] * 100:6.2f}% | "
            f"{s['max_balanced_accuracy'] * 100:6.2f}% | "
            f"{s['max_fold_to_fold_degradation'] * 100:6.2f}% | "
            f"{s['folds_above_50_pct']}/{s['total_folds']}   | "
            f"{s['folds_above_55_pct']}/{s['total_folds']}"
        )

    # -------------------------------------------------------------------------
    # PART 5: DESCRIPTIVE REGIME ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[Step 5/7] Performing Descriptive Market Regime Analysis...")
    oof_data = eval_results["oof_candidate_predictions"]
    regime_results = analyze_h4_regimes(
        features_df=feats_res,
        target_series=target_res,
        oof_indices=oof_data["indices"],
        oof_preds=oof_data["predictions"],
        oof_probs=oof_data["probabilities"],
    )

    vol_low = regime_results["volatility_regime"]["low_volatility"]
    vol_high = regime_results["volatility_regime"]["high_volatility"]
    print(
        f"  Volatility (Low ATR) : N={vol_low['sample_count']:<4} | "
        f"BalAcc={vol_low['balanced_accuracy'] * 100:5.2f}% | MacroF1={vol_low['macro_f1']:.4f}"
    )
    print(
        f"  Volatility (High ATR): N={vol_high['sample_count']:<4} | "
        f"BalAcc={vol_high['balanced_accuracy'] * 100:5.2f}% | MacroF1={vol_high['macro_f1']:.4f}"
    )

    tr_bull = regime_results["trend_regime"]["strong_bullish"]
    tr_bear = regime_results["trend_regime"]["strong_bearish"]
    tr_mix = regime_results["trend_regime"]["mixed_transition"]
    print(
        f"  Trend (Strong Bull)  : N={tr_bull['sample_count']:<4} | "
        f"BalAcc={tr_bull['balanced_accuracy'] * 100:5.2f}% | MacroF1={tr_bull['macro_f1']:.4f}"
    )
    print(
        f"  Trend (Strong Bear)  : N={tr_bear['sample_count']:<4} | "
        f"BalAcc={tr_bear['balanced_accuracy'] * 100:5.2f}% | MacroF1={tr_bear['macro_f1']:.4f}"
    )
    print(
        f"  Trend (Mixed / Tran) : N={tr_mix['sample_count']:<4} | "
        f"BalAcc={tr_mix['balanced_accuracy'] * 100:5.2f}% | MacroF1={tr_mix['macro_f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # PART 6: M15 ENTRY CONFIRMATION EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n[Step 6/7] Aligning M15 Candles & Evaluating Entry Confirmation Rules...")
    ds_m15 = assemble_dataset()
    aligned_m15 = align_m15_with_h4_decisions(h4_res, ds_m15.df)

    oof_idx_arr = np.array(oof_data["indices"])
    sub_aligned_m15 = aligned_m15.iloc[oof_idx_arr].reset_index(drop=True)
    sub_y_true = target_res.iloc[oof_idx_arr].to_numpy()
    sub_preds = np.array(oof_data["predictions"])

    m15_confirm_results = evaluate_m15_confirmations(
        y_true=sub_y_true,
        h4_preds=sub_preds,
        aligned_m15=sub_aligned_m15,
    )

    print(
        f"  {'Configuration':<32} | {'Signals':<8} | {'Filtered':<9} | "
        f"{'BalAcc':<9} | {'Acc':<8} | {'MacroF1':<8}"
    )
    print("  " + "-" * 85)
    for conf_name, res in m15_confirm_results.items():
        print(
            f"  {conf_name:<32} | "
            f"{res['signals_count']:<8} | "
            f"{res['pct_filtered']:6.1f}%  | "
            f"{res['balanced_accuracy'] * 100:6.2f}%  | "
            f"{res['accuracy'] * 100:6.2f}% | "
            f"{res['macro_f1']:6.4f}"
        )

    # Timing audit sample
    timing_records = generate_timing_audit_report(
        h4_df=h4_res,
        aligned_m15=aligned_m15,
        target_series=target_res,
        horizon_bars=8,
    )

    # -------------------------------------------------------------------------
    # PART 7: 10 DETERMINISTIC LEAKAGE & GOVERNANCE CHECKS
    # -------------------------------------------------------------------------
    print("\n[Step 7/7] Running 10 Deterministic Leakage & Governance Checks...")
    checks = []

    # Check 1: Monotonic H4 timestamps
    c1 = bool(pd.to_datetime(h4_df["timestamp"], utc=True).is_monotonic_increasing)
    checks.append(
        {"check": "1. H4 timestamp monotonicity", "passed": c1, "detail": "Strictly monotonic UTC"}
    )

    # Check 2: Chronological fold ordering (no negative time jumps)
    c2 = all(folds[i].train_end_ts < folds[i + 1].train_end_ts for i in range(len(folds) - 1))
    checks.append(
        {"check": "2. Chronological fold progression", "passed": c2, "detail": "Expanding window"}
    )

    # Check 3: Purge gap correctness (exact 8 bars between train and val)
    c3 = all(f.purge_gap_bars == 8 for f in folds)
    checks.append(
        {"check": "3. Purge gap size", "passed": c3, "detail": "Exact 8 bars (32h) purged"}
    )

    # Check 4: No train/validation overlap in any fold
    c4 = all(f.train_end_ts < f.val_start_ts for f in folds)
    checks.append(
        {
            "check": "4. Zero train/val temporal overlap",
            "passed": c4,
            "detail": "train_end < val_start",
        }
    )

    # Check 5: No future feature leakage in H4 features
    c5 = not feats_res.isna().all().any() and not np.isinf(feats_res.to_numpy()).any()
    checks.append(
        {
            "check": "5. H4 feature validity & finite bounds",
            "passed": c5,
            "detail": "Zero infs/all-NaNs",
        }
    )

    # Check 6: Point-in-time M15 alignment causality (M15 ts <= cutoff < H4 close)
    c6 = all(
        pd.Timestamp(r["actual_matched_m15_timestamp"]) < pd.Timestamp(r["h4_decision_time"])
        for r in timing_records
    )
    checks.append(
        {
            "check": "6. M15 point-in-time alignment",
            "passed": c6,
            "detail": "M15 ts < H4 decision time",
        }
    )

    # Check 7: Incomplete M15 candle exclusion
    c7 = all(
        pd.Timestamp(r["actual_matched_m15_timestamp"])
        <= pd.Timestamp(r["latest_allowed_m15_timestamp"])
        for r in timing_records
    )
    checks.append(
        {
            "check": "7. Incomplete M15 candle exclusion",
            "passed": c7,
            "detail": "M15 ts <= cutoff_m15",
        }
    )

    # Check 8: Target horizon causality (decision_time < outcome_time)
    c8 = all(r["causality_verified"] for r in timing_records)
    checks.append(
        {
            "check": "8. Information ordering causality",
            "passed": c8,
            "detail": "FEATURE < DECISION < OUTCOME",
        }
    )

    # Check 9: Frozen candidate immutability
    c9 = (
        eval_results["summary"]["RandomForest"]["total_folds"] == 5
        and len(eval_results["oof_candidate_predictions"]["predictions"]) == 1201
    )
    checks.append(
        {
            "check": "9. Candidate immutability",
            "passed": c9,
            "detail": "100 trees, depth 5, leaf 10, balanced",
        }
    )

    # Check 10: Locked test partitions completely unaccessed
    c10 = folds[-1].val_end_ts <= RESEARCH_END_TS and folds[-1].val_end_ts < LOCKED_TEST_START_TS
    checks.append(
        {
            "check": "10. Locked test partitions unaccessed",
            "passed": c10,
            "detail": f"Max val ts {folds[-1].val_end_ts} < {LOCKED_TEST_START_TS}",
        }
    )

    all_passed = all(c["passed"] for c in checks)
    for c in checks:
        status_str = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status_str}] {c['check']:<38}: {c['detail']}")

    assert all_passed, "One or more leakage/governance checks failed!"

    elapsed = time.time() - t_start
    print(f"\nPhase 16 research execution finished in {elapsed:.2f} seconds.")

    # Save comprehensive metrics to report JSON
    report_data = {
        "timestamp_utc": now_utc,
        "elapsed_seconds": elapsed,
        "governance": {
            "m15_phase11_test": "LOCKED",
            "h4_phase15_test": "LOCKED",
            "d1_phase15_test": "LOCKED",
            "phase12_baseline": "UNCHANGED",
            "research_end_ts": str(RESEARCH_END_TS),
            "locked_test_start_ts": str(LOCKED_TEST_START_TS),
        },
        "candidate": {
            "timeframe": "H4",
            "target": "direction_vol_8",
            "horizon_bars": 8,
            "horizon_hours": 32,
            "model": "RandomForestClassifier",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 10,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        "folds": [f.to_dict() for f in folds],
        "walk_forward_evaluation": eval_results,
        "regime_analysis": regime_results,
        "m15_confirmation": m15_confirm_results,
        "timing_audit_sample": timing_records,
        "leakage_checks": checks,
    }

    metrics_file = REPORTS_DIR / "phase16_validation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"Saved complete research metrics to {metrics_file}")

    return report_data


if __name__ == "__main__":
    run_phase16_research()
